#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <nvToolsExt.h>
#include "energy_monitor.h"

// Error check macro for CUDA API calls
#define CHECK_CUDA_ERROR(call) {                                        \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error in call at file '%s' line %d: %s\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

// Comparator for qsort to sort by absolute values (descending)
int compare_abs_desc(const void *a, const void *b) {
    float abs_a = fabsf(*(const float*)a);
    float abs_b = fabsf(*(const float*)b);
    return (abs_b > abs_a) - (abs_b < abs_a);  // Descending order
}

// Function to perform top-k sparsification using qsort for efficiency
void compress_gradients_topk(float* gradients, int num_elements, float k) {
    int k_elements = (int)(k * num_elements);  // Retain the top k% gradients
    if (k_elements > 0) {
        float *abs_gradients = (float*)malloc(num_elements * sizeof(float));
        if (!abs_gradients) {
            fprintf(stderr, "Memory allocation error\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < num_elements; i++) {
            abs_gradients[i] = gradients[i];
        }

        qsort(abs_gradients, num_elements, sizeof(float), compare_abs_desc);
        float threshold = fabsf(abs_gradients[k_elements - 1]);

        for (int i = 0; i < num_elements; i++) {
            if (fabsf(gradients[i]) < threshold) {
                gradients[i] = 0.0f;
            }
        }

        free(abs_gradients);
    }
}

// CUDA kernel to initialize GPU memory with random floats
__global__ void init_random(float* d_data, int num_elements, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        d_data[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

// Thread-safe queue for passing data between threads
template<typename T>
class SafeQueue {
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;

public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(std::move(item));
        cond.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] { return !queue.empty(); });
        T item = std::move(queue.front());
        queue.pop();
        return item;
    }
};

// Structure to hold data for each iteration
struct IterationData {
    float* h_data;
    size_t size;
    int iteration;
};

// Compression thread function
void compress_thread_func(SafeQueue<IterationData>& queue) {
    while (true) {
        IterationData data = queue.pop();
        if (data.iteration == -1) break;  // Exit signal

        char compress_marker[64];
        sprintf(compress_marker, "CPU Compression Iteration %d", data.iteration);
        nvtxRangePush(compress_marker);

        clock_t start_time = clock();
        compress_gradients_topk(data.h_data, data.size / sizeof(float), 0.1);
        clock_t end_time = clock();
        double compression_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("Time taken to compress data (iteration %d): %f seconds\n", 
               data.iteration, compression_time);

        free(data.h_data);
        nvtxRangePop();
    }
}

int main() {
    // NVTX event attributes setup
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.colorType = NVTX_COLOR_ARGB;

    // Define colors for different operations
    const uint32_t COLOR_GENERATE = 0xFF00FF00;  // Green
    const uint32_t COLOR_COPY = 0xFFFF0000;      // Red
    const uint32_t COLOR_COMPRESS = 0xFF0000FF;  // Blue

    nvtxRangePush("Total Execution");
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // nvtxRangePush("Energy Monitoring Setup");
    // start_energy_monitoring();
    // nvtxRangePop();

    const size_t data_size = 200 * 1024 * 1024;  // 200MB
    const int num_elements = data_size / sizeof(float);
    const int num_iterations = 10;

    nvtxRangePush("CUDA Stream Creation");
    cudaStream_t stream1, stream2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
    nvtxRangePop();

    nvtxRangePush("GPU Memory Allocation");
    float *d_data1, *d_data2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data1, data_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data2, data_size));
    nvtxRangePop();

    SafeQueue<IterationData> compression_queue;
    std::thread compression_thread(compress_thread_func, std::ref(compression_queue));

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long seed = time(NULL);

    nvtxRangePush("Processing Loop");
    for (int i = 0; i < num_iterations; i++) {
        char iteration_marker[64];
        sprintf(iteration_marker, "Iteration %d", i);
        nvtxRangePush(iteration_marker);

        float* d_current = (i % 2 == 0) ? d_data1 : d_data2;
        float* d_previous = (i % 2 == 0) ? d_data2 : d_data1;
        cudaStream_t current_stream = (i % 2 == 0) ? stream1 : stream2;
        cudaStream_t previous_stream = (i % 2 == 0) ? stream2 : stream1;

        // Data generation
        eventAttrib.color = COLOR_GENERATE;
        eventAttrib.message.ascii = "Data Generation";
        nvtxRangePushEx(&eventAttrib);
        
        clock_t start_gen = clock();
        init_random<<<num_blocks, block_size, 0, current_stream>>>(
            d_current, num_elements, seed + i);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(current_stream));
        clock_t end_gen = clock();
        double gen_time = (double)(end_gen - start_gen) / CLOCKS_PER_SEC;
        printf("Time taken to generate data (iteration %d): %f seconds\n", i, gen_time);
        nvtxRangePop();

        if (i > 0) {
            // Data copy
            eventAttrib.color = COLOR_COPY;
            eventAttrib.message.ascii = "GPU to CPU Copy";
            nvtxRangePushEx(&eventAttrib);

            float* h_data = (float*)malloc(data_size);
            clock_t start_copy = clock();
            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_previous, data_size, 
                                           cudaMemcpyDeviceToHost, previous_stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(previous_stream));
            clock_t end_copy = clock();
            double copy_time = (double)(end_copy - start_copy) / CLOCKS_PER_SEC;
            printf("Time taken to copy data (iteration %d): %f seconds\n", i-1, copy_time);
            nvtxRangePop();

            compression_queue.push({h_data, data_size, i - 1});
        }
        nvtxRangePop(); // End iteration marker
    }
    nvtxRangePop(); // End processing loop

    // Process final iteration
    nvtxRangePush("Final Iteration Processing");
    {
        eventAttrib.color = COLOR_COPY;
        eventAttrib.message.ascii = "Final GPU to CPU Copy";
        nvtxRangePushEx(&eventAttrib);

        float* h_data = (float*)malloc(data_size);
        clock_t start_copy = clock();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, 
            (num_iterations % 2 == 0) ? d_data2 : d_data1,
            data_size, cudaMemcpyDeviceToHost,
            (num_iterations % 2 == 0) ? stream2 : stream1));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(
            (num_iterations % 2 == 0) ? stream2 : stream1));
        clock_t end_copy = clock();
        double copy_time = (double)(end_copy - start_copy) / CLOCKS_PER_SEC;
        printf("Time taken to copy final data: %f seconds\n", copy_time);
        nvtxRangePop();

        compression_queue.push({h_data, data_size, num_iterations - 1});
    }
    nvtxRangePop();

    // Cleanup
    nvtxRangePush("Cleanup");
    compression_queue.push({nullptr, 0, -1});
    compression_thread.join();
    CHECK_CUDA_ERROR(cudaFree(d_data1));
    CHECK_CUDA_ERROR(cudaFree(d_data2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    nvtxRangePop();

    // Final reporting
    nvtxRangePush("Results Reporting");
    // stop_energy_monitoring();
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>
                         (total_end_time - total_start_time).count();

    printf("\nFinal Results:\n");
    printf("Total time taken: %ld milliseconds\n", total_duration);
    // printf("CPU Energy: %.6f J\n", cpu_energy);
    // printf("GPU 0 Energy: %.6f J\n", gpu_energy[0]);
    // printf("GPU 1 Energy: %.6f J\n", gpu_energy[1]);
    // printf("Total GPU Energy: %.6f J\n", gpu_energy[0] + gpu_energy[1]);
    // printf("Total Energy: %.6f J\n", cpu_energy + gpu_energy[0] + gpu_energy[1]);
    nvtxRangePop();

    nvtxRangePop(); // End Total Execution
    return 0;
}