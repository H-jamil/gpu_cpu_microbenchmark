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
#include <chrono>  // Add this include for high-resolution timing


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
        // Create a copy of gradients for sorting based on absolute values
        float *abs_gradients = (float*)malloc(num_elements * sizeof(float));
        if (!abs_gradients) {
            fprintf(stderr, "Memory allocation error\n");
            exit(EXIT_FAILURE);
        }

        // Copy gradients for sorting
        for (int i = 0; i < num_elements; i++) {
            abs_gradients[i] = gradients[i];
        }

        // Sort gradients based on absolute values (in descending order)
        qsort(abs_gradients, num_elements, sizeof(float), compare_abs_desc);

        // Determine the threshold (the k-th largest absolute value)
        float threshold = fabsf(abs_gradients[k_elements - 1]);

        // Zero out gradients below the threshold
        for (int i = 0; i < num_elements; i++) {
            if (fabsf(gradients[i]) < threshold) {
                gradients[i] = 0.0f;
            }
        }

        // Free the temporary array
        free(abs_gradients);
    }
}

// CUDA kernel to initialize GPU memory with random floats using curand
__global__ void init_random(float* d_data, int num_elements, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Initialize the random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random float between -1 and 1
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

// Function to compress data (to be run in a separate thread)
void compress_thread_func(SafeQueue<IterationData>& queue) {
    while (true) {
        IterationData data = queue.pop();
        if (data.iteration == -1) break;  // Signal to exit

        clock_t start_time = clock();
        compress_gradients_topk(data.h_data, data.size / sizeof(float), 0.1);
        clock_t end_time = clock();
        double compression_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("Time taken to compress 200MB of data (iteration %d): %f seconds\n", data.iteration, compression_time);

        free(data.h_data);
    }
}

int main() {
    auto total_start_time = std::chrono::high_resolution_clock::now();
    const size_t data_size = 200 * 1024 * 1024;  // 200MB in bytes
    const int num_elements = data_size / sizeof(float);
    const int num_iterations = 10;

    // Create two CUDA streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // Allocate two sets of GPU memory
    float *d_data1, *d_data2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data1, data_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data2, data_size));

    // Create a queue and start the compression thread
    SafeQueue<IterationData> compression_queue;
    std::thread compression_thread(compress_thread_func, std::ref(compression_queue));

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long seed = time(NULL);

    for (int i = 0; i < num_iterations; i++) {
        float* d_current = (i % 2 == 0) ? d_data1 : d_data2;
        float* d_previous = (i % 2 == 0) ? d_data2 : d_data1;
        cudaStream_t current_stream = (i % 2 == 0) ? stream1 : stream2;
        cudaStream_t previous_stream = (i % 2 == 0) ? stream2 : stream1;

        // Measure time for data generation
        clock_t start_gen = clock();
        init_random<<<num_blocks, block_size, 0, current_stream>>>(d_current, num_elements, seed + i);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(current_stream));
        clock_t end_gen = clock();
        double gen_time = (double)(end_gen - start_gen) / CLOCKS_PER_SEC;
        printf("Time taken to generate data (iteration %d): %f seconds\n", i, gen_time);

        if (i > 0) {
            // Allocate CPU memory for the previous iteration's data
            float* h_data = (float*)malloc(data_size);

            // Measure time for data copy
            clock_t start_copy = clock();
            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_previous, data_size, cudaMemcpyDeviceToHost, previous_stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(previous_stream));
            clock_t end_copy = clock();
            double copy_time = (double)(end_copy - start_copy) / CLOCKS_PER_SEC;
            printf("Time taken to copy data (iteration %d): %f seconds\n", i - 1, copy_time);

            // Queue the data for compression
            compression_queue.push({h_data, data_size, i - 1});
        }
    }

    // Process the last iteration
    float* h_data = (float*)malloc(data_size);
    clock_t start_copy = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, (num_iterations % 2 == 0) ? d_data2 : d_data1, data_size, cudaMemcpyDeviceToHost, (num_iterations % 2 == 0) ? stream2 : stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize((num_iterations % 2 == 0) ? stream2 : stream1));
    clock_t end_copy = clock();
    double copy_time = (double)(end_copy - start_copy) / CLOCKS_PER_SEC;
    printf("Time taken to copy data (iteration %d): %f seconds\n", num_iterations - 1, copy_time);
    compression_queue.push({h_data, data_size, num_iterations - 1});

    // Signal the compression thread to exit
    compression_queue.push({nullptr, 0, -1});

    // Wait for the compression thread to finish
    compression_thread.join();

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_data1));
    CHECK_CUDA_ERROR(cudaFree(d_data2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();
    printf("Total time taken for all iterations: %ld milliseconds\n", total_duration);

    return 0;
}