#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <chrono>
#include <nvToolsExt.h>
// #include "energy_monitor.h"

#define CHECK_CUDA_ERROR(call) {                                        \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error in call at file '%s' line %d: %s\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

// Previous kernel definitions remain the same...
__global__ void init_random(float* d_data, int num_elements, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        d_data[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void compress_gradients_topk(float* gradients, int num_elements, float k) {
    // ... (kernel code remains the same) ...
}

int main() {
    // Initialize NVTX attributes
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.colorType = NVTX_COLOR_ARGB;

    // Define colors for different operations
    const uint32_t COLOR_GENERATE = 0xFF00FF00;  // Green
    const uint32_t COLOR_COPY_GPU1_CPU = 0xFFFF0000;  // Red
    const uint32_t COLOR_COPY_CPU_GPU2 = 0xFF0000FF;  // Blue
    const uint32_t COLOR_COMPRESS = 0xFFFF00FF;  // Purple

    nvtxRangePush("Total Execution");
    
    auto total_start_time = std::chrono::high_resolution_clock::now();

    nvtxRangePush("Energy Monitoring Setup");
    // start_energy_monitoring();
    nvtxRangePop();

    const size_t data_size = 200 * 1024 * 1024;
    const int num_elements = data_size / sizeof(float);
    const int num_iterations = 10;
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    nvtxRangePush("Memory Allocation");
    float *d_data_gpu1, *d_data_gpu2, *h_data;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_gpu1, data_size));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_gpu2, data_size));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_data, data_size));
    nvtxRangePop();

    nvtxRangePush("Stream Creation");
    cudaStream_t stream_gpu1, stream_gpu2;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_gpu1));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_gpu2));
    nvtxRangePop();

    unsigned long long seed = time(NULL);

    nvtxRangePush("Processing Loop");
    for (int i = 0; i < num_iterations; i++) {
        char iteration_marker[64];
        sprintf(iteration_marker, "Iteration %d", i);
        nvtxRangePush(iteration_marker);

        // GPU1: Generate data
        eventAttrib.color = COLOR_GENERATE;
        eventAttrib.message.ascii = "Data Generation on GPU1";
        nvtxRangePushEx(&eventAttrib);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        init_random<<<num_blocks, block_size, 0, stream_gpu1>>>(d_data_gpu1, num_elements, seed + i);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu1));
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - GPU1 data generation: %ld ms\n", i, duration.count());
        nvtxRangePop();

        // GPU1 to CPU copy
        eventAttrib.color = COLOR_COPY_GPU1_CPU;
        eventAttrib.message.ascii = "GPU1 to CPU Transfer";
        nvtxRangePushEx(&eventAttrib);
        
        start_time = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data_gpu1, data_size, 
                                       cudaMemcpyDeviceToHost, stream_gpu1));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu1));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - GPU1 to CPU transfer: %ld ms\n", i, duration.count());
        nvtxRangePop();

        // CPU to GPU2 copy
        eventAttrib.color = COLOR_COPY_CPU_GPU2;
        eventAttrib.message.ascii = "CPU to GPU2 Transfer";
        nvtxRangePushEx(&eventAttrib);
        
        start_time = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaSetDevice(1));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data_gpu2, h_data, data_size, 
                                       cudaMemcpyHostToDevice, stream_gpu2));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu2));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - CPU to GPU2 transfer: %ld ms\n", i, duration.count());
        nvtxRangePop();

        // GPU2: Compression
        eventAttrib.color = COLOR_COMPRESS;
        eventAttrib.message.ascii = "Compression on GPU2";
        nvtxRangePushEx(&eventAttrib);
        
        start_time = std::chrono::high_resolution_clock::now();
        compress_gradients_topk<<<num_blocks, block_size, 0, stream_gpu2>>>(
            d_data_gpu2, num_elements, 0.1f);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu2));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - GPU2 compression: %ld ms\n", i, duration.count());
        nvtxRangePop();

        printf("\n");
        nvtxRangePop(); // End iteration marker
    }
    nvtxRangePop(); // End processing loop

    nvtxRangePush("Cleanup and Results");
    // stop_energy_monitoring();
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_start_time).count();

    printf("\nFinal Results:\n");
    printf("Total time: %ld milliseconds\n", total_duration);
    // printf("CPU Energy: %.6f J\n", cpu_energy);
    // printf("GPU 0 Energy: %.6f J\n", gpu_energy[0]);
    // printf("GPU 1 Energy: %.6f J\n", gpu_energy[1]);
    // printf("Total GPU Energy: %.6f J\n", gpu_energy[0] + gpu_energy[1]);
    // printf("Total Energy: %.6f J\n", cpu_energy + gpu_energy[0] + gpu_energy[1]);

    // Cleanup
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaFree(d_data_gpu1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_gpu1));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    CHECK_CUDA_ERROR(cudaFree(d_data_gpu2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_gpu2));
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    nvtxRangePop();

    nvtxRangePop(); // End Total Execution
    return 0;
}