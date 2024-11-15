#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <chrono>
#include <nvToolsExt.h> // Changed to older NVTX header
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

// CUDA kernel to initialize GPU memory with random floats using curand
__global__ void init_random(float* d_data, int num_elements, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        d_data[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

// CUDA kernel to perform top-k sparsification
__global__ void compress_gradients_topk(float* gradients, int num_elements, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory to store local max values
    __shared__ float local_max[256];
    
    float thread_max = 0.0f;
    
    // Each thread finds its local max
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float abs_val = fabsf(gradients[i]);
        if (abs_val > thread_max) {
            thread_max = abs_val;
        }
    }
    
    // Store thread_max in shared memory
    local_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Reduce to find block max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            local_max[threadIdx.x] = fmaxf(local_max[threadIdx.x], local_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block max to global memory
    if (threadIdx.x == 0) {
        atomicMax((int*)&gradients[num_elements - 1], __float_as_int(local_max[0]));
    }
    
    __syncthreads();
    
    // Use the global max as threshold
    float threshold = __int_as_float(gradients[num_elements - 1]);
    threshold *= k;  // Adjust threshold based on k
    
    // Zero out gradients below the threshold
    for (int i = idx; i < num_elements - 1; i += blockDim.x * gridDim.x) {
        if (fabsf(gradients[i]) < threshold) {
            gradients[i] = 0.0f;
        }
    }
}

int main() {
    nvtxRangePush("Total Execution");
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Start energy monitoring with NVTX marker
    // nvtxRangePush("Energy Monitoring Setup");
    // start_energy_monitoring();
    // nvtxRangePop();

    const size_t data_size = 200 * 1024 * 1024;  // 200MB in bytes
    const int num_elements = data_size / sizeof(float);
    const int num_iterations = 10;

    // Memory allocation range
    float *d_data;
    nvtxRangePush("GPU Memory Allocation");
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, data_size));
    nvtxRangePop();

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long seed = time(NULL);

    // Main processing loop with NVTX markers
    nvtxRangePush("Processing Loop");
    
    for (int i = 0; i < num_iterations; i++) {
        // Mark iteration start
        char iteration_marker[32];
        sprintf(iteration_marker, "Iteration %d Start", i);
        nvtxMarkA(iteration_marker);
        
        // Data generation phase
        nvtxRangePush("Data Generation");
        auto start_gen = std::chrono::high_resolution_clock::now();
        init_random<<<num_blocks, block_size>>>(d_data, num_elements, seed + i);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto end_gen = std::chrono::high_resolution_clock::now();
        auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
        printf("Time taken to generate data (iteration %d): %ld milliseconds\n", i, gen_time);
        nvtxRangePop();

        // Data compression phase
        nvtxRangePush("Data Compression");
        auto start_compress = std::chrono::high_resolution_clock::now();
        compress_gradients_topk<<<num_blocks, block_size>>>(d_data, num_elements, 0.1f);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto end_compress = std::chrono::high_resolution_clock::now();
        auto compress_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_compress - start_compress).count();
        printf("Time taken to compress data (iteration %d): %ld milliseconds\n", i, compress_time);
        nvtxRangePop();
        
        // Mark iteration end
        sprintf(iteration_marker, "Iteration %d End", i);
        nvtxMarkA(iteration_marker);
    }
    nvtxRangePop(); // End Processing Loop

    // Cleanup phase
    nvtxRangePush("Cleanup");
    CHECK_CUDA_ERROR(cudaFree(d_data));
    // stop_energy_monitoring();
    nvtxRangePop();

    // Final timing and energy reporting
    nvtxRangePush("Results Reporting");
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();
    
    printf("Total time taken for all iterations: %ld milliseconds\n", total_duration);
    // printf("CPU Energy Consumed: %.6f J\n", cpu_energy);
    // printf("GPU 0 Energy Consumed: %.6f J\n", gpu_energy[0]);
    // printf("GPU 1 Energy Consumed: %.6f J\n", gpu_energy[1]);
    // printf("Total GPU Energy Consumed: %.6f J\n", gpu_energy[0] + gpu_energy[1]);
    // printf("Total Energy Consumed: %.6f J\n", cpu_energy + gpu_energy[0] + gpu_energy[1]);
    nvtxRangePop();

    nvtxRangePop(); // End Total Execution
    return 0;
}