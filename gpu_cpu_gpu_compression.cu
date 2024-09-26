#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <chrono>

#define CHECK_CUDA_ERROR(call) {                                        \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error in call at file '%s' line %d: %s\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

__global__ void init_random(float* d_data, int num_elements, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        d_data[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void compress_gradients_topk(float* gradients, int num_elements, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float local_max[256];
    float thread_max = 0.0f;
    
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float abs_val = fabsf(gradients[i]);
        if (abs_val > thread_max) {
            thread_max = abs_val;
        }
    }
    
    local_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            local_max[threadIdx.x] = fmaxf(local_max[threadIdx.x], local_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicMax((int*)&gradients[num_elements - 1], __float_as_int(local_max[0]));
    }
    
    __syncthreads();
    
    float threshold = __int_as_float(gradients[num_elements - 1]);
    threshold *= k;
    
    for (int i = idx; i < num_elements - 1; i += blockDim.x * gridDim.x) {
        if (fabsf(gradients[i]) < threshold) {
            gradients[i] = 0.0f;
        }
    }
}

int main() {
    const size_t data_size = 200 * 1024 * 1024;  // 200MB in bytes
    const int num_elements = data_size / sizeof(float);
    const int num_iterations = 10;
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // Allocate memory on GPU1, GPU2, and CPU
    float *d_data_gpu1, *d_data_gpu2, *h_data;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_gpu1, data_size));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_gpu2, data_size));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_data, data_size));  // Pinned memory for faster transfers

    // Create streams for each GPU
    cudaStream_t stream_gpu1, stream_gpu2;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_gpu1));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_gpu2));

    unsigned long long seed = time(NULL);

    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds duration;

        // GPU1: Generate data
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        init_random<<<num_blocks, block_size, 0, stream_gpu1>>>(d_data_gpu1, num_elements, seed + i);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu1));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - GPU1 data generation: %ld ms\n", i, duration.count());

        // GPU1 to CPU copy
        start_time = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data_gpu1, data_size, cudaMemcpyDeviceToHost, stream_gpu1));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu1));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - GPU1 to CPU transfer: %ld ms\n", i, duration.count());

        // CPU to GPU2 copy
        start_time = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaSetDevice(1));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data_gpu2, h_data, data_size, cudaMemcpyHostToDevice, stream_gpu2));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu2));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - CPU to GPU2 transfer: %ld ms\n", i, duration.count());

        // GPU2: Compression
        start_time = std::chrono::high_resolution_clock::now();
        compress_gradients_topk<<<num_blocks, block_size, 0, stream_gpu2>>>(d_data_gpu2, num_elements, 0.1f);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_gpu2));
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
        printf("Iteration %d - GPU2 compression: %ld ms\n", i, duration.count());

        printf("\n");
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();
    printf("Total time taken for all iterations: %ld milliseconds\n", total_duration);

    // Cleanup
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaFree(d_data_gpu1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_gpu1));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    CHECK_CUDA_ERROR(cudaFree(d_data_gpu2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_gpu2));
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));

    return 0;
}
