// #pragma once
#include "kernels.h"

#include <hip/hip_runtime.h>

__global__ void vecAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__device__ __forceinline__ uint64_t u64Load(const uint64_t* addr) {
    // to bypass cache: refer to https://clang.llvm.org/docs/LanguageExtensions.html
    return __builtin_nontemporal_load(addr);
}

__global__ void pchaseKernel(uint64_t* array, long long num_itr) {
    uint64_t j = threadIdx.x + blockIdx.x * blockDim.x;

    for (long long i = 0; i < num_itr; i++) {
        uint64_t* nextAddr = &array[j];
        
        j = u64Load(nextAddr); 
    }

    // Don't mark this as dead code ;)
    if (j == UINT64_MAX) {
        array[0] = j;
    }
}

__global__ void copyKernel(
    float *in, 
    float *out, 
    long long num_floats,
    long long num_iter,
    uint* cu_ids,
    unsigned long long *start,
    unsigned long long *end
){
    size_t start_idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t step = gridDim.x * blockDim.x;

    for(size_t j = 0; j < num_iter; j++){
        for(size_t i = start_idx; i < num_floats; i += step){
            out[i] = in[i];
        }
    }
}

__global__ void copyKernelDbg(float *in, float *out, long long num_floats, long long num_iter) {
    size_t start_idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t step = gridDim.x * blockDim.x;

    for(size_t j = 0; j < num_iter; j++){
        for(size_t i = start_idx; i < num_floats; i += step){
            out[i] = in[i];
        }
    }
}

__global__ void copyKernelVectorized(
    float *in,
    float *out,
    long long num_floats,
    long long num_itr,
    uint* cu_ids,
    unsigned long long *start,
    unsigned long long *end
){
    using vec_t = float4;  // float4 groups 4 floats in one structure
    size_t num_vecs = num_floats / 4;
    
    vec_t *in_vec = reinterpret_cast<vec_t*>(in);
    float result = 0;

    size_t vec_start = threadIdx.x + blockDim.x * blockIdx.x;
    size_t vec_step = gridDim.x * blockDim.x;

    for(size_t j = 0; j < num_itr; j++){
        for(size_t i = vec_start; i < num_vecs; i += vec_step){
            float4 val = ((float4 *)in_vec)[i];
            result += val.x;
            result += val.y;
            result += val.z;
            result += val.w;
        }
    }
    out[threadIdx.x] = result;
}

__global__ void copyKernelStrided(
    float *in,
    float *out,
    long long num_floats,
    long long num_itr,
    size_t stride,
    uint* cu_ids,
    unsigned long long *start,
    unsigned long long *end
){
    size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t total_threads = gridDim.x * blockDim.x;

    for (size_t j = 0; j < num_itr; j++){
        for(size_t i = thread_id * stride; i < num_floats; i += stride * total_threads){
            out[i] = in[i];
        }
    }
}

inline __device__ void addByte(uint *sharedWavefrontHist, uint data){
    atomicAdd(sharedWavefrontHist + data, 1);
}
inline __device__ void addWord(uint *sharedWavefrontHist, uint data){
    addByte(sharedWavefrontHist, (data >> 0) & 0xFF);
    addByte(sharedWavefrontHist, (data >> 8) & 0xFF);
    addByte(sharedWavefrontHist, (data >> 16) & 0xFF);
    addByte(sharedWavefrontHist, (data >> 24) & 0xFF);
}

// inline __device__ void addQuad(uint *sharedWavefrontHist, uint data){
//     addWord(sharedWavefrontHist, (data >> 0) & 0xFFFFFFFF);
//     addWord(sharedWavefrontHist, (data >> 32) & 0xFFFFFFFF);
// }

__global__ void noopKernel() {}

__global__ void histogram256Kernel(uint *d_PartialHist, uint *d_Data, uint dataCount){
    __shared__ uint sharedHist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *sharedWavefrontHist = sharedHist + (threadIdx.x >> LOG2_WF_SIZE) * HISTOGRAM256_BIN_COUNT;

#pragma unroll
    for(uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++){
        sharedHist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WF_SIZE);
    __syncthreads();

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for(uint i = idx; i < dataCount; i += stride){
        uint data = d_Data[i];
        addWord(sharedWavefrontHist, data);
    }
    __syncthreads();

    for(uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE){
        uint sum = 0;
        for(uint wf = 0; wf < WF_COUNT; wf++){
            sum += sharedHist[wf * HISTOGRAM256_BIN_COUNT + bin] & TAG_MASK;
        }
        d_PartialHist[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

// kernels for fdtd
__constant__ float stencil[RADIUS + 1];

__global__ void finiteDifferencesKernel(float *output, const float *input, const int dimx, const int dimy, const int dimz, long long num_itr){
    const int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadIdy = blockIdx.y * blockDim.y + threadIdx.y;
    const int localThreadIdx = threadIdx.x;
    const int localThreadIdy = threadIdx.y;
    const int workThreadIdx = blockDim.x;
    const int workThreadIdy = blockDim.y;

    for (long long i = 0; i < num_itr; i++){
        bool validRead = true;
        bool validWrite = true;

        __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];
        const int strideY = dimx + 2 * RADIUS;
        const int strideZ = (dimy + 2 * RADIUS) * strideY;
        
        int inputIndex = 0;
        int outputIndex = 0;

        inputIndex += RADIUS * strideZ + RADIUS * strideY + RADIUS;
        inputIndex += globalThreadIdy * strideY + globalThreadIdx;

        float infront[RADIUS];
        float behind[RADIUS];
        float current;

        const int tx = localThreadIdx + RADIUS;
        const int ty = localThreadIdy + RADIUS;

        if((globalThreadIdx >= dimx + RADIUS) || (globalThreadIdy >= dimy + RADIUS)){
            validRead = false;
        }

        if((globalThreadIdx >= dimx) || (globalThreadIdy >= dimy)){
            validWrite = false;
        }

        for(int i = RADIUS - 2; i >= 0; i--){
            if (validRead) behind[i] = input[inputIndex];
            inputIndex -= strideZ;
        }
        if (validRead) current = input[inputIndex];

        outputIndex = inputIndex;
        inputIndex += strideZ;

        for(int i = 0; i < RADIUS; i++){
            if (validRead) infront[i] = input[inputIndex];
            inputIndex += strideZ;
        }

    #pragma unroll 9
        for(int z = 0; z < dimz; z++){
            for(int j = RADIUS - 1; j > 0; j--)
                behind[j] = behind[j - 1];
            behind[0] = current;
            current = infront[0];

    #pragma unroll 4
            for(int j = 0; j < RADIUS - 1; j++)
                infront[j] = infront[j + 1];

            if (validRead) infront[RADIUS - 1] = input[inputIndex];
            inputIndex += strideZ;
            outputIndex += strideZ;

            __syncthreads();

            if(localThreadIdy < RADIUS){
                tile[localThreadIdy][tx] = input[outputIndex - RADIUS * strideY];
                tile[localThreadIdy + workThreadIdy + RADIUS][tx] = input[outputIndex + workThreadIdy * strideY];
            }

            if(localThreadIdx < RADIUS){
                tile[ty][localThreadIdx] = input[outputIndex - RADIUS];
                tile[ty][localThreadIdx + workThreadIdx + RADIUS] = input[outputIndex + workThreadIdx];
            }

            tile[ty][tx] = current;
            __syncthreads();

            float value = stencil[0] * current;

    #pragma unroll 4
            for(int j = 1; j <= RADIUS; j++){
                value += stencil[j] * (infront[j - 1] + behind[j - 1] + tile[ty - j][tx] + tile[ty + j][tx] + tile[ty][tx - j] + tile[ty][tx + j]);
            }

            if (validWrite) output[outputIndex] = value;
        }
    }
}