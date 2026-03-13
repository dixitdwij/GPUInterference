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
