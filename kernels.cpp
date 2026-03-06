#pragma once

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