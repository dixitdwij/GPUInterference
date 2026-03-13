#pragma once
#include <hip/hip_runtime.h>

// definitions for the histogram kernel
#define HISTOGRAM256_BIN_COUNT 256
#define WAVEFRONT_SIZE 64
#define LOG2_WF_SIZE 6U
#define WF_COUNT 7 //might change it to 6
#define TAG_MASK 0xFFFFFFFFU
#define HISTOGRAM256_THREADBLOCK_SIZE (WF_COUNT * WAVEFRONT_SIZE)
#define HISTOGRAM256_THREADBLOCK_MEMORY (WF_COUNT * HISTOGRAM256_BIN_COUNT)
#define UINT_BITS 32
#define PARTIAL_HISTOGRAM256_COUNT 240
#define UMUL(a, b) ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))

__global__ void copyKernel(
    float *in, 
    float *out, 
    long long num_floats,
    long long num_iter,
    uint* cu_ids,
    unsigned long long *start,
    unsigned long long *end
 );

 __global__ void copyKernelVectorized(
    float *in,
    float *out,
    long long num_floats,
    long long num_itr,
    uint* cu_ids,
    unsigned long long *start,
    unsigned long long *end
 );

 __global__ void copyKernelStrided(
    float *in,
    float *out,
    long long num_floats,
    long long num_itr,
    size_t stride,
    uint* cu_ids,
    unsigned long long *start,
    unsigned long long *end
 );

 __global__ void histogram256Kernel(uint *d_PartialHist, uint *d_Data, uint dataCount);