#include <hip/hip_runtime.h>

#pragma once

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