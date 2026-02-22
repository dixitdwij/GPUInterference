#include <iostream>
#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <hip/hip_runtime.h>

int n = 64;
int grid = 1;
int block = 1;

// https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/error_handling.html 
#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

__global__ void add_vectors(float *v1, float *v2, float *result, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = idx; i < n; i += gridDim.x * blockDim.x)
        result[i] = v1[i] + v2[i];
}

void init_vector(float *v){
    for(int i = 0; i < n; i++)
        v[i] = static_cast<float>(i+1);
}

void print_result(float *v){
    for(int i = 0; i < n; i++){
        printf("%f ", v[i]);
    }
}

int main(){
    float *h_v1, *h_v2, *h_result;
    float *d_v1, *d_v2, *d_result;
    
    h_v1 = (float*)malloc(sizeof(float)*n);
    h_v2 = (float*)malloc(sizeof(float)*n);
    h_result = (float*)malloc(sizeof(float)*n);

    HIP_CHECK(hipMalloc(&d_v1, sizeof(float)*n));
    HIP_CHECK(hipMalloc(&d_v2, sizeof(float)*n));
    HIP_CHECK(hipMalloc(&d_result, sizeof(float)*n));

    init_vector(h_v1);
    init_vector(h_v2);

    HIP_CHECK(hipMemcpy(d_v1, h_v1, n * (sizeof(float)), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v2, h_v2, n * (sizeof(float)), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(add_vectors, grid, block, 0, 0, d_v1, d_v2, d_result, n);

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_result, d_result, n * sizeof(float), hipMemcpyDeviceToHost));

    print_result(h_result);

    HIP_CHECK(hipFree(d_v1));
    HIP_CHECK(hipFree(d_v2));
    HIP_CHECK(hipFree(d_result));
    
    free(h_v1);
    free(h_v2);
    free(h_result);
}