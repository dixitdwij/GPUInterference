#include <iostream>
#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <hip/hip_runtime.h>

int n = 4;
int block_size = 2;

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

__global__ void matrix_multiplication(float *m1, float *m2, float *result, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; //col
    int idy = threadIdx.y + blockDim.y * blockIdx.y; //row

    if((idx < n) && (idy < n)){
        float c = 0;
        for(int i = 0; i < n; i++){
            c += m1[idy * n + i] * m2[n * i + idx];
        }
        result[idy * n + idx] = c;
    }
}

void matrix_initialization(float *m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            m[i * n + j] = static_cast<float>(i+1);
    }
}

void print_matrix(float *m){
     for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            printf("%f ", m[i * n + j]);
        printf("\n");
    }
}

int main(){
    float *h_m1, *h_m2, *h_result;
    float *d_m1, *d_m2, *d_result;

    h_m1 = (float*)malloc(sizeof(float) * n * n);
    h_m2 = (float*)malloc(sizeof(float) * n * n);
    h_result = (float*)malloc(sizeof(float) * n * n);

    matrix_initialization(h_m1);
    matrix_initialization(h_m2);

    HIP_CHECK(hipMalloc(&d_m1, n*n*(sizeof(float))));
    HIP_CHECK(hipMalloc(&d_m2, n*n*(sizeof(float))));
    HIP_CHECK(hipMalloc(&d_result, n*n*(sizeof(float))));

    HIP_CHECK(hipMemcpy(d_m1, h_m1, n * n * (sizeof(float)), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m2, h_m2, n * n * (sizeof(float)), hipMemcpyHostToDevice));

    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (n + block.y-1)/block.y);

    hipLaunchKernelGGL(matrix_multiplication, grid, block, 0, 0, d_m1, d_m2, d_result, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_result, d_result, n * n * sizeof(float), hipMemcpyDeviceToHost));

    print_matrix(h_result);

    HIP_CHECK(hipFree(d_m1));
    HIP_CHECK(hipFree(d_m2));
    HIP_CHECK(hipFree(d_result));
    
    free(h_m1);
    free(h_m2);
    free(h_result);
    return 0;
}
