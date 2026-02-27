#include <iostream>
#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "utils.cpp"

int n = 64;
int grid = 1;
int block = 1;

__global__ void add_vectors(float *v1, float *v2, float *result, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = idx; i < n; i += gridDim.x * blockDim.x)
        result[i] = v1[i] + v2[i];
}

void print_result(float *v){
    for(int i = 0; i < n; i++){
        printf("%f ", v[i]);
    }
}

int main(){
    HipVector v1(n, true);
    HipVector v2(n, true);
    HipVector result(n, false);

    v1.toDevice();
    v2.toDevice();

    HipStream stream;

    hipLaunchKernelGGL(add_vectors, grid, block, 0, stream.stream, v1.d_vec, v2.d_vec, result.d_vec, n);

    stream.synchronize();

    result.toHost();

    print_result(result.vec);

    return 0;
}