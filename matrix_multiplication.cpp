#include <iostream>
#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "utils.cpp"

int n = 4;
int block_size = 2;

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

void print_matrix(float *m){
     for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            printf("%f ", m[i * n + j]);
        printf("\n");
    }
}

int main(){
    HipVector m1(n*n, true);
    HipVector m2(n*n, true);
    HipVector result(n*n, false);

    m1.toDevice();
    m2.toDevice();

    HipStream stream;

    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (n + block.y-1)/block.y);

    hipLaunchKernelGGL(matrix_multiplication, grid, block, 0, stream.stream,
                       m1.d_vec, m2.d_vec, result.d_vec, n);

    stream.synchronize();

    result.toHost();

    print_matrix(result.vec);

    return 0;
}
