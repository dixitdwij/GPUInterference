#include <iostream>
#include "utils.cpp"
#include "kernels.cpp"
#include "kernels.h"

using namespace std;

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

int main(){
    size_t num_elements = 64;
    const int N = 1 << 24;
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    const int radius = 4;
    const size_t outerDimX = num_elements + 2 * radius;
    const size_t outerDimY = num_elements + 2 * radius;
    const size_t outerDimZ = num_elements + 2 * radius;
    const size_t volumeSize = outerDimX * outerDimY * outerDimZ;

    int fdtd_padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + fdtd_padding;

    printf("num_elements is %zu, volumeSize is %zu, paddedVolumeSize is %zu\n", num_elements, volumeSize, paddedVolumeSize);

    
    float *A, *B;

    HipErrorCheck(hipMalloc(&A, paddedVolumeSize * sizeof(float)));
    HipErrorCheck(hipMalloc(&B, paddedVolumeSize * sizeof(float)));

    A += fdtd_padding;
    B += fdtd_padding;

    dim3 block_dim(k_blockDimX, k_blockSizeMin / k_blockDimX);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (num_elements + block_dim.y - 1) / block_dim.y);

    HipStream stream1;
    
    HipStartStop event1(&stream1);

    std::vector<float> times;

    for(int i = 0; i < 10; i++){
        event1.startTiming();
        finiteDifferencesKernel<<<grid_dim, block_dim, 0, stream1.stream>>>(A, B, num_elements, num_elements, num_elements, 100);
        HipErrorCheck();
        event1.stopTiming();

        float time = event1.elapsedTime();
        times.push_back(time);
        cout << "Time is " << time << "ms" << endl;
    }

    HipErrorCheck(hipFree(A - fdtd_padding));
    HipErrorCheck(hipFree(B - fdtd_padding));

    // FiniteDifferencesKernel<<<grid_dim, block_dim, 0, stream>>>(A[client_id], B[client_id], num_elements, num_elements, num_elements, num_itr);

    return 0;
}
