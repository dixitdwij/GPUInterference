#include <iostream>
#include "utils.cpp"
#include "kernels.cpp"
#include "kernels.h"
using namespace std;

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
            sum += sharedHist[wf * HISTOGRAM256_BIN_COUNT + bin];
        }
        d_PartialHist[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

int main(){

    // memory allocation for histogram data and input data
    uint *d_PartialHistograms, *d_Data;
    uint dataCount = 40U;
    HipStream stream1;


    // allocate memory for the histogram

    uint* h_Data = (uint *)malloc(dataCount * sizeof(uint));
    for (uint i = 0; i < dataCount; i++) {
        h_Data[i] = rand() % 256;
    }

    HipErrorCheck(hipMalloc((void **)&d_Data, dataCount * sizeof(uint)));
    HipErrorCheck(hipMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    HipErrorCheck(hipMemcpy(d_Data, h_Data, dataCount * sizeof(uint), hipMemcpyHostToDevice));
    
    HipStartStop event1(&stream1);

    std::vector<float> times;

    for(int i = 0; i < 10; i++){
        float time1;

        event1.startTiming();
        histogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE, 0, stream1.stream>>>(d_PartialHistograms, d_Data, dataCount);
        event1.stopTiming();
        time1 = event1.elapsedTime();
        cout << "Time is " << time1 << "ms" <<endl;

        if (i >= 1){
            times.push_back(time1);
        }
    }
    
    HipErrorCheck(hipFree(d_PartialHistograms));
    HipErrorCheck(hipFree(d_Data));
    free(h_Data);
    return 0;
}