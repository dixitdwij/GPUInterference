#include <iostream>
#include "utils.cpp"
#include "kernels.cpp"
#include "kernels.h"
using namespace std;



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