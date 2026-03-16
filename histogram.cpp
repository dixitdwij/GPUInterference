#include <iostream>
#include "utils.cpp"
#include "kernels.cpp"
#include "kernels.h"
using namespace std;

struct histogram{
    uint *d_PartialHistograms, *d_Data;
    uint dataCount;
    HipStream stream;
    uint *h_Data;

    void allocate_memory(){
        h_Data = (uint *)malloc(dataCount * sizeof(uint));
        for (uint i = 0; i < dataCount; i++) {
            h_Data[i] = rand() % 256;
        }
        HipErrorCheck(hipMalloc((void **)&d_Data, dataCount * sizeof(uint)));
        HipErrorCheck(hipMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
        HipErrorCheck(hipMemcpy(d_Data, h_Data, dataCount * sizeof(uint), hipMemcpyHostToDevice));
    };

    void free_memory(){
        HipErrorCheck(hipFree(d_PartialHistograms));
        HipErrorCheck(hipFree(d_Data));
        free(h_Data);
    };

    void run_kernel(){
        allocate_memory();

        HipStartStop event(&stream);
        std::vector<float> times;
        for(int i = 0; i < 10; i++){
            float time;

            event.startTiming();
            histogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE, 0, stream.stream>>>(d_PartialHistograms, d_Data, dataCount);
            event.stopTiming();
            time = event.elapsedTime();

            cout << "Time is " << time << "ms" <<endl;
            times.push_back(time);
        }
        free_memory();
    };
};

int main(){

    // memory allocation for histogram data and input data
    uint *d_PartialHistograms, *d_Data;
    uint dataCount = 40U;
    HipStream stream1;
    uint *h_Data;
    
    histogram histo = {d_PartialHistograms, d_Data, dataCount, stream1, h_Data};

    histo.run_kernel();
    
    return 0;
}