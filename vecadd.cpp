#include "kernels.cpp"
#include "utils.cpp"
#include <iostream>
#include <vector>

// struct HipVector {
//     float *vec, *d_vec;
//     int size;

//     HipVector(int n, bool init = true) : size(n) {
//         vec = new float[n];
//         HipErrorCheck(hipMalloc(&d_vec, n * sizeof(float)));

//         if (init) {
//             for (int i = 0; i < size; i++) {
//                 vec[i] = static_cast<float>(i);
//             }
//         }
//     }

//     ~HipVector() {
//         delete[] vec;
//         HipErrorCheck(hipFree(d_vec));
//     }

//     void toDevice() {
//         HipErrorCheck(hipMemcpy(d_vec, vec, size * sizeof(float), hipMemcpyHostToDevice));
//     }

//     void toHost() {
//         HipErrorCheck(hipMemcpy(vec, d_vec, size * sizeof(float), hipMemcpyDeviceToHost));
//     }
// };

// struct HipStartStop {
//     hipEvent_t start, stop;

//     HipStartStop() {
//         HipErrorCheck(hipEventCreate(&start));
//         HipErrorCheck(hipEventCreate(&stop));
//     }

//     ~HipStartStop() {
//         HipErrorCheck(hipEventDestroy(start));
//         HipErrorCheck(hipEventDestroy(stop));
//     }

//     void startTiming() {
//         HipErrorCheck(hipEventRecord(start));
//     }

//     void stopTiming() {
//         // why sync after record?
//         HipErrorCheck(hipEventRecord(stop));
//         HipErrorCheck(hipEventSynchronize(stop));
//     }

//     float elapsedTime() {
//         float milliseconds = 0;
//         HipErrorCheck(hipEventElapsedTime(&milliseconds, start, stop));
//         return milliseconds;
//     }
// };

int main() {
    const int N = 1 << 24;
    HipVector ha(N), hb(N), hc(N, false);

    ha.toDevice();
    hb.toDevice();
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Timing
    HipStartStop timer;
    
    timer.startTiming();
    vecAdd<<<numBlocks, blockSize>>>(ha.d_vec, hb.d_vec, hc.d_vec, N);
    timer.stopTiming(nullptr); 
        
    float milliseconds = timer.elapsedTime();
    
    hc.toHost();
    
    // Verify
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (hc.vec[i] != ha.vec[i] + hb.vec[i]) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    
    return 0;
}
