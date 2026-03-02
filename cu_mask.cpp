#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h> // Required for hipExtStreamCreateWithCUMask
#include "utils.cpp"
#include "kernels.cpp"

int main() {
    // Each uint32_t covers 32 CUs. MI210 has 104 CUs (confirm)
    uint32_t cuMaskSize = 4;
    uint32_t cuMask[4] = {0, 0, 0, 0};

    // Only first 4 CUs
    cuMask[0] = 0x0F; 
    hipStream_t masked_stream;
    
    // Create the stream using the HIP extension for CU masking
    HipErrorCheck(hipExtStreamCreateWithCUMask(&masked_stream, cuMaskSize, cuMask));
    std::cout << "Masked stream created." << std::endl;

    const int N = 1 << 30;
    HipVector ha(N), hb(N), hc(N, false);

    ha.toDevice();
    hb.toDevice();

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    std::cout << "Launching vecAdd" << std::endl;

    HipStartStop timer;
    timer.startTiming();

    vecAdd<<<numBlocks, blockSize, 0, masked_stream>>>(ha.d_vec, hb.d_vec, hc.d_vec, N);

    HipErrorCheck(hipStreamSynchronize(masked_stream));
    timer.stopTiming(nullptr); 

    float milliseconds = timer.elapsedTime();
    
    hc.toHost();

    // bool correct = true;
    // for (int i = 0; i < N; i++) {
    //     if (hc.vec[i] != ha.vec[i] + hb.vec[i]) {
    //         correct = false;
    //         break;
    //     }
    // }

    // std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    // Cleanup the masked stream
    HipErrorCheck(hipStreamDestroy(masked_stream));

    return 0;
}