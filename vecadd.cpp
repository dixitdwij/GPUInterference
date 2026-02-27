#include "kernels.cpp"
#include "utils.cpp"
#include <iostream>
#include <vector>

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
