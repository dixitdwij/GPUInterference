#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h> // Required for hipExtStreamCreateWithCUMask
#include "utils.cpp"
#include "kernels.cpp"

struct cu_mask{
    uint32_t cuMaskSize;
    uint32_t cuMask[4];
    HipStream masked_stream; 
    const int N;
    HipVector ha, hb, hc;

    cu_mask(const int n): cuMaskSize(4), cuMask{0x0F, 0, 0, 0}, masked_stream(cuMaskSize, cuMask), N(n), ha(N), hb(N), hc(N, false) {}

    void run_kernel(){
        ha.toDevice();
        hb.toDevice();

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;

        HipStartStop timer(&masked_stream);
        timer.startTiming();

        vecAdd<<<numBlocks, blockSize, 0, masked_stream.stream>>>(ha.d_vec, hb.d_vec, hc.d_vec, N);
        timer.stopTiming();

        float milliseconds = timer.elapsedTime();
        hc.toHost();

        std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    }
};

int main() {
    cu_mask program(1 << 30);
    program.run_kernel();
    return 0;
}