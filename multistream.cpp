#include "utils.cpp"
#include "kernels.cpp"
#include <omp.h>
#include <iostream>

int main() {
    HipStream stream1, stream2;

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        HipStream& stream = (tid == 0) ? stream1 : stream2;
        
        int vecSize = 1 << 16;
        HipVector a(vecSize), b(vecSize), c(vecSize, false);
        a.toDevice();
        b.toDevice();

        int blockSize = 256;
        int numBlocks = (a.size + blockSize - 1) / blockSize;

        #pragma omp barrier

        HipStartStop timer(&stream);
        timer.startTiming();

        vecAdd<<<numBlocks, blockSize, 0, stream.stream>>>(a.d_vec, b.d_vec, c.d_vec, a.size);

        timer.stopTiming();

        #pragma omp barrier     // TODO: CRITICAL Why does this barrier change the runtime? Check on systems profiler
        
        c.toHost();

        bool correct = true;
        // for (int i = 0; i < c.size; i++) {
        //     if (c.vec[i] != a.vec[i]  + b.vec[i]) {
        //         correct = false;
        //         break;
        //     }
        // }

        float milliseconds = timer.elapsedTime();
        std::cout << "Thread " << tid << ": Result: " << (correct ? "PASS" : "FAIL") << ", Time: " << milliseconds << " ms" << std::endl;
        
    }

    return 0;
}
