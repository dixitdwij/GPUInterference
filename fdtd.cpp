#include <iostream>
#include "utils.cpp"
#include "kernels.cpp"
#include "kernels.h"

using namespace std;


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

    return 0;
}
