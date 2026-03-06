#include <algorithm>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <chrono>
#include <cassert>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>

#include "utils.cpp"
#include "kernels.cpp"

using namespace std;

enum KERNEL_TYPE{
    COPY
};

void run_kernel(
    int client_id,
    HipVector* A,
    HipVector* B,
    KERNEL_TYPE ktype,
    int num_itr,
    int num_floats,
    dim3 grid_dim,
    dim3 block_dim,
    hipStream_t stream,
    uint* cu_ids,
    unsigned long long* gpu_start,
    unsigned long long* gpu_stop
){
    switch(ktype){
        case COPY:
            copyKernel<<<grid_dim, block_dim, 0, stream>>>(
                A->d_vec, B->d_vec, num_floats, 
                num_itr, cu_ids, gpu_start, gpu_stop
            );
            break;
        default:
            printf("Invalid Kernel type!\n");
            break;

    }
}

float run_single_kernel(
    int num_blocks,
    int num_threads,
    int num_itr_a,
    KERNEL_TYPE ktype,
    size_t num_floats,
    HipStream* stream1,
    HipStartStop* event1
){
    dim3 grid_dim(num_blocks);
    dim3 block_dim(num_threads);

    std::vector<float> times;
    const int N = 1 << 24;
    HipVector h_a(N), h_b(N, false); 

    for(int i = 0; i < 10; i++){
        float time1;

        event1->startTiming();
        run_kernel(0, &h_a, &h_b, ktype, num_itr_a, num_floats, grid_dim, block_dim, stream1->stream, NULL, NULL, NULL);
        event1->stopTiming();
        time1 = event1->elapsedTime();

        cout << "Time is " << time1 << "ms" <<endl;

        if (i >= 1){
            times.push_back(time1);
        }

    }
    return 0.0;
}

int main()
{
    const int N = 1 << 24; 

    HipStream stream1;

    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    int num_itr_a = 15;
    size_t num_floats = 40;

    dim3 grid_dim(num_blocks);
    dim3 block_dim(num_threads);

    std::vector<float> times;
    HipVector h_a(N), h_b(N, false); 
    h_a.toDevice();

    HipStartStop event1(&stream1);
    run_single_kernel(num_blocks, num_threads, num_itr_a, KERNEL_TYPE::COPY, num_floats, &stream1, &event1);

    h_b.toHost();
    return 0;

}