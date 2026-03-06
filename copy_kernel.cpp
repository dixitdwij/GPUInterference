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

HipStream stream1;
HipStream stream2;
HipStartStop event1(&stream1);
HipStartStop event2(&stream2);

float *A[2], *B[2], *C[2], *D[2];

enum KERNEL_TYPE{
    COPY
};

void run_kernel(
    int client_id,
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
                A[client_id], B[client_id], num_floats, 
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
    size_t num_floats
){
    dim3 grid_dim(num_blocks);
    dim3 block_dim(num_threads);

    std::vector<float> times;
    const int N = 1 << 24;
    HipVector h_a(N), h_b(N); 

    for(int i = 0; i < 10; i++){
        float time1;

        event1.startTiming();
        run_kernel(0, ktype, num_itr_a, num_floats, grid_dim, block_dim, stream1.stream, NULL, NULL, NULL);
        event1.stopTiming();
        HipErrorCheck(hipDeviceSynchronize());
        time1 = event1.elapsedTime();

        cout << "Time is " << time1 << "ms" <<endl;

        if (i >= 1){
            times.push_back(time1);
        }

    }
    return 0.0;
}

int main(int argc, char *argv[]){

    const int N = 1 << 24; 

    int kernel_type = 0;
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    int num_itr_a = 15;
    size_t num_floats = 40;

    float avg_time = run_single_kernel(num_blocks, num_threads, num_itr_a, static_cast<KERNEL_TYPE>(kernel_type), num_floats);

    std::cout << "-------------- avg time is " << avg_time << endl;
}