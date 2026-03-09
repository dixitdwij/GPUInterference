#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "utils.cpp"

__global__ void mul_fp32_ilp1_hip(
    float *a,
    float *b,     // Unused kept for signature matching
    float *c,
    long long num_itr,
    uint* sm_ids,
    unsigned long long *start,
    unsigned long long *end
)
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        sm_ids[blockIdx.x] = __smid();
        start[blockIdx.x] = wall_clock64();
    }

    float op1 = a[threadIdx.x];
    float op3 = 1.0f;

    long long unrolled_itr = num_itr / 16;
    for (long long i = 0; i < unrolled_itr; i++)
    {
        #pragma unroll 16
        for (int j = 0; j < 16; j++) {
            op3 = __fmul_rn(op1, op3);
        }
    }

    c[threadIdx.x] = op3;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        end[blockIdx.x] = wall_clock64();
    }
}

int main() {
    int threads = 256; 
    int blocks = 1; 
    long long num_itr = 16000000; // 16 million iterations (multiple of 16)

    size_t float_bytes = threads * blocks * sizeof(float);
    size_t block_bytes = blocks * sizeof(unsigned long long);
    size_t sm_bytes = blocks * sizeof(uint);

    std::vector<float> h_a(threads * blocks, 1.0f); 
    std::vector<float> h_b(threads * blocks, 1.0f);
    std::vector<float> h_c(threads * blocks, 0.0f);
    std::vector<unsigned long long> h_start(blocks, 0);
    std::vector<unsigned long long> h_end(blocks, 0);
    std::vector<uint> h_sm_ids(blocks, 0);

    float *d_a, *d_b, *d_c;
    unsigned long long *d_start, *d_end;
    uint *d_sm_ids;

    HipErrorCheck(hipMalloc(&d_a, float_bytes));
    HipErrorCheck(hipMalloc(&d_b, float_bytes));
    HipErrorCheck(hipMalloc(&d_c, float_bytes));
    HipErrorCheck(hipMalloc(&d_start, block_bytes));
    HipErrorCheck(hipMalloc(&d_end, block_bytes));
    HipErrorCheck(hipMalloc(&d_sm_ids, sm_bytes));

    HipErrorCheck(hipMemcpy(d_a, h_a.data(), float_bytes, hipMemcpyHostToDevice));
    HipErrorCheck(hipMemcpy(d_b, h_b.data(), float_bytes, hipMemcpyHostToDevice));

    std::cout << "Launching ILP1 kernel with " << num_itr << " iterations..." << std::endl;

    hipLaunchKernelGGL(mul_fp32_ilp1_hip, dim3(blocks), dim3(threads), 0, 0, 
                       d_a, d_b, d_c, num_itr, d_sm_ids, d_start, d_end);
    HipErrorCheck(hipDeviceSynchronize());

    HipErrorCheck(hipMemcpy(h_c.data(), d_c, float_bytes, hipMemcpyDeviceToHost));
    HipErrorCheck(hipMemcpy(h_start.data(), d_start, block_bytes, hipMemcpyDeviceToHost));
    HipErrorCheck(hipMemcpy(h_end.data(), d_end, block_bytes, hipMemcpyDeviceToHost));
    HipErrorCheck(hipMemcpy(h_sm_ids.data(), d_sm_ids, sm_bytes, hipMemcpyDeviceToHost));

    unsigned long long start_tick = h_start[0];
    unsigned long long end_tick = h_end[0];
    unsigned long long elapsed_ticks = end_tick - start_tick;

    // wall_clock64() assumed at 100 MHz on modern AMD GPUs.
    double tick_freq_mhz = 100.0; 
    double elapsed_ns = elapsed_ticks * (1000.0 / tick_freq_mhz);
    double ns_per_instruction = elapsed_ns / num_itr;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Executed on Compute Unit: " << h_sm_ids[0] << std::endl;
    std::cout << "Total hardware ticks:     " << elapsed_ticks << std::endl;
    std::cout << "Estimated total time:     " << elapsed_ns << " ns" << std::endl;
    std::cout << "Time per FMUL (Latency):  " << std::fixed << std::setprecision(3) << ns_per_instruction << " ns" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Note: To find pipeline depth in clock cycles, multiply the" << std::endl;
    std::cout << "'Time per FMUL' by your MI210's peak engine clock (in GHz)." << std::endl;

    HipErrorCheck(hipFree(d_a));
    HipErrorCheck(hipFree(d_b));
    HipErrorCheck(hipFree(d_c));
    HipErrorCheck(hipFree(d_start));
    HipErrorCheck(hipFree(d_end));
    HipErrorCheck(hipFree(d_sm_ids));

    return 0;
}