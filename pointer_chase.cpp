#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include "utils.cpp"
#include "kernels.cpp"

int main() {
    // array to exceed L2/L3 cache sizes (e.g., 256 MB)
    size_t N = 32 * 1024 * 1024; 
    size_t bytes = N * sizeof(uint64_t);

    std::vector<uint64_t> h_array(N);

    // 2. Initialize array with a random permutation cycle.
    // This ensures we jump around memory unpredictably, defeating the prefetcher.
    std::cout << "Initializing random pointer chain... " << std::flush;
    std::iota(h_array.begin(), h_array.end(), 0);
    
    std::mt19937 rng(42); // Fixed seed for reproducibility
    // single massive cycle
    for (size_t i = N - 1; i > 0; --i) {
        size_t j = rng() % (i + 1);
        std::swap(h_array[i], h_array[j]);
    }
    std::cout << "Done." << std::endl;

    // 3. Allocate device memory and copy
    uint64_t* d_array;
    HipErrorCheck(hipMalloc(&d_array, bytes));
    HipErrorCheck(hipMemcpy(d_array, h_array.data(), bytes, hipMemcpyHostToDevice));

    // 4. Configure launch parameters
    // We use a small number of threads to prevent memory bandwidth 
    // saturation from masking the raw latency.
    int threads = 64; 
    int blocks = 1; 
    long long num_itr = 1000000;

    // 5. Setup HIP events for precise timing
    hipEvent_t start, stop;
    HipErrorCheck(hipEventCreate(&start));
    HipErrorCheck(hipEventCreate(&stop));

    std::cout << "Launching pointer chase kernel..." << std::endl;
    
    // Warmup run
    hipLaunchKernelGGL(pchaseKernel, dim3(blocks), dim3(threads), 0, 0, d_array, 1000);
    HipErrorCheck(hipDeviceSynchronize());

    // Timed run
    HipErrorCheck(hipEventRecord(start));
    hipLaunchKernelGGL(pchaseKernel, dim3(blocks), dim3(threads), 0, 0, d_array, num_itr);
    HipErrorCheck(hipEventRecord(stop));
    HipErrorCheck(hipEventSynchronize(stop));

    // 6. Calculate results
    float milliseconds = 0;
    HipErrorCheck(hipEventElapsedTime(&milliseconds, start, stop));

    // Latency per hop calculation: (Total Time in ns) / (Iterations)
    // Note: We divide by iterations, but *not* threads, because all threads 
    // are executing parallel independent pointer chases.
    double total_ns = milliseconds * 1e6;
    double ns_per_hop = total_ns / num_itr;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total Time:      " << milliseconds << " ms" << std::endl;
    std::cout << "Iterations:      " << num_itr << std::endl;
    std::cout << "Threads:         " << threads << std::endl;
    std::cout << "Avg Latency/Hop: " << ns_per_hop << " ns" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 7. Cleanup
    HipErrorCheck(hipFree(d_array));
    HipErrorCheck(hipEventDestroy(start));
    HipErrorCheck(hipEventDestroy(stop));

    return 0;
}