#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include "utils.cpp"

int main() {
    // Increase size to ~134.2 million elements to saturate the MI210
    // 1 << 27 elements * 8 bytes (float2) = ~1.07 GB of data
    size_t N = 1ULL << 27; 
    size_t bytes = N * sizeof(hipfftComplex); // Single-precision complex (float2)

    std::cout << "Allocating " << bytes / (1024.0 * 1024.0) << " MB of memory..." << std::endl;

    // Allocate Host Memory
    std::vector<hipfftComplex> h_data(N);
    for (size_t i = 0; i < N; i++) {
        h_data[i].x = 1.0f; // Real part
        h_data[i].y = 0.0f; // Imaginary part
    }

    // Allocate Device (MI210) Memory
    hipfftComplex* d_data;
    HipErrorCheck(hipMalloc(&d_data, bytes));

    // Copy Data: Host -> Device
    HipErrorCheck(hipMemcpy(d_data, h_data.data(), bytes, hipMemcpyHostToDevice));

    // Create the hipFFT Plan
    std::cout << "Creating hipFFT plan..." << std::endl;
    hipfftHandle plan;
    // hipfftPlan1d(plan, size, type, batch_size)
    if (hipfftPlan1d(&plan, N, HIPFFT_C2C, 1) != HIPFFT_SUCCESS) {
        std::cerr << "Plan creation failed!" << std::endl;
        return 1;
    }

    // WARM-UP RUN
    std::cout << "Running warm-up execution..." << std::endl;
    if (hipfftExecC2C(plan, d_data, d_data, HIPFFT_FORWARD) != HIPFFT_SUCCESS) {
        std::cerr << "Warm-up execution failed!" << std::endl;
        return 1;
    }
    // Wait for the GPU to finish the warm-up
    HipErrorCheck(hipDeviceSynchronize());


    // TIMED RUN
    hipEvent_t start, stop;
    HipErrorCheck(hipEventCreate(&start));
    HipErrorCheck(hipEventCreate(&stop));

    std::cout << "Running timed FFT..." << std::endl;
    
    // Start timer
    HipErrorCheck(hipEventRecord(start));

    // Execute the FFT (Forward transform)
    if (hipfftExecC2C(plan, d_data, d_data, HIPFFT_FORWARD) != HIPFFT_SUCCESS) {
        std::cerr << "Timed execution failed!" << std::endl;
        return 1;
    }

    // Stop timer
    HipErrorCheck(hipEventRecord(stop));
    HipErrorCheck(hipEventSynchronize(stop)); // Wait for the timed run to finish

    // Calculate Elapsed Time
    float milliseconds = 0;
    HipErrorCheck(hipEventElapsedTime(&milliseconds, start, stop));

    std::cout << "\n========================================" << std::endl;
    std::cout << "FFT Size: " << N << " elements" << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;


    // Copy Data: Device -> Host
    HipErrorCheck(hipMemcpy(h_data.data(), d_data, bytes, hipMemcpyDeviceToHost));

    // Cleanup
    hipfftDestroy(plan);
    HipErrorCheck(hipFree(d_data));
    HipErrorCheck(hipEventDestroy(start));
    HipErrorCheck(hipEventDestroy(stop));

    // Output a small sample to verify (DC component will be large: 1.34218e+08)
    std::cout << "Verification Check:" << std::endl;
    std::cout << "FFT Sample Output [0]: " << h_data[0].x << " + " << h_data[0].y << "i" << std::endl;
    std::cout << "FFT Sample Output [1]: " << h_data[1].x << " + " << h_data[1].y << "i" << std::endl;

    return 0;
}

// hipcc ftt_kernel.cpp -o bin/fft_kernel -I/opt/rocm-7.2.0/include -I/opt/rocm-7.2.0/include/hipfft -I/opt/rocm-7.2.0/include/rocfft -L/opt/rocm-7.2.0/lib -lhipfft -lrocfft --std=c++20
