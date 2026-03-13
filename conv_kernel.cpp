#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include "utils.cpp" 

int main() {
    // Heavy Workload Dimensions (e.g., a massive CNN layer)
    int n = 128;         // Batch size
    int c = 128;         // Input channels
    int in_h = 112;      // Image height
    int in_w = 112;      // Image width
    int k = 256;         // Output channels (number of filters)
    int filter_h = 3;    // Filter height
    int filter_w = 3;    // Filter width

    size_t in_bytes = (size_t)n * c * in_h * in_w * sizeof(float);
    size_t weight_bytes = (size_t)k * c * filter_h * filter_w * sizeof(float);

    std::cout << "Allocating Tensors..." << std::endl;
    std::cout << "Input Memory:  " << in_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Weight Memory: " << weight_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    // Fill with 1.0f s
    std::vector<float> h_input(in_bytes / sizeof(float), 1.0f);
    std::vector<float> h_weight(weight_bytes / sizeof(float), 1.0f);

    float *d_input, *d_weight, *d_output;
    HipErrorCheck(hipMalloc(&d_input, in_bytes));
    HipErrorCheck(hipMalloc(&d_weight, weight_bytes));
    
    HipErrorCheck(hipMemcpy(d_input, h_input.data(), in_bytes, hipMemcpyHostToDevice));
    HipErrorCheck(hipMemcpy(d_weight, h_weight.data(), weight_bytes, hipMemcpyHostToDevice));

    // Initialize MIOpen and Descriptors
    std::cout << "Initializing MIOpen..." << std::endl;
    miopenHandle_t handle;
    HipErrorCheck(miopenCreate(&handle));

    miopenTensorDescriptor_t in_desc, weight_desc, out_desc;
    HipErrorCheck(miopenCreateTensorDescriptor(&in_desc));
    HipErrorCheck(miopenCreateTensorDescriptor(&weight_desc));
    HipErrorCheck(miopenCreateTensorDescriptor(&out_desc));

    HipErrorCheck(miopenSet4dTensorDescriptor(in_desc, miopenFloat, n, c, in_h, in_w));
    HipErrorCheck(miopenSet4dTensorDescriptor(weight_desc, miopenFloat, k, c, filter_h, filter_w));

    miopenConvolutionDescriptor_t conv_desc;
    HipErrorCheck(miopenCreateConvolutionDescriptor(&conv_desc));
    // Padding=0, Stride=1, Dilation=1
    HipErrorCheck(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, 0, 0, 1, 1, 1, 1));

    // Calculate Output Dimensions
    int out_n, out_c, out_h, out_w;
    HipErrorCheck(miopenGetConvolutionForwardOutputDim(conv_desc, in_desc, weight_desc, &out_n, &out_c, &out_h, &out_w));
    
    HipErrorCheck(miopenSet4dTensorDescriptor(out_desc, miopenFloat, out_n, out_c, out_h, out_w));

    size_t out_bytes = (size_t)out_n * out_c * out_h * out_w * sizeof(float);
    std::cout << "Output Memory: " << out_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    HipErrorCheck(hipMalloc(&d_output, out_bytes));

    // Allocate Workspace & Find the Best Algorithm
    size_t workspace_size;
    HipErrorCheck(miopenConvolutionForwardGetWorkSpaceSize(handle, weight_desc, in_desc, conv_desc, out_desc, &workspace_size));
    std::cout << "Workspace Mem: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;

    void* d_workspace;
    HipErrorCheck(hipMalloc(&d_workspace, workspace_size));

    std::cout << "\nSearching DB" << std::endl;
    int algo_count;
    miopenConvAlgoPerf_t perf_result;
    
    // MIOpen will run several algorithms on the MI210 and pick the fastest
    HipErrorCheck(miopenFindConvolutionForwardAlgorithm(
        handle, in_desc, d_input, weight_desc, d_weight, conv_desc, out_desc, d_output,
        1, &algo_count, &perf_result, d_workspace, workspace_size, false));

    std::cout << "Selected Algorithm ID: " << perf_result.fwd_algo << " (Search Time: " << perf_result.time << " ms)" << std::endl;

    // Timed Execution
    std::cout << "\nExecuting optimal convolution..." << std::endl;
    
    hipEvent_t start, stop;
    HipErrorCheck(hipEventCreate(&start));
    HipErrorCheck(hipEventCreate(&stop));

    HipErrorCheck(hipDeviceSynchronize());
    HipErrorCheck(hipEventRecord(start));

    float alpha = 1.0f, beta = 0.0f;
    HipErrorCheck(miopenConvolutionForward(
        handle, &alpha, in_desc, d_input, weight_desc, d_weight, conv_desc,
        perf_result.fwd_algo, &beta, out_desc, d_output, d_workspace, workspace_size));

    HipErrorCheck(hipEventRecord(stop));
    HipErrorCheck(hipEventSynchronize(stop));

    float ms = 0;
    HipErrorCheck(hipEventElapsedTime(&ms, start, stop));
    std::cout << "-> Execution Time: " << ms << " ms" << std::endl;

    // Copy Back and Verify
    std::vector<float> h_output(out_bytes / sizeof(float));
    HipErrorCheck(hipMemcpy(h_output.data(), d_output, out_bytes, hipMemcpyDeviceToHost));

    // A 3x3 filter looking at 128 input channels sums up (3 * 3 * 128) = 1152 elements.
    // Since input and weights are all 1.0, every output element should be exactly 1152.0.
    std::cout << "\nOutput Dimensions: " << out_n << "x" << out_c << "x" << out_h << "x" << out_w << std::endl;
    std::cout << "Verification Check (Should all be 1152):" << std::endl;
    std::cout << "Output[0]: " << h_output[0] << std::endl;
    std::cout << "Output[1]: " << h_output[1] << std::endl;
    std::cout << "Output[2]: " << h_output[2] << std::endl;

    miopenDestroyTensorDescriptor(in_desc);
    miopenDestroyTensorDescriptor(weight_desc);
    miopenDestroyTensorDescriptor(out_desc);
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroy(handle);
    
    HipErrorCheck(hipFree(d_input));
    HipErrorCheck(hipFree(d_weight));
    HipErrorCheck(hipFree(d_output));
    HipErrorCheck(hipFree(d_workspace));
    HipErrorCheck(hipEventDestroy(start));
    HipErrorCheck(hipEventDestroy(stop));

    return 0;
}


// hipcc conv_kernel.cpp -o bin/conv_kernel -I/opt/rocm-7.2.0/include -L/opt/rocm-7.2.0/lib -lMIOpen --std=c++20