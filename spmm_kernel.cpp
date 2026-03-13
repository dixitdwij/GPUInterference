#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include "utils.cpp"

int main() {
    // Heavy Workload Dimensions
    // Matrix A: 65536 x 65536 (Sparse)
    // Matrix B: 65536 x 1024  (Dense)
    // Matrix C: 65536 x 1024  (Dense)
    int m = 65536;
    int k = 65536;
    int n = 1024;
    int nnz_per_row = 128; // Degree of sparsity
    int nnz = m * nnz_per_row;

    size_t dense_b_bytes = (size_t)k * n * sizeof(float);
    size_t dense_c_bytes = (size_t)m * n * sizeof(float);

    std::cout << "Allocating Memory & Generating CSR Data on CPU..." << std::endl;
    std::cout << "Dense Matrix B & C Memory: ~" << (dense_b_bytes + dense_c_bytes) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Procedurally generate Sparse Matrix A (CSR Format)
    std::vector<int>   h_csr_row_offsets(m + 1);
    std::vector<int>   h_csr_col_indices(nnz);
    std::vector<float> h_csr_values(nnz, 1.0f); // All non-zeros are 1.0

    for (int i = 0; i < m; i++) {
        h_csr_row_offsets[i] = i * nnz_per_row;
        for (int j = 0; j < nnz_per_row; j++) {
            // Distribute column indices to simulate a banded/random sparse structure
            h_csr_col_indices[i * nnz_per_row + j] = (i + j * 7) % k; 
        }
    }
    h_csr_row_offsets[m] = nnz;

    // Dense Matrix B (all 1.0s) and C (initialized to 0)
    std::vector<float> h_B((size_t)k * n, 1.0f);
    std::vector<float> h_C((size_t)m * n, 0.0f);

    // Device Memory Allocation
    int *d_csr_row_offsets, *d_csr_col_indices;
    float *d_csr_values, *d_B, *d_C;

    HipErrorCheck(hipMalloc(&d_csr_row_offsets, (m + 1) * sizeof(int)));
    HipErrorCheck(hipMalloc(&d_csr_col_indices, nnz * sizeof(int)));
    HipErrorCheck(hipMalloc(&d_csr_values, nnz * sizeof(float)));
    HipErrorCheck(hipMalloc(&d_B, dense_b_bytes));
    HipErrorCheck(hipMalloc(&d_C, dense_c_bytes));

    // Copy to Device
    std::cout << "Copying data to MI210..." << std::endl;
    HipErrorCheck(hipMemcpy(d_csr_row_offsets, h_csr_row_offsets.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HipErrorCheck(hipMemcpy(d_csr_col_indices, h_csr_col_indices.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
    HipErrorCheck(hipMemcpy(d_csr_values, h_csr_values.data(), nnz * sizeof(float), hipMemcpyHostToDevice));
    HipErrorCheck(hipMemcpy(d_B, h_B.data(), dense_b_bytes, hipMemcpyHostToDevice));

    // Initialize hipSPARSE & Descriptors
    hipsparseHandle_t handle;
    HipErrorCheck(hipsparseCreate(&handle));

    hipsparseSpMatDescr_t matA;
    hipsparseDnMatDescr_t matB, matC;

    HipErrorCheck(hipsparseCreateCsr(&matA, m, k, nnz,
                                     d_csr_row_offsets, d_csr_col_indices, d_csr_values,
                                     HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, 
                                     HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));   

    HipErrorCheck(hipsparseCreateDnMat(&matB, k, n, n, d_B, HIP_R_32F, HIPSPARSE_ORDER_ROW));
    HipErrorCheck(hipsparseCreateDnMat(&matC, m, n, n, d_C, HIP_R_32F, HIPSPARSE_ORDER_ROW));

    // Allocate Workspace for SpMM
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;

    HipErrorCheck(hipsparseSpMM_bufferSize(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, HIP_R_32F,
        HIPSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    void* d_buffer = nullptr;
    if (bufferSize > 0) {
        HipErrorCheck(hipMalloc(&d_buffer, bufferSize));
    }

    // Warm-up & Timed Execution
    std::cout << "Running warm-up..." << std::endl;
    HipErrorCheck(hipsparseSpMM(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, HIP_R_32F,
        HIPSPARSE_SPMM_ALG_DEFAULT, d_buffer));
    HipErrorCheck(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HipErrorCheck(hipEventCreate(&start));
    HipErrorCheck(hipEventCreate(&stop));

    std::cout << "Executing Timed SpMM..." << std::endl;
    HipErrorCheck(hipEventRecord(start));

    HipErrorCheck(hipsparseSpMM(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, HIP_R_32F,
        HIPSPARSE_SPMM_ALG_DEFAULT, d_buffer));

    HipErrorCheck(hipEventRecord(stop));
    HipErrorCheck(hipEventSynchronize(stop));

    float ms = 0;
    HipErrorCheck(hipEventElapsedTime(&ms, start, stop));
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Execution Time: " << ms << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Copy Back and Verify
    HipErrorCheck(hipMemcpy(h_C.data(), d_C, dense_c_bytes, hipMemcpyDeviceToHost));

    // Because A has `nnz_per_row` ones per row, and B is filled entirely with ones,
    // every element in the output matrix C should exactly equal `nnz_per_row` (which is 128).
    std::cout << "Verification Check (Should all be " << nnz_per_row << "):" << std::endl;
    std::cout << "Output[0, 0]: " << h_C[0] << std::endl;
    std::cout << "Output[0, 1]: " << h_C[1] << std::endl;
    std::cout << "Output[m/2, n/2]: " << h_C[(m/2) * n + (n/2)] << std::endl;

    // Cleanup
    HipErrorCheck(hipsparseDestroySpMat(matA));
    HipErrorCheck(hipsparseDestroyDnMat(matB));
    HipErrorCheck(hipsparseDestroyDnMat(matC));
    HipErrorCheck(hipsparseDestroy(handle));

    HipErrorCheck(hipFree(d_csr_row_offsets));
    HipErrorCheck(hipFree(d_csr_col_indices));
    HipErrorCheck(hipFree(d_csr_values));
    HipErrorCheck(hipFree(d_B));
    HipErrorCheck(hipFree(d_C));
    if (d_buffer) HipErrorCheck(hipFree(d_buffer));
    HipErrorCheck(hipEventDestroy(start));
    HipErrorCheck(hipEventDestroy(stop));

    return 0;
}

// hipcc spmm_kernel.cpp -o bin/spmm_kernel -I/opt/rocm-7.2.0/include -I/opt/rocm-7.2.0/include/hipsparse -L/opt/rocm-7.2.0/lib -lhipsparse -lrocsparse --std=c++20
