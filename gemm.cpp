#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "utils.cpp"

#define CHECK_HIPBLAS(error) \
    if (error != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLAS error code: " << error << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    int m = 3; 
    int k = 4;
    int n = 2;

    // C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    HipVector A(m * k, false);
    for (int i = 0; i < A.size; i++) A.vec[i] = 1.0f;
    HipVector B(k * n, false);
    for (int i = 0; i < B.size; i++) B.vec[i] = 2.0f;
    HipVector C(m * n, false);
    for (int i = 0; i < C.size; i++) C.vec[i] = 0.0f;

    A.toDevice();
    B.toDevice();
    C.toDevice();

    // hipBLAS Handle
    hipblasHandle_t handle;
    CHECK_HIPBLAS(hipblasCreate(&handle));

    int lda = m;
    int ldb = k;
    int ldc = m;

    CHECK_HIPBLAS(hipblasSgemm(handle, 
                               HIPBLAS_OP_N,   
                               HIPBLAS_OP_N,   // Don't transpose
                               m, n, k,        
                               &alpha,         
                               A.d_vec, lda,       
                               B.d_vec, ldb,       
                               &beta,         
                               C.d_vec, ldc));    

    HipErrorCheck(hipDeviceSynchronize()); if (HipErrorCheck) { std::cerr << HipErrorCheck.what() << std::endl; exit(EXIT_FAILURE); }

    C.toHost();

    // verify (Expected: 1.0 * 4 * 2.0 = 8.0) TODO: full verify
    std::cout << "Top-left element of C: " << C.vec[0] << std::endl;

    CHECK_HIPBLAS(hipblasDestroy(handle));

    return 0;
}