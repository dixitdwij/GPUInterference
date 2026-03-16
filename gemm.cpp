#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "utils.cpp"

struct GemmParam {
    int m, n, k;
    float alpha, beta;

    HipVector A, B, C;
    hipblasHandle_t& handle;
    
    GemmParam(hipblasHandle_t& handle, int m, int n, int k, float alpha, float beta, bool random_init)
        : handle(handle), m(m), n(n), k(k), alpha(alpha), beta(beta),
        A(m*k, random_init), B(k*n, random_init), C(m*n, random_init) {
        // Assuming that the handle is created and lifetime managed by the caller
        // The caller to initialise the vectors (lifecycle managed by the struct)
    }

    void toDevice() {
        A.toDevice();
        B.toDevice();
        C.toDevice();
    }

    void toHost() {
        C.toHost();
    }

    inline int lda() const { return m; }
    inline int ldb() const { return k; }
    inline int ldc() const { return m; }

    float launch(HipStream* s) {
        HipStartStop timer(s);

        timer.startTiming();

        HipErrorCheck(hipblasSgemm(this->handle, 
                            HIPBLAS_OP_N,   
                            HIPBLAS_OP_N,   // Don't transpose
                            this->m, this->n, this->k,        
                            &this->alpha,         
                            this->A.d_vec, this->lda(),       
                            this->B.d_vec, this->ldb(),       
                            &this->beta,         
                            this->C.d_vec, this->ldc())); 
        
        timer.stopTiming();
        return timer.elapsedTime();
    }
};

int main() {
    int m = 3; 
    int k = 4;
    int n = 2;

    hipblasHandle_t handle;
    HipErrorCheck(hipblasCreate(&handle));

    HipStream stream; 
    HipErrorCheck(hipblasSetStream(handle, stream.get())); 

    GemmParam params(handle, m, n, k, 1.0f, 0.0f, false);

    for (int i = 0; i < params.A.size; i++) params.A.vec[i] = 1.0f;
    for (int i = 0; i < params.B.size; i++) params.B.vec[i] = 2.0f;
    for (int i = 0; i < params.C.size; i++) params.C.vec[i] = 0.0f;

    params.toDevice();

    int warmup_iters = 5;
    int profile_iters = 10;

    std::cout << "Running " << warmup_iters << " warmup iterations..." << std::endl;
    for (int i = 0; i < warmup_iters; i++) {
        params.launch(&stream); 
    }

    std::cout << "Running " << profile_iters << " profiling iterations..." << std::endl;
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < profile_iters; i++) {
        total_time_ms += params.launch(&stream);
    }

    std::cout << "Average kernel execution time: " 
              << (total_time_ms / profile_iters) << " ms" << std::endl;

    params.toHost();
    std::cout <<  params.C.vec[0] << std::endl;

    HipErrorCheck(hipblasDestroy(handle));
    return 0;
}