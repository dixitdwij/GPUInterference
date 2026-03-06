#pragma once

#include <hip/hip_runtime.h>
#include <source_location>
#include <string>


thread_local std::string __hipLastErrorMsg_;

struct {
    const auto& operator()(hipError_t err, const std::source_location& location = std::source_location::current()) const {
        if (err != hipSuccess) {
            __hipLastErrorMsg_ = "HIP Error: " + std::string(hipGetErrorString(err)) + 
                                 " at " + location.file_name() + ":" + std::to_string(location.line());
        } else {
            __hipLastErrorMsg_.clear();
        }
        return *this;
    }

    const auto& operator()(const std::source_location& location = std::source_location::current()) const {
        return (*this)(hipGetLastError(), location);
    }

    explicit operator bool() const {
        return !__hipLastErrorMsg_.empty();
    }

    const char* what() const {
        return __hipLastErrorMsg_.c_str();
    }
} HipErrorCheck;


struct HipStream {
    hipStream_t stream;

    HipStream() {
        HipErrorCheck(hipStreamCreate(&stream));
    }

    HipStream(uint32_t size, uint32_t* mask) {
        HipErrorCheck(hipExtStreamCreateWithCUMask(&stream, size, mask));
    }

    void synchronize() {
        HipErrorCheck(hipStreamSynchronize(stream));
    }

    ~HipStream() {
        HipErrorCheck(hipStreamDestroy(stream));
    }
};


struct HipVector {
    float *vec, *d_vec;
    int size;

    HipVector(int n, bool init = true) : size(n) {
        vec = new float[n];
        HipErrorCheck(hipMalloc(&d_vec, n * sizeof(float)));

        if (init) {
            for (int i = 0; i < size; i++) {
                vec[i] = static_cast<float>(i);
            }
        }
    }

    ~HipVector() {
        delete[] vec;
        HipErrorCheck(hipFree(d_vec));
    }

    void toDevice() {
        HipErrorCheck(hipMemcpy(d_vec, vec, size * sizeof(float), hipMemcpyHostToDevice));
    }

    void toHost() {
        HipErrorCheck(hipMemcpy(vec, d_vec, size * sizeof(float), hipMemcpyDeviceToHost));
    }
};

struct HipStartStop {
    hipEvent_t start, stop;
    HipStream* stream;

    HipStartStop(HipStream* s = nullptr) : stream(s) {
        HipErrorCheck(hipEventCreate(&start));
        HipErrorCheck(hipEventCreate(&stop));
    }

    ~HipStartStop() {
        HipErrorCheck(hipEventDestroy(start));
        HipErrorCheck(hipEventDestroy(stop));
    }

    void startTiming() {
        if (this->stream) {
            HipErrorCheck(hipEventRecord(start, this->stream->stream));
        } else {
            HipErrorCheck(hipEventRecord(start));
        }
    }

    void stopTiming() {
        if (this->stream) {
            HipErrorCheck(hipEventRecord(stop, this->stream->stream));
            this->stream->synchronize();
        } else {
            HipErrorCheck(hipEventRecord(stop));
            HipErrorCheck(hipEventSynchronize(stop));
        }
    }

    float elapsedTime() {
        float milliseconds = 0;
        HipErrorCheck(hipEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};