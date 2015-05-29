#ifndef CUDA_KUKRI
#define CUDA_KUKRI

#include <cuda_runtime.h>

#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
    const char *file,
    int line,
    bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define blasErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cublasStatus_t code,
    const char *file,
    int line,
    bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            _cudaGetErrorEnum(code), file, line);
        exit(code);
    }
}

namespace kukri{
    typedef unsigned short half;

    void array_half2float_host(float *h_dst, half *h_src, size_t size);
    void array_half2float_device(float *d_dst, half *d_src, size_t size);
    __global__ void _array_half2float_kernel(float *d_dst, half *d_src, size_t size);

    void array_float2half_host(half *h_dst, float *h_src, size_t size);
    void array_float2half_device(half *d_dst, float *d_src, size_t size);
    __global__ void _array_float2half_kernel(half *d_dst, float *d_src, size_t size);

    class Timer {
    public:
        float t;
        cudaEvent_t m_start;
        cudaEvent_t m_stop;

        Timer() { t = 0; }

        void tic();
        float toc();     
        float get_val() { return t; }
    };

    class Recorder {
    public:
        double max_abs;
        double sum;
        size_t count;
        Recorder(){
            max_abs = 0; sum = 0; count = 0;
        }
        void update(double val){
            double t = fabs(val);
            if (t > max_abs) {
                max_abs = t;
            }
            sum += val;
            count++;
        }
        double get_max_abs() { return max_abs; }
        double get_avg() { return sum / count; }
        double get_sum() { return sum; }
    };
}

#endif