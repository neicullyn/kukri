#ifndef CUDA_KUKRI
#define CUDA_KUKRI

#include <cuda_runtime.h>

#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

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
        cudaEvent_t m_start;
        cudaEvent_t m_stop;
        void tic();
        float toc();        
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
            double t = abs(val);
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