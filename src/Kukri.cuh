#ifndef CUDA_KUKRI
#define CUDA_KUKRI

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#include "helpers.h"


namespace kukri{
    typedef unsigned short half;

    void array_half2float_host(float *h_dst, half *h_src, size_t size);
    void array_half2float_device(float *d_dst, half *d_src, size_t size);
    __global__ void _array_half2float_kernel(float *d_dst, half *d_src, size_t size);

    void array_float2half_host(half *h_dst, float *h_src, size_t size);
    void array_float2half_device(half *d_dst, float *d_src, size_t size);
    __global__ void _array_float2half_kernel(half *d_dst, float *d_src, size_t size);

    void half_mm_v1(const half *d_A, const half *d_B, half *d_C, int M, int N, int K);
    __global__ void _half_mm_v1_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K);

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