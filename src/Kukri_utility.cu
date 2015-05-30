#include "Kukri.cuh"

using namespace kukri;

void kukri::Timer::tic() {
    gpuErrChk(cudaEventCreate(&m_start));
    gpuErrChk(cudaEventCreate(&m_stop));
    gpuErrChk(cudaEventRecord(m_start));
}

float kukri::Timer::toc() {
    gpuErrChk(cudaEventRecord(m_stop));
    gpuErrChk(cudaEventSynchronize(m_stop));
    gpuErrChk(cudaEventElapsedTime(&t, m_start, m_stop));
    gpuErrChk(cudaEventDestroy(m_start));
    gpuErrChk(cudaEventDestroy(m_stop));
    return t;
}

void kukri::array_half2float_host(float *h_dst, half *h_src, size_t size) {
    // Convert an array of half to an array of float, both of which are in host
    float *d_dst;
    half *d_src;

    gpuErrChk(cudaMalloc(&d_dst, size * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_src, size * sizeof(half)));

    gpuErrChk(cudaMemcpy(d_src, h_src, size * sizeof(half), cudaMemcpyHostToDevice));

    kukri::array_half2float_device(d_dst, d_src, size);

    gpuErrChk(cudaMemcpy(h_dst, d_dst, size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_dst);
    cudaFree(d_src);
}

void kukri::array_half2float_device(float *d_dst, half *d_src, size_t size) {
    int block_size = 512;
    int grid_size = 256;
    kukri::_array_half2float_kernel <<<grid_size, block_size>>> (d_dst, d_src, size);
    gpuErrChk(cudaGetLastError());
}

__global__ void kukri::_array_half2float_kernel(float *d_dst, half *d_src, size_t size) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t step_size = blockDim.x * gridDim.x;
    for (; index < size; index += step_size) {
        d_dst[index] = __half2float(d_src[index]);
    }
}

void kukri::array_float2half_host(half *h_dst, float *h_src, size_t size) {
    // Convert an array of float to an array of half, both of which are in host
    half *d_dst;
    float *d_src;

    gpuErrChk(cudaMalloc(&d_dst, size * sizeof(half)));
    gpuErrChk(cudaMalloc(&d_src, size * sizeof(float)));

    gpuErrChk(cudaMemcpy(d_src, h_src, size * sizeof(float), cudaMemcpyHostToDevice));

    kukri::array_float2half_device(d_dst, d_src, size);

    cudaMemcpy(h_dst, d_dst, size * sizeof(half), cudaMemcpyDeviceToHost);

    gpuErrChk(cudaFree(d_dst));
    gpuErrChk(cudaFree(d_src));
}

void kukri::array_float2half_device(half *d_dst, float *d_src, size_t size) {
    int block_size = 512;
    int grid_size = 256;

    kukri::_array_float2half_kernel<<<grid_size, block_size>>>(d_dst, d_src, size);
    gpuErrChk(cudaGetLastError());
}

__global__ void kukri::_array_float2half_kernel(half *d_dst, float *d_src, size_t size) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t step_size = blockDim.x * gridDim.x;
    for (; index < size; index += step_size) {
        d_dst[index] = __float2half_rn(d_src[index]);
    }
}
