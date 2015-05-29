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

void kukri::half_mm_v1(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    dim3 grid_size;
    dim3 block_size;

    // Each thread handles 1 element
    // Each block handles 16x16 elements

    block_size.x = 16;
    block_size.y = 16;
    block_size.z = 1;

    grid_size.x = (M + block_size.x - 1) / block_size.x;
    grid_size.y = (N + block_size.y - 1) / block_size.y;
    grid_size.z = 1;

    kukri::_half_mm_v1_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
}

__global__ void kukri::_half_mm_v1_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float valf, af, bf;
    half valh, ah, bh;
    if (i < M && j < N) {
        valf = 0;
        for (int k = 0; k < K; k++) {
            ah = d_A[k * M + i];
            bh = d_B[j * K + k];

            af = __half2float(ah);
            bf = __half2float(bh);

            valf += af * bf;
        }
        valh = __float2half_rn(valf);
        d_C[j * M + i] = valh;
    }
}