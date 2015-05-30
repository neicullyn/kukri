#include "Kukri.cuh"

using namespace kukri;

void kukri::half_mm_v01(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
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

    kukri::_half_mm_v01_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
}

__global__ void kukri::_half_mm_v01_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float valf;
    half valh, ah, bh;
    if (i < M && j < N) {
        valf = 0;
        for (int k = 0; k < K; k++) {
            ah = d_A[k * M + i];
            bh = d_B[j * K + k];
            
            valf += kukri::_half_mul_2float(ah, bh);
        }
        valh = __float2half_rn(valf);
        d_C[j * M + i] = valh;
    }
}
