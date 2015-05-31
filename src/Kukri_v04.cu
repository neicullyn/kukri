#include "Kukri.cuh"

using namespace kukri;
#define _BOX_V04 64
#define _BLOCK_SIZE_X_V04 64
#define _BLOCK_SIZE_Y_V04 8
#define _STRID_Y_V04 _BLOCK_SIZE_Y_V04
#define _N_LINE_Y_V04 (_BOX_V04 / _BLOCK_SIZE_Y_V04)

void kukri::half_mm_v04(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    dim3 grid_size;
    dim3 block_size;

    // Each thread handles 1 element
    // Each block handles 16x16 elements

    block_size.x = _BLOCK_SIZE_X_V04;
    block_size.y = _BLOCK_SIZE_Y_V04;
    block_size.z = 1;

    grid_size.x = (M + _BOX_V04 - 1) / _BOX_V04;
    grid_size.y = (N + _BOX_V04 - 1) / _BOX_V04;
    grid_size.z = 1;

    int n_iter = (K + _BOX_V04 - 1) / _BOX_V04;

    //printf("%d %d %d | %d %d %d\n", block_size.x, block_size.y, block_size.z, grid_size.x, grid_size.y, grid_size.z);
    kukri::_half_mm_v04_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, n_iter);
}

__global__ void kukri::_half_mm_v04_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ float buf_A[(_BOX_V04 + 1) * _BOX_V04];
    __shared__ float buf_B[(_BOX_V04 + 1) * _BOX_V04];

    int m_offset = _BOX_V04 * blockIdx.x;
    int n_offset = _BOX_V04 * blockIdx.y;

    int m_limit = MIN(M - m_offset, _BOX_V04);
    int n_limit = MIN(N - n_offset, _BOX_V04);

    int x = threadIdx.x;

    float val[_N_LINE_Y_V04];
    int yf[_N_LINE_Y_V04];
    float * buf_B_yo[_N_LINE_Y_V04];

    float a, b;

    for (int i = 0; i < _N_LINE_Y_V04; i++) {
        val[i] = 0;
    }

    for (int i = 0; i < _N_LINE_Y_V04; i++) {
        yf[i] = threadIdx.y + i * _STRID_Y_V04;
    }

    for (int i = 0; i < _N_LINE_Y_V04; i++) {
        buf_B_yo[i] = buf_B + yf[i] * (_BOX_V04 + 1);
    }

    for (int i_iter = 0; i_iter < n_iter; i_iter++) {
        // Loading the block into shared memory

        int k_offset = _BOX_V04 * i_iter;
        int k_limit = MIN(K - k_offset, _BOX_V04);        

        if (x < m_limit) {
            for (int y = threadIdx.y; y < k_limit; y += _STRID_Y_V04) {
                buf_A[IDX2C(x, y, _BOX_V04 + 1)] = __half2float(d_A[IDX2C(x + m_offset, y + k_offset, M)]);
            }
        }

        if (x < k_limit) {
            for (int y = threadIdx.y; y < n_limit; y += _STRID_Y_V04) {
                buf_B[IDX2C(x, y, _BOX_V04 + 1)] = __half2float(d_B[IDX2C(x + k_offset, y + n_offset, K)]);
            }
        }

        __syncthreads();

        if (x < m_limit) {
            for (int k = 0; k < k_limit; k++) {         
                for (int i = 0; i < _N_LINE_Y_V04; i++) {
                    int y = yf[i];                    

                    if (y < n_limit) {                    
                        // float b = buf_B[IDX2C(k, y, _BOX_V04 + 1)];
                        a = buf_A[IDX2C(x, k, _BOX_V04 + 1)];
                        b = *(buf_B_yo[i] + k);
                        
                        val[i] += a * b;
                    }
                }
            }
        }        
    }
    //__syncthreads();

    if (x < m_limit) {
        for (int i = 0; i < _N_LINE_Y_V04; i++) {
            int y = yf[i];
            if (y < n_limit) {
                d_C[IDX2C(x+m_offset, y+n_offset, M)] = __float2half_rn(val[i]);
            }
        }
    }

}
