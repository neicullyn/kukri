#include "Kukri.cuh"

using namespace kukri;
#define _BOX_V02 64
#define _BLOCK_SIZE_X_V02 64
#define _BLOCK_SIZE_Y_V02 8
#define _STRID_Y_V02 _BLOCK_SIZE_Y_V02
#define _N_LINE_Y_V02 (_BOX_V02 / _BLOCK_SIZE_Y_V02)

void kukri::half_mm_v02(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    dim3 grid_size;
    dim3 block_size;

    // Each thread handles 1 element
    // Each block handles 16x16 elements

    block_size.x = _BLOCK_SIZE_X_V02;
    block_size.y = _BLOCK_SIZE_Y_V02;
    block_size.z = 1;

    grid_size.x = (M + _BOX_V02 - 1) / _BOX_V02;
    grid_size.y = (N + _BOX_V02 - 1) / _BOX_V02;
    grid_size.z = 1;

    int n_iter = (K + _BOX_V02 - 1) / _BOX_V02;

    //printf("%d %d %d | %d %d %d\n", block_size.x, block_size.y, block_size.z, grid_size.x, grid_size.y, grid_size.z);
    kukri::_half_mm_v02_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, n_iter);
}

__global__ void kukri::_half_mm_v02_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ float buf_A[_BOX_V02 * _BOX_V02];
    __shared__ float buf_B[_BOX_V02 * _BOX_V02];

    int m_offset = _BOX_V02 * blockIdx.x;
    int n_offset = _BOX_V02 * blockIdx.y;

    int m_limit = MIN(M - m_offset, _BOX_V02);
    int n_limit = MIN(N - n_offset, _BOX_V02);

    int x = threadIdx.x;

    float val[_N_LINE_Y_V02];

    for (int i = 0; i < _N_LINE_Y_V02; i++) {
        val[i] = 0;
    }

    for (int i_iter = 0; i_iter < n_iter; i_iter++) {
        // Loading the block into shared memory

        int k_offset = _BOX_V02 * i_iter;
        int k_limit = MIN(K - k_offset, _BOX_V02);        

        if (x < k_limit) {
            for (int y = threadIdx.y; y < m_limit; y += _STRID_Y_V02) {
                // Note that buf_A and buf_B are transposed
                buf_A[IDX2C(x, y, _BOX_V02)] = __half2float(d_A[IDX2C(y + m_offset, x + k_offset, M)]);
            }
        }

        if (x < n_limit) {
            for (int y = threadIdx.y; y < k_limit; y += _STRID_Y_V02) {
                // Note that buf_A and buf_B are transposed
                buf_B[IDX2C(x, y, _BOX_V02)] = __half2float(d_B[IDX2C(y + k_offset, x + n_offset, K)]);
            }
        }

        __syncthreads();

        if (x < n_limit) {
            // Need to change to using register
            // Not sure where the data is stored now
            for (int i = 0; i < _N_LINE_Y_V02; i++) {
                int y = threadIdx.y + i * _STRID_Y_V02;
                if (y < m_limit) {
                    for (int k = 0; k < k_limit; k++) {
                        float a = buf_A[IDX2C(k, y, _BOX_V02)];
                        float b = buf_B[IDX2C(x, k, _BOX_V02)];
                        val[i] += a * b;
                    }   
                }
            }
        }        
    }
    __syncthreads();

    if (x < n_limit) {
        for (int i = 0; i < _N_LINE_Y_V02; i++) {
            int y = threadIdx.y + i * _STRID_Y_V02;
            if (y < m_limit) {
                d_C[IDX2C(y+m_offset, x+n_offset, M)] = __float2half_rn(val[i]);
                //d_C[IDX2C(y+m_offset, x+n_offset, M)] = __float2half_rn(n_offset);
            }
        }
    }
}