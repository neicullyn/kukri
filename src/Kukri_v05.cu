#include "Kukri.cuh"

using namespace kukri;
#define _BOX_V05 64
#define _BLOCK_SIZE_X_V05 64
#define _BLOCK_SIZE_Y_V05 8
#define _STRID_Y_V05 _BLOCK_SIZE_Y_V05
#define _N_LINE_Y_V05 ((_BOX_V05 + _BLOCK_SIZE_Y_V05 - 1) / _BLOCK_SIZE_Y_V05)


void kukri::half_mm_v05(const half *d_A, size_t pitch_A, const half *d_B, size_t pitch_B, half *d_C, int M, int N, int K) {
    dim3 grid_size;
    dim3 block_size;

    // Each thread handles 1 element
    // Each block handles 16x16 elements

    block_size.x = _BLOCK_SIZE_X_V05;
    block_size.y = _BLOCK_SIZE_Y_V05;
    block_size.z = 1;

    grid_size.x = (M + _BOX_V05 - 1) / _BOX_V05;
    grid_size.y = (N + _BOX_V05 - 1) / _BOX_V05;
    grid_size.z = 1;

    int n_iter = (K + _BOX_V05 - 1) / _BOX_V05;

    size_t offset_A;
    size_t offset_B;

    kukri::_half_mm_v05_kernel<<<grid_size, block_size>>>(d_A, pitch_A / sizeof(kukri::half), d_B, pitch_B / sizeof(kukri::half), d_C, M, N, K, n_iter);

}

__global__ void kukri::_half_mm_v05_kernel(const half *d_A, int ld_A, const half *d_B, int ld_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ float buf_A[(_BOX_V05 + 1) * _BOX_V05];
    __shared__ float buf_B[(_BOX_V05 + 1) * _BOX_V05];

    int m_offset = _BOX_V05 * blockIdx.x;
    int n_offset = _BOX_V05 * blockIdx.y;

    int m_limit = MIN(M - m_offset, _BOX_V05);
    int n_limit = MIN(N - n_offset, _BOX_V05);

    int x = threadIdx.x;

    float val[_N_LINE_Y_V05];
    int yf[_N_LINE_Y_V05];


    for (int i = 0; i < _N_LINE_Y_V05; i++) {
        val[i] = 0;
    }

    for (int i = 0; i < _N_LINE_Y_V05; i++) {
        yf[i] = threadIdx.y + i * _STRID_Y_V05;
    }

    for (int i_iter = 0; i_iter < n_iter; i_iter++) {
        // Loading the block into shared memory

        int k_offset = _BOX_V05 * i_iter;
        int k_limit = MIN(K - k_offset, _BOX_V05);    

        for (int i = 0; i < _N_LINE_Y_V05; i++){
            int y = yf[i];
            buf_A[IDX2C(x, y, _BOX_V05 + 1)] = 0;
            buf_B[IDX2C(x, y, _BOX_V05 + 1)] = 0;
        }

        if (x < m_limit) {
            for (int y = threadIdx.y; y < k_limit; y += _STRID_Y_V05) {
                buf_A[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(d_A[IDX2C(x + m_offset, y + k_offset, ld_A)]);
            }
        }

        if (x < k_limit) {
            for (int y = threadIdx.y; y < n_limit; y += _STRID_Y_V05) {
                buf_B[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(d_B[IDX2C(x + k_offset, y + n_offset, ld_B)]);
            }
        }

        __syncthreads();

        
        for (int k = 0; k < _BOX_V05; k++) {
            float a = buf_A[IDX2C(x, k, _BOX_V05 + 1)];

            for (int i = 0; i < _N_LINE_Y_V05; i++) {
                int y = yf[i];
                float b = buf_B[IDX2C(k, y, _BOX_V05 + 1)];
                val[i] += a * b;
            }
        }
        __syncthreads();
    }


    if (x < m_limit) {
        for (int i = 0; i < _N_LINE_Y_V05; i++) {
            int y = yf[i];
            if (y < n_limit) {
                d_C[IDX2C(x+m_offset, y+n_offset, M)] = __float2half_rn(val[i]);
            }
        }
    }

}
