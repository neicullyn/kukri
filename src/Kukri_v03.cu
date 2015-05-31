#include "Kukri.cuh"

using namespace kukri;
#define _BOX_V03 64
#define _BLOCK_SIZE_X_V03 64
#define _BLOCK_SIZE_Y_V03 8
#define _STRIP_Y_V03 _BLOCK_SIZE_Y_V03
#define _N_LINE_Y_V03 (_BOX_V03 / _BLOCK_SIZE_Y_V03)

void kukri::half_mm_v03(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    dim3 grid_size;
    dim3 block_size;

    // Each thread handles 1 element
    // Each block handles 16x16 elements

    block_size.x = _BLOCK_SIZE_X_V03;
    block_size.y = _BLOCK_SIZE_Y_V03;
    block_size.z = 1;

    grid_size.x = (M + _BOX_V03 - 1) / _BOX_V03;
    grid_size.y = (N + _BOX_V03 - 1) / _BOX_V03;
    grid_size.z = 1;

    int n_iter = (K + _BOX_V03 - 1) / _BOX_V03;

    //printf("%d %d %d | %d %d %d\n", block_size.x, block_size.y, block_size.z, grid_size.x, grid_size.y, grid_size.z);
    kukri::_half_mm_v03_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, n_iter);
}

__global__ void kukri::_half_mm_v03_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ float buf_A[_BOX_V03 * _BOX_V03];
    __shared__ float buf_B[_BOX_V03 * _BOX_V03];

    int m_offset = _BOX_V03 * blockIdx.x;
    int n_offset = _BOX_V03 * blockIdx.y;

    int m_limit = MIN(M - m_offset, _BOX_V03);
    int n_limit = MIN(N - n_offset, _BOX_V03);

    int x = threadIdx.x;

    float val[_N_LINE_Y_V03];
    int yf[_N_LINE_Y_V03];
    int yo[_N_LINE_Y_V03];

    for (int i = 0; i < _N_LINE_Y_V03; i++) {
        val[i] = 0;
    }

    for (int i = 0; i < _N_LINE_Y_V03; i++) {
        yf[i] = threadIdx.y + i * _STRIP_Y_V03;
    }

    for (int i = 0; i < _N_LINE_Y_V03; i++) {
        yo[i] = threadIdx.y * _BOX_V03 + i * _STRIP_Y_V03 * _BOX_V03;
    }

    for (int i_iter = 0; i_iter < n_iter; i_iter++) {
        // Loading the block into shared memory

        int k_offset = _BOX_V03 * i_iter;
        int k_limit = MIN(K - k_offset, _BOX_V03);        

        if (x < k_limit) {
            for (int y = threadIdx.y; y < m_limit; y += _STRIP_Y_V03) {
                // Note that buf_A and buf_B are transposed
                buf_A[IDX2C(x, y, _BOX_V03)] = __half2float(d_A[IDX2C(y + m_offset, x + k_offset, M)]);
                //buf_A[IDX2C(x, y, _BOX_V03)] = __half2float(1);
            }
        }

        if (x < n_limit) {
            for (int y = threadIdx.y; y < k_limit; y += _STRIP_Y_V03) {
                // Note that buf_A and buf_B are transposed
                buf_B[IDX2C(x, y, _BOX_V03)] = __half2float(d_B[IDX2C(y + k_offset, x + n_offset, K)]);
                //buf_B[IDX2C(x, y, _BOX_V03)] = __half2float(1);
            }
        }

        __syncthreads();

        if (x < n_limit) {
            for (int k = 0; k < k_limit; k++) {
                float b = buf_B[IDX2C(x, k, _BOX_V03)];               

                for (int i = 0; i < _N_LINE_Y_V03; i++) {
                    int y = yf[i];                    

                    if (y < m_limit) {                    
                        // a =  buf_A[IDX2C(k, y, _BOX_V03)]
                        //   = buf_A[y * _BOX_V03 + k]
                        //   = buf_A[yo[i] + k]
                        float a = buf_A[yo[i] + k];
                        val[i] += a * b;
                    }
                }
            }
        }        
    }
    //__syncthreads();

    //Test code

    if (x < n_limit) {
        for (int i = 0; i < _N_LINE_Y_V03; i++) {
            int y = yf[i];
            if (y < m_limit) {
                d_C[IDX2C(y+m_offset, x+n_offset, M)] = __float2half_rn(val[i]);
                //d_C[IDX2C(y+m_offset, x+n_offset, M)] = __float2half_rn(n_offset);
            }
        }
    }
}
