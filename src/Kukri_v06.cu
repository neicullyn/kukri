#include "Kukri.cuh"

using namespace kukri;
#define _BOX_V06 64
#define _BLOCK_SIZE_X_V06 64
#define _BLOCK_SIZE_Y_V06 8
#define _STRID_Y_V06 _BLOCK_SIZE_Y_V06
#define _N_LINE_Y_V06 ((_BOX_V06 + _BLOCK_SIZE_Y_V06 - 1) / _BLOCK_SIZE_Y_V06)


void kukri::half_mm_v06(const half *d_A, size_t pitch_A, const half *d_B, size_t pitch_B, half *d_C, int M, int N, int K) {
    dim3 grid_size;
    dim3 block_size;

    // Each thread handles 1 element
    // Each block handles 16x16 elements

    block_size.x = _BLOCK_SIZE_X_V06;
    block_size.y = _BLOCK_SIZE_Y_V06;
    block_size.z = 1;

    grid_size.x = (M + _BOX_V06 - 1) / _BOX_V06;
    grid_size.y = (N + _BOX_V06 - 1) / _BOX_V06;
    grid_size.z = 1;

    int n_iter = (K + _BOX_V06 - 1) / _BOX_V06;

    //printf("%d %d %d\n", M, N, K);
    if (!((M % _BOX_V06 == 0) && (N % _BOX_V06 == 0) && (K % _BOX_V06 == 0))) {
        printf("Dimension error...\n");
        exit(-1);
    }

    
    kukri::_half_mm_v06_kernel<<<grid_size, block_size>>>(d_A, pitch_A / sizeof(kukri::half), d_B, pitch_B / sizeof(kukri::half), d_C, M, N, K, n_iter);

}

// Assume M, N ,K are all mutiples of _BOX_V06
__global__ void kukri::_half_mm_v06_kernel(const half *d_A, int ld_A, const half *d_B, int ld_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ float buf_A[(_BOX_V06 + 1) * _BOX_V06];
    __shared__ float buf_B[(_BOX_V06 + 1) * _BOX_V06];

    int m_offset = _BOX_V06 * blockIdx.x;
    int n_offset = _BOX_V06 * blockIdx.y;

    int x = threadIdx.x;

    float val[_N_LINE_Y_V06];
    int yf[_N_LINE_Y_V06];


    for (int i = 0; i < _N_LINE_Y_V06; i++) {
        val[i] = 0;
    }

    for (int i = 0; i < _N_LINE_Y_V06; i++) {
        yf[i] = threadIdx.y + i * _STRID_Y_V06;
    }


    for (int i_iter = 0; i_iter < n_iter; i_iter++) {
        // Loading the block into shared memory

        int k_offset = _BOX_V06 * i_iter;


        for (int i = 0; i < _N_LINE_Y_V06; i++) {
            int y = yf[i];
            buf_A[IDX2C(x, y, _BOX_V06 + 1)] = __half2float(d_A[IDX2C(x + m_offset, y + k_offset, ld_A)]);
            buf_B[IDX2C(x, y, _BOX_V06 + 1)] = __half2float(d_B[IDX2C(x + k_offset, y + n_offset, ld_B)]);
        }

        __syncthreads();

        
        for (int k = 0; k < _BOX_V06; k++) {
            float a = buf_A[IDX2C(x, k, _BOX_V06 + 1)];
            for (int i = 0; i < _N_LINE_Y_V06; i++) {
                int y = yf[i];
                float b = buf_B[IDX2C(k, y, _BOX_V06 + 1)];
                val[i] += a * b;
            }
        }
        __syncthreads();
    }


    for (int i = 0; i < _N_LINE_Y_V06; i++) {
        int y = yf[i];
        d_C[IDX2C(x+m_offset, y+n_offset, M)] = __float2half_rn(val[i]);
    }
}
