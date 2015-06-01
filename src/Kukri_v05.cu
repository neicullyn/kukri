#include "Kukri.cuh"

using namespace kukri;
#define _BOX_V05 64
#define _BLOCK_SIZE_X_V05 64
#define _BLOCK_SIZE_Y_V05 8
#define _STRID_Y_V05 _BLOCK_SIZE_Y_V05
#define _N_LINE_Y_V05 ((_BOX_V05 + _BLOCK_SIZE_Y_V05 - 1) / _BLOCK_SIZE_Y_V05)

texture<half, cudaTextureType2D, cudaReadModeElementType> tex_A;
texture<half, cudaTextureType2D, cudaReadModeElementType> tex_B;


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

    //cudaChannelFormatDesc channel = cudaCreateChannelDescHalf1();

    //

    //tex_A.filterMode = cudaFilterModePoint;
    //tex_A.addressMode[0] = cudaAddressModeBorder;
    //tex_A.addressMode[1] = cudaAddressModeBorder;
    //tex_A.channelDesc = channel; //IMPORTANT
    //tex_A.normalized = false;

    //tex_B.filterMode = cudaFilterModePoint;
    //tex_B.addressMode[0] = cudaAddressModeBorder;
    //tex_B.addressMode[1] = cudaAddressModeBorder;
    //tex_B.channelDesc = channel; //IMPORTANT
    //tex_B.normalized = false;

    size_t offset_A;
    size_t offset_B;

    //gpuErrChk(cudaBindTexture2D(&offset_A, &tex_A, d_A, &channel, M, K, pitch_A));
    //gpuErrChk(cudaBindTexture2D(&offset_B, &tex_B, d_B, &channel, K, N, pitch_B));

    //printf("%d %d %d | %d %d %d\n", block_size.x, block_size.y, block_size.z, grid_size.x, grid_size.y, grid_size.z);
    kukri::_half_mm_v05_kernel<<<grid_size, block_size>>>(d_A, pitch_A / sizeof(kukri::half), d_B, pitch_B / sizeof(kukri::half), d_C, M, N, K, n_iter);

    //cudaUnbindTexture(&tex_A);
    //cudaUnbindTexture(&tex_B);
}

__global__ void kukri::_half_mm_v05_kernel(const half *d_A, int ld_A, const half *d_B, int ld_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ float buf_A[(_BOX_V05 + 1) * _BOX_V05];
    __shared__ float buf_B[(_BOX_V05 + 1) * _BOX_V05];

    int m_offset = _BOX_V05 * blockIdx.x;
    int n_offset = _BOX_V05 * blockIdx.y;

    int m_limit = MIN(M - m_offset, _BOX_V05);
    int n_limit = MIN(N - n_offset, _BOX_V05);

    int x = threadIdx.x;

    float val1[_N_LINE_Y_V05];
    float val2[_N_LINE_Y_V05];

    int yf[_N_LINE_Y_V05];

    float *a1_base;
    float *a2_base;

    float *b1_base;
    float *b2_base;

    for (int i = 0; i < _N_LINE_Y_V05; i++) {
        val1[i] = 0;
        val2[i] = 0;
    }

    for (int i = 0; i < _N_LINE_Y_V05; i++) {
        yf[i] = threadIdx.y + i * _STRID_Y_V05;
    }

    for (int i_iter = 0; i_iter < n_iter; i_iter++) {
        // Loading the block into shared memory

        int k_offset = _BOX_V05 * i_iter;
        int k_limit = MIN(K - k_offset, _BOX_V05);        

        if (x < m_limit) {
            for (int y = threadIdx.y; y < k_limit; y += _STRID_Y_V05) {
                buf_A[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(d_A[IDX2C(x + m_offset, y + k_offset, ld_A)]);
                //buf_A[IDX2C(x, y, _BOX_V05 + 1)] = y;
                //buf_A[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(tex2D(tex_A, y + k_offset, x + m_offset));
                //buf_A[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(tex2D(tex_A, 0, 0));
            }
        }

        if (x < k_limit) {
            for (int y = threadIdx.y; y < n_limit; y += _STRID_Y_V05) {
                buf_B[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(d_B[IDX2C(x + k_offset, y + n_offset, ld_B)]);
                //buf_B[IDX2C(x, y, _BOX_V05 + 1)] = y;
                //buf_B[IDX2C(x, y, _BOX_V05 + 1)] = __half2float(tex2D(tex_B, y + n_offset, x + m_offset));
            }
        }

        __syncthreads();

        if (x < m_limit) {
            int k = 0;
            int l = 1;
            a1_base = buf_A + IDX2C(x, 0, _BOX_V05 + 1);
            a2_base = buf_A + IDX2C(x, 1, _BOX_V05 + 1);
            for (; l < k_limit; k += 2, l +=2) {
                float a1 = *a1_base;
                float a2 = *a2_base;

                b1_base = buf_B + IDX2C(k, threadIdx.y, _BOX_V05 + 1);
                b2_base = buf_B + IDX2C(l, threadIdx.y, _BOX_V05 + 1);

                for (int i = 0; i < _N_LINE_Y_V05; i++) {
                    int y = yf[i];                    

                    if (y < n_limit) {                    
                        float b1 = *b1_base;
                        float b2 = *b2_base;
                        val1[i] += a1 * b1;
                        val2[i] += a2 * b2;

                        b1_base += IDX2C(0, _STRID_Y_V05, _BOX_V05 + 1) - IDX2C(0, 0, _BOX_V05 + 1);
                        b2_base += IDX2C(0, _STRID_Y_V05, _BOX_V05 + 1) - IDX2C(0, 0, _BOX_V05 + 1);

                    }
                }

                a1_base += IDX2C(0, 2, _BOX_V05 + 1) - IDX2C(0, 0, _BOX_V05 + 1);
                a2_base += IDX2C(0, 2, _BOX_V05 + 1) - IDX2C(0, 0, _BOX_V05 + 1);
            }
            if (k < k_limit) {
                float a1 = *a1_base;

                for (int i = 0; i < _N_LINE_Y_V05; i++) {
                    int y = yf[i];

                    if (y < n_limit) {
                        float b1 = buf_B[IDX2C(k, y, _BOX_V05 + 1)];
                        val1[i] += a1 * b1;
                    }
                }
            }
        }        

        __syncthreads();
    }


    if (x < m_limit) {
        half *c_base = d_C +  IDX2C(x+m_offset, threadIdx.y + n_offset, M);
        for (int i = 0; i < _N_LINE_Y_V05; i++) {
            int y = yf[i];
            if (y < n_limit) {
                *c_base = __float2half_rn(val1[i] + val2[i]);
                c_base += M * _STRID_Y_V05;
            }
        }
    }

}
