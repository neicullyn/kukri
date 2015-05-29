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

inline __device__ float kukri::_half_mul_2float(half a, half b) {
    float af, bf;
    af = __half2float(a);
    bf = __half2float(b);
    return af * bf;
}

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

#define _BOX_V02 64
#define _BLOCK_SIZE_X_V02 64
#define _BLOCK_SIZE_Y_V02 8
#define _STRIP_Y_V02 _BLOCK_SIZE_Y_V02
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

    printf("%d %d %d | %d %d %d\n", block_size.x, block_size.y, block_size.z, grid_size.x, grid_size.y, grid_size.z);
    kukri::_half_mm_v02_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, n_iter);
}

__global__ void kukri::_half_mm_v02_kernel(const half *d_A, const half *d_B, half *d_C, int M, int N, int K, int n_iter) {
    __shared__ half buf_A[_BOX_V02 * _BOX_V02];
    __shared__ half buf_B[_BOX_V02 * _BOX_V02];

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
            for (int y = threadIdx.y; y < m_limit; y += _STRIP_Y_V02) {
                // Note that buf_A and buf_B are transposed
                buf_A[IDX2C(x, y, _BOX_V02)] = d_A[IDX2C(y + m_offset, x + k_offset, M)];
            }
        }

        if (x < n_limit) {
            for (int y = threadIdx.y; y < k_limit; y += _STRIP_Y_V02) {
                // Note that buf_A and buf_B are transposed
                buf_B[IDX2C(x, y, _BOX_V02)] = d_B[IDX2C(y + k_offset, x + n_offset, K)];
            }
        }

        __syncthreads();

        if (x < n_limit) {
            // Need to change to using register
            // Not sure where the data is stored now
            for (int i = 0; i < _N_LINE_Y_V02; i++) {
                int y = threadIdx.y + i * _STRIP_Y_V02;
                if (y < m_limit) {
                    for (int k = 0; k < k_limit; k++) {
                        half a = buf_A[IDX2C(k, y, _BOX_V02)];
                        half b = buf_B[IDX2C(x, k, _BOX_V02)];
                        val[i] += kukri::_half_mul_2float(a, b);
                    }   
                }
            }
        }        
    }

    if (x < n_limit) {
        for (int i = 0; i < _N_LINE_Y_V02; i++) {
            int y = threadIdx.y + i * _STRIP_Y_V02;
            if (y < m_limit) {
                d_C[IDX2C(y, x, M)] = __float2half_rn(val[i]);
            }
        }
    }
}