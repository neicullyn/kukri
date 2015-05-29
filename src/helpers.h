#ifndef CUDA_KUKRI_HELPERS
#define CUDA_KUKRI_HELPERS

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    default:
        return "<unknown>";
    }
}

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
    const char *file,
    int line,
    bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define blasErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cublasStatus_t code,
    const char *file,
    int line,
    bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            _cudaGetErrorEnum(code), file, line);
        exit(code);
    }
}

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#endif