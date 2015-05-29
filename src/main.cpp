#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <cublas_v2.h>
#include "Kukri.cuh"


void kukri_float2half2float_test(size_t n_rows, size_t n_cols) {
    float *A_input, *A_back;
    kukri::half *A_half;
    
    printf("\n\n");
    printf("---float2half and half2float---\n");

    A_input = new float[n_rows * n_cols];
    A_back = new float[n_rows * n_cols];
    A_half = new kukri::half[n_rows * n_cols];    

    printf("Generating input matrix : %ldx%ld\n", n_rows, n_cols);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution(0.5, 1.5);

    for (size_t i = 0; i < n_rows * n_cols; i++) {
        A_input[i] = distribution(gen);
    }


    kukri::Timer tmr;
    float tmr_result;

    printf("float2half...");
    tmr.tic();
    kukri::array_float2half_host(A_half, A_input, n_rows * n_cols);
    tmr_result = tmr.toc();
    printf(" %f ms\n", tmr_result);

    printf("half2float...");
    tmr.tic();
    kukri::array_half2float_host(A_back, A_half, n_rows * n_cols);
    tmr_result = tmr.toc();
    printf(" %f ms\n", tmr_result);

    // Record absolute error
    kukri::Recorder rcd_err;
    // Record relative error
    kukri::Recorder rcd_rerr;

    for (size_t i = 0; i < n_rows * n_cols; i++) {
        double err = A_input[i] - A_back[i];
        double rerr = err / A_input[i];
        
        rcd_err.update(err);
        if (!isnan(rerr)) {
            rcd_rerr.update(rerr * rerr);
        }
        
    }
    
    printf("Absolute Error (Maximum) : %lf\n", rcd_err.get_max_abs());
    printf("Relative Error (RMSE) : %lf\n", sqrt(rcd_rerr.get_avg()));

    delete [] A_input;
    delete [] A_back;
    delete [] A_half;    
}

void blas_mm(float *h_A_blas, float *h_B_blas, float *h_C_blas, int m, int n, int k) {
    printf("\n");
    printf("-cuBlas-\n");
    kukri::Timer h2d;
    kukri::Timer mm;
    kukri::Timer d2h;

    printf("Initiating BLAS...\n");
    float *d_A_blas;
    float *d_B_blas;
    float *d_C_blas;

    gpuErrChk(cudaMalloc(&d_A_blas, m * k * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_B_blas, k * n * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_C_blas, m * n * sizeof(float)));

    cublasHandle_t handle;
    blasErrChk(cublasCreate(&handle));

    printf("h2d...");
    h2d.tic();
    blasErrChk(cublasSetVector(m * k, sizeof(float), h_A_blas, 1, d_A_blas, 1));
    blasErrChk(cublasSetVector(k * n, sizeof(float), h_B_blas, 1, d_B_blas, 1));
    h2d.toc();
    printf("%f ms\n", h2d.get_val());

    printf("mm ...");
    mm.tic();
    float alpha = 1;
    float beta = 1;
    blasErrChk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        d_A_blas, m, d_B_blas, k, &beta, d_C_blas, m));
    mm.toc();
    printf("%f ms\n", mm.get_val());

    printf("d2h...");
    d2h.tic();
    blasErrChk(cublasGetVector(m * n, sizeof(float), d_C_blas, 1, h_C_blas, 1));
    d2h.toc();
    printf("%f ms\n", d2h.get_val());

    printf("overall: %f ms\n", h2d.get_val(), mm.get_val(), d2h.get_val());

    gpuErrChk(cudaFree(d_A_blas));
    gpuErrChk(cudaFree(d_B_blas));
    gpuErrChk(cudaFree(d_C_blas));
}

void kukri_mm_test(size_t n_rows_A, size_t n_cols_A, size_t n_rows_B, size_t n_cols_B) {
    printf("\n\n");
    printf("---Matrix Multiplication---\n");

    if (n_cols_A != n_rows_B) {
        printf("Matrix Dimensions Dismatch\n");
        exit(-1);
    }

    int m = n_rows_A;
    int n = n_cols_B;
    int k = n_cols_A;

    printf("\n");
    printf("Initiating Matrices...\n");
    // A: (n_rows_A x n_cols_A)
    // B: (n_rows_B x n_cols_B)
    // C: (n_rows_A x n_cols_B)

    float *h_A_blas, *h_B_blas, *h_C_blas;
    kukri::half *h_A_kukri_half, *h_B_kukri_half, *h_C_kukri_half;
    float *h_A_kukri, *h_B_kukri, *h_C_kukri, *h_C_kukri_naive;

    h_A_blas = new float[n_rows_A * n_cols_A];
    h_B_blas = new float[n_rows_B * n_cols_B];
    h_C_blas = new float[n_rows_A * n_cols_B];

    h_A_kukri_half = new kukri::half[n_rows_A * n_cols_A];
    h_B_kukri_half = new kukri::half[n_rows_B * n_cols_B];
    h_C_kukri_half = new kukri::half[n_rows_A * n_cols_B];

    h_A_kukri = new float[n_rows_A * n_cols_A];
    h_B_kukri = new float[n_rows_B * n_cols_B];
    h_C_kukri = new float[n_rows_A * n_cols_B];
    h_C_kukri_naive = new float[n_rows_A * n_cols_B];


    printf("Assigning the values\n");

    std::default_random_engine gen;
    std::normal_distribution<float> distribution(0, 1);

    for (size_t i = 0; i < n_rows_A * n_cols_A; i++) {
        h_A_blas[i] = distribution(gen);
    }

    for (size_t i = 0; i < n_rows_B * n_cols_B; i++) {
        h_B_blas[i] = distribution(gen);
    }

    blas_mm(h_A_blas, h_B_blas, h_C_blas, m, n, k);


    delete[] h_A_blas;
    delete[] h_B_blas;
    delete[] h_C_blas;

    delete[] h_A_kukri_half;
    delete[] h_B_kukri_half;
    delete[] h_C_kukri_half;

    delete[] h_A_kukri;
    delete[] h_B_kukri;
    delete[] h_C_kukri;
    delete[] h_C_kukri_naive;
}

int main(int argc, char *argv[]) {
    printf("\n");
    printf("*****Kukri Test*****\n");
    // Check the arguments
    if (argc != 4) {        
        printf("Usage: (n_rows_A) (n_cols_A/n_rows_B) (n_cols_B)\n");
        return -1;
    }

    int n_rows_A = atoi(argv[1]);
    int n_cols_A = atoi(argv[2]);
    int n_rows_B = atoi(argv[2]);
    int n_cols_B = atoi(argv[3]);


    kukri_float2half2float_test(n_rows_A, n_cols_A);
    kukri_mm_test(n_rows_A, n_cols_A, n_rows_B, n_cols_B);




}