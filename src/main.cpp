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

void naive_mm(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_C[j * M + i] = 0;
            for (int k = 0; k < K; k++) {
                h_C[j * M + i] += h_A[k * M + i] * h_B[j * K + k];
            }
        }
    }
}

void blas_mm_test(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    printf("\n");
    printf("---cuBlas---\n");
    printf("M = %d, N = %d, K = %d\n", M, N, K);
    kukri::Timer h2d;
    kukri::Timer mm;
    kukri::Timer d2h;

    printf("Initiating BLAS...\n");
    float *d_A;
    float *d_B;
    float *d_C;

    gpuErrChk(cudaMalloc(&d_A, M * K * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_B, K * N * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_C, M * N * sizeof(float)));

    cublasHandle_t handle;
    blasErrChk(cublasCreate(&handle));

    printf("h2d...");
    h2d.tic();
    blasErrChk(cublasSetVector(M * K, sizeof(float), h_A, 1, d_A, 1));
    blasErrChk(cublasSetVector(K * N, sizeof(float), h_B, 1, d_B, 1));
    h2d.toc();
    printf("%f ms\n", h2d.get_val());

    printf("mm ...");
    mm.tic();
    float alpha = 1;
    float beta = 0;
    blasErrChk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
        d_A, M, d_B, K, &beta, d_C, M));
    mm.toc();
    printf("%f ms\n", mm.get_val());

    printf("d2h...");
    d2h.tic();
    blasErrChk(cublasGetVector(M * N, sizeof(float), d_C, 1, h_C, 1));
    d2h.toc();
    printf("%f ms\n", d2h.get_val());

    printf("overall: %f ms\n", h2d.get_val() + mm.get_val() + d2h.get_val());

    gpuErrChk(cudaFree(d_A));
    gpuErrChk(cudaFree(d_B));
    gpuErrChk(cudaFree(d_C));
}

void blas_mm_test(int size) {
    float *h_A = new float[size * size];
    float *h_B = new float[size * size];
    float *h_C_naive = new float[size * size];
    float *h_C = new float[size * size];

    std::default_random_engine gen;
    std::normal_distribution<float> distribution(0, 1);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_A[j * size + i] = distribution(gen);
            h_B[j * size + i] = distribution(gen);
        }
    }

    naive_mm(h_A, h_B, h_C_naive, size, size, size);

    blas_mm_test(h_A, h_B, h_C, size, size, size);

    bool flag = true;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float diff = h_C[j * size + i] - h_C_naive[j * size + i];
            if (abs(diff) > 1e-6) {
                printf("Test fails: i = %d, j = %d, C[i,j] = %f, C_naive[i,j] = %f\n",
                        i, j, h_C[j * size + i], h_C_naive[j * size + i]);
                flag = false;
            }
        }
    }

    if (flag) {
        printf("[Test Pass]\n");
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_naive;
}

void kukri_mm_test(kukri::half_mm_func_t func, kukri::half *h_A, kukri::half *h_B, kukri::half *h_C, int M, int N, int K, char *test_name=NULL) {
    printf("\n");
    if (test_name == NULL) {
        printf("---Kukri---\n"); 
    } else {
        printf("---Kukri [%s]---\n", test_name);
    }
        
    printf("M = %d, N = %d, K = %d\n", M, N, K);
    kukri::Timer h2d;
    kukri::Timer mm;
    kukri::Timer d2h;

    printf("Initiating Kukri...\n");
    kukri::half *d_A;
    kukri::half *d_B;
    kukri::half *d_C;

    gpuErrChk(cudaMalloc(&d_A, M * K * sizeof(kukri::half)));
    gpuErrChk(cudaMalloc(&d_B, K * N * sizeof(kukri::half)));
    gpuErrChk(cudaMalloc(&d_C, M * N * sizeof(kukri::half)));


    printf("h2d...");
    h2d.tic();
    gpuErrChk(cudaMemcpy(d_A, h_A, M * K * sizeof(kukri::half), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_B, h_B, K * N * sizeof(kukri::half), cudaMemcpyHostToDevice));
    h2d.toc();
    printf("%f ms\n", h2d.get_val());

    printf("mm ...");
    mm.tic();
    float alpha = 1;
    float beta = 0;
    func(d_A, d_B, d_C, M, N, K);
    gpuErrChk(cudaGetLastError());
    mm.toc();
    printf("%f ms\n", mm.get_val());

    printf("d2h...");
    d2h.tic();
    gpuErrChk(cudaMemcpy(h_C, d_C, M * N * sizeof(kukri::half), cudaMemcpyDeviceToHost));
    d2h.toc();
    printf("%f ms\n", d2h.get_val());

    printf("overall: %f ms\n", h2d.get_val() + mm.get_val() + d2h.get_val());

    gpuErrChk(cudaFree(d_A));
    gpuErrChk(cudaFree(d_B));
    gpuErrChk(cudaFree(d_C));
}

void kukri_mm_test(kukri::half_mm_func_t func, int size, char *test_name=NULL) {

    float *h_A = new float[size * size];
    float *h_B = new float[size * size];
    float *h_C_naive = new float[size * size];
    float *h_C = new float[size * size];

    kukri::half *h_Ah = new kukri::half[size * size];
    kukri::half *h_Bh = new kukri::half[size * size];
    kukri::half *h_Ch = new kukri::half[size * size];

    std::default_random_engine gen;
    std::normal_distribution<float> distribution(0, 1);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_A[j * size + i] = distribution(gen);
            h_B[j * size + i] = distribution(gen);
        }
    }

    kukri::array_float2half_host(h_Ah, h_A, size * size);
    kukri::array_float2half_host(h_Bh, h_B, size * size);

    naive_mm(h_A, h_B, h_C_naive, size, size, size);

    kukri_mm_test(func, h_Ah, h_Bh, h_Ch, size, size, size, test_name);

    kukri::array_half2float_host(h_C, h_Ch, size * size);

    bool flag = true;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float diff = h_C[j * size + i] - h_C_naive[j * size + i];
            if (abs(diff) > 1e-6) {
                printf("Test fails: i = %d, j = %d, C[i,j] = %f, C_naive[i,j] = %f\n",
                    i, j, h_C[j * size + i], h_C_naive[j * size + i]);
                flag = false;
            }
        }
    }

    if (flag) {
        printf("[Test Pass]\n");
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_naive;
}

void mm_test(size_t n_rows_A, size_t n_cols_A, size_t n_rows_B, size_t n_cols_B) {
    printf("\n\n");
    printf("===Matrix Multiplication===\n");

    if (n_cols_A != n_rows_B) {
        printf("Matrix Dimensions Dismatch\n");
        exit(-1);
    }

    int M = n_rows_A;
    int N = n_cols_B;
    int K = n_cols_A;

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

    kukri::array_float2half_host(h_A_kukri_half, h_A_blas, M * K);
    kukri::array_float2half_host(h_B_kukri_half, h_B_blas, K * N);

    printf("Assigning the values\n");

    std::default_random_engine gen;
    std::normal_distribution<float> distribution(0, 1);

    for (size_t i = 0; i < n_rows_A * n_cols_A; i++) {
        h_A_blas[i] = distribution(gen);
    }

    for (size_t i = 0; i < n_rows_B * n_cols_B; i++) {
        h_B_blas[i] = distribution(gen);
    }

    kukri::array_float2half_host(h_A_kukri_half, h_A_blas, M * K);
    kukri::array_float2half_host(h_B_kukri_half, h_B_blas, K * N);

    // Test cublasSgemm
    blas_mm_test(h_A_blas, h_B_blas, h_C_blas, M, N, K);


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


    //kukri_float2half2float_test(n_rows_A, n_cols_A);
    int single_test_size = 1000;
    blas_mm_test(single_test_size);
    kukri_mm_test(kukri::half_mm_v1, single_test_size, "Naive Half");

    //mm_test(n_rows_A, n_cols_A, n_rows_B, n_cols_B);




}