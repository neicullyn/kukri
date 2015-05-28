#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include "Kukri.cuh"
#include "cublas_v2.h"

void kukri_float2half2float_test(size_t n_rows, size_t n_cols) {
    float *A_input, *A_back;
    kukri::half *A_half;
    
    printf("\n");
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

void kukri_mm_test(size_t n_rows_A, size_t n_cols_A, size_t n_rows_B, size_t n_cols_B) {
    // A: (n_rows_A x n_cols_A)
    // B: (n_rows_B x n_cols_B)
    // C: (n_rows_A x n_cols_B)
    if (n_cols_A != n_rows_B) {
        printf("Matrix dimensions dismatch\n");
        exit(-1);
    }

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
    if (argc != 5) {        
        printf("Usage: (n_repeats) (n_rows_A) (n_cols_A/n_rows_B) (n_cols_B)\n");
        return -1;
    }

    int n_repeats = atoi(argv[1]);
    int n_rows_A = atoi(argv[2]);
    int n_cols_A = atoi(argv[3]);
    int n_rows_B = atoi(argv[3]);
    int n_cols_B = atoi(argv[4]);


    kukri_float2half2float_test(n_rows_A, n_cols_A);
    kukri_mm_test(n_rows_A, n_cols_A, n_rows_B, n_cols_B);




}