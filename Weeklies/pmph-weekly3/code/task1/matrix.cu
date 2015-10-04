#include <stdio.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <matrix_kernels.cu.h>
#endif

#define RAND_FLOAT(min,max) (min + static_cast <float> (rand()) /(static_cast <float> (RAND_MAX/(max-min))))

void matrix_fill_random(float *mat[], int m, int n, float min, float max) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      mat[i][j] = RAND_FLOAT(min,max);
}

void matrix_transpose_seq(float *out[], float *in[], int m, int n) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      out[j][i] = in[i][j];
}

void matrix_transpose_omp(float *out[], float *in[], int m, int n) {
  #if defined(_OPENMP)
  #pragma omp parallel for
  #endif
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      out[j][i] = in[i][j];
}

void matrix_transpose_cuda_naive(float *out[], float *in[], int m, int n) {
  #ifndef __CUDACC__
    // If the CUDA compiler is not used, fall back to OMP implementation.
    matrix_transpose_omp(h_out, h_in, int m, int n);
  #else
    // Invoke kernel
    matrix_transpose_naive_kernel<float><<<  >>>(out, in, int m, int n);
  #endif
}
