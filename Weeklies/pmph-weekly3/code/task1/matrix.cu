#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <matrix_kernels.cu.h>
#endif

#include "matrix.cu.h"

#define EPSILON 0.00001

#define RAND_FLOAT(min,max) (min + static_cast <float> (rand()) /(static_cast <float> (RAND_MAX/(max-min))))
#define APPROX_EQUAL(a,b,epsilon) (fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon))

template <typename T>
void matrix_set_element(matrix_t<T> mat, int i, int j, T value) {
  mat[i*mat.width+j] = value;
}

template <typename T>
T matrix_get_element(matrix_t<T> mat, int i, int j) {
  return mat[i*mat.width+j];
}

template <typename T>
bool matrix_is_equal(matrix_t<T> a, matrix_t<T> b) {
  if (a.width != b.width || a.length != b.length)
    return false;

  bool equal = true;

  for (int i = 0; i < a.height; ++i) {
    for (int j = 0; j < a.width; ++j) {
      if (!APPROX_EQUAL(matrix_get_element(a,i,j), matrix_get_element(b,i,j), EPSILON)) {
        equal = false;
        break;
      }
    }
  }

  return equal;
}

template <typename T>
void matrix_fill_random(matrix_t<T> mat, float min, float max) {
  for (int i = 0; i < mat.height; ++i)
    for (int j = 0; j < mat.width; ++j)
      matrix_set_element(mat, i, j, RAND_FLOAT(min,max));
}

template <typename T>
void matrix_transpose_seq(matrix_t<T> out, matrix_t<T> in) {
  for (int i = 0; i < in.height; ++i) {
    for (int j = 0; j < in.width; ++j) {
      matrix_set_element(out, j, i, matrix_get_element(in, i, j));
    }
  }
}

template <typename T>
void matrix_transpose_omp(matrix_t<T> out, matrix_t<T> in) {
  #if defined(_OPENMP)
  #pragma omp parallel for
  #endif
  for (int i = 0; i < in.height; ++i) {
    for (int j = 0; j < in.width; ++j) {
      matrix_set_element(out, j, i, matrix_get_element(in, i, j));
    }
  }
}

template <typename T>
void matrix_transpose_cuda_naive(matrix_t<T> out, matrix_t<T> in) {
  #ifndef __CUDACC__
    // If the CUDA compiler is not used, fall back to OMP implementation.
    matrix_transpose_omp<T>(out, in);
  #else
    // Invoke kernel
    matrix_transpose_naive_kernel<float><<<  >>>(out, in, int m, int n);
  #endif
}

int main(int argc, char *argv[]) {
  return 0;
}
