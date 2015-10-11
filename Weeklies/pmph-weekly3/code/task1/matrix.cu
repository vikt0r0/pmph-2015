#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __CUDACC__
#include "matrix_kernels.cu.h"
#include <cuda_runtime.h>
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
  return mat.elements[i*mat.width+j];
}

template <typename T>
bool matrix_is_equal(matrix_t<T> a, matrix_t<T> b) {
  if (a.width != b.width || a.height != b.height)
    return false;

  bool equal = true;

  for (int i = 0; i < a.height; ++i) {
    for (int j = 0; j < a.width; ++j) {
      if (matrix_get_element(a,i,j) != matrix_get_element(b,i,j)) {
        equal = false;
        break;
      }
    }
  }
  return equal;
}

bool matrix_is_equal(matrix_t<float> a, matrix_t<float> b) {
  if (a.width != b.width || a.height != b.height)
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

void matrix_fill_random(matrix_t<float> mat, float min, float max) {
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
void matrix_transpose_cuda_naive(const unsigned int block_size, matrix_t<T> out, const matrix_t<T> in) {
  #ifndef __CUDACC__
    // If the CUDA compiler is not used, fall back to OMP implementation.
    matrix_transpose_omp<T>(out, in);
  #else
    // Set up and invoke kernel
    unsigned int num_blocks_x, num_blocks_y;

    num_blocks_x = ((in.width % block_size) == 0) ?
                     in.width / block_size     :
                     in.width / block_size + 1 ;

    num_blocks_y = ((in.height % block_size) == 0) ?
                     in.height / block_size     :
                     in.height / block_size + 1 ;

    dim3 blockDim(block_size, block_size);
    dim3 gridDim(num_blocks_x, num_blocks_y);

    matrix_transpose_naive_kernel<T><<< blockDim, gridDim >>>(out, in);
    cudaThreadSynchronize();
  #endif
}
