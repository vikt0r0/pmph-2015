#ifndef MATRIX_KERNELS
#define MATRIX_KERNELS

#include "matrix.cu.h"

template <class T>
__global__ void
matrix_transpose_naive_kernel(matrix_t<T> d_out, matrix_t<T> d_in) {
}

#endif
