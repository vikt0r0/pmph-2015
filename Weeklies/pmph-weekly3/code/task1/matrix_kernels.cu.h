#ifndef MATRIX_KERNELS
#define MATRIX_KERNELS

#include "matrix.cu.h"

template <class T>
__device__ float getElement(matrix_t<T> mat, int i, int j) {
  return mat.elements[mat.width * i + j];
}

template <class T>
__device__ void setElement(matrix_t<T> mat, int i, int j, T val) {
  mat.elements[mat.width * i + j] = val;
}

template <class T>
__global__ void
matrix_transpose_naive_kernel(matrix_t<T> d_out, matrix_t<T> d_in) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= d_in.width || y >= d_in.height)
    return;

  T e = getElement(d_in, y, x);
  setElement(d_out, x, y);
}

#endif
