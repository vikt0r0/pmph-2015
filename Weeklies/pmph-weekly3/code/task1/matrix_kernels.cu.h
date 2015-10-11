#ifndef _MATRIX_KERNELS
#define _MATRIX_KERNELS

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
  setElement<T>(d_out, x, y, e);
}

template <class T, unsigned int TILE_SIZE>
__global__ void matrix_transpose_tiled_kernel(matrix_t<T> d_out, matrix_t<T> d_in) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE];

  int j = blockIdx.x * TILE_SIZE + threadIdx.x;
  int i = blockIdx.y * TILE_SIZE + threadIdx.y;

  if( j >= d_in.width || i >= d_in.height )
    return;

  tile[threadIdx.y][threadIdx.x] = getElement(d_in, i, j);

  __syncthreads();

  i = blockIdx.y*TILE_SIZE + threadIdx.x;
  j = blockIdx.x*TILE_SIZE + threadIdx.y;

  if( j < d_in.width && i < d_in.height ) {
    T elem = tile[threadIdx.x][threadIdx.y];
    setElement(d_out, j, i, elem);
  }
 }

#endif // _MATRIX_KERNELS
