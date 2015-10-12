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

    T e = getElement<T>(d_in, x, y);
    setElement<T>(d_out, x, y, e);
  d_out.elements[x] = 0;
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
    setElement<T>(d_out, j, i, elem);
  }
 }

template <class T>
__global__ void matrix_mult_naive_kernel(matrix_t<T> a, matrix_t<T> b, matrix_t<T> r) {
  const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
  T res = 0;
  // Check if we are within bounds
  if (col < r.width && row < r.height) {
    for (int i = 0; i < a.width; ++i)
      res += getElement<T>(a, row, i) * getElement<T>(b, i, col);
    setElement<T>(r, row, col, res);
  }
}

template <typename T, unsigned int TILE_SIZE>
__global__ void matrix_mult_tiled_kernel(matrix_t<T> a, matrix_t<T> b, matrix_t<T> r) {

    T tmp = 0;

    int i = blockIdx.y*TILE_SIZE + threadIdx.y;
    int j = blockIdx.x*TILE_SIZE + threadIdx.x;

    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k < (TILE_SIZE + a.width - 1) / TILE_SIZE; k++) {

         if (k*TILE_SIZE + threadIdx.x < a.width && j < a.width) 
           a_shared[threadIdx.y][threadIdx.x] = getElement(a, i, k * TILE_SIZE + threadIdx.x);
         else
           a_shared[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_SIZE + threadIdx.y < b.height && j < b.width)
           b_shared[threadIdx.y][threadIdx.x] = getElement(b, k * TILE_SIZE + threadIdx.y, j);
         else
           b_shared[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_SIZE; ++n)
           tmp += a_shared[threadIdx.y][n] * b_shared[n][threadIdx.x];

         __syncthreads();
    }

    if (i < r.height && j < r.width)
        setElement(r, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x*blockDim.x+threadIdx.x, tmp);
}

#endif // _MATRIX_KERNELS
