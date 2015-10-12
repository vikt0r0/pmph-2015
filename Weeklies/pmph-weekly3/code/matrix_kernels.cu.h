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
matrix_transpose_naive_kernel2(matrix_t<T> d_out, matrix_t<T> d_in) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= d_in.width || y >= d_in.height)
    return;

  T e = getElement<T>(d_in, y, x);
  setElement<T>(d_out, x, y, e);
}

template <class T>
__global__ void matrix_transpose_naive_kernel(matrix_t<T> d_out, matrix_t<T> d_in) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if( j >= d_in.width || i >= d_in.height )
    return;

  T elem = getElement(d_in, i, j);

  i = blockIdx.y*blockDim.y + threadIdx.x;
  j = blockIdx.x*blockDim.x + threadIdx.y;

  if( j < d_in.width && i < d_in.height ) {
    setElement<T>(d_out, j, i, elem);
  }
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

/*
  __shared__ T a_shared[TILE_SIZE][TILE_SIZE], b_shared[TILE_SIZE][TILE_SIZE];

  int ii = blockIdx.y * TILE_SIZE;
  int jj = blockIdx.x * TILE_SIZE;

  int tidy = threadIdx.y, i = tidy+ii;
  int tidx = threadIdx.x, j = tidx+jj;

  T tmp = 0;

  for(int kk=0; kk<blockDim.x dim; kk+=TILE_SIZE) {
    // Copy to shared memory
    a_shared[tidyx, tidx] = (i< && kk+tidx<U)       ? getElement(a, i, kk+tidx) : 0;
    b_shared[tidy, tidx]  = (j<r.width && kk+tidy<U) ? getElement(b, kk+tidy j)  : 0;

    __syncthreads();

    // Gather results
    for(int k=0; k<TILE_SIZE; k++) {
      tmp += a_shared[tidy][k] * a_shared[k][tidx]
    }

    __syncthreads();
  }

  // Write to global
  if (i<r.height && j<r.width)
    setElement(r,i,j,tmp);
*/
}

#endif // _MATRIX_KERNELS
