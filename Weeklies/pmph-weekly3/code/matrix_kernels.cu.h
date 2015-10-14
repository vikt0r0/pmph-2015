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
__global__ void matrix_transpose_naive_kernel(matrix_t<T> d_out, matrix_t<T> d_in) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if( j >= d_in.width || i >= d_in.height )
    return;

  T elem = getElement(d_in, i, j);
  setElement(d_out, j, i, elem);
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
  const unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int i = blockDim.y * blockIdx.y + threadIdx.y;
  T res = 0.0;
  // Check if we are within bounds
  if (j < r.width && i < r.height) {
    for (int k = 0; k < a.width; ++k)
      res += getElement<T>(a, i, k) * getElement<T>(b, k, j);
    setElement<T>(r, i, j, res);
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

         if (k*TILE_SIZE + threadIdx.x < a.width && i < a.height) 
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
        setElement(r, i, j, tmp);
}

__global__ void sqrt_squared_sum_naive_kernel(matrix_t<float> a, matrix_t<float> b) {
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  float accum = getElement(a, i, 0) * getElement(a, i, 0);
  setElement<float>(b, i, 0, accum);

  if (i < a.height) {
    for (int j = 1; j < b.width; ++j) {
      float tmpA = getElement(a, i, j);
      accum = sqrtf(accum) + tmpA*tmpA;
      setElement(b, i, j, accum); 
    }
  }
}

__global__ void sqrt_squared_sum_coalesced_kernel(matrix_t<float> a, matrix_t<float> b) {
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  float accum = getElement(a, 0, i) * getElement(a, 0, i);
  setElement<float>(b, 0, i, accum);

  if (i < a.width) {
    for (int j = 1; j < b.height; ++j) {
      float tmpA = getElement(a, j, i);
      accum = sqrtf(accum) + tmpA*tmpA;
      setElement(b, j, i, accum); 
    }
  }
}

#endif // _MATRIX_KERNELS
