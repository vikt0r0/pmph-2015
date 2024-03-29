#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "matrix.cu.h"
#ifdef __CUDACC__
#include "matrix_kernels.cu.h"
#include <cuda_runtime.h>
#endif

#define EPSILON 0.00001
#define MATRIX_SIZE 1024
#define TILE_SIZE 32

#define RAND_FLOAT(min,max) (min + static_cast <float> (rand()) /(static_cast <float> (RAND_MAX/(max-min))))
#define APPROX_EQUAL(a,b,epsilon) (fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon))

struct timeval t_start, t_end, t_diff;

template <typename T>
void matrix_set_element(matrix_t<T> mat, int i, int j, T value) {
  mat.elements[i*mat.width+j] = value;
}

template <typename T>
T matrix_get_element(matrix_t<T> mat, int i, int j) {
  return mat.elements[i*mat.width+j];
}

template <typename T>
bool matrix_is_equal(matrix_t<T> a, matrix_t<T> b) {
  if (a.width != b.width || a.height != b.height) {
    return false;
  }

  bool equal = true;

  for (int i = 0; i < a.height; ++i) {
    for (int j = 0; j < a.width; ++j) {
      if (matrix_get_element(a,i,j) != matrix_get_element(b,i,j)) {
        equal = false;
      }
    }
  }
  return equal;
}

void print_matrix(matrix_t<float> m) {
  int width = m.width;
  int height = m.height;
  for (int i=0; i < height; ++i) {
    for (int j=0; j < width; ++j) {
      printf("%06.3f\t", m.elements[i*width+j]);
    }
    printf("\n");
  }
}

template <>
bool matrix_is_equal(matrix_t<float> a, matrix_t<float> b) {
  if (a.width != b.width || a.height != b.height) {
    return false;
  }

  bool equal = true;

  for (int i = 0; i < a.height; ++i) {
    for (int j = 0; j < a.width; ++j) {
      if (!APPROX_EQUAL(matrix_get_element(a,i,j), matrix_get_element(b,i,j), EPSILON)) {
        equal = false;
        //printf("At i=%d, j=%d - Expected %f, got %f\n", i, j, matrix_get_element(a,i,j), matrix_get_element(b,i,j));
      }
    }
  }
  return equal;
}

void matrix_fill_random_float(matrix_t<float> mat, float min, float max) {
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

    num_blocks_x = (in.width + block_size - 1) / block_size;
    num_blocks_y = (in.height + block_size  - 1) / block_size;

    dim3 blockDim(block_size, block_size);
    dim3 gridDim(num_blocks_x, num_blocks_y);

    matrix_transpose_naive_kernel<T><<< blockDim, gridDim >>>(out, in);
    cudaThreadSynchronize();
  #endif
}

template <typename T>
void matrix_transpose_cuda_tiled(matrix_t<T> out, const matrix_t<T> in) {
  #ifndef __CUDACC__
    // If the CUDA compiler is not used, fall back to OMP implementation.
    matrix_transpose_omp<T>(out, in);
  #else
    // Set up and invoke kernel
    unsigned int num_blocks_x, num_blocks_y;

    num_blocks_x = (in.width + TILE_SIZE  - 1) / TILE_SIZE;
    num_blocks_y = (in.height + TILE_SIZE  - 1) / TILE_SIZE;
 
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(num_blocks_x, num_blocks_y);

    matrix_transpose_tiled_kernel<T, TILE_SIZE><<< gridDim, blockDim >>>(out, in);
    cudaThreadSynchronize();
  #endif
}

template <typename T>
void matrix_mult_seq(matrix_t<T> a, matrix_t<T> b, matrix_t<T> r) {
  for (int i = 0; i < r.height; ++i) {
    for (int j = 0; j < r.width; ++j) {
      T res = 0;
      for (int k = 0; k < a.width; ++k) {
        res += a.elements[i * a.width + k] * b.elements[k * b.width + j];
      }
      r.elements[i * r.width + j] = res;
    }
  }
}

template <typename T>
void matrix_mult_cuda_naive(const unsigned int block_size, matrix_t<T> a, matrix_t<T> b, matrix_t<T> r) {
  #ifdef __CUDACC__
  // Allocate and specify dimensions
  unsigned int num_blocks_x, num_blocks_y;
  num_blocks_x = (r.width + block_size - 1) / block_size;
  num_blocks_y = (r.height + block_size  - 1) / block_size;

  dim3 blockDim(block_size, block_size);
  dim3 gridDim(num_blocks_x, num_blocks_y);
  // Invoke the kernel
  matrix_mult_naive_kernel<<< gridDim, blockDim>>>(a,b,r);
  cudaThreadSynchronize(); 
  #endif
}

template <typename T>
void matrix_mult_cuda_tiled(matrix_t<T> a, matrix_t<T> b, matrix_t<T> r) {
  #ifdef __CUDACC__
  unsigned int block_size = TILE_SIZE;
  // Allocate and specify dimensions
  unsigned int num_blocks_x, num_blocks_y;
  num_blocks_x = (r.width + TILE_SIZE - 1) / TILE_SIZE;
  num_blocks_y = (r.height + TILE_SIZE  - 1) / TILE_SIZE;

  dim3 blockDim(block_size, block_size);
  dim3 gridDim(num_blocks_x, num_blocks_y);
  // Invoke the kernel
  matrix_mult_tiled_kernel<T, TILE_SIZE><<< gridDim, blockDim >>>(a,b,r);
  cudaThreadSynchronize(); 
  #endif
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

void timer_start() {
  gettimeofday(&t_start, NULL);
}

unsigned long int timer_stop() {
  unsigned long int elapsed;
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  return elapsed;
}

// Transpose using sequential implementation
matrix_t<float> test_transpose_seq(matrix_t<float> m_in) {
  unsigned long int elapsed;
  matrix_t<float> m_out_seq;
  m_out_seq.width = m_in.height;
  m_out_seq.height = m_in.width;
  m_out_seq.elements = (float*) malloc(m_out_seq.width * m_out_seq.height * sizeof(float));
  timer_start();
  matrix_transpose_seq<float>(m_out_seq, m_in);
  elapsed = timer_stop();
  printf("Sequential implementation of transpose finished in %lu microseconds!\n", elapsed);
  return m_out_seq;
}

// Transpose using OMP implementation
void test_transpose_omp(matrix_t<float> m_in, matrix_t<float> m_out_seq) {
  #if defined(_OPENMP)
  unsigned long int elapsed;
  matrix_t<float> m_out_omp;
  m_out_omp.width = m_in.height;
  m_out_omp.height = m_in.width;
  m_out_omp.elements = (float*) malloc(m_out_omp.width * m_out_omp.height * sizeof(float));
  timer_start();
  matrix_transpose_omp<float>(m_out_omp, m_in);
  elapsed = timer_stop();
  if (matrix_is_equal(m_out_seq, m_out_omp)) {
    printf("OMP implementation of transpose produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("OMP implementation of transpose produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  free(m_out_omp.elements);
  #else
  printf("OMP not supported by the current compiler... Skipping...\n");
  #endif
}

// Transpose using naive CUDA implementation
void test_transpose_cuda_naive(matrix_t<float> in, matrix_t<float> m_out_seq) {
  #ifdef __CUDACC__
  unsigned long int elapsed;
  // Device structs
  matrix_t<float> d_out, d_in, out;
  d_out.height = out.height = d_in.width  = in.width;
  d_out.width  = out.width  = d_in.height = in.height;
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_out.elements),
    d_out.width * d_out.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_in.elements),
    d_in.width * d_in.height * sizeof(float)
  );
  cudaMemcpy(
    d_in.elements, in.elements,
    d_in.width * d_in.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  out.elements = (float*) malloc(
    out.width * out.height * sizeof(float)
  );
  timer_start();
  matrix_transpose_cuda_naive<float>(512, d_out, d_in);
  elapsed = timer_stop();
  cudaMemcpy(
    out.elements, d_out.elements,
    out.width * out.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(out, m_out_seq)) {
    printf("CUDA implementation (naive) of transpose produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (naive) of transpose produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  cudaFree(d_in.elements);
  cudaFree(d_out.elements);
  free(out.elements);
  #else
  printf("CUDA not supported by the current compiler... Skipping...\n");
  #endif
}
// Transpose using tiled CUDA implementation
void test_transpose_cuda_tiled(matrix_t<float> in, matrix_t<float> m_out_seq) {
  #ifdef __CUDACC__
  unsigned long int elapsed;
  // Device structs
  matrix_t<float> d_out, d_in, out;
  d_out.height = out.height = d_in.width  = in.width;
  d_out.width  = out.width  = d_in.height = in.height;
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_out.elements),
    d_out.width * d_out.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_in.elements),
    d_in.width * d_in.height * sizeof(float)
  );
  cudaMemcpy(
    d_in.elements, in.elements,
    d_in.width * d_in.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  out.elements = (float*) malloc(
    out.width * out.height * sizeof(float)
  );
  timer_start();
  matrix_transpose_cuda_tiled<float>(d_out, d_in);
  elapsed = timer_stop();
  cudaMemcpy(
    out.elements, d_out.elements,
    out.width * out.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(out, m_out_seq)) {
    printf("CUDA implementation (tiled) of transpose produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (tiled) of transpose produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  cudaFree(d_in.elements);
  cudaFree(d_out.elements);
  free(out.elements);
  #else
  printf("CUDA not supported by the current compiler... Skipping...\n");
  #endif
}

void test_transpose_all() {
  // Create input matrices
  matrix_t<float> m_in;
  m_in.width = MATRIX_SIZE;
  m_in.height = MATRIX_SIZE;
  m_in.elements = (float*) malloc(m_in.width * m_in.height * sizeof(float));
  matrix_fill_random_float(m_in,10.0,10.0);

  // Get output matrix
  matrix_t<float> m_out_seq = test_transpose_seq(m_in);  

  // Test transpose using OMP
  test_transpose_omp(m_in, m_out_seq);

  // Test transpose using naive CUDA implementation
  test_transpose_cuda_naive(m_in, m_out_seq);

  // Test transpose using tiled CUDA implementation
  test_transpose_cuda_tiled(m_in, m_out_seq);  

  free(m_in.elements);
}

// Multiply using sequential implementation
matrix_t<float> test_mat_mult_seq(matrix_t<float> a, matrix_t<float> b) {
  unsigned long elapsed;
  matrix_t<float> out;
  out.height = a.height;
  out.width = b.width;
  out.elements = (float*) malloc(out.width * out.height * sizeof(float));
  timer_start();
  matrix_mult_seq<float>(a, b, out);
  elapsed = timer_stop();
  printf("Sequential implementation of multiply finished in %lu microseconds!\n", elapsed);
  return out;
}

// Multiply using naive CUDA implementation
void test_mat_mult_cuda_naive(matrix_t<float> a, matrix_t<float> b, matrix_t<float> m_out_seq) {
  #ifdef __CUDACC__
  // Device structs
  unsigned long elapsed;
  matrix_t<float> d_in_0, d_in_1, d_out, out;
  d_in_0 = a;
  d_in_1 = b;
  out.height = d_out.height = d_in_0.height;
  out.width  = d_out.width  = d_in_1.width;
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_in_0.elements),
    d_in_0.width * d_in_0.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_in_1.elements),
    d_in_1.width * d_in_1.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_out.elements),
    d_out.width * d_out.height * sizeof(float)
  );
  out.elements = (float*) malloc(
    out.width * out.height * sizeof(float)
  );
  cudaMemcpy(
    d_in_0.elements, a.elements,
    d_in_0.width * d_in_0.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  cudaMemcpy(
    d_in_1.elements, b.elements,
    d_in_1.width * d_in_1.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  timer_start();
  matrix_mult_cuda_naive<float>(512, d_in_0, d_in_1, d_out);
  elapsed = timer_stop();
  cudaMemcpy(
    out.elements, d_out.elements,
    out.width * out.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(out, m_out_seq)) {
    printf("CUDA implementation (naive) of multiply produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (naive) of multiply produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  cudaFree(d_in_0.elements);
  cudaFree(d_in_1.elements);
  cudaFree(d_out.elements);
  free(out.elements);
  #else
  printf("CUDA not supported by the current compiler... Skipping...\n");
  #endif
}

// Multiply using tiled CUDA implementation
void test_mat_mult_cuda_tiled(matrix_t<float> a, matrix_t<float> b, matrix_t<float> m_out_seq) {
  #ifdef __CUDACC__
  unsigned long elapsed;
  // Device structs
  matrix_t<float> d_in_0, d_in_1, d_out, out;
  d_in_0 = a;
  d_in_1 = b;
  out.height = d_out.height = d_in_0.height;
  out.width  = d_out.width  = d_in_1.width;
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_in_0.elements),
    d_in_0.width * d_in_0.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_in_1.elements),
    d_in_1.width * d_in_1.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_out.elements),
    d_out.width * d_out.height * sizeof(float)
  );
  out.elements = (float*) malloc(
    out.width * out.height * sizeof(float)
  );
  cudaMemcpy(
    d_in_0.elements, a.elements,
    d_in_0.width * d_in_0.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  cudaMemcpy(
    d_in_1.elements, b.elements,
    d_in_1.width * d_in_1.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  timer_start();
  matrix_mult_cuda_tiled<float>(d_in_0, d_in_1, d_out);
  elapsed = timer_stop();
  cudaMemcpy(
    out.elements, d_out.elements,
    out.width * out.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(out, m_out_seq)) {
    printf("CUDA implementation (tiled) of multiply produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (tiled) of multiply produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  cudaFree(d_in_0.elements);
  cudaFree(d_in_1.elements);
  cudaFree(d_out.elements);
  free(out.elements);
  #else
  printf("CUDA not supported by the current compiler... Skipping...\n");
  #endif
}

void test_multiply_all() {
  // Create input matrices
  matrix_t<float> m_in_0;
  m_in_0.width = MATRIX_SIZE;
  m_in_0.height = MATRIX_SIZE;
  m_in_0.elements = (float*) malloc(m_in_0.width * m_in_0.height * sizeof(float));
   matrix_fill_random_float(m_in_0,0.0,10.0);

  matrix_t<float> m_in_1;
  m_in_1.width = MATRIX_SIZE;
  m_in_1.height = MATRIX_SIZE;
  m_in_1.elements = (float*) malloc(m_in_1.width * m_in_1.height * sizeof(float));
  matrix_fill_random_float(m_in_1,0.0,10.0);

  // Get output matrix
  matrix_t<float> m_out_seq = test_mat_mult_seq(m_in_0, m_in_1);  

  // Test naive CUDA implementation
  test_mat_mult_cuda_naive(m_in_0, m_in_1, m_out_seq);  

  // Test tiled CUDA implementation
  test_mat_mult_cuda_tiled(m_in_0, m_in_1, m_out_seq);  

  free(m_in_0.elements);
  free(m_in_1.elements);
  free(m_out_seq.elements);
}


matrix_t<float> sqrt_squared_sum_all_seq(matrix_t<float> a) {
  unsigned int elapsed;

  matrix_t<float> b;
  b = a;
  b.elements = (float*) malloc(b.width * b.height * sizeof(float));

  timer_start();
  float accum;
  for (int i = 0; i < a.height; ++i) {
    accum = matrix_get_element(a, i, 0) * matrix_get_element(a, i, 0);
    matrix_set_element(b, i, 0, accum);
    for (int j = 1; j < b.width; ++j) {
      float tmpA = matrix_get_element(a, i, j);
      accum = sqrt(accum) + tmpA * tmpA;
      matrix_set_element(b, i, j, accum);
    }
  }

  elapsed = timer_stop();
  printf("Sequential implementation of squareroot sums finished in %lu microseconds!\n", elapsed);

  return b;
}

matrix_t<float> sqrt_squared_sum_omp(matrix_t<float> a) {
  unsigned int elapsed;

  matrix_t<float> b;
  b = a;
  b.elements = (float*) malloc(b.width * b.height * sizeof(float));

  timer_start();
  float accum[a.height];
  #if defined(_OPENMP)
  #pragma omp parallel for
  #endif
  for (int i = 0; i < a.height; ++i) {
    accum[i] = matrix_get_element(a, i, 0) * matrix_get_element(a, i, 0);
    matrix_set_element(b, i, 0, accum[i]);
    for (int j = 1; j < b.width; ++j) {
      float tmpA = matrix_get_element(a, i, j);
      accum[i] = sqrt(accum[i]) + tmpA * tmpA;
      matrix_set_element(b, i, j, accum[i]);
    }
  }

  elapsed = timer_stop();
  printf("OMP implementation of squareroot sums finished in %lu microseconds!\n", elapsed);

  return b;
}

// Squareroot sums using OMP implementation
void test_sqrt_squared_sum_omp(matrix_t<float> m_in, matrix_t<float> b_seq) {
  #if defined(_OPENMP)
  unsigned long int elapsed;
  timer_start();
  matrix_t<float> b_out_omp = sqrt_squared_sum_omp<float>(m_in);
  elapsed = timer_stop();
  if (matrix_is_equal(b_seq, b_out_omp)) {
    printf("OMP implementation of squareroot-square-sums produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("OMP implementation of squareroot-square-sums produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  free(b_out_omp.elements);
  #else
  printf("OMP not supported by the current compiler... Skipping...\n");
  #endif
}


void sqrt_squared_sum_cuda_naive(const unsigned int block_size, matrix_t<float> a, matrix_t<float> b) {
  #ifdef __CUDACC__
    // Set up and invoke kernel
    unsigned int num_blocks;

    num_blocks = (a.height + block_size - 1) / block_size;

    sqrt_squared_sum_naive_kernel<<< num_blocks, block_size >>>(a, b);
    cudaThreadSynchronize();
  #endif
}

void sqrt_squared_sum_cuda_coalesced(const unsigned int block_size, matrix_t<float> a, matrix_t<float> b) {
  #ifdef __CUDACC__
    // Set up and invoke kernel
    unsigned int num_blocks;

    num_blocks = (a.height + block_size - 1) / block_size;
 
    sqrt_squared_sum_coalesced_kernel<<< num_blocks, block_size >>>(a, b);
    cudaThreadSynchronize();
  #endif
}

void test_sqrt_squared_sum_cuda_naive(matrix_t<float> a, matrix_t<float> b) {
  #ifdef __CUDACC__
  unsigned long elapsed;
  // Device structs
  matrix_t<float> d_a, d_b, b_out;
  d_a = a;
  d_b = b_out = b;
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_a.elements),
    d_a.width * d_a.height * sizeof(float)
  );
  cudaMemcpy(
    d_a.elements, a.elements,
    d_a.width * d_a.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  cudaMalloc(
    (void**) &(d_b.elements),
    d_b.width * d_b.height * sizeof(float)
  );
  b_out.elements = (float*) malloc(
    b_out.width * b_out.height * sizeof(float)
  );
  timer_start();
  sqrt_squared_sum_cuda_naive(512, d_a, d_b);
  elapsed = timer_stop();
  cudaMemcpy(
    b_out.elements, d_b.elements,
    b_out.width * b_out.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(b_out, b)) {
    printf("CUDA implementation (naive) of squareroot sums produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (naive) of squareroot sums produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  cudaFree(d_b.elements);
  cudaFree(d_a.elements);
  free(b_out.elements);
  #else
  printf("CUDA not supported by the current compiler... Skipping...\n");
  #endif
}

void test_sqrt_squared_sum_cuda_coalesced(matrix_t<float> a, matrix_t<float> b) {
  #ifdef __CUDACC__
  unsigned long elapsed;
  // Device structs
  matrix_t<float> d_a, d_aT, d_b, d_bT, b_out;
  d_a = a;
  d_b = b_out = b;
  d_aT.width = d_a.height;
  d_aT.height = d_a.width;
  d_bT.width = d_b.height;
  d_bT.height = d_b.width;
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_a.elements),
    d_a.width * d_a.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_aT.elements),
    d_aT.width * d_aT.height * sizeof(float)
  );
  cudaMemcpy(
    d_a.elements, a.elements,
    d_a.width * d_a.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  cudaMalloc(
    (void**) &(d_b.elements),
    d_b.width * d_b.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_bT.elements),
    d_bT.width * d_bT.height * sizeof(float)
  );
  b_out.elements = (float*) malloc(
    b_out.width * b_out.height * sizeof(float)
  );
  timer_start();
  matrix_transpose_cuda_tiled<float>(d_aT, d_a);
  sqrt_squared_sum_cuda_coalesced(512, d_aT, d_bT);
  matrix_transpose_cuda_tiled<float>(d_b, d_bT);
  elapsed = timer_stop();
  cudaMemcpy(
    b_out.elements, d_b.elements,
    b_out.width * b_out.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(b_out, b)) {
    printf("CUDA implementation (naive) of squareroot sums produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (naive) of squareroot sums produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  cudaFree(d_b.elements);
  cudaFree(d_a.elements);
  free(b_out.elements);
  #else
  printf("CUDA not supported by the current compiler... Skipping...\n");
  #endif
}
void test_sqrt_squared_sum_all() {
  // Create input matrices
  matrix_t<float> a;
  a.width = 64;
  a.height = MATRIX_SIZE;
  a.elements = (float*) malloc(a.width * a.height * sizeof(float));
  matrix_fill_random_float(a,1.0,1.0);

  // Get output matrix
  matrix_t<float> b = sqrt_squared_sum_all_seq(a);  

  // Test OMP implementation
  test_sqrt_squared_sum_omp(a, b);
  
  // Test naive CUDA implementation
  test_sqrt_squared_sum_cuda_naive(a, b);  

  // Test tiled CUDA implementation
  test_sqrt_squared_sum_cuda_coalesced(a, b);  


  free(a.elements);
  free(b.elements);
}


int main(int argc, char *argv[]) {
  test_transpose_all();
  test_sqrt_squared_sum_all();
  test_multiply_all();
  return 0;
}
