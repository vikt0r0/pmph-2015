#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifdef __CUDACC__
#include "matrix_kernels.cu.h"
#include <cuda_runtime.h>
#endif

#include "matrix.cu.h"

#define EPSILON 0.00001
#define MATRIX_SIZE 1024

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

template <>
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

int main(int argc, char *argv[]) {
  printf("hejsa");
  unsigned long int elapsed;
  // Create input matrix
  matrix_t<float> m_in;
  m_in.width = MATRIX_SIZE;
  m_in.height = MATRIX_SIZE;
  matrix_fill_random_float(m_in,0.0,1000.0);
  // Transpose using sequential implementation
  matrix_t<float> m_out_seq;
  m_out_seq.width = m_in.height;
  m_out_seq.height = m_in.width;
  m_out_seq.elements = (float*) malloc(m_out_seq.width * m_out_seq.height * sizeof(float));
  timer_start();
  matrix_transpose_seq<float>(m_out_seq, m_in);
  elapsed = timer_stop();
  printf("Sequential implementation of transpose finished in %lu microseconds!\n", elapsed);
  // Transpose using OMP implementation
  #if defined(_OPENMP)
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
  #else
  printf("OMP not supported by the current compiler... Skipping...");
  #endif
  // Transpose using naive CUDA implementation
  #ifdef __CUDACC__
  // Device structs
  matrix_t<float> d_m_out_cuda_naive, d_m_in_cuda_naive, m_out_cuda_naive;
  d_m_in_cuda_naive.width   = m_in.width;
  d_m_in_cuda_naive.height  = m_in.height;
  d_m_out_cuda_naive.width  = m_in.height;
  d_m_out_cuda_naive.height = m_in.width;
  m_out_cuda_naive.width    = m_in.height;
  m_out_cuda_naive.height   = m_in.width;
  m_out_cuda_naive.elements = (float*) malloc(
    m_out_cuda_naive.width * m_out_cuda_naive.height * sizeof(float)
  );
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_m_in_cuda_naive.elements),
    d_m_in_cuda_naive.width * d_m_in_cuda_naive.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_m_out_cuda_naive.elements),
    d_m_out_cuda_naive.width * d_m_out_cuda_naive.height * sizeof(float)
  );
  cudaMemcpy(
    d_m_in_cuda_naive.elements, m_in.elements,
    d_m_in_cuda_naive.width * d_m_in_cuda_naive.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  timer_start();
  matrix_transpose_cuda_naive<float>(512,d_m_out_cuda_naive, d_m_in_cuda_naive);
  elapsed = timer_stop();
  cudaMemcpy(
    m_out_cuda_naive.elements, d_m_out_cuda_naive.elements,
    d_m_out_cuda_naive.width * d_m_out_cuda_naive.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  if (matrix_is_equal(m_out_seq, m_out_cuda_naive)) {
    printf("CUDA implementation (naive) of transpose produced the CORRECT result in %lu microseconds!\n", elapsed);
  } else {
    printf("CUDA implementation (naive) of transpose produced an INCORRECT result in %lu microseconds!\n", elapsed);
  }
  #else
  printf("CUDA not supported by the current compiler... Skipping...");
  #endif
  return 0;
}
