#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.cu.h"

#define MATRIX_SIZE 1024

struct timeval t_start, t_end, t_diff;

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
  unsigned long int elapsed;
  // Create input matrix
  matrix_t<float> m_in;
  m_in.width = MATRIX_SIZE;
  m_in.height = MATRIX_SIZE;
  matrix_fill_random(m_in,0.0,1000.0);
  // Transpose using sequential implementation
  matrix_t<float> m_out_seq;
  m_out_seq.width = m_in.height;
  m_out_seq.height = m_in.width;
  m_out_seq.elements = malloc(m_out_seq.width * m_out_seq.height * sizeof(float));
  timer_start();
  matrix_transpose_seq(m_out_seq, m_in);
  elapsed = timer_stop();
  printf("Sequential implementation of transpose took %lu microseconds!\n", elapsed);
  // Transpose using OMP implementation
  #if defined(_OPENMP)
  matrix_t<float> m_out_omp;
  m_out_omp.width = m_in.height;
  m_out_omp.height = m_in.width;
  m_out_omp.elements = malloc(m_out_omp.width * m_out_omp.height * sizeof(float));
  timer_start();
  matrix_transpose_omp(m_out_omp, m_in);
  elapsed = timer_stop();
  printf("OMP implementation of transpose took %lu microseconds!\n", elapsed);
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
  m_out_cuda_naive.elements = malloc(
    m_out_cuda_naive.width * m_out_cuda_naive.height * sizeof(float)
  );
  // Copy input array to device
  cudaMalloc(
    (void**) &(d_m_in_cuda_naive.elements),
    d_m_in_cuda.width * d_m_in_cuda.height * sizeof(float)
  );
  cudaMalloc(
    (void**) &(d_m_out_cuda_naive.elements),
    d_m_out_cuda.width * d_m_out_cuda.height * sizeof(float)
  );
  cudaMemcpy(
    d_m_in_cuda_naive.elements, m_in.elements,
    d_m_in_cuda_naive.width * d_m_in_cuda_naive.height * sizeof(float),
    cudaMemcpyHostToDevice
  );
  timer_start();
  matrix_transpose_cuda_naive(d_m_out_cuda_naive, d_m_in_cuda_naive);
  elapsed = timer_stop();
  printf("CUDA implementation (naive) of transpose took %lu microseconds!\n", elapsed);
  cudaMemcpy(
    m_out_cuda_naive.elements, d_m_out_cuda_naive.elements,
    d_m_out_cuda_naive.width * d_m_out_cuda_naive.height * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  #else
  printf("CUDA not supported by the current compiler... Skipping...");
  #endif
  return 0;
}
