#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "matrixMult.h"

// Thread block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 32

__global__ void matrixMultKernel(Matrix a, Matrix b, Matrix r) {
  const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
  float res = 0.0;
  // Check if we are within bounds
  if (col < r.width && row < r.height) {
    for (int i = 0; i < a.width; ++i)
      res += a.elements[row * a.width + i] * b.elements[i * b.width + col];
    r.elements[row * r.width + col] = res;
  }
}

int matrixMult(Matrix a, Matrix b, Matrix r) {
  cudaError_t err;
  size_t size;
  
  // Allocate device memory for the operands and the result
  Matrix d_a, d_b, d_r;
  d_a = a; d_b = b; d_r = r;
  
  // Allocate and initialize device matrix a
  size = d_a.width*d_a.height;
  err = cudaMalloc(&d_a.elements, size * sizeof(float));  
  if (err != cudaSuccess)
    return MAT_MULT_STATUS_ERROR;
  err = cudaMemcpy(d_a.elements, a.elements, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_a.elements);
    return MAT_MULT_STATUS_ERROR;
  }

  // Allocate and initialize device matrix b
  size = d_b.width*d_b.height;
  err = cudaMalloc(&d_b.elements, size * sizeof(float));  
  if (err != cudaSuccess) {
    cudaFree(d_a.elements);
    return MAT_MULT_STATUS_ERROR;
  }
  err = cudaMemcpy(d_b.elements, b.elements, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    return MAT_MULT_STATUS_ERROR;
  }

  // Allocate and initialize device matrix r
  size = d_r.width*d_r.height;
  err = cudaMalloc(&d_r.elements, size * sizeof(float));  
  if (err != cudaSuccess) {
    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    return MAT_MULT_STATUS_ERROR;
  }

  // Allocate and specify dimensions
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid(( b.width + dimBlock.x - 1 ) / dimBlock.x,
               ( a.height + dimBlock.y - 1) / dimBlock.y);

  // Invoke the kernel
  matrixMultKernel<<<dimGrid, dimBlock>>>(d_a,d_b,d_r);
  err = cudaThreadSynchronize();
 
  // Copy result to host
  int result = MAT_MULT_STATUS_SUCCESS;
  if (err != cudaSuccess) {
    result = MAT_MULT_STATUS_ERROR;
  } else {
    err = cudaMemcpy(r.elements, d_r.elements, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      result = MAT_MULT_STATUS_ERROR;
    }
  }
  // Clean up
  cudaFree(d_a.elements);
  cudaFree(d_b.elements);
  cudaFree(d_r.elements);

  return result;
}

void matrixMultSeq(Matrix a, Matrix b, Matrix r) {
  for (int i = 0; i < r.height; ++i) {
    for (int j = 0; j < r.width; ++j) {
      float res = 0.0;
      for (int k = 0; k < a.width; ++k) {
        res += a.elements[i * a.width + k] * b.elements[k * b.width + j];
      }
      r.elements[i * r.width + j] = res;
    }
  }
}

void printMatrix(Matrix m) {
  int width = m.width;
  int height = m.height;
  for (int i=0; i < height; ++i) {
    for (int j=0; j < width; ++j) {
      printf("%06.3f\t", m.elements[i*width+j]);
    }
    printf("\n");
  }
}

// Verify the product of a * b == c
bool verifyProduct(Matrix a, Matrix b, Matrix c) {
  Matrix r;
  r.height = a.height;
  r.width = b.width;
  r.elements = (float*) malloc(r.height * r.width * sizeof(float));
  matrixMultSeq(a,b,r);
  // Compare results
  bool equal = true;
  for (int i = 0; i < r.width * r.height; ++i) {
    equal = equal && (r.elements[i] == c.elements[i]);
  }
  free(r.elements);
  return equal;
}

void usage() {
  printf("Usage: `matrixMult 5 10 7` will create a 5 x 10 matrix and a 10 x 7 matrix with random values in the range [0..9] and multiply them.\n");
}


int main(int argc, char *argv[]) {
  // Check arguments
  if (argc < 4) {
    usage();
    return -1;
  }
  
  // Parse arguments
  int m, n, o;
  m = atoi(argv[1]);
  n = atoi(argv[2]);
  o = atoi(argv[3]);

  // Initialize matrices
  Matrix a, b, r;

  // Initialize sizes and allocate data arrays
  a.height = m; a.width = n;
  b.height = n; b.width = o;
  r.height = a.height; r.width = b.width;
  a.elements = (float*) malloc(a.width * a.height * sizeof(float));
  b.elements = (float*) malloc(b.width * b.height * sizeof(float));
  r.elements = (float*) malloc(r.width * r.height * sizeof(float));
  for (int i = 0; i < a.height; ++i)
    for (int j = 0; j < a.width; ++j)
      a.elements[i*a.width+j] = (float) (rand() % 10);
  for (int i = 0; i < b.height; ++i)
    for (int j = 0; j < b.width; ++j)
      b.elements[i*b.width+j] = (float) (rand() % 10);

  // Perform the multiplication
  int err = matrixMult(a, b, r);

  // Print the result
  printf("Matrix A (%d by %d):\n", a.height, a.width);
  printMatrix(a);
  printf("\nMatrix B (%d by %d):\n", b.height, b.width);
  printMatrix(b);
  printf("\nMatrix R (%d by %d):\n", r.height, r.width);
  printMatrix(r);

  printf("\nChecking result... ");

  if (verifyProduct(a,b,r))
    printf("The computed matrix product verified as correct.\n");
  else
    printf("The computed matrix product is incorrect!\n");

  return err;
}
