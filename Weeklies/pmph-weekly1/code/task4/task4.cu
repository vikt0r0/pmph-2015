#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>

#define NUM_ELEMENTS 753411
#define EPSILON 0.000005
#define NUM_THREADS_PER_BLOCK 256
#define NUM_BLOCKS NUM_ELEMENTS/NUM_THREADS_PER_BLOCK+1

void functionSerial(float* d_in, float *d_out) {
  float x;
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    x = d_in[i]; x = (x/(x-2.3));
    d_out[i] = x * x * x;
  }
}

__global__ void functionKernel(float* d_in, float *d_out) {
    const unsigned int lid = threadIdx.x; // local id inside a block
    const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
    float x;
    if (gid < NUM_ELEMENTS) {
      x = d_in[gid]; x = (x/(x-2.3));
      d_out[gid] = x * x * x;
    }
}

bool equal(float f0, float f1) {
    return fabs(f0 - f1) < EPSILON; 
}

int timeval_subtract(struct timeval* result, struct timeval* t2,struct timeval* t1) {
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
                    (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

int main(int argc, char** argv) {
    unsigned long int elapsed_gpu, elapsed_cpu, elapsed_gpu_memcpy;
    struct timeval t_start, t_end, t_diff;
    unsigned int mem_size = NUM_ELEMENTS*sizeof(float);

    // allocate host memory
    float* h_in = (float*) malloc(mem_size);
    float* h_out_gpu = (float*) malloc(mem_size);
    float* h_out_cpu = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<NUM_ELEMENTS; ++i){
        h_in[i] = (float)i+1.0;
    }
    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    gettimeofday(&t_start, NULL);
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed_gpu_memcpy = t_diff.tv_sec*1e6+t_diff.tv_usec;    

    // execute the kernel on GPU
    gettimeofday(&t_start, NULL);
    functionKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_in, d_out);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed_gpu = t_diff.tv_sec*1e6+t_diff.tv_usec;    

    // copy result from ddevice to host 
    gettimeofday(&t_start, NULL);
    cudaMemcpy(h_out_gpu, d_out, mem_size, cudaMemcpyDeviceToHost);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed_gpu_memcpy += t_diff.tv_sec*1e6+t_diff.tv_usec;    

    // compute the function on CPU
    gettimeofday(&t_start, NULL);
    functionSerial(h_in, h_out_cpu); 
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed_cpu = t_diff.tv_sec*1e6+t_diff.tv_usec;    

    // check if result is valid
    bool valid = true;
    for(unsigned int i=0; i<NUM_ELEMENTS; ++i)
      valid = valid && equal(h_out_gpu[i], h_out_cpu[i]);

    // report back
    if (valid)
      printf("VALID\n");
    else
      printf("INVALID\n");

    printf("GPU took %d microseconds (%.2fms)\n",elapsed_gpu_memcpy+elapsed_gpu,(elapsed_gpu_memcpy+elapsed_gpu)/1000.0);
    printf("-- of which  %d microseconds (%.2fms) was spent copying memory\n",elapsed_gpu_memcpy,elapsed_gpu_memcpy/1000.0);
    printf("-- of which  %d microseconds (%.2fms) was spent in the kernel\n",elapsed_gpu,elapsed_gpu/1000.0);
    printf("CPU took a total of  %d microseconds (%.2fms)\n",elapsed_cpu,elapsed_cpu/1000.0);

    // clean-up memory
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);
    cudaFree(d_in);
    cudaFree(d_out);
}
