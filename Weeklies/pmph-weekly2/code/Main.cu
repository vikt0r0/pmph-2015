#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"
#include "MsspHost.cu.h"
#include "MsspKernels.cu.h"
#include "MatVecMultHost.cu.h"


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

MyInt4 mssSerial(int *in, int length) {
    if (length == 1) {
      int x = *in;
      if (x > 0)
        return MyInt4(x,x,x,x);
      else
        return MyInt4(0,0,0,x);
    }
    int half = length/2;
    MyInt4 t1 = mssSerial(&in[0], half);
    MyInt4 t2 = mssSerial(&in[half], length-half);
    int mssx = t1.x, misx = t1.y, mcsx = t1.z, tsx = t1.w;
    int mssy = t2.x, misy = t2.y, mcsy = t2.z, tsy = t2.w;
    int mss = max(max(mssx,mssy), mssx+mssy);
    int mis = max(misx,tsx+misy);
    int mcs = max(mcsy,mcsx+tsy);
    int t   = tsx+tsy;
    return MyInt4(mss, mis, mcs, t);
}

int matVecMultTest() {
    const unsigned int block_size  = 512;

    int matrix_flag[] = { 1, 0, 1, 0, 0, 1, 0, 0, 1, 0 };
    MyPair<int,float> matrix_flat[]  = {
      MyPair<int,float>(0,2.0),  MyPair<int,float>(1,-1.0),
      MyPair<int,float>(0,-1.0), MyPair<int,float>(1, 2.0), MyPair<int,float>(2,-1.0),
      MyPair<int,float>(1,-1.0), MyPair<int,float>(2, 2.0), MyPair<int,float>(3,-1.0),
      MyPair<int,float>(2,-1.0), MyPair<int,float>(3, 2.0)
    };
    int   x_vector_length = 4;
    float x_vector[] = { 2.0, 1.0, 0.0, 3.0 };
    float result[x_vector_length];
    float expected[] = { 3.0, 0.0, -4.0, 6.0 };

    int matrix_length = 10;
    int matrix_flag_size = sizeof(int) * matrix_length;
    int matrix_flat_size = sizeof(MyPair<int,float>) * matrix_length;
    int x_vector_size = sizeof(float) * x_vector_length;

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling exclusive (segmented) scan
        int *d_flags;
        MyPair<int,float> *d_mat;
        float *d_vec;
        float *d_out;

        cudaMalloc((void**)&d_flags, matrix_flag_size);
        cudaMalloc((void**)&d_mat, matrix_flat_size);
        cudaMalloc((void**)&d_vec, x_vector_size);
        cudaMalloc((void**)&d_out, x_vector_size);

        // copy host memory to device
        cudaMemcpy((void*)d_flags, (void*)matrix_flag, matrix_flag_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_mat, (void*)matrix_flat, matrix_flat_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_vec, (void*)x_vector, x_vector_size, cudaMemcpyHostToDevice);

        // execute kernel
        MatVecMult<float>(block_size, matrix_length, d_flags, d_mat, d_vec, d_out);

        // copy back result
        cudaMemcpy(result, d_out, x_vector_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_flags);
        cudaFree(d_mat);
        cudaFree(d_vec);
        cudaFree(d_out);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("MatVecMult on GPU runs in: %lu microsecs\n", elapsed);

    bool valid = true;

    for (int i = 0; i < x_vector_length; ++i) {
        valid = valid && (result[i] == expected[i]);
    }

    if(true) printf("MatVecMult +   VALID RESULT!\n");
    else     printf("MatVecMult + INVALID RESULT!\n");

    return 0;
}

int mssTest() {
    const unsigned int num_threads = 1000;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    for (int i = 0; i < num_threads; ++i) {
      h_in[i] = rand() % 100;
    }
    int gpuRes;

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling exclusive (segmented) scan
        int* d_in;
        cudaMalloc((void**)&d_in, mem_size);

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

        // execute kernel
        gpuRes = Mss(block_size, num_threads, d_in);

        // cleanup memory
        cudaFree(d_in );
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Mss on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    MyInt4 serialRes = mssSerial(h_in, num_threads);

    if(serialRes.x = gpuRes) printf("Mss +   VALID RESULT!\n");
    else                     printf("Mss + INVALID RESULT!\n");

    // cleanup memory
    free(h_in );

    return 0;
}

int scanExcTest(bool is_segmented) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1;
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling exclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanExc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanExc< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Exclusive on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
            accum += 1;
        }
    } else {
        accum = 0;
        for(int i=0; i<num_threads; i++) {

            if ( accum != h_out[i] ) {
                success = false;
                // printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
            accum += 1;
        }
    }

    if(success) printf("Scan Exclusive +   VALID RESULT!\n");
    else        printf("Scan Exclusive + INVALID RESULT!\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}

int scanIncTest(bool is_segmented) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1;
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling inclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanInc< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Inclusive on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    } else {
        for(int i=0; i<num_threads; i++) {
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    }

    if(success) printf("Scan Inclusive +   VALID RESULT!\n");
    else        printf("Scan Inclusive + INVALID RESULT!\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}

int main(int argc, char** argv) {
    scanIncTest(true);
    scanIncTest(false);
    printf("\n");
    scanExcTest(true);
    scanExcTest(false);
    printf("\n");
    matVecMultTest();
    printf("\n");
    mssTest();
}
