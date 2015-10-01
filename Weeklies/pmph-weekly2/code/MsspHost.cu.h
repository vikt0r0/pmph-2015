#ifndef MSSP_HOST
#define MSSP_HOST

#include "MsspKernels.cu.h"
#include "ScanHost.cu.h"

#include <sys/time.h>
#include <time.h>

/**
 * block_size is the size of the cuda block (must be a multiple
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates
 *                start of a segment.
 * d_out      is the output GPU array -- if you want
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator
 *                and should have an implementation similar to
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates,
 *                e.g., float or int.
 */
int Mss( const unsigned int  block_size,
         const unsigned long d_size,
         int* d_in  //device
) {
    int res;
    unsigned int num_blocks;
    //unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    MyInt4 *inp_lift;
    MyInt4 *out_lift;
    cudaMalloc((void**)&inp_lift, d_size*sizeof(MyInt4));
    cudaMalloc((void**)&out_lift, d_size*sizeof(MyInt4));

    // Map to lift the input array
    msspTrivialMap <<< num_blocks, block_size >>> (d_in, inp_lift, d_size);
    cudaThreadSynchronize();

    // If we scan, we can simply return the last element, which is the reduction
    scanInc<MsspOp, MyInt4>(block_size, d_size, inp_lift, out_lift);
    cudaThreadSynchronize();

    // Get the result
    cudaMemcpy(&res, &out_lift[d_size-1].x, sizeof(int), cudaMemcpyDeviceToHost);

    return res;
}

#endif //MSSP_HOST
