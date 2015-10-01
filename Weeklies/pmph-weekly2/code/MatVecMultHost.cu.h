#ifndef MATVECMULT_HOST
#define MATVECMULT_HOST

#include "MatVecMultKernels.cu.h"
#include "ScanHost.cu.h"
#include "ScanKernels.cu.h"

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
template <class T>
void MatVecMult( const unsigned int  block_size,
                const unsigned long d_size,
                int                *d_flags,    //device
                MyPair<int, T>     *d_mat,      //device,
                T                  *d_vec,      //device,
                T                  *d_out       //device
) {
    unsigned int num_blocks;
    //unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    T *d_prods;
    cudaMalloc((void**)&d_prods, d_size*sizeof(T));

    //r Compute the products
    matVecMultProdsKernel<T> <<< num_blocks, block_size >>> (d_size, d_mat, d_vec, d_prods);
    cudaThreadSynchronize();

    T *d_prods_sums;
    cudaMalloc((void**)&d_prods_sums, d_size*sizeof(T));

    // Scan each subarray so we can return the reductions on (+,0) on
    // the products, that is, the last element of each scanned subarray.
    sgmScanInc<Add<T>, T>(block_size, d_size, d_prods, d_flags, d_prods_sums);
    cudaThreadSynchronize();

    cudaFree(d_prods);

    // Compute the result indices
    int *d_prods_sums_res_inds;
    cudaMalloc((void**)&d_prods_sums_res_inds, d_size*sizeof(int));
    scanInc<Add<int>,int>(block_size, d_size, d_flags, d_prods_sums_res_inds);

    // Return the last element of each subarray
    write_lastSgmElem <<< num_blocks, block_size >>> (d_prods_sums, d_prods_sums_res_inds, d_flags, d_size, d_out);
}

#endif //MATVECMULT_HOST
