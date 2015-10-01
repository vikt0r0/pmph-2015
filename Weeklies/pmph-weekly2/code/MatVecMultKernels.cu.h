#ifndef MATVECMULT_KERNELS
#define MATVECMULT_KERNELS

#include <cuda_runtime.h>

template <class T1, class T2>
class MyPair {
  public:
    T1 x; T2 y;
    MyPair (T1 f, T2 s)
    {
      x = f; y = s;
    }
};

/**
 * This implements the ps calculation of the matric-vector multiplication
 * found in PrimesQuicksort.hs:
 * d_size   the size of the flags array/the flat sparse matrix
 * d_flags  the flags denoting starts of the irregular arrays
 * d_mat    the sparse matrix
 * d_out    flat array of the products of all pairwise multiplications
 **/
template <class T>
__global__ void
matVecMultProdsKernel(const unsigned long d_size,
                      MyPair<int, T>     *d_mat,
                      T                  *d_vec,
                      T                  *d_out
                     ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    T a, b, res;

    // Check if we are within bounds of the array
    if (gid >= d_size) return;

    // Determine the corresponding element of the input vector
    a = d_mat[gid].y;
    b = d_vec[d_mat[gid].x];

    // Perform the multiplication
    d_out[gid] = a * b;
}

#endif //MATVECMULT_KERNELS
