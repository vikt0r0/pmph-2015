#ifndef MATRIX_KERNELS
#define MATRIX_KERNELS

template <class T>
__global__ void
matrix_transpose_naive(const unsigned long d_size,
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

#endif
