#ifndef SCAN_KERS
#define SCAN_KERS

#include <cuda_runtime.h>

template<class T>
class Add {
  public:
    typedef T BaseType;
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }
};

template<class T>
class Mul {
  public:
    typedef T BaseType;
    static __device__ inline T identity()                      { return (T)1;    }
    static __device__ inline T apply(const T& t1, const T& t2) { return t1 * t2; }
};

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP, class T>
__device__ inline
T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]);
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T>
__device__ inline
T scanIncBlock(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & 31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP,T>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); }
    __syncthreads();

    // Scan the first warp, i.e. the end-of-warp results
    if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
    __syncthreads();

    // Take scanned last values from the previous warp and apply to all elements
    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    // Return the last value
    return val;
}

template<class OP, class T>
__global__ void
scanIncKernel(T* d_in, T* d_out, unsigned int d_size) {
    extern __shared__ char sh_mem1[];
    volatile T* sh_memT = (volatile T*)sh_mem1;
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    T el    = (gid < d_size) ? d_in[gid] : OP::identity();
    sh_memT[tid] = el;
    __syncthreads();
    T res   = scanIncBlock < OP, T >(sh_memT, tid);
    if (gid < d_size) d_out[gid] = res;
}

/***************************************/
/***      Scan Exclusive Kernel      ***/
/***************************************/
template<class OP, class T>
__global__ void
scanExcFixUp(T* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    // First element of exclusive scan should be the identity element.
    if (gid == 0)
      d_out[0] = OP::identity();
}

/***********************************************************/
/*** Kernels to copy/distribute the end of block results ***/
/***********************************************************/

template<class T>
__global__ void
copyEndOfBlockKernel(T* d_in, T* d_out, unsigned int d_out_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid < d_out_size)
        d_out[gid] = d_in[ blockDim.x*(gid+1) - 1];
}

template<class OP, class T>
__global__ void
distributeEndBlock(T* d_in, T* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid < d_size && blockIdx.x > 0)
        d_out[gid] = OP::apply(d_in[blockIdx.x-1], d_out[gid]);
}

template<class T>
__global__ void
shiftRightByOne(T* d_in, T* d_out, T ne, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if      (gid == 0)      d_out[gid] = ne;
    else if (gid < d_size)  d_out[gid] = d_in[gid-1];
}

/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template<class OP, class T, class F>
__device__ inline
T sgmScanIncWarp(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]); }
        flg[idx] = flg[idx-1] | flg[idx];
    }
    if (lane >= 2)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]); }
        flg[idx] = flg[idx-2] | flg[idx];
    }
    if (lane >= 4)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]); }
        flg[idx] = flg[idx-4] | flg[idx];
    }
    if (lane >= 8)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]); }
        flg[idx] = flg[idx-8] | flg[idx];
    }
    if (lane >= 16)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]); }
        flg[idx] = flg[idx-16] | flg[idx];
    }

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T, class F>
__device__ inline
T sgmScanIncBlock(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst= (warpid<<5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP,T>(ptr,flg,idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T&>(ptr[warplst]);

    // 2b: warp_flag is the OR-reduction of the flags
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    bool warp_flag = flg[warplst]!=0 || !warp_is_open;
    bool will_accum= warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (lane == 31) {
        ptr[warpid] = warp_total; //ptr[idx];
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    //
    if (warpid == 0) sgmScanIncWarp<OP,T>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum) {
        val = OP::apply(ptr[warpid-1], val);
    }
    return val;
}

template<class OP, class T>
__global__ void
sgmScanIncKernel(T* d_in, int* flags, T* d_out,
                          int* f_rec, T* d_rec, unsigned int d_size) {
    extern __shared__ char sh_mem[];
    volatile T*   vals_sh = (volatile T*)sh_mem;
    volatile int* flag_sh = (int*) (vals_sh + blockDim.x);
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    int fl;
    if (gid < d_size) { vals_sh[tid] = d_in[gid];      fl = flags[gid]; }
    else              { vals_sh[tid] = OP::identity(); fl = 0;          }
    flag_sh[tid] = fl;
    __syncthreads();
    T res = sgmScanIncBlock <OP, T>(vals_sh, flag_sh, tid);
    if (gid < d_size) d_out [gid] = res;

    // set the flags and data for the recursive step!
    if(tid == 0)  { f_rec[blockIdx.x] = 0; }
    __syncthreads();
    if(fl  >  0)  { f_rec[blockIdx.x] = 1; }
    if(tid == (blockDim.x - 1)) { d_rec[blockIdx.x] = res; }
}

template<class OP, class T>
__global__ void
sgmDistributeEndBlock(T* d_rec_in, T* d_out, int* f_inds, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid < d_size && blockIdx.x > 0) {
        if(f_inds[gid] == 0)
            d_out[gid] = OP::apply(d_rec_in[blockIdx.x-1], d_out[gid]);
    }
}

/*************************************************/
/*************************************************/
/***     Segmented Exclusive Scan Helpers      ***/
/*************************************************/
/*************************************************/
template<class OP, class T>
__global__ void
sgmScanExcAdj(T* d_in, int* flags, T* d_adj, const unsigned long d_size)  {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    d_adj[gid] = flags[gid] ? OP::identity() : d_in[gid];
}

////////////////////////////////////////
////////////////////////////////////////

/**
 * d_in     the input array
 * flags    the flag array
 * d_out    the result array: if the corresponding flag is set then ne
 *                            else the previous element of d_in
 * ne       is the neutral element
 * d_size   if the size of the input, flag, and output arrays
 **/
template<class T>
__global__ void
sgmShiftRightByOne(T* d_in, int*flags, T* d_out, T ne, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        // ... fill in the blanks ...
    }
}

/**
 * tmp_scan  the segmented scan (+) version of the flat matrix,
 *               i.e., the last element of each segment is the
 *               result vector value.
 * tmp_inds  IF `gid' is the index of the LAST ELEMENT of a SEGMENT (row)
 *               then (tmp_inds[gid]-1) is the index in `vct_res' where that
 *               element should be stored, i.e.,
 *               vct_res[tmp_inds[gid]-1] = tmp_scan[gid]
 * flags_d   denote the starts of the rows of the flat matrix, i.e.,
 *               flags[gid]!=0 then a new row starts at position `gid'.
 *               It follows that an index `gid' corresponds to the
 *               last element of a row if flags_d[gid+1] != 0
 * tot_size  the total number of elements of the flat matrix
 * vct_res   the result vector
 */
__global__ void
write_lastSgmElem(float* tmp_scan, int* tmp_inds, int* flags_d, int tot_size, float* vct_res) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < tot_size) {
        if (flags_d[gid+1] != 0)
          vct_res[tmp_inds[gid]-1] = tmp_scan[gid];
    }
}


#endif //SCAN_KERS
