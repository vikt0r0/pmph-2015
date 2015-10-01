#ifndef MSSP_KERS
#define MSSP_KERS

#include <cuda_runtime.h>

class MyInt4 {
  public:
    int x; int y; int z; int w;

    __device__ __host__ inline MyInt4() {
        x = 0; y = 0; z = 0; w = 0;
    }

    __device__ __host__ inline MyInt4(const int& a, const int& b, const int& c, const int& d) {
        x = a; y = b; z = c; w = d;
    }

    __device__ __host__ inline MyInt4(const MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
    }

    volatile __device__ __host__ inline MyInt4& operator=(const MyInt4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
        return *this;
    }
};

class MsspOp {
  public:
    typedef MyInt4 BaseType;
    static __device__ inline MyInt4 identity() { return MyInt4(0,0,0,0); }
    static __device__ inline MyInt4 apply(volatile MyInt4& t1, volatile MyInt4& t2) {
        int mssx = t1.x, misx = t1.y, mcsx = t1.z, tsx = t1.w;
        int mssy = t2.x, misy = t2.y, mcsy = t2.z, tsy = t2.w;
        int mss = max(max(mssx,mssy), mssx+mssy);
        int mis = max(misx,tsx+misy);
        int mcs = max(mcsy,mcsx+tsy);
        int t   = tsx+tsy;
        return MyInt4(mss, mis, mcs, t);
    }
};

/**
 * This implements the map from MSSP:
 * inp_d    the original array (of ints)
 * inp_lift the result array, in which an integer x in inp_d
 *              should be transformed to MyInt4(x,x,x,x) if x > 0
 *                                and to MyInt4(0,0,0,x) otherwise
 * inp_size is the size of the original (and output) array
 *              in number of int (MyInt4) elements
 **/
__global__ void
msspTrivialMap(int* inp_d, MyInt4* inp_lift, int inp_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < inp_size) {
      int x = inp_d[gid];
      inp_lift[gid] = (x > 0) ? MyInt4(x,x,x,x) : MyInt4(0,0,0,x);
    }
}


#endif //MSSP_KERS
