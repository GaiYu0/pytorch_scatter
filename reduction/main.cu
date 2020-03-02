#include <assert.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#define DTYPE double

void maximum(DTYPE *x, std::vector<DTYPE> &y, int n, int d) {
  y.resize(n);
  #pragma opm parallel
  for (int i = 0; i < n; i++) {
    y[i] = 0;
    for (int j = 0; j < d; j++) {
      y[i] = x[i * n + j] < y[i] ? y[i] : x[i * n + j];
    }
  }
}

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T> inline T max(T lhs, T rhs) {
  return lhs < rhs ? rhs : lhs;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    // mySum += g_idata[i];
    mySum = max(mySum, g_idata[j * gridDim.y + i]);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    // if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];
    if (nIsPow2 || i + blockSize < n) mySum = max(mySum, g_idata[j * gridDim.y + i + blockSize]);

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 256];
    sdata[tid] = mySum = max(mySum, sdata[tid + 256]);
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 128];
    sdata[tid] = mySum = max(mySum, sdata[tid + 128]);
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 64];
    sdata[tid] = mySum = max(mySum, sdata[tid + 64]);
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    // if (blockSize >= 64) mySum += sdata[tid + 32];
    if (blockSize >= 64) mySum = max(mySum, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      // mySum += tile32.shfl_down(mySum, offset);
      mySum = max(mySum, tile32.shfl_down(mySum, offset));
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[j * gridDim.y + blockIdx.x] = mySum;
}

inline void cudaAssert(cudaError_t code) {
   if (code != cudaSuccess) {
      std::cerr << "cudaAssert: " << cudaGetErrorString(code) << std::endl;
   }
}

#define N_THREADS 1024
int main() {
  int N = 32;
  int D = 3 * 224 * 224;
  int E = D / N_THREADS + !(D % N_THREADS);

  DTYPE *x, *y;
  cudaMallocManaged(&x, N * D * sizeof(DTYPE));
  cudaMallocManaged(&y, N * E * sizeof(DTYPE));

  #pragma omp parallel
  for (int i = 0; i < N; i++) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::normal_distribution<> normal_dist(0, 1);
    for (int j = 0; j < D; j++) {
      x[i * N + j] = normal_dist(rng);
    }
  }
  std::vector<DTYPE> max;
  maximum(x, max, N, D);

  dim3 d_block(N_THREADS, 1, 1);
  dim3 d_grid(N, E, 1);
  int sm_size = (N_THREADS <= 32) ? 2 * N_THREADS * sizeof(DTYPE) : N_THREADS * sizeof(DTYPE);

  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  cudaDeviceSynchronize();

  clock_t t0 = clock();
  reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
  cudaAssert(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  double t = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
  std::cout << t << std::endl;

  std::vector<DTYPE> maxx;
  maximum(x, maxx, N, D);
  for (int i = 0; i < N; i++) {
    assert(max[i] == maxx[i]);
  }
}
