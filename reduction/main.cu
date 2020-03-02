#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <random>
#include "reduction.h"

#define DTYPE double

DTYPE maximum(DTYPE *x, int n) {
  DTYPE max;
  for (int i = 0; i < n; i++) {
    max = x[i] < max ? max : x[i];
  }
  return max;
}

/*
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  // if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
  if (i + blockDim.x < n) mySum = max(mySum, g_idata[i + blockDim.x]);

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = max(mySum, sdata[tid + s]);
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}
*/

int main() {
  int N = 1 << 25;
  int N_THREADS = 512;
  int N_BLOCKS = N / N_THREADS;
  int KERNEL_ID = 6;

  DTYPE *x, *y;
  cudaMallocManaged(&x, N * sizeof(DTYPE));
  cudaMallocManaged(&y, N_BLOCKS * sizeof(DTYPE));

  // srand(time(NULL));
  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<> normal_dist(0, 1);
  for (int i = 0; i < N; i++) {
    // x[i] = rand();
    x[i] = normal_dist(rng);
  }
  std::cout << "max: " << maximum(x, N) << std::endl;

  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  cudaDeviceSynchronize();

  clock_t t0 = clock();
  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  cudaDeviceSynchronize();
  double t = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
  std::cout << t << std::endl;

  std::cout << "max: " << maximum(y, N_BLOCKS) << std::endl;
}
