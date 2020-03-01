#include "reduction.h"

int main() {
  int N = 1 << 20;
  int N_BLOCKS = 128;
  int N_THREADS = 1024;
  int KERNEL_ID = 0;

  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&x, N / N_BLOCKS * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
  }

  reduce(N, N_THREADS, N_BLOCKS, KERNEL_ID, x, y);
  cudaDeviceSynchronize();
}
