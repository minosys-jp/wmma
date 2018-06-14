#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <omp.h>
#include "header.h"

using namespace std;

void calc_fma_gold(float c[], half a[], half b[], int n, int m, int k) {
  int i, j, ks;

  #pragma omp parallel for private(i, j, ks)
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      for (ks = 0; ks < k; ++ks) {
        c[EL(i, j, n)] += __half2float(a[EL(ks, j, k)]) * __half2float(b[EL(i, ks, n)]);
      }
    }
  }
}

void print_matrix(const char *title, float a[], int m, int n) {
  std::cout << title << ":" << endl << "[";
  for (int i = 0; i < n; ++i) {
    cout << "[";
    for (int j = 0; j < m; ++j) {
      cout << a[EL(i, j, m)];
      if (j < m - 1)
        cout << ",";
    }
    cout << "]";
    if (i < n - 1)
      cout << ",";
    cout << endl;
  }
  cout << "]" << endl;
}

void float2half_matrix(half h[], float f[], int n, int m) {
  int i, j;

  #pragma omp parallel for private(i, j)
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      h[EL(i, j, m)] = __float2half(f[EL(i, j, m)]);
    }
  }
}

int main() {
  half *ah, *bh;
  float *a, *b, *c, *c_gold;
  int i, j;

  a = new float [SIZE(NN, KK)];
  b = new float [SIZE(KK, MM)];
  c_gold = new float [SIZE(NN, MM)];
  checkError(cudaMallocManaged((void **)&ah, sizeof(half) * SIZE(NN, KK)));
  checkError(cudaMallocManaged((void **)&bh, sizeof(half) * SIZE(KK, MM)));
  cudaError(cudaMallocManaged((void **)&c, sizeof(float) * SIZE(NN, MM)));

  for (i = 0; i < KK; ++i) {
    for (j = 0; j < NN; ++j) {
      a[EL(i, j, KK)] = (j >= i)?(1.f/16.f):0.f;
    }
  }
  #pragma omp parallel for private(j)
  for (j = 0; j < MM; ++j) {
    for (int k = 0; k < KK; ++k) {
      b[EL(j, k, MM)] = (k >= j)?(1.f/16.f):0.f;
    }
  }
  #pragma omp parallel for private(i)
  for (i = 0; i < SIZE(NN, MM); ++i) {
    c[i] = c_gold[i] = 0.0f;
  }

  float2half_matrix(ah, a, NN, KK);
  float2half_matrix(bh, b, KK, MM);

  cout << "start wmma version..." << endl;
  for (i = 0; i < 3; ++i) {
    #pragma omp parallel for private(j)
    for (j = 0; j < SIZE(NN, MM); ++j) {
      c[j] = 0.0f;
    }
    cudaEvent_t start, end;
    float ms;
    checkError(cudaEventCreate(&start));
    checkError(cudaEventCreate(&end));
    checkError(cudaEventRecord(start));
    calc_fma(c, ah, bh, NN, MM, KK, i == 2);
    checkError(cudaEventRecord(end));
    checkError(cudaEventSynchronize(end));
    checkError(cudaEventElapsedTime(&ms, start, end));
//    print_matrix("c", c, NN, MM);
    cout << "calc_fm elapsed time ";
    cout << ((i == 2)?"with TensorCore":"without TensorCore");
    cout << ":" << ms << "ms" << endl;
  }

  cout << "start CPU gold..." << endl;
  struct timespec clk_start, clk_end;
  clock_gettime(CLOCK_REALTIME, &clk_start);
  calc_fma_gold(c_gold, ah, bh, NN, MM, KK);
  clock_gettime(CLOCK_REALTIME, &clk_end);
  double startd = clk_start.tv_sec + (double)clk_start.tv_nsec / 1000000000.0;
  double endd = clk_end.tv_sec + (double)clk_end.tv_nsec / 1000000000.0;
  double elapsedd = (endd - startd) * 1000;
//  print_matrix("c_gold", c_gold, NN, MM);
  cout << "calc_fm_gold elapsed time:" << elapsedd << "m" << endl;
  return 0;
}

