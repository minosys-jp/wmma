#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
#include "header.h"
#include <cstdio>

// wmma のテスト
// 単一 warp に 16x16 の計算を実行する

using namespace nvcuda;

#define M (16)
#define N (16)
#define K (16)
#define KTILE (1)
#define BULKF (sizeof(float4)/sizeof(float))
#define BULKH (sizeof(float4)/sizeof(half))

#define B(x, d) ((x + d - 1) / d)

__global__
void wmma_fma(float c[], half a[], half b[], int n, int m, int k, bool bTensorCore) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  const half2 z8[4] = {make_half2(0.f, 0.f), make_half2(0.f, 0.f), make_half2(0.f, 0.f), make_half2(0.f, 0.f)};

  // 結果行列を初期化する
  __shared__ float cc[SIZE(N, M)];
  if (n % BULKF == 0 && x / BULKF < n / BULKF) {
    if (x % BULKF == 0) {
      *(float4*)&cc[EL(threadIdx.x, threadIdx.y, N)] = (y < m)?*(float4*)&c[EL(x, y, n)]:make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
    cc[EL(threadIdx.x, threadIdx.y, N)] = (y < m && x < n)?c[EL(x, y, n)]:0.0f;
  }
  __syncthreads();

  // 行列をタイル化して部分行列の FMA を計算する
  for (int kk = 0; kk < B(k, K); kk += KTILE) {
    __shared__ half ah[KTILE][SIZE(K, M)], bh[KTILE][SIZE(N, K)];
    for (int kt = 0; kt < KTILE; ++kt) {
      for (int ks = 0; ks < K; ++ks) {
        int z = (kk + kt) * K + ks;

        // shared memory にロード: 整形および転置も行う
        if (threadIdx.x == 0)
          if (k % BULKH == 0 && z / BULKH < k / BULKH) {
            if (ks % BULKH == 0)
              *(float4*)&ah[kt][EL(ks, threadIdx.y, K)] = (y < m)?*(float4*)&a[EL(z, y, k)]:*(float4*)&z8[0];
          } else {
            ah[kt][EL(ks, threadIdx.y, K)] = (y < m && z < k)?a[EL(z, y, k)]:__float2half(0.f);
          }
        if (threadIdx.y == 0) {
          if (n % BULKH == 0 && x / BULKH < n / BULKH) {
            if (x % BULKH == 0) {
            *(float4*)&bh[kt][EL(threadIdx.x, ks, N)] = (z < k)?*(float4*)&b[EL(x, z, n)]:*(float4*)&z8[0];
            }
          } else {
            bh[kt][EL(threadIdx.x, ks, N)] = (x < n && z < k)?b[EL(x, z, n)]:__float2half(0.0f);
          }
        }
      }
    }
    __syncthreads();

    if (!bTensorCore) {
      float ctmp = cc[EL(threadIdx.x, threadIdx.y, N)];
      for (int kt = 0; kt < KTILE; ++kt) {
        for (int ks = 0; ks < K; ++ks) {
          ctmp += __half2float(ah[kt][EL(ks, threadIdx.y, K)]) * __half2float(bh[kt][EL(threadIdx.x, ks, M)]);
        }
      }
      cc[EL(threadIdx.x, threadIdx.y, N)] = ctmp;
   } else {
      // wmma にロード
      wmma::fragment<wmma::matrix_a, N, M, K, half, wmma::row_major> a_frag[KTILE];
      wmma::fragment<wmma::matrix_b, N, M, K, half, wmma::row_major> b_frag[KTILE];
      wmma::fragment<wmma::accumulator, N, M, K, float> c_frag[KTILE];

      // fragment を初期化
      for (int kt = 0; kt < KTILE; ++kt) {
        if ((kt + kk) * K >= k) continue;
        wmma::load_matrix_sync(a_frag[kt], ah[kt], K);
        wmma::load_matrix_sync(b_frag[kt], bh[kt], M);
        if (kt == 0) {
          wmma::load_matrix_sync(c_frag[kt], cc, N, wmma::mem_row_major);
        } else {
          wmma::fill_fragment(c_frag[kt], 0.0f);
        }
      }

      // (M, K) x (K, N)  + (M, N) の行列演算を実行
      for (int kt = 0; kt < KTILE; ++kt) {
        if ((kt + kk) * K >= k) continue;
        wmma::mma_sync(c_frag[kt], a_frag[kt], b_frag[kt], c_frag[kt]);
      }

      // 結果を shared memory に保存
      __shared__ float ctmp[KTILE][SIZE(N, M)];
      for (int kt = 0; kt < KTILE; ++kt) {
        if ((kt + kk) * K >= k) continue;
        wmma::store_matrix_sync(ctmp[kt], c_frag[kt], N, wmma::mem_row_major);
      }
      __syncthreads();
      for (int kt = 1; kt < KTILE; ++kt) {
        if ((kt + kk) * K >= k) continue;
        ctmp[0][EL(threadIdx.x, threadIdx.y, N)] += ctmp[kt][EL(threadIdx.x, threadIdx.y, N)];
      }
      cc[EL(threadIdx.x, threadIdx.y, N)] = ctmp[0][EL(threadIdx.x, threadIdx.y, N)];
    }
    __syncthreads();
  } 

  // 得られた部分行列を結果に戻す
  if (y < m) {
    if (n % BULKF == 0 && x / BULKF < n / BULKF) {
      if (x % BULKF == 0) {
        *(float4*)&c[EL(x, y, n)] = *(float4*)&cc[EL(threadIdx.x, threadIdx.y, N)];
      }
    } else if (x < n) {
      c[EL(x, y, n)] = cc[EL(threadIdx.x, threadIdx.y, N)];
    }
  }
}

__host__
void calc_fma(float c[], half a[], half b[], int n, int m, int k, bool bTensorCore) {
  dim3 threadN(N, M);
  dim3 blockN(B(n, N), B(m, M));
  wmma_fma<<<blockN, threadN>>>(c, a, b, n, m, k, bTensorCore);
  cudaDeviceSynchronize();
}

