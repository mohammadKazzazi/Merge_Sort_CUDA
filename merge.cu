// // ONLY MODIFY THIS FILE

#include "gputimer.h"
#include "gpuerrors.h"
#include "merge_sort.h"

#include <algorithm>
#include <stdint.h>
#include <limits.h>

#define TX threadIdx.x
#define BDX blockDim.x
#define BX  blockIdx.x

// =================== Tunables ===================
#ifndef RUN_SIZE
#define RUN_SIZE 2048          // 2048 reduces merge passes vs 1024
#endif

#ifndef TPB
#define TPB 1024                // more threads per block -> higher throughput
#endif
// =================================================

// Cached global load helper (__ldg on cc>=3.5)
__device__ __forceinline__ int ldgc(const int* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

/*
 * Canonical merge-path binary search.
 * Find a in [max(0, diag-lenB), min(diag, lenA)] such that:
 *   (a==0 || B[b]   >= A[a-1])   and
 *   (b==0 || A[a]   >  B[b-1])
 * where b = diag - a.
 */
__device__ __forceinline__ int binary_search_diag(const int* A, int lenA,
                                                  const int* B, int lenB,
                                                  int diag)
{
    int lo = max(0, diag - lenB);
    int hi = min(diag, lenA);

    while (lo <= hi) {
        int a = (lo + hi) >> 1;
        int b = diag - a;

        int A_im1 = (a > 0)    ? ldgc(A + a - 1) : INT_MIN;
        int A_i   = (a < lenA) ? ldgc(A + a)     : INT_MAX;
        int B_im1 = (b > 0)    ? ldgc(B + b - 1) : INT_MIN;
        int B_i   = (b < lenB) ? ldgc(B + b)     : INT_MAX;

        if (A_im1 <= B_i && B_im1 < A_i) {
            return a;
        } else if (A_im1 > B_i) {
            hi = a - 1;
        } else {
            lo = a + 1;
        }
    }
    return lo; // fallback
}

// -------- Stage 1: Data conditioning in shared memory --------
// Renamed from bitonic_sort_shared
__device__ __forceinline__ void local_data_rearrangement(volatile int* data_block, int block_size)
{
  // 'block_size' must be a power of 2 for this procedure
  for (int level_width = 2; level_width <= block_size; level_width <<= 1) {
    for (int step_dist = level_width >> 1; step_dist > 0; step_dist >>= 1) {
      for (int i = TX; i < block_size; i += BDX) {
        int partner_idx = i ^ step_dist;
        if (partner_idx > i) {
          int direction_flag = ((i & level_width) == 0);
          int val1 = data_block[i];
          int val2 = data_block[partner_idx];
          if ((val1 > val2) == direction_flag) {
            data_block[i] = val2;
            data_block[partner_idx] = val1;
          }
        }
      }
      __syncthreads();
    }
  }
}

__launch_bounds__(TPB)
__global__ void kernel_init_runs(const int* __restrict__ in,
                                 int* __restrict__ out,
                                 int n)
{
  // The size of this array depends on the compile-time constant RUN_SIZE
  __shared__ int s[RUN_SIZE];

  const int base = BX * RUN_SIZE;
  if (base >= n) return;

  // Load with +INF padding to full tile
  for (int i = TX; i < RUN_SIZE; i += BDX) {
    int g = base + i;
    s[i] = (g < n) ? in[g] : INT_MAX;
  }
  __syncthreads();

  // Perform the in-place data processing on the local tile
  local_data_rearrangement(s, RUN_SIZE);
  __syncthreads();

  // Store back valid range
  for (int i = TX; i < RUN_SIZE; i += BDX) {
    int g = base + i;
    if (g < n) out[g] = s[i];
  }
}


// -------- Stage 2+: per-thread fixed slice via merge-path --------
__launch_bounds__(TPB)
__global__ void kernel_merge_path(const int* __restrict__ src,
                                  int* __restrict__ dst,
                                  int n,
                                  int run)
{
    const int pairSize = run << 1;
    const int left     = BX * pairSize;
    if (left >= n) return;

    const int mid   = min(left + run, n);
    const int right = min(left + pairSize, n);

    const int lenA = mid  - left;      // may be 0
    const int lenB = right - mid;      // may be 0
    const int outLen = lenA + lenB;
    if (outLen == 0) return;

    const int* A = src + left;
    const int* B = src + mid;

    // Partition [0, outLen) into near-equal chunks per thread.
    const int T = BDX;
    int chunk = (outLen + T - 1) / T;   // ceil
    if (chunk == 0) chunk = 1;

    const int diagLo = min(TX * chunk, outLen);
    const int diagHi = min(diagLo + chunk, outLen);
    if (diagLo >= diagHi) return;

    const int aLo = binary_search_diag(A, lenA, B, lenB, diagLo);
    const int bLo = diagLo - aLo;

    int ai = aLo;
    int bi = bLo;
    int outPos = left + diagLo;
    int toEmit = diagHi - diagLo;

    #pragma unroll 4
    while (toEmit--) {
        int av = (ai < lenA) ? ldgc(A + ai) : INT_MAX;
        int bv = (bi < lenB) ? ldgc(B + bi) : INT_MAX;
        if (av <= bv) { dst[outPos++] = av; ++ai; }
        else          { dst[outPos++] = bv; ++bi; }
    }
}

// ---------------- Host orchestration ----------------
void gpuKernel(const int* f, int* result, int n, double* gpu_kernel_time)
{
    if (n <= 0) { *gpu_kernel_time = 0.0; return; }

    int *d_a = NULL, *d_b = NULL;
    HANDLE_ERROR(cudaMalloc(&d_a, n * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_b, n * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_a, f, n * sizeof(int), cudaMemcpyHostToDevice));

    // Prefer L1 for these two kernels
    cudaFuncSetCacheConfig(kernel_init_runs, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(kernel_merge_path, cudaFuncCachePreferL1);

    GpuTimer timer;
    HANDLE_ERROR(cudaDeviceSynchronize());
    timer.Start();

    // 1) initial sorted runs -> d_b
    {
        int blocks = (n + RUN_SIZE - 1) / RUN_SIZE;
        kernel_init_runs<<<blocks, TPB>>>(d_a, d_b, n);
        HANDLE_ERROR(cudaGetLastError());
    }

    // 2) iterative merges (ping-pong between d_b and d_a)
    int* src = d_b;
    int* dst = d_a;
    int passes = 0; // Count the number of merge passes

    for (int run = RUN_SIZE; run < n; run <<= 1) {
        int blocks = (n + (run << 1) - 1) / (run << 1);

        kernel_merge_path<<<blocks, TPB>>>(src, dst, n, run);
        HANDLE_ERROR(cudaGetLastError());

        int* tmp = src; src = dst; dst = tmp;
        passes++; // Increment pass counter
    }
    

    HANDLE_ERROR(cudaDeviceSynchronize());
    timer.Stop();
    *gpu_kernel_time = timer.Elapsed();

    // Copy directly from the correct final buffer (`src`) to the host.
    // This avoids a potentially large and slow device-to-device copy.
    HANDLE_ERROR(cudaMemcpy(result, src, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
}