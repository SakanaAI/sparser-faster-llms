/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include "hybrid_sp.h"
#include "perf_instrumentation.h"

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>


#define ERROR_CHECK 0
#define TILE_DIM_X 128       // 128x128 Tiles
#define TILE_DIM_Y 128       // 128x128 Tiles


#define VEC_SIZE 8

// Fragment register to tile coord mapping
__forceinline__ __device__ int atomic_read_i32(const int* p) {
    // atomic "read" that has proper visibility under contention
    return atomicAdd((int*)p, 0);
}

__forceinline__ __device__ void populate_dense(
    int g_row, int g_col, __nv_bfloat16 val,
    __nv_bfloat16* __restrict__ tail_dense, int dense_ld,
    int* __restrict__ tail_dense_map,
    int* __restrict__ tail_dense_map_reverse,
    int* __restrict__ tail_dense_counter,
    int tail_cap, int discard
) {
    if (discard == 1) return;

    // Fast path: already mapped
    int dr = atomic_read_i32(&tail_dense_map[g_row]);
    if (dr >= 0 && dr < tail_cap) {
        tail_dense[(size_t)dr * (size_t)dense_ld + (size_t)g_col] = val;
        return;
    }
    if (dr >= tail_cap) return;  // stale/corrupt mapping

    // Try to claim mapping if unmapped (-1)
    if (dr == -1) {
        int old = atomicCAS(&tail_dense_map[g_row], -1, -2); // -2 = in-progress
        if (old == -1) {
            // We won: allocate dense row id
            int new_dr = atomicAdd(tail_dense_counter, 1);

            if (new_dr < tail_cap) {
                // Publish with atomicExch so others see it
                atomicExch(&tail_dense_map[g_row], new_dr);
                dr = new_dr;
                tail_dense_map_reverse[dr] = g_row;
            } else {
                // Out of capacity: publish terminal "failed" state
                atomicExch(&tail_dense_map[g_row], -3);
                return;
            }

            tail_dense[(size_t)dr * (size_t)dense_ld + (size_t)g_col] = val;
            return;
        } else {
            dr = old; // someone else has state already
        }
    }

    // Someone else is assigning (dr == -2): wait bounded
    if (dr == -2) {
        while (dr == -2 ) {
            dr = atomic_read_i32(&tail_dense_map[g_row]);
        }
        if (dr < 0 || dr >= tail_cap) {
            return; // unmapped/failed/corrupt
        }
    }

    // dr >= 0 mapped, else negative => failed
    if (dr >= 0 && dr < tail_cap) {
        tail_dense[(size_t)dr * (size_t)dense_ld + (size_t)g_col] = val;
    }
}


__forceinline__ __device__ void populate_dense_known(
    int g_row, int g_col, __nv_bfloat16 val,
    __nv_bfloat16* __restrict__ tail_dense, int dense_ld,
    const int* __restrict__ tail_dense_map
) {
    int dr = atomic_read_i32((int*)&tail_dense_map[g_row]);
    if (dr >= 0) {
        tail_dense[(size_t)dr * (size_t)dense_ld + (size_t)g_col] = val;
    }
}

__global__ void promote_overflow_rows_ell_into_tail_dense(
    const uint16_t*      __restrict__ ell_cols,       // [M_rows * ell_stride]
    const __nv_bfloat16* __restrict__ ell_vals,       // [M_rows * ell_stride]
    int*                 __restrict__ row_counters,   // [M_rows] (ELL+tail count)
    int                  M_rows,

    __nv_bfloat16*       __restrict__ tail_dense,     // [tail_dense_rows * dense_ld]
    int                  dense_ld,                    // num cols in tail_dense
    const int*           __restrict__ tail_dense_map, // [M_rows], -1 for non-overflow
    int                  ell_w,                       // runtime ELL overflow threshold
    int                  ell_stride                   // ELL buffer stride for this object (_ell_stride)
) {
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int nnz = row_counters[row];
    if (nnz <= ell_w) return;     // not an overflow row => nothing to do

    int dr = tail_dense_map[row];
    if (dr < 0) return;               // should not happen if mapping is correct

    const uint16_t*      r_cols = ell_cols + (size_t)row * ell_stride;
    const __nv_bfloat16* r_vals = ell_vals + (size_t)row * ell_stride;

    // Vectorize: 8 entries per iteration (16B idx + 16B vals)
    int k0 = (threadIdx.x * 8);
    int stride = (blockDim.x * 8);

    for (int k = k0; k < ell_stride; k += stride) {
        // safe because ELL_WIDTH is typically multiple of 8; otherwise add a tail path
        int4 idx_raw = *reinterpret_cast<const int4*>(r_cols + k); // 8x u16
        int4 val_raw = *reinterpret_cast<const int4*>(r_vals + k); // 8x bf16

        const uint16_t* idx8 = reinterpret_cast<const uint16_t*>(&idx_raw);
        const __nv_bfloat16* v8 = reinterpret_cast<const __nv_bfloat16*>(&val_raw);

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int col = (int)idx8[t];
            if ((unsigned)col < (unsigned)dense_ld) {
                tail_dense[(size_t)dr * dense_ld + (size_t)col] = v8[t];
            }
        }
    }
}


// Scatter tail dense rows back into the full output matrix.
// One block per row; non-mapped rows exit immediately.
__global__ void scatter_tail_dense_rows_into_full(
    __nv_bfloat16*       __restrict__ A,             // [M_rows, N_cols] output
    int                  M_rows,
    int                  N_cols,
    const __nv_bfloat16* __restrict__ B,             // [tail_rows, N_cols] dense tail
    int                  tail_rows,
    const int*           __restrict__ tail_dense_map // [M_rows], -1 if not mapped
) {
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int dr = tail_dense_map[row];
    if (dr < 0 || (unsigned)dr >= (unsigned)tail_rows) return;

    const __nv_bfloat16* src = B + (size_t)dr  * N_cols;
    __nv_bfloat16*       dst = A + (size_t)row * N_cols;

    constexpr int vec_elems = 8;
    int vecs = N_cols / vec_elems;
    int rem  = N_cols % vec_elems;

    for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
        int4 x = *reinterpret_cast<const int4*>(src + v * vec_elems);
        *reinterpret_cast<int4*>(dst + v * vec_elems) = x;
    }
    if (rem && threadIdx.x == 0) {
        int base = vecs * vec_elems;
        for (int i = 0; i < rem; ++i) dst[base + i] = src[base + i];
    }
}


__global__ void gather_rows_by_map_bf16_int4(
    const __nv_bfloat16* __restrict__ A,   // [M, cols]
    __nv_bfloat16*       __restrict__ C,   // [R, cols]
    const int*           __restrict__ map, // [R] sparse_row -> dense_row, -1 if unused
    int M,
    int R,
    int cols
) {
    int sr = blockIdx.x;
    if (sr >= M) return;

    int dr = map[sr];
    if (dr < 0) return;
    if (dr >= R) {
        return;
    }

    const __nv_bfloat16* __restrict__ src = A + (size_t)sr * cols;
    __nv_bfloat16*       __restrict__ dst = C + (size_t)dr * cols;

    // Vectorized path: 8 bf16 = 16 bytes = int4
    int vec_cols = cols & ~7; // floor to multiple of 8

    for (int j = threadIdx.x * 8; j < vec_cols; j += blockDim.x * 8) {
        const int4 v = *reinterpret_cast<const int4*>(src + j);
        *reinterpret_cast<int4*>(dst + j) = v;
    }

    // tail (cols not multiple of 8)
    for (int j = vec_cols + threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = src[j];
    }
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Helper to cast vector types easily
union OutputPack {
    __nv_bfloat16 bf[8];
    int4 vec;
};

__global__ void ell_spmm_rowmajor_b_rowwise_optimized(
    const __nv_bfloat16* __restrict__ A_vals,     // [M_rows x ell_stride]
    const uint16_t* __restrict__ A_idxs,          // [M_rows x ell_stride]
    const int* __restrict__ row_counts,           // [M_rows]
    const __nv_bfloat16* __restrict__ B,          // [K_rows x N_cols]
    __nv_bfloat16* __restrict__ C,                // [M_rows x N_cols]
    int M_rows,
    int K_rows,
    int N_cols,
    int ell_stride,           // ELL buffer stride for this object (_ell_stride)
    int overflow_threshold    // rows with nnz > this are in dense tail (g_ell_width_regular)
) {
    // Dynamic shared memory: [ell_stride * bf16] | [ell_stride * u16]
    extern __shared__ char _smem[];
    __nv_bfloat16* sh_vals = reinterpret_cast<__nv_bfloat16*>(_smem);
    uint16_t*      sh_idxs = reinterpret_cast<uint16_t*>(_smem + ell_stride * sizeof(__nv_bfloat16));

    // Grid-stride loop: each block processes multiple rows when gridDim.x < M_rows
    for (int row = blockIdx.x; row < M_rows; row += gridDim.x) {
    int nnz = row_counts[row];
    if (nnz <= 0 || nnz > overflow_threshold) continue;

    // Pointers to Global Memory for this row
    const __nv_bfloat16* A_row_vals_g = A_vals + (size_t)row * ell_stride;
    const uint16_t* A_row_idxs_g = A_idxs + (size_t)row * ell_stride;

    // Cooperative Load into Shared Memory
    for (int k = threadIdx.x; k < nnz; k += blockDim.x) {
        sh_idxs[k] = A_row_idxs_g[k];
        sh_vals[k] = A_row_vals_g[k];
    }

    // Barrier to ensure shared memory is populated
    __syncthreads();

    // Main Computation Loop: each thread processes an 8-element chunk of the output column
    for (int n_out = threadIdx.x * VEC_SIZE; n_out < N_cols; n_out += VEC_SIZE * blockDim.x) {

        float2 acc[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = make_float2(0.f, 0.f);

        // Iterate exactly nnz times (no internal break)
        // Since 'nnz' is in a register and constant for the block, this is efficient
        //#pragma unroll 4
        for (int k = 0; k < nnz; ++k) {
            // Read from Shared Memory (Fast, low latency)
            __nv_bfloat16 a_val = sh_vals[k];
            uint16_t    col_idx = sh_idxs[k];

            // Calculate pointer to B row
            const __nv_bfloat16* B_row_ptr = B + (size_t)col_idx * N_cols + n_out;

            // Vectorized Load from B (128-bit load)
            // reinterpret_cast to int4 forces a single 128-bit LDG instruction
            int4 b_vec_raw = *reinterpret_cast<const int4*>(B_row_ptr);
            
            // Math Setup (similar to original)
            __nv_bfloat16* b_vec = reinterpret_cast<__nv_bfloat16*>(&b_vec_raw);
            __nv_bfloat162* b_pairs = reinterpret_cast<__nv_bfloat162*>(b_vec);

            float a= __bfloat162float(a_val);

            // Accumulate
            // Note: On Ampere+, consider __hmma or __hfma2 instructions 
            // if strict bf16 accumulation is allowed, but f32 acc is standard for precision.
            float2 b_f32;
            
            b_f32 = __bfloat1622float2(b_pairs[0]);
            acc[0].x = fmaf(a, b_f32.x, acc[0].x);
            acc[0].y = fmaf(a, b_f32.y, acc[0].y);

            b_f32 = __bfloat1622float2(b_pairs[1]);
            acc[1].x = fmaf(a, b_f32.x, acc[1].x);
            acc[1].y = fmaf(a, b_f32.y, acc[1].y);

            b_f32 = __bfloat1622float2(b_pairs[2]);
            acc[2].x = fmaf(a, b_f32.x, acc[2].x);
            acc[2].y = fmaf(a, b_f32.y, acc[2].y);

            b_f32 = __bfloat1622float2(b_pairs[3]);
            acc[3].x = fmaf(a, b_f32.x, acc[3].x);
            acc[3].y = fmaf(a, b_f32.y, acc[3].y);
        }

        // 5. Vectorized Store
        // Pack results back into int4 for a single 128-bit write
        OutputPack out;
        out.bf[0] = __float2bfloat16(acc[0].x);
        out.bf[1] = __float2bfloat16(acc[0].y);
        out.bf[2] = __float2bfloat16(acc[1].x);
        out.bf[3] = __float2bfloat16(acc[1].y);
        out.bf[4] = __float2bfloat16(acc[2].x);
        out.bf[5] = __float2bfloat16(acc[2].y);
        out.bf[6] = __float2bfloat16(acc[3].x);
        out.bf[7] = __float2bfloat16(acc[3].y);

        __nv_bfloat16* C_ptr = C + (size_t)row * N_cols + n_out;
        *reinterpret_cast<int4*>(C_ptr) = out.vec;
    }
    } // end grid-stride loop
}
// --------------------------------------------------------------------------
// IDEA 1: Warp-per-row spmm
//   - 8 warps per block, each warp owns one sparse row
//   - ELL entries loaded into registers via coalesced lane-load + __shfl_sync broadcast
//   - No shared memory, no __syncthreads
//   - Each lane handles VEC_SIZE=8 output cols with stride WARP_SIZE*VEC_SIZE=256
//   - Works for any nnz: loads ELL in batches of 32 (one per warp lane)
// --------------------------------------------------------------------------
#define WARPS_PER_BLOCK_WPR 8
#define WARP_SIZE_WPR 32

// Zero only rows with nnz == 0; rows processed by ELL or tail kernels are untouched.
// Uses the same WPR grid (M/8 blocks, 8 warps/block) so each warp handles one row.
// When all rows have nnz > 0 (benchmark default), every warp branches out immediately.
__global__ void zero_empty_rows(
    __nv_bfloat16*   __restrict__ C,
    const int*       __restrict__ row_counts,
    int M_rows,
    int N_cols
) {
    const int warp_id = threadIdx.x / WARP_SIZE_WPR;
    const int lane    = threadIdx.x % WARP_SIZE_WPR;
    const int row     = blockIdx.x * WARPS_PER_BLOCK_WPR + warp_id;
    if (row >= M_rows) return;
    if (row_counts[row] > 0) return;  // handled by main kernel — nothing to do

    __nv_bfloat16* C_row = C + (size_t)row * N_cols;
    const int4 zero = {0, 0, 0, 0};
    for (int i = lane * VEC_SIZE; i < N_cols; i += WARP_SIZE_WPR * VEC_SIZE)
        *reinterpret_cast<int4*>(C_row + i) = zero;
}

__global__ void ell_spmm_warp_per_row(
    const __nv_bfloat16* __restrict__ A_vals,
    const uint16_t*      __restrict__ A_idxs,
    const int*           __restrict__ row_counts,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    int M_rows,
    int K_rows,
    int N_cols,
    int ell_stride,
    int overflow_threshold
) {
    static_assert(VEC_SIZE == 8, "VEC_SIZE must be 8");

    const int warp_id = threadIdx.x / WARP_SIZE_WPR;
    const int lane    = threadIdx.x % WARP_SIZE_WPR;

    const int row = blockIdx.x * WARPS_PER_BLOCK_WPR + warp_id;
    if (row >= M_rows) return;

    const int nnz = row_counts[row];
    if (nnz <= 0 || nnz > overflow_threshold) return;

    const __nv_bfloat16* A_row_vals = A_vals + (size_t)row * ell_stride;
    const uint16_t*      A_row_idxs = A_idxs + (size_t)row * ell_stride;
    __nv_bfloat16*       C_row      = C      + (size_t)row * N_cols;

    // Each lane handles cols [lane*8, lane*8+8) with stride 256
    for (int n_out = lane * VEC_SIZE; n_out < N_cols; n_out += WARP_SIZE_WPR * VEC_SIZE) {
        float2 acc[4] = {};

        // Process ELL entries in batches of 32 (one load per lane per batch)
        for (int k_base = 0; k_base < nnz; k_base += WARP_SIZE_WPR) {
            const int k_local = k_base + lane;

            // Coalesced warp load: lane k_base+i loads entry k_base+i
            uint16_t batch_col = (k_local < nnz) ? A_row_idxs[k_local] : 0;
            float    batch_val = (k_local < nnz) ? __bfloat162float(A_row_vals[k_local]) : 0.f;

            const int batch_size = min(WARP_SIZE_WPR, nnz - k_base);
            #pragma unroll 8
            for (int j = 0; j < batch_size; ++j) {
                const uint16_t col_idx = __shfl_sync(0xffffffff, batch_col, j);
                const float    a_val   = __shfl_sync(0xffffffff, batch_val,  j);

                const __nv_bfloat16* B_row_ptr = B + (size_t)col_idx * N_cols + n_out;
                int4 b_raw = *reinterpret_cast<const int4*>(B_row_ptr);
                __nv_bfloat162* b_pairs = reinterpret_cast<__nv_bfloat162*>(&b_raw);

                float2 b_f32;
                b_f32 = __bfloat1622float2(b_pairs[0]);
                acc[0].x = fmaf(a_val, b_f32.x, acc[0].x);
                acc[0].y = fmaf(a_val, b_f32.y, acc[0].y);
                b_f32 = __bfloat1622float2(b_pairs[1]);
                acc[1].x = fmaf(a_val, b_f32.x, acc[1].x);
                acc[1].y = fmaf(a_val, b_f32.y, acc[1].y);
                b_f32 = __bfloat1622float2(b_pairs[2]);
                acc[2].x = fmaf(a_val, b_f32.x, acc[2].x);
                acc[2].y = fmaf(a_val, b_f32.y, acc[2].y);
                b_f32 = __bfloat1622float2(b_pairs[3]);
                acc[3].x = fmaf(a_val, b_f32.x, acc[3].x);
                acc[3].y = fmaf(a_val, b_f32.y, acc[3].y);
            }
        }

        OutputPack out;
        out.bf[0] = __float2bfloat16(acc[0].x); out.bf[1] = __float2bfloat16(acc[0].y);
        out.bf[2] = __float2bfloat16(acc[1].x); out.bf[3] = __float2bfloat16(acc[1].y);
        out.bf[4] = __float2bfloat16(acc[2].x); out.bf[5] = __float2bfloat16(acc[2].y);
        out.bf[6] = __float2bfloat16(acc[3].x); out.bf[7] = __float2bfloat16(acc[3].y);
        *reinterpret_cast<int4*>(C_row + n_out) = out.vec;
    }
}
// --------------------------------------------------------------------------
// IDEA 3: Persistent blocks (work-stealing queue)
//   Launch sm_count * 8 blocks (max occupancy for 256-thread blocks on H100).
//   Each block loops: thread 0 atomicAdd's work_counter by WARPS_PER_BLOCK_WPR
//   to claim a batch of rows; all warps process their rows; repeat until done.
//   Eliminates wave-scheduling overhead: every SM stays busy from launch to end.
// --------------------------------------------------------------------------
__global__ void ell_spmm_persistent(
    const __nv_bfloat16* __restrict__ A_vals,
    const uint16_t*      __restrict__ A_idxs,
    const int*           __restrict__ row_counts,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    int*                              work_counter,
    int M_rows,
    int K_rows,
    int N_cols,
    int ell_stride,
    int overflow_threshold
) {
    static_assert(VEC_SIZE == 8, "VEC_SIZE must be 8");

    __shared__ int s_row_base;

    const int warp_id = threadIdx.x / WARP_SIZE_WPR;
    const int lane    = threadIdx.x % WARP_SIZE_WPR;

    while (true) {
        // Thread 0 claims next batch of WARPS_PER_BLOCK_WPR rows
        if (threadIdx.x == 0)
            s_row_base = atomicAdd(work_counter, WARPS_PER_BLOCK_WPR);
        __syncthreads();

        const int row_base = s_row_base;
        if (row_base >= M_rows) return;  // all warps exit together

        const int row = row_base + warp_id;
        if (row < M_rows) {
            const int nnz = row_counts[row];
            if (nnz > 0 && nnz <= overflow_threshold) {
                const __nv_bfloat16* A_row_vals = A_vals + (size_t)row * ell_stride;
                const uint16_t*      A_row_idxs = A_idxs + (size_t)row * ell_stride;
                __nv_bfloat16*       C_row      = C      + (size_t)row * N_cols;

                for (int n_out = lane * VEC_SIZE; n_out < N_cols; n_out += WARP_SIZE_WPR * VEC_SIZE) {
                    float2 acc[4] = {};

                    for (int k_base = 0; k_base < nnz; k_base += WARP_SIZE_WPR) {
                        const int k_local = k_base + lane;

                        uint16_t batch_col = (k_local < nnz) ? A_row_idxs[k_local] : 0;
                        float    batch_val = (k_local < nnz) ? __bfloat162float(A_row_vals[k_local]) : 0.f;

                        const int batch_size = min(WARP_SIZE_WPR, nnz - k_base);
                        #pragma unroll 8
                        for (int j = 0; j < batch_size; ++j) {
                            const uint16_t col_idx = __shfl_sync(0xffffffff, batch_col, j);
                            const float    a_val   = __shfl_sync(0xffffffff, batch_val,  j);

                            const __nv_bfloat16* B_row_ptr = B + (size_t)col_idx * N_cols + n_out;
                            int4 b_raw = *reinterpret_cast<const int4*>(B_row_ptr);
                            __nv_bfloat162* b_pairs = reinterpret_cast<__nv_bfloat162*>(&b_raw);

                            float2 b_f32;
                            b_f32 = __bfloat1622float2(b_pairs[0]);
                            acc[0].x = fmaf(a_val, b_f32.x, acc[0].x); acc[0].y = fmaf(a_val, b_f32.y, acc[0].y);
                            b_f32 = __bfloat1622float2(b_pairs[1]);
                            acc[1].x = fmaf(a_val, b_f32.x, acc[1].x); acc[1].y = fmaf(a_val, b_f32.y, acc[1].y);
                            b_f32 = __bfloat1622float2(b_pairs[2]);
                            acc[2].x = fmaf(a_val, b_f32.x, acc[2].x); acc[2].y = fmaf(a_val, b_f32.y, acc[2].y);
                            b_f32 = __bfloat1622float2(b_pairs[3]);
                            acc[3].x = fmaf(a_val, b_f32.x, acc[3].x); acc[3].y = fmaf(a_val, b_f32.y, acc[3].y);
                        }
                    }

                    OutputPack out_pack;
                    out_pack.bf[0] = __float2bfloat16(acc[0].x); out_pack.bf[1] = __float2bfloat16(acc[0].y);
                    out_pack.bf[2] = __float2bfloat16(acc[1].x); out_pack.bf[3] = __float2bfloat16(acc[1].y);
                    out_pack.bf[4] = __float2bfloat16(acc[2].x); out_pack.bf[5] = __float2bfloat16(acc[2].y);
                    out_pack.bf[6] = __float2bfloat16(acc[3].x); out_pack.bf[7] = __float2bfloat16(acc[3].y);
                    *reinterpret_cast<int4*>(C_row + n_out) = out_pack.vec;
                }
            }
        }
        __syncthreads();  // ensure all warps finish before next atomicAdd
    }
}

// --------------------------------------------------------------------------
// OPTIMIZATION A3: L2 Cache Persistence Helper
// --------------------------------------------------------------------------

static void set_l2_persist_for_matrix(
    const void* ptr,
    size_t num_bytes,
    cudaStream_t stream,
    float hit_ratio = 1.0f
) {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = num_bytes;
    attr.accessPolicyWindow.hitRatio = hit_ratio;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaError_t err = cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    );

    // Ignore errors on older GPUs that don't support this
    if (err != cudaSuccess && err != cudaErrorNotSupported) {
        cudaGetLastError();  // Clear error
    }
}

static void reset_l2_persist(cudaStream_t stream) {
    cudaStreamAttrValue reset_attr = {};
    reset_attr.accessPolicyWindow.num_bytes = 0;

    cudaError_t err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &reset_attr);
    // Ignore errors on older GPUs that don't support this
    if (err != cudaSuccess && err != cudaErrorNotSupported) {
        cudaGetLastError();  // Clear error
    }
}


// --------------------------------------------------------------------------
// NEW KERNEL: Compute per-row tail counts FROM OVERFLOW COO
//   tail_counts[row] = # of entries in overflow_rows with that row
//   Uses overflow_counter; no host-side nnz needed.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// HOST: L2 PERSISTENCE (OPTIONAL)
// --------------------------------------------------------------------------
void set_matrix_b_persistence(const void* ptr, size_t size, cudaStream_t stream) {
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr  = (void*)ptr;
    attr.accessPolicyWindow.num_bytes = size;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaError_t err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    TORCH_CHECK(err == cudaSuccess, "cudaStreamSetAttribute failed: ", cudaGetErrorString(err));
}

hybrid_sp_t::hybrid_sp_t(int M, int N, torch::Device device)
    : hybrid_sp_t(M, N, device, g_ell_width_regular, g_tail_rows_regular) {}

hybrid_sp_t::hybrid_sp_t(int M, int N, torch::Device device, int ell_w, int tcap) {
    auto options_tint16 = at::TensorOptions().dtype(torch::kInt16).device(device);
    auto options_tint32 = at::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bf16 = at::TensorOptions().dtype(torch::kBFloat16).device(device);
    this->_ell_stride = ell_w;
    this->_tail_cap   = tcap;
    this->_ell_col_indices = at::empty({M * ell_w}, options_tint16);
    this->_ell_values = at::empty({M * ell_w}, options_bf16);
    this->_row_counters = at::zeros({M}, options_tint32);
    this->_overflow_counter = at::zeros({}, options_tint32);
    this->_tail_dense = at::zeros({tcap, N}, options_bf16);
    this->_tail_dense_map = at::full({M}, -1, options_tint32);
    this->_tail_dense_map_reverse = at::full({tcap}, -1, options_tint32);
    this->_dense_active_rows = tcap;
    this->hN = at::empty({1}, at::TensorOptions().device(at::kCPU).dtype(at::kInt).pinned_memory(true));
}

hybrid_sp_t::hybrid_sp_t(const hybrid_sp_t& sp) {
    this->_ell_col_indices = sp._ell_col_indices;
    this->_ell_values = sp._ell_values;
    this->_row_counters = sp._row_counters;
    this->_overflow_counter = sp._overflow_counter;
    this->_tail_dense = sp._tail_dense;
    this->_tail_dense_map = sp._tail_dense_map;
    this->_tail_dense_map_reverse = sp._tail_dense_map_reverse;
    this->_dense_active_rows = sp._dense_active_rows;
    this->_ell_stride = sp._ell_stride;
    this->_tail_cap   = sp._tail_cap;
}

// Reset ELL values while reusing col-index and tail structure from a shared sparsity pattern.
void hybrid_sp_t::reset_vals() {
    auto options_bf16 = at::TensorOptions().dtype(torch::kBFloat16).device(this->_ell_values.device());
    // We have counters for each row, the values don't need to be initialized
    this->_ell_values = at::empty({this->_ell_values.numel()}, options_bf16);
}

// --------------------------------------------------------------------------
// OVERFLOW TAIL KERNEL
// For rows where row_counts[row] > ell_width_threshold: atomically allocates
// a tail_dense slot, sets up tail_dense_map / tail_dense_map_reverse, then
// copies the full dense row to tail_dense with implicit ReLU implemented
// as a sign-bit mask (no fp conversions).
// Grid: M_rows blocks; non-overflow rows exit immediately after one load.
// Block: 128 threads (4 warps).
// --------------------------------------------------------------------------
__global__ void populate_overflow_tail_kernel(
    const __nv_bfloat16* __restrict__ dense,                  // [M_rows × N_cols]
    const int*           __restrict__ row_counts,             // [M_rows]
    int*                 __restrict__ overflow_counter,       // atomic slot allocator
    __nv_bfloat16*       __restrict__ tail_dense,             // [TAIL_CAPACITY_ROWS × N_cols]
    int*                 __restrict__ tail_dense_map,         // [M_rows], -1 = no overflow
    int*                 __restrict__ tail_dense_map_reverse, // [TAIL_CAPACITY_ROWS]
    int M_rows, int N_cols,
    int ell_width_threshold,   // runtime ELL overflow threshold (<= ELL_WIDTH)
    int tail_cap,              // runtime dense tail capacity (<= TAIL_CAPACITY_ROWS)
    int discard)               // if 1, count overflow but skip tail writes
{
    const int row = (int)blockIdx.x;
    if (row >= M_rows) return;
    if (row_counts[row] <= ell_width_threshold) return;  // not an overflow row

    // Thread 0 allocates the dense slot; result broadcast via smem
    __shared__ int smem_dr;
    if (threadIdx.x == 0) {
        int new_dr = atomicAdd(overflow_counter, 1);  // always count
        if (discard != 1 && new_dr < tail_cap) {
            tail_dense_map[row]              = new_dr;
            tail_dense_map_reverse[new_dr]   = row;
            smem_dr = new_dr;
        } else {
            smem_dr = -1;  // no mapping; tail_dense_map stays -1
        }
    }
    __syncthreads();
    const int dr = smem_dr;
    if (dr < 0) return;

    // Vectorized copy with implicit ReLU: zero out any element where sign bit = 1.
    // Uses 128-bit loads/stores (8 bf16 per thread per iteration).
    const __nv_bfloat16* src = dense     + (size_t)row * N_cols;
    __nv_bfloat16*       dst = tail_dense + (size_t)dr  * N_cols;

    for (int base = threadIdx.x * VEC_SIZE; base < N_cols; base += blockDim.x * VEC_SIZE) {
        int4 raw = *reinterpret_cast<const int4*>(src + base);
        uint16_t* u = reinterpret_cast<uint16_t*>(&raw);
        #pragma unroll
        for (int vi = 0; vi < VEC_SIZE; vi++) {
            if (u[vi] & 0x8000u) u[vi] = 0u;  // ReLU: zero negatives
        }
        *reinterpret_cast<int4*>(dst + base) = raw;
    }
}

// Forward declarations
template<int WARPS_PER_BLOCK>
__global__ void dense_to_ell_kernel(
    const __nv_bfloat16* __restrict__, uint16_t* __restrict__,
    __nv_bfloat16* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__, int, int, int, int);

template<int WARPS_PER_ROW>
__global__ void dense_to_ell_mwpr_kernel(
    const __nv_bfloat16* __restrict__, uint16_t* __restrict__,
    __nv_bfloat16* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__, int, int, int, int);


// WARPS_PER_ROW config for dense_to_ell:
//   0       = original multi-row-per-block kernel (WARPS_PER_BLOCK=4)
//   1,2,4,8 = multi-warp-per-row kernel (N warps cooperate on one row)
static int g_ell_create_warps_per_row = 0;
void set_ell_create_warps_per_row(int v) { g_ell_create_warps_per_row = v; }

// Per-operation ELL overflow thresholds — these also determine buffer allocation size.
// Buffer stride = max(regular, transpose) so both ops fit in the same hybrid_sp_t.
int g_ell_width_regular   = ELL_WIDTH;
int g_ell_width_transpose = ELL_WIDTH;
// Per-operation dense tail capacity — buffer size = max(regular, transpose).
int g_tail_rows_regular   = TAIL_CAPACITY_ROWS;
int g_tail_rows_transpose = TAIL_CAPACITY_ROWS;
// Discard overflow mode: 1 = drop overflow rows instead of writing to dense tail
int g_discard_overflow    = 0;

void set_ell_width_regular(int v)   { g_ell_width_regular   = (v > 0) ? v : ELL_WIDTH; }
void set_ell_width_transpose(int v) { g_ell_width_transpose = (v > 0) ? v : ELL_WIDTH; }
void set_tail_rows_regular(int v)   { g_tail_rows_regular   = (v > 0) ? v : TAIL_CAPACITY_ROWS; }
void set_tail_rows_transpose(int v) { g_tail_rows_transpose = (v > 0) ? v : TAIL_CAPACITY_ROWS; }
void set_discard_overflow(int v)    { g_discard_overflow    = v; }

// --------------------------------------------------------------------------
// HOST FUNCTION: create ELL directly from a dense gate-activation matrix.
// Used for the gate projection path.
// --------------------------------------------------------------------------

void create_hybrid_sparse_from_dense(
    const at::Tensor& dense,   // [M, N] bfloat16 row-major
    hybrid_sp_t* sp,
    at::Tensor& l0, at::Tensor& l1,
    int M, int N,
    cudaStream_t stream)
{
    PERF_START("create_hybrid_sparse_from_dense:total", stream);

    const auto* d_ptr   = static_cast<const __nv_bfloat16*>(dense.data_ptr());
    auto*       e_cols  = sp->ell_col_indices();
    auto*       e_vals  = sp->ell_values();
    auto*       e_rcnt  = sp->row_counters();
    auto*       l0_ptr  = l0.defined() ? static_cast<float*>(l0.data_ptr()) : nullptr;
    auto*       l1_ptr  = l1.defined() ? static_cast<float*>(l1.data_ptr()) : nullptr;

    PERF_START("create_hybrid_sparse_from_dense:dense_to_ell", stream);
    const int ell_stride = sp->_ell_stride;
    if (g_ell_create_warps_per_row == 0) {
        // Original: multiple rows per block (4 warps per block, 1 warp per row)
        constexpr int WPB = 4;
        size_t d2e_smem = (size_t)WPB * ell_stride * (sizeof(uint16_t) + sizeof(__nv_bfloat16));
        dense_to_ell_kernel<WPB><<<(M + WPB - 1) / WPB, WPB * 32, d2e_smem, stream>>>(
            d_ptr, e_cols, e_vals, e_rcnt, l0_ptr, l1_ptr, M, N, ell_stride, ell_stride);
    } else {
        // Multi-warp-per-row: WARPS_PER_ROW warps cooperate on one row
        auto mwpr_smem = [ell_stride](int wpr) -> size_t {
            return (size_t)wpr * ell_stride * (sizeof(uint16_t) + sizeof(__nv_bfloat16));
        };
        switch (g_ell_create_warps_per_row) {
            case 2: dense_to_ell_mwpr_kernel<2><<<M,  64, mwpr_smem(2), stream>>>(d_ptr, e_cols, e_vals, e_rcnt, l0_ptr, l1_ptr, M, N, ell_stride, ell_stride); break;
            case 4: dense_to_ell_mwpr_kernel<4><<<M, 128, mwpr_smem(4), stream>>>(d_ptr, e_cols, e_vals, e_rcnt, l0_ptr, l1_ptr, M, N, ell_stride, ell_stride); break;
            case 8: dense_to_ell_mwpr_kernel<8><<<M, 256, mwpr_smem(8), stream>>>(d_ptr, e_cols, e_vals, e_rcnt, l0_ptr, l1_ptr, M, N, ell_stride, ell_stride); break;
            default: dense_to_ell_mwpr_kernel<1><<<M,  32, mwpr_smem(1), stream>>>(d_ptr, e_cols, e_vals, e_rcnt, l0_ptr, l1_ptr, M, N, ell_stride, ell_stride); break;
        }
    }
    PERF_STOP("create_hybrid_sparse_from_dense:dense_to_ell");

#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "dense_to_ell: Fatal error: (%s at %s:%d)\n",
                cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error in dense_to_ell_kernel");
    }
#endif

    // For overflow rows (row_counts > ELL_WIDTH): allocate tail_dense slot,
    // set up maps, and copy the full dense row with implicit ReLU.
    PERF_START("create_hybrid_sparse_from_dense:overflow_tail", stream);
    populate_overflow_tail_kernel<<<M, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dense.data_ptr()),
        sp->row_counters(),
        sp->overflow_counter(),
        sp->tail_dense(),
        sp->tail_dense_map(),
        sp->tail_dense_map_reverse(),
        M, N,
        g_ell_width_regular,
        g_tail_rows_regular,
        g_discard_overflow
    );
    PERF_STOP("create_hybrid_sparse_from_dense:overflow_tail");

    PERF_STOP("create_hybrid_sparse_from_dense:total");
}


// Original warp-based kernel with perfect memory coalescing
__global__ void dense_ab_as_sparse_hybrid_unified_optimized(
    const __nv_bfloat16* __restrict__ A,          // [M x K]
    const __nv_bfloat16* __restrict__ B_T,        // [N x K] (Transposed)
    const uint16_t* __restrict__ ell_cols,        // [M x ell_stride]
    const int* __restrict__ row_counts,           // [M]
    __nv_bfloat16* __restrict__ C_ell_vals,       // [M x ell_stride]
    const float* __restrict__ init_val,           // scalar
    int M_rows,
    int K,
    int N_cols,
    int ell_stride,        // ELL buffer stride for this object (_ell_stride)
    int overflow_threshold // rows with nnz > this are overflow (g_ell_width_regular)
) {
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int nnz_total = row_counts[row];
    if (nnz_total <= 0) return;
    if (nnz_total > overflow_threshold) return;

    // 1. Shared Memory Cache for Row A
    extern __shared__ __nv_bfloat16 sh_A[];

    const __nv_bfloat16* A_row_ptr = A + (size_t)row * K;

    // --- Phase 1: Cooperative Load A into Shared Memory ---
    int num_vec_k = K / 8;
    int tail_k    = K % 8;

    int4* sh_A_vec = reinterpret_cast<int4*>(sh_A);
    const int4* A_row_vec = reinterpret_cast<const int4*>(A_row_ptr);

    // All threads in block help copy
    for (int i = threadIdx.x; i < num_vec_k; i += blockDim.x) {
        sh_A_vec[i] = A_row_vec[i];
    }

    // Handle remaining elements if K is not multiple of 8
    if (tail_k > 0) {
        int tail_start = num_vec_k * 8;
        for (int i = tail_start + threadIdx.x; i < K; i += blockDim.x) {
            sh_A[i] = A_row_ptr[i];
        }
    }

    __syncthreads();

    // --- Phase 2: Warp-based Dot Products (perfect coalescing) ---
    const int lane_id   = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    float base_init = (init_val) ? *init_val : 0.0f;

    // Each warp iterates over a subset of the output columns (ELL entries)
    for (int out_idx = warp_id; out_idx < nnz_total; out_idx += num_warps) {

        // 1. Identify which column of B to read
        int col = (int)ell_cols[(size_t)row * ell_stride + out_idx];
        if (col < 0 || col >= N_cols) continue;

        // Pointer to B column (which is a row in B_T)
        const __nv_bfloat16* B_row_ptr = B_T + (size_t)col * K;

        float acc = base_init;

        // 2. Vectorized Dot Product Loop with perfect coalescing
        int k_vec_idx = lane_id;

        // Main Loop (Vectorized)
        while (k_vec_idx < num_vec_k) {
            // Load A from Shared Memory (Fast, Low Latency)
            int4 a_raw = *reinterpret_cast<const int4*>(&sh_A[k_vec_idx * 8]);

            // Load B from Global Memory (Perfect coalescing across warp)
            int4 b_raw = *reinterpret_cast<const int4*>(&B_row_ptr[k_vec_idx * 8]);

            __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&a_raw);
            __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(&b_raw);

            // Accumulate 8 elements
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                float2 af = __bfloat1622float2(a2[t]);
                float2 bf = __bfloat1622float2(b2[t]);
                acc = fmaf(af.x, bf.x, acc);
                acc = fmaf(af.y, bf.y, acc);
            }

            k_vec_idx += 32; // Stride by warp size
        }

        // 3. Tail Loop (Scalar)
        if (tail_k > 0) {
            int k_idx = num_vec_k * 8 + lane_id;
            if (k_idx < K) {
                float a = __bfloat162float(sh_A[k_idx]);
                float b = __bfloat162float(B_row_ptr[k_idx]);
                acc = fmaf(a, b, acc);
            }
        }

        // 4. Warp Reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }

        // 5. Write Result
        if (lane_id == 0) {
            C_ell_vals[(size_t)row * ell_stride + out_idx] = __float2bfloat16(acc);
        }
    }
}

__device__ __forceinline__ bool bf16_nonzero_mask(__nv_bfloat16 v) {
    // compare raw bits against 0 (covers +0 and -0 too)
    return (__bfloat16_as_ushort(v) & 0x7FFF) != 0;
}

__global__ void tail_dense_masked_add_inplace(
    __nv_bfloat16*       __restrict__ out_tail,     // [rows, N]
    const __nv_bfloat16*       __restrict__ tail,     // [rows, N]
    const __nv_bfloat16* __restrict__ dense_out,    // [rows, N]
    int rows,
    int N,
    const float*         __restrict__ init_val      // device scalar
) {
    // each int4 = 16 bytes = 8 bf16 values
    const int64_t total_vec = (int64_t)rows * (int64_t)N / 8;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_vec) return;

    float init = init_val ? *init_val : 0.0f;

    // vector index -> bf16 element offset
    int64_t elem0 = tid * 8;

    // reinterpret as int4 for vector load/store
    int4 out_raw = *reinterpret_cast<const int4*>(out_tail  + elem0);
    int4 tail_raw = *reinterpret_cast<const int4*>(tail  + elem0);
    int4 den_raw = *reinterpret_cast<const int4*>(dense_out + elem0);

    __nv_bfloat16* out8 = reinterpret_cast<__nv_bfloat16*>(&out_raw);
    __nv_bfloat16* tail8 = reinterpret_cast<__nv_bfloat16*>(&tail_raw);
    __nv_bfloat16* den8 = reinterpret_cast<__nv_bfloat16*>(&den_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        bool m = bf16_nonzero_mask(tail8[i]);        // (out != 0)
        if (m) {
            float d = __bfloat162float(den8[i]);
            out8[i] = __float2bfloat16(init + d);
        } else {
            out8[i] = __float2bfloat16(0.0f);       // IMPORTANT: zero else-branch
        }
    }

    *reinterpret_cast<int4*>(out_tail + elem0) = out_raw;
}

void new_product_as_sparse_sma(hybrid_sp_t* out, at::Tensor const& a, at::Tensor const& b, at::Tensor const& init_val, int M, int N, int K, cudaStream_t stream) {
    PERF_START("new_product_as_sparse_total", stream);

    const __nv_bfloat16* A_ptr  =
        reinterpret_cast<const __nv_bfloat16*>(a.contiguous().data_ptr());
    const __nv_bfloat16* BT_ptr =
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());

    const float* init_ptr =
        init_val.data_ptr<float>();  // 1-element tensor

    dim3 block(256);  // 256 threads = 8 warps for more parallel outputs
    dim3 grid(M);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);

    // OPTIMIZATION: Enable L2 cache persistence for B_T matrix (reused across all rows)
    size_t BT_size = (size_t)N * K * sizeof(__nv_bfloat16);
    if (BT_size < 4 * 1024 * 1024) {  // Only if B_T fits in L2
        set_l2_persist_for_matrix(BT_ptr, BT_size, stream, 1.0f);
    }

    PERF_START("new_product_as_sparse:dense_ab_kernel", stream);
    dense_ab_as_sparse_hybrid_unified_optimized<<<grid, block, smem_bytes, stream>>>(
        A_ptr, BT_ptr,
        out->ell_col_indices(),
        out->row_counters(),
        out->ell_values(),
        init_ptr,
        M,
        K,
        N,
        out->_ell_stride,
        out->_ell_stride
    );
    PERF_STOP("new_product_as_sparse:dense_ab_kernel");

    // Reset L2 policy
    if (BT_size < 4 * 1024 * 1024) {
        reset_l2_persist(stream);
    }
#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "Dense_ab_as_sparse: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__); 
        TORCH_CHECK(false, "error on hybrid");
    }
#endif
    if(out->_dense_active_rows) {
        // Product with dense and accomulate it in out
        // Gather A, B rows
        auto options_bf16 = at::TensorOptions().dtype(torch::kBFloat16).device(a.device());
        // Dont need to zero, I am copying complete rows
        const int tcap = out->_tail_cap;
        at::Tensor dense = torch::empty({tcap, K}, options_bf16);
        __nv_bfloat16* dense_ptr  =
            reinterpret_cast<__nv_bfloat16*>(dense.data_ptr());

        PERF_START("new_product_as_sparse:gather_rows", stream);
        gather_rows_by_map_bf16_int4<<<M, 128, 0, stream>>>(
            A_ptr, dense_ptr, out->tail_dense_map(),
            M, tcap, K);
        PERF_STOP("new_product_as_sparse:gather_rows");
#if ERROR_CHECK
        __err = cudaDeviceSynchronize();
        if (__err != cudaSuccess) {
            fprintf(stderr, "gather_rows: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
            TORCH_CHECK(false, "error on hybrid");
        }
#endif
        if(out->_dense_active_rows > 0) {
        PERF_START("new_product_as_sparse:dense_matmul", stream);
    
        // Compute dense [M, K] @ b.T [K, N] where b is [N, K]
        at::Tensor dense_active = dense.narrow(0, 0, out->_dense_active_rows);  // [M, K]
    
        // Allocate output
        at::Tensor dense_out = torch::empty({out->_dense_active_rows, N},
                                           torch::TensorOptions()
                                               .dtype(torch::kBFloat16)
                                               .device(a.device()));
    
        dense_out = torch::matmul(dense_active, b.transpose(0, 1));

        PERF_STOP("new_product_as_sparse:dense_matmul");

        TORCH_CHECK(out->_tail_dense.is_contiguous(), "out_tail_dense must be contiguous");
        TORCH_CHECK(dense_out.is_contiguous(), "dense_out must be contiguous");

        const int vecs_per_row = N / 8;
        const int64_t total_vec = (int64_t)out->_dense_active_rows * (int64_t)vecs_per_row;
        const int threads = 256;
        const int blocks  = (int)((total_vec + threads - 1) / threads);
        // Clone before modifying: _tail_dense may be aliased across P/R/T
        auto tail_dense = out->_tail_dense.clone();
        out->_tail_dense = torch::empty_like(tail_dense);
        PERF_START("new_product_as_sparse:masked_add", stream);
        tail_dense_masked_add_inplace<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out->_tail_dense.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(tail_dense.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(dense_out.data_ptr()),
            out->_dense_active_rows,
            N,
            init_ptr
        );
        PERF_STOP("new_product_as_sparse:masked_add");
        }
#if ERROR_CHECK
        __err = cudaDeviceSynchronize();
        if (__err != cudaSuccess) {
            fprintf(stderr,
                    "masked_add_inplace: Fatal error: (%s at %s:%d)\n",
                    cudaGetErrorString(__err), __FILE__, __LINE__);
            TORCH_CHECK(false, "error in new_product_as_sparse_sma");
        }
#endif
    }
    PERF_STOP("new_product_as_sparse_total");
}


__device__ __forceinline__ bool bf16_is_pos_nonzero(uint16_t u)
{
    // Fast predicate: sign==0 and not +0
    // NOTE: This will treat +NaN as "positive". If NaNs exist and you need IEEE semantics,
    // use a real compare: (__bfloat162float(*(__nv_bfloat16*)&u) > 0.0f)
    return ((u & 0x8000u) == 0u) && ((u & 0x7fffu) != 0u);
}



// --------------------------------------------------------------------------
// FAST DENSE→ELL KERNEL
// One warp per row. No slot-assignment atomics: warp-ballot prefix scan.
// Overflow rows: ELL filled for first ELL_WIDTH active elements; row_counts
// stores the true total NNZ. Overflow tail handling deferred.
// l0 / l1:
//   l0 += 1/M_rows per active element, l1 += val/M_rows per active element.
// --------------------------------------------------------------------------
template<int WARPS_PER_BLOCK>
__global__ void dense_to_ell_kernel(
    const __nv_bfloat16* __restrict__ dense,     // [M_rows × N_cols]
    uint16_t*            __restrict__ ell_cols,  // [M_rows × ell_stride]
    __nv_bfloat16*       __restrict__ ell_vals,  // [M_rows × ell_stride]
    int*                 __restrict__ row_counts, // [M_rows]
    float*               __restrict__ l0_out,    // nullable scalar accumulator
    float*               __restrict__ l1_out,    // nullable scalar accumulator
    int M_rows, int N_cols,
    int ell_stride,        // ELL buffer stride for this object (_ell_stride)
    int overflow_threshold // ELL capacity for this op (g_ell_width_regular)
)
{
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int row     = (int)blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Per-warp smem slices — dynamic, laid out as [WARPS_PER_BLOCK][ell_stride]
    // Layout: [WPB * ell_stride * u16] | [WPB * ell_stride * bf16]
    extern __shared__ char _smem_d2e[];
    uint16_t*      smem_cols = reinterpret_cast<uint16_t*>(_smem_d2e);
    __nv_bfloat16* smem_vals = reinterpret_cast<__nv_bfloat16*>(
        _smem_d2e + WARPS_PER_BLOCK * ell_stride * sizeof(uint16_t));

    if (row >= M_rows) return;

    const __nv_bfloat16* row_ptr = dense + (size_t)row * N_cols;
    int   slot_cnt = 0;  // identical value on all lanes (counts total active seen)
    float l0_acc   = 0.0f;
    float l1_acc   = 0.0f;

    // Scan row in batches of VEC_SIZE*32 = 256 elements per warp
    for (int base_col = 0; base_col < N_cols; base_col += 32 * VEC_SIZE) {
        const int lane_base = base_col + lane * VEC_SIZE;
        const bool in_bounds = (lane_base + VEC_SIZE - 1) < N_cols;

        // 128-bit vectorized load of 8 bf16 elements (16-byte aligned)
        __nv_bfloat16 elems[VEC_SIZE];
        if (in_bounds) {
            *reinterpret_cast<int4*>(elems) =
                *reinterpret_cast<const int4*>(row_ptr + lane_base);
        }

        // Each sub-element: ballot across the full warp, then prefix-rank write
        #pragma unroll
        for (int vi = 0; vi < VEC_SIZE; vi++) {
            const bool active = in_bounds &&
                bf16_is_pos_nonzero(__bfloat16_as_ushort(elems[vi]));

            const uint32_t mask  = __ballot_sync(0xFFFFFFFF, active);
            const int      count = __popc(mask);
            const int      rank  = __popc(mask & ((1u << lane) - 1u));

            if (active) {
                l0_acc += 1.0f / (float)M_rows;
                l1_acc += __bfloat162float(elems[vi]) / (float)M_rows;
                const int s = slot_cnt + rank;
                if (s < overflow_threshold) {
                    smem_cols[warp_id * ell_stride + s] = (uint16_t)(lane_base + vi);
                    smem_vals[warp_id * ell_stride + s] = elems[vi];
                }
                // overflow elements are silently counted only
            }
            slot_cnt += count;  // same on all lanes
        }
    }

    // Warp-reduce l0/l1 to lane 0
    for (int offset = 16; offset > 0; offset >>= 1) {
        l0_acc += __shfl_down_sync(0xFFFFFFFF, l0_acc, offset);
        l1_acc += __shfl_down_sync(0xFFFFFFFF, l1_acc, offset);
    }

    // __ballot_sync calls act as warp-level barriers; smem writes are visible
    // to all lanes before we enter the writeback loop.
    const int actual_nnz = (slot_cnt < overflow_threshold) ? slot_cnt : overflow_threshold;
    uint16_t*      out_cols = ell_cols + (size_t)row * ell_stride;
    __nv_bfloat16* out_vals = ell_vals + (size_t)row * ell_stride;
    for (int k = lane; k < actual_nnz; k += 32) {
        out_cols[k] = smem_cols[warp_id * ell_stride + k];
        out_vals[k] = smem_vals[warp_id * ell_stride + k];
    }

    if (lane == 0) {
        row_counts[row] = slot_cnt;
        if (l0_out) atomicAdd(l0_out, l0_acc);
        if (l1_out) atomicAdd(l1_out, l1_acc);
    }
}

// --------------------------------------------------------------------------
// MULTI-WARP PER ROW dense→ELL KERNEL
// WARPS_PER_ROW warps cooperate on a single row.
//   - Each warp scans N_cols/WARPS_PER_ROW columns independently.
//   - Ballot-based slot assignment (no atomics) within each warp.
//   - After scan: inter-warp exclusive prefix sum → per-warp global ELL slot bases.
//   - Each warp scatters its compacted elements to global ELL at its slot base.
// With WARPS_PER_ROW=4 each warp does 4× fewer ballot calls than single-warp.
// --------------------------------------------------------------------------
template<int WARPS_PER_ROW>
__global__ void dense_to_ell_mwpr_kernel(
    const __nv_bfloat16* __restrict__ dense,     // [M_rows × N_cols]
    uint16_t*            __restrict__ ell_cols,  // [M_rows × ell_stride]
    __nv_bfloat16*       __restrict__ ell_vals,  // [M_rows × ell_stride]
    int*                 __restrict__ row_counts, // [M_rows]
    float*               __restrict__ l0_out,    // nullable scalar accumulator
    float*               __restrict__ l1_out,    // nullable scalar accumulator
    int M_rows, int N_cols,
    int ell_stride,        // ELL buffer stride for this object (_ell_stride)
    int overflow_threshold // ELL capacity for this op (g_ell_width_regular)
)
{
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;   // 0 .. WARPS_PER_ROW-1
    const int row     = (int)blockIdx.x;
    if (row >= M_rows) return;

    // Dynamic smem for per-warp col/val staging: [WPR * ell_stride * u16] | [WPR * ell_stride * bf16]
    // Fixed-size smem for warp metadata stays static.
    extern __shared__ char _smem_mwpr[];
    uint16_t*      smem_cols = reinterpret_cast<uint16_t*>(_smem_mwpr);
    __nv_bfloat16* smem_vals = reinterpret_cast<__nv_bfloat16*>(
        _smem_mwpr + WARPS_PER_ROW * ell_stride * sizeof(uint16_t));
    __shared__ int           warp_cnt[WARPS_PER_ROW];
    __shared__ int           warp_base[WARPS_PER_ROW];
    __shared__ float         smem_l0[WARPS_PER_ROW], smem_l1[WARPS_PER_ROW];

    // Partition columns across warps (ceiling division, last warp may get fewer)
    const int col_per_warp = (N_cols + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
    const int col_start    = warp_id * col_per_warp;
    const int col_end      = min(col_start + col_per_warp, N_cols);

    const __nv_bfloat16* row_ptr = dense + (size_t)row * N_cols;
    int   write_cnt = 0;  // ELL slots written to local smem (capped at ELL_WIDTH)
    int   total_cnt = 0;  // true NNZ found in this warp's column region
    float l0_acc = 0.f, l1_acc = 0.f;

    for (int base_col = col_start; base_col < col_end; base_col += 32 * VEC_SIZE) {
        const int lane_base = base_col + lane * VEC_SIZE;
        const bool in_bounds = (lane_base + VEC_SIZE - 1) < col_end;

        __nv_bfloat16 elems[VEC_SIZE];
        if (in_bounds)
            *reinterpret_cast<int4*>(elems) =
                *reinterpret_cast<const int4*>(row_ptr + lane_base);

        #pragma unroll
        for (int vi = 0; vi < VEC_SIZE; vi++) {
            const bool active = in_bounds &&
                bf16_is_pos_nonzero(__bfloat16_as_ushort(elems[vi]));
            const uint32_t mask  = __ballot_sync(0xFFFFFFFF, active);
            const int      count = __popc(mask);
            const int      rank  = __popc(mask & ((1u << lane) - 1u));
            if (active) {
                l0_acc += 1.f / (float)M_rows;
                l1_acc += __bfloat162float(elems[vi]) / (float)M_rows;
                const int s = write_cnt + rank;
                if (s < overflow_threshold) {
                    smem_cols[warp_id * ell_stride + s] = (uint16_t)(lane_base + vi);
                    smem_vals[warp_id * ell_stride + s] = elems[vi];
                }
            }
            total_cnt += count;
            write_cnt += count;
        }
    }
    write_cnt = min(write_cnt, overflow_threshold);

    // Reduce l0/l1 to lane 0, then expose warp metadata via smem
    for (int off = 16; off > 0; off >>= 1) {
        l0_acc += __shfl_down_sync(0xFFFFFFFF, l0_acc, off);
        l1_acc += __shfl_down_sync(0xFFFFFFFF, l1_acc, off);
    }
    if (lane == 0) {
        warp_cnt[warp_id] = total_cnt;
        smem_l0[warp_id]  = l0_acc;
        smem_l1[warp_id]  = l1_acc;
    }

    __syncthreads();  // warp_cnt / smem_l0 / smem_l1 now visible

    // Warp 0, lane 0: exclusive prefix sum → warp_base[]; commit row-level outputs
    if (warp_id == 0 && lane == 0) {
        int base = 0;
        float gl0 = 0.f, gl1 = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_ROW; w++) {
            warp_base[w]  = base;
            base         += warp_cnt[w];
            gl0          += smem_l0[w];
            gl1          += smem_l1[w];
        }
        row_counts[row] = base;
        if (l0_out) atomicAdd(l0_out, gl0);
        if (l1_out) atomicAdd(l1_out, gl1);
    }

    __syncthreads();  // warp_base[] now visible to all warps

    // Each warp scatters its local smem to global ELL at its assigned slot base
    const int global_base = warp_base[warp_id];
    if (global_base < overflow_threshold) {
        const int allowed      = min(write_cnt, overflow_threshold - global_base);
        uint16_t*      out_cols = ell_cols + (size_t)row * ell_stride;
        __nv_bfloat16* out_vals = ell_vals + (size_t)row * ell_stride;
        for (int k = lane; k < allowed; k += 32) {
            out_cols[global_base + k] = smem_cols[warp_id * ell_stride + k];
            out_vals[global_base + k] = smem_vals[warp_id * ell_stride + k];
        }
    }
}


__global__ void transpose_hybrid_ell_dense(
    // Input A (hybrid: ELL + overflow-as-dense-rows)
    const uint16_t*      __restrict__ A_ell_cols,
    const __nv_bfloat16* __restrict__ A_ell_vals,
    const int*           __restrict__ A_row_counts,
    const __nv_bfloat16* __restrict__ A_tail_dense,              // [TAIL_ROWS x dense_ld]
    const int*           __restrict__ A_tail_dense_map_reverse,  // [TAIL_ROWS], maps dense_row -> sparse_row (or -1)
    int*                 __restrict__ A_tail_dense_rows,                // number of rows in A_tail_dense / map_reverse
    int                  A_dense_ld,                        // number of cols in A_tail_dense (typically N_cols)
    int M_rows,
    int N_cols,

    // Output A^T (hybrid)
    uint16_t*            __restrict__ AT_ell_cols,
    __nv_bfloat16*       __restrict__ AT_ell_vals,
    int*                 __restrict__ AT_row_counts,
    int*                 __restrict__ AT_overflow_counter,
    __nv_bfloat16*       __restrict__ tail_dense,        // [TAIL_ROWS_T x dense_ld_T] (your AT tail dense storage)
    int                  dense_ld,                        // number of cols in A_tail_dense (typically N_cols)
    int*                 __restrict__ tail_dense_map,     // [N_cols] map for AT overflow rows (keyed by AT row)
    int*                 __restrict__ tail_dense_map_reverse,     // [N_cols] map for AT overflow rows (keyed by AT row)

    // NEW: Optional precomputed tail maps for reuse
    const int*           __restrict__ precomputed_tail_dense_map,
    bool                 use_precomputed_maps,
    int                  ell_w,          // ELL overflow threshold for output (AT._ell_stride)
    int                  in_ell_stride,  // ELL buffer stride of input A
    int                  out_ell_stride, // ELL buffer stride of output AT
    int                  tail_cap,       // dense tail capacity for output AT
    int                  discard         // if 1, drop overflow elements
) {
    // --------------------------
    // 1) Transpose NON-overflow rows from ELL
    // --------------------------
    for (int row = blockIdx.x; row < M_rows; row += gridDim.x) {
        int nnz_row = A_row_counts[row];
        if (nnz_row <= 0) continue;

        // IMPORTANT: rows with nnz > ell_w were materialized into A_tail_dense
        // (including their ELL portion), so we must NOT process their ELL here.
        if (nnz_row > ell_w) continue;

        const int ell_n = nnz_row; // <= ELL_WIDTH

        for (int k = threadIdx.x; k < ell_n; k += blockDim.x) {
            uint16_t col = A_ell_cols[(size_t)row * in_ell_stride + k];
            __nv_bfloat16 val = A_ell_vals[(size_t)row * in_ell_stride + k];

            const int out_row = (int)col;  // row index in A^T
            const int out_col = row;       // col index in A^T
            if ((unsigned)out_row >= (unsigned)N_cols) continue;
            int pos = atomicAdd(&AT_row_counts[out_row], 1);
            if (pos < ell_w) {
                size_t addr = (size_t)out_row * out_ell_stride + pos;
                AT_ell_cols[addr] = (uint16_t)out_col;
                AT_ell_vals[addr] = val;
            } else {
                if (use_precomputed_maps) {
                    populate_dense_known(out_row, out_col, val,
                                       tail_dense, /*dense_ld_AT=*/M_rows,
                                       precomputed_tail_dense_map);
                } else {
                    populate_dense(out_row, out_col, val,
                                   tail_dense, /*dense_ld_AT=*/M_rows,
                                   tail_dense_map, tail_dense_map_reverse, AT_overflow_counter,
                                   tail_cap, discard);
                }
            }
        }
    }
    //return;
    // --------------------------
    // 2) transpose overflow rows by scanning A_tail_dense
    //    SIMD: int4 loads => 8 bf16 per thread per iteration
    // --------------------------
    int a_dense_rows = *A_tail_dense_rows;
    for (int dense_row = blockIdx.x; dense_row < a_dense_rows; dense_row += gridDim.x) {
        const int src_row = A_tail_dense_map_reverse[dense_row];
        if (src_row < 0) continue;                 // unused dense row
        if ((unsigned)src_row >= (unsigned)M_rows) continue;

        const __nv_bfloat16* __restrict__ src = A_tail_dense + (size_t)dense_row * A_dense_ld;

        // each thread processes chunks of 8 columns (vector width = 8 bf16)
        // NOTE: iterate over A's column count (A_dense_ld), NOT AT's leading dim (dense_ld).
        // A_tail_dense rows have A_dense_ld elements; dense_ld is AT's (transposed) width.
        for (int col0 = threadIdx.x * 8; col0 < A_dense_ld; col0 += blockDim.x * 8) {

            if (col0 + 7 < A_dense_ld) {
                // int4 = 16 bytes = 8 bf16
                int4 raw = *reinterpret_cast<const int4*>(src + col0);

                // quick reject: if all 16 bytes are 0 => all 8 bf16 are 0
                // (works because bf16 zero is 0x0000)
                if ((raw.x | raw.y | raw.z | raw.w) == 0) {
                    continue;
                }

                // unpack 8 bf16 lanes from 4x32-bit words
                uint32_t w0 = (uint32_t)raw.x;
                uint32_t w1 = (uint32_t)raw.y;
                uint32_t w2 = (uint32_t)raw.z;
                uint32_t w3 = (uint32_t)raw.w;

                uint16_t e[8];
                e[0] = (uint16_t)(w0 & 0xFFFF);
                e[1] = (uint16_t)(w0 >> 16);
                e[2] = (uint16_t)(w1 & 0xFFFF);
                e[3] = (uint16_t)(w1 >> 16);
                e[4] = (uint16_t)(w2 & 0xFFFF);
                e[5] = (uint16_t)(w2 >> 16);
                e[6] = (uint16_t)(w3 & 0xFFFF);
                e[7] = (uint16_t)(w3 >> 16);

                // build an 8-bit mask of non-zeros
                unsigned mask = 0;
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    mask |= (unsigned)(e[t] != 0) << t;
                }
                if (!mask) continue;

                // iterate set bits (only for non-zero lanes)
                while (mask) {
                    int t = __ffs(mask) - 1;
                    mask &= (mask - 1);

                    const int out_row = col0 + t;    // row in A^T
                    if ((unsigned)out_row >= (unsigned)N_cols) continue;

                    // load value (safe, aligned chunk already in registers but simplest is reread)
                    __nv_bfloat16 val = src[out_row];

                    const int out_col = src_row;     // col in A^T

                    int pos = atomicAdd(&AT_row_counts[out_row], 1);
                    if (pos < ell_w) {
                        size_t addr = (size_t)out_row * out_ell_stride + pos;
                        AT_ell_cols[addr] = (uint16_t)out_col;
                        AT_ell_vals[addr] = val;
                    } else {
                        if (use_precomputed_maps) {
                            populate_dense_known(out_row, out_col, val,
                                               tail_dense, /*dense_ld_AT=*/M_rows,
                                               precomputed_tail_dense_map);
                        } else {
                            populate_dense(out_row, out_col, val,
                                           tail_dense, /*dense_ld_AT=*/M_rows,
                                           tail_dense_map, tail_dense_map_reverse, AT_overflow_counter,
                                           tail_cap, discard);
                        }
                    }
                }
            } else {
                // tail columns (scalar)
                for (int c = col0; c < A_dense_ld; ++c) {
                    __nv_bfloat16 val = src[c];
                    // zero test: check bits directly (faster than float convert)
                    if (*reinterpret_cast<const uint16_t*>(&val) == 0) continue;

                    const int out_row = c;
                    if ((unsigned)out_row >= (unsigned)N_cols) continue;

                    const int out_col = src_row;

                    int pos = atomicAdd(&AT_row_counts[out_row], 1);
                    if (pos < ell_w) {
                        size_t addr = (size_t)out_row * out_ell_stride + pos;
                        AT_ell_cols[addr] = (uint16_t)out_col;
                        AT_ell_vals[addr] = val;
                    } else {
                        if (use_precomputed_maps) {
                            populate_dense_known(out_row, out_col, val,
                                               tail_dense, /*dense_ld_AT=*/M_rows,
                                               precomputed_tail_dense_map);
                        } else {
                            populate_dense(out_row, out_col, val,
                                           tail_dense, /*dense_ld_AT=*/M_rows,
                                           tail_dense_map, tail_dense_map_reverse, AT_overflow_counter,
                                           tail_cap, discard);
                        }
                    }
                }
            }
        }
    }
}



void transpose_hybrid_dense(
    const hybrid_sp_t& A,   // input
    hybrid_sp_t&       AT,  // output (already allocated with N_rows=M_cols, etc.)
    int M_rows,
    int N_cols,
    cudaStream_t stream,
    const int* precomputed_tail_dense_map,
    const int* precomputed_tail_dense_map_reverse)
{
    PERF_START("transpose_hybrid_dense_total", stream);

    bool use_precomputed = (precomputed_tail_dense_map != nullptr);

    // If precomputed maps are provided, copy them to AT
    if (use_precomputed && g_discard_overflow != 1) {
        cudaError_t err = cudaMemcpyAsync(AT.tail_dense_map(),
                       precomputed_tail_dense_map,
                       N_cols * sizeof(int),
                       cudaMemcpyDeviceToDevice,
                       stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
        err = cudaMemcpyAsync(AT.tail_dense_map_reverse(),
                       precomputed_tail_dense_map_reverse,
                       AT._tail_cap * sizeof(int),
                       cudaMemcpyDeviceToDevice,
                       stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
    }

    auto device = A._ell_col_indices.get_device();
    int num_sms;
    cudaError_t err2 = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(err2 == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err2));
    dim3 block(128);
    dim3 grid(min(M_rows, num_sms*4));  // or whatever you want

    PERF_START("transpose_hybrid_dense:transpose_kernel", stream);
    transpose_hybrid_ell_dense<<<grid, block, 0, stream>>>(
        A.ell_col_indices(),
        A.ell_values(),
        A.row_counters(),
        A.tail_dense(),
        A.tail_dense_map_reverse(),
        A.overflow_counter(),
        N_cols,
        M_rows,
        N_cols,
        AT.ell_col_indices(),
        AT.ell_values(),
        AT.row_counters(),
        AT.overflow_counter(),
        AT.tail_dense(),
        M_rows,
        AT.tail_dense_map(),
        AT.tail_dense_map_reverse(),
        use_precomputed ? precomputed_tail_dense_map : nullptr,
        use_precomputed,
        AT._ell_stride,      // ell_w (overflow threshold = output stride)
        A._ell_stride,       // in_ell_stride
        AT._ell_stride,      // out_ell_stride
        AT._tail_cap,
        g_discard_overflow
    );
    PERF_STOP("transpose_hybrid_dense:transpose_kernel");
#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "Transpose: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error on transpose");
    }
#endif
    // Copy the exceeding elements to the dense matrix
    if (g_discard_overflow != 1) {
        PERF_START("transpose_hybrid_dense:promote_overflow", stream);
        promote_overflow_rows_ell_into_tail_dense<<<N_cols, 128, 0, stream>>>(
            AT.ell_col_indices(),
            AT.ell_values(),
            AT.row_counters(),
            N_cols,
            AT.tail_dense(),
            M_rows,
            AT.tail_dense_map(),
            AT._ell_stride,      // ell_w
            AT._ell_stride       // ell_stride
        );
        PERF_STOP("transpose_hybrid_dense:promote_overflow");
    }
#if ERROR_CHECK
    __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "Transpose Promote 1: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error on transpose");
    }
#endif

    PERF_STOP("transpose_hybrid_dense_total");
}


int op_id = 0;

void sparse_dense_gemm_hybrid_dense(at::Tensor& out, hybrid_sp_t* A, const at::Tensor& B, int M, int N, int K, bool transpose_dense_part, cudaStream_t stream, const at::Tensor& B_fp32_cache) {
    PERF_START("sparse_dense_gemm_total", stream);

    //// Do a dense dense matmul of the overflow part

    // OPTIMIZATION A3: Enable L2 cache persistence for B matrix.
    // Use hitRatio = min(1, L2_size / B_size) so that on hardware where B fits
    // (or nearly fits) in L2, the accessed rows are kept resident across forward
    // passes.  On small GPUs the ratio approaches 1.0; on H100 (50 MB L2,
    // ~56 MB B) it is ~0.89, meaning almost all accessed rows stay persistent.
    size_t B_size = B.numel() * B.element_size();
    {
        int l2_bytes = 0;
        cudaError_t err = cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
        TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
        float hit_ratio = (l2_bytes > 0 && B_size > 0)
                        ? fminf(1.0f, (float)l2_bytes / (float)B_size)
                        : 1.0f;
        set_l2_persist_for_matrix(B.data_ptr(), B_size, stream, hit_ratio);
    }

    // ELL kernel for non-overflow rows (nnz <= ELL_WIDTH)
    PERF_START("sparse_dense_gemm:ell_spmm", stream);
    const int ell_stride = A->_ell_stride;
    size_t spmm_smem = (size_t)ell_stride * (sizeof(__nv_bfloat16) + sizeof(uint16_t));
    ell_spmm_rowmajor_b_rowwise_optimized<<<M, 256, spmm_smem, stream>>>(
        A->ell_values(), A->ell_col_indices(), A->row_counters(),
        static_cast<__nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        M, K, N,
        ell_stride,
        ell_stride
    );
    PERF_STOP("sparse_dense_gemm:ell_spmm");

    // Reset L2 persistence policy after kernel
    reset_l2_persist(stream);
#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "ELL Prod 1: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error on hybrid ell");
    }
#endif
    at::Tensor dense;
    PERF_START("sparse_dense_gemm:dense_matmul", stream);
    if(A->_dense_active_rows) {
        // Use native PyTorch matmul with bfloat16 precision
        at::Tensor A_tail_bf16 = A->_tail_dense.narrow(0, 0, A->_dense_active_rows);
        at::Tensor B_bf16 = B.reshape({K, N});
    
        // Allocate output tensor
        dense = torch::empty({A->_dense_active_rows, N},
                            torch::TensorOptions()
                                .dtype(torch::kBFloat16)
                                .device(B.device()));
    
        dense = torch::matmul(A_tail_bf16, B_bf16);

        PERF_STOP("sparse_dense_gemm:dense_matmul");
#if ERROR_CHECK
        __err = cudaDeviceSynchronize();
        if (__err != cudaSuccess) {
            int device = B.get_device();
            fprintf(stderr, "OP %d, device %d Error in dense-matmul ell (%d, %d) -> %d\n", op_id, device, M, N, A->_dense_active_rows);
            fprintf(stderr, "DENSEMATMUL ELL: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
            TORCH_CHECK(false, "error on hybrid dense");
        }
#endif
        PERF_START("sparse_dense_gemm:scatter", stream);
        scatter_tail_dense_rows_into_full<<<M, 128, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            M, N,
            static_cast<__nv_bfloat16*>(dense.data_ptr()),
            A->_dense_active_rows,
            A->tail_dense_map()
        );
        PERF_STOP("sparse_dense_gemm:scatter");
#if ERROR_CHECK
        __err = cudaDeviceSynchronize();
        op_id++;
        if (__err != cudaSuccess) {
            int device = B.get_device();
            fprintf(stderr, "OP %d, device %d Error in scatter ell (%d, %d) -> %d\n",op_id, device, M, N, A->_dense_active_rows);
            fprintf(stderr, "scatter ELL: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
            TORCH_CHECK(false, "error on hybrid scatter");
        }
#endif
    }
    PERF_STOP("sparse_dense_gemm_total");
}


at::Tensor ell_spmm_raw(
    at::Tensor ell_vals,
    at::Tensor ell_cols,
    at::Tensor row_counts,
    at::Tensor B,
    int64_t M, int64_t K, int64_t N,
    int64_t ell_stride,
    int64_t overflow_threshold
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    at::Tensor out = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(B.device()));

    // Zero only rows with nnz == 0; all other rows are written by the main kernel.
    // When nnz > 0 for all rows, this kernel returns immediately in every warp.
    {
        int zero_grid = ((int)M + WARPS_PER_BLOCK_WPR - 1) / WARPS_PER_BLOCK_WPR;
        zero_empty_rows<<<zero_grid, WARPS_PER_BLOCK_WPR * WARP_SIZE_WPR, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(row_counts.data_ptr()),
            (int)M, (int)N
        );
    }

    // Dispatch heuristic:
    //   ORIG_COLS_PER_PASS = 256*VEC_SIZE = 2048 (threads × cols/thread in original kernel).
    //   WPR wins when N % 2048 != 0 (original wastes threads on last pass) or N < 2048.
    //   Persistent wins when N is a multiple of 2048 > 2048: per-warp work is large enough
    //     to amortize atomicAdd overhead, and it avoids wave-scheduling latency.
    //   At N=2048: original tiles perfectly in 1 pass and outperforms both alternatives.
    const int ORIG_COLS_PER_PASS = 256 * VEC_SIZE;  // = 2048
    const bool n_is_exact_multiple = (N % ORIG_COLS_PER_PASS) == 0;
    if (!n_is_exact_multiple || N < ORIG_COLS_PER_PASS) {
        // WPR: non-multiple of 2048 or very small N
        int grid = ((int)M + WARPS_PER_BLOCK_WPR - 1) / WARPS_PER_BLOCK_WPR;
        ell_spmm_warp_per_row<<<grid, WARPS_PER_BLOCK_WPR * WARP_SIZE_WPR, 0, stream>>>(
            static_cast<__nv_bfloat16*>(ell_vals.data_ptr()),
            static_cast<uint16_t*>(ell_cols.data_ptr()),
            static_cast<int*>(row_counts.data_ptr()),
            static_cast<__nv_bfloat16*>(B.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            (int)M, (int)K, (int)N,
            (int)ell_stride,
            (int)overflow_threshold
        );
    } else if (N == ORIG_COLS_PER_PASS) {
        // Original: N=2048, perfect tiling, all 256 threads active, 1 pass
        size_t spmm_smem = (size_t)ell_stride * (sizeof(__nv_bfloat16) + sizeof(uint16_t));
        ell_spmm_rowmajor_b_rowwise_optimized<<<M, 256, spmm_smem, stream>>>(
            static_cast<__nv_bfloat16*>(ell_vals.data_ptr()),
            static_cast<uint16_t*>(ell_cols.data_ptr()),
            static_cast<int*>(row_counts.data_ptr()),
            static_cast<__nv_bfloat16*>(B.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            (int)M, (int)K, (int)N,
            (int)ell_stride,
            (int)overflow_threshold
        );
    } else {
        // Persistent: N is a multiple of 2048 larger than 2048 (e.g. 4096, 6144, ...)
        // Allocate work-stealing counter from device memory pool (fast, pooled allocation)
        at::Tensor counter = torch::zeros({1},
            torch::TensorOptions().dtype(torch::kInt32).device(B.device()));
        int device_id = B.get_device();
        int sm_count = 0;
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
        TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
        const int BLOCKS_PER_SM = 4;
        int grid = sm_count * BLOCKS_PER_SM;
        ell_spmm_persistent<<<grid, WARPS_PER_BLOCK_WPR * WARP_SIZE_WPR, 0, stream>>>(
            static_cast<__nv_bfloat16*>(ell_vals.data_ptr()),
            static_cast<uint16_t*>(ell_cols.data_ptr()),
            static_cast<int*>(row_counts.data_ptr()),
            static_cast<__nv_bfloat16*>(B.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(counter.data_ptr()),
            (int)M, (int)K, (int)N,
            (int)ell_stride,
            (int)overflow_threshold
        );
    }
    return out;
}

at::Tensor ell_spmm_raw_orig(
    at::Tensor ell_vals,
    at::Tensor ell_cols,
    at::Tensor row_counts,
    at::Tensor B,
    int64_t M, int64_t K, int64_t N,
    int64_t ell_stride,
    int64_t overflow_threshold
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    at::Tensor out = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(B.device()));

    size_t spmm_smem = (size_t)ell_stride * (sizeof(__nv_bfloat16) + sizeof(uint16_t));
    ell_spmm_rowmajor_b_rowwise_optimized<<<M, 256, spmm_smem, stream>>>(
        static_cast<__nv_bfloat16*>(ell_vals.data_ptr()),
        static_cast<uint16_t*>(ell_cols.data_ptr()),
        static_cast<int*>(row_counts.data_ptr()),
        static_cast<__nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)M, (int)K, (int)N,
        (int)ell_stride,
        (int)overflow_threshold
    );
    return out;
}

at::Tensor ell_spmm_raw_persistent(
    at::Tensor ell_vals,
    at::Tensor ell_cols,
    at::Tensor row_counts,
    at::Tensor B,
    int64_t M, int64_t K, int64_t N,
    int64_t ell_stride,
    int64_t overflow_threshold
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    at::Tensor out = torch::zeros({(long)M, (long)N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(B.device()));

    // Work-stealing counter, zeroed on device
    at::Tensor counter = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(B.device()));

    // Launch: sm_count * 8 blocks (max occupancy for 256-thread blocks on H100)
    int device_id = B.get_device();
    int sm_count = 0;
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
    // 8 blocks per SM: 256 threads / 32 = 8 warps; H100 can host up to 32 warps/SM
    // but 8 blocks * 8 warps = 64 warps > 32, so cap at 4 blocks/SM to stay in limits
    const int BLOCKS_PER_SM = 4;
    int grid = sm_count * BLOCKS_PER_SM;

    ell_spmm_persistent<<<grid, WARPS_PER_BLOCK_WPR * WARP_SIZE_WPR, 0, stream>>>(
        static_cast<__nv_bfloat16*>(ell_vals.data_ptr()),
        static_cast<uint16_t*>(ell_cols.data_ptr()),
        static_cast<int*>(row_counts.data_ptr()),
        static_cast<__nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int*>(counter.data_ptr()),
        (int)M, (int)K, (int)N,
        (int)ell_stride,
        (int)overflow_threshold
    );
    return out;
}


__forceinline__ __device__ bool int4_all_zero(const int4& v) {
    return (v.x | v.y | v.z | v.w) == 0;
}

// out_ell:   [M * ell_stride] bf16
// out_dense: [DENSE_ROWS * N]  (bf16)  (DENSE_ROWS is tail capacity)
// A_ell/B_ell: [M * ell_stride]
// A_dense/B_dense: [DENSE_ROWS * N]
// row_counts: [M] total nnz in original row
// tail_dense_map: [M] maps sparse row -> dense row id (or -1)
__global__ void sparse_elementwise_product(
    __nv_bfloat16*       __restrict__ out_ell,
    __nv_bfloat16*       __restrict__ out_dense,
    const __nv_bfloat16* __restrict__ A_ell,
    const __nv_bfloat16* __restrict__ A_dense,
    const __nv_bfloat16* __restrict__ B_ell,
    const __nv_bfloat16* __restrict__ B_dense,
    const int*           __restrict__ row_counts,
    const int*           __restrict__ tail_dense_map,
    const int*           __restrict__ b_tail_dense_map,
    int M,
    int N,
    const float* __restrict__ acc_init,
    int ell_stride,        // ELL buffer stride for this object (_ell_stride)
    int overflow_threshold // rows with nnz > this are in dense tail
) {
    constexpr int VEC = 8; // bf16 per int4
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    // one warp per sparse row
    const int row = (int)blockIdx.x * warps_per_block + warp;
    if (row >= M) return;

    int nnz_total = row_counts[row];
    if (nnz_total <= 0) return;

    float init = acc_init ? *acc_init : 0.0f;

    // --------------------------
    // Case 1: ELL-only row
    // --------------------------
    if (nnz_total <= overflow_threshold) {
        const __nv_bfloat16* a_row = A_ell   + (size_t)row * ell_stride;
        const __nv_bfloat16* b_row = B_ell   + (size_t)row * ell_stride;
        __nv_bfloat16*       o_row = out_ell + (size_t)row * ell_stride;

        // round nnz_total up to vec8 boundary (still <= overflow_threshold <= ell_stride)
        int nnz_vec = (nnz_total + (VEC - 1)) & ~(VEC - 1);
        if (nnz_vec > overflow_threshold) nnz_vec = overflow_threshold;

        // warp-stride in vec8: 32 lanes * 8 = 256 bf16
        for (int base = lane * VEC; base < nnz_vec; base += 32 * VEC) {
            const int4 a_raw = *reinterpret_cast<const int4*>(a_row + base);
            const int4 b_raw = *reinterpret_cast<const int4*>(b_row + base);

            int4 o_raw;
            const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a_raw);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b_raw);
            __nv_bfloat162* o2       = reinterpret_cast<__nv_bfloat162*>(&o_raw);
            __nv_bfloat162 init2      = __bfloat162bfloat162(__float2bfloat16(init));

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                o2[t] = a2[t] * b2[t] + init2;
            }

            *reinterpret_cast<int4*>(o_row + base) = o_raw;
        }

        // If you truly want to overwrite all ELL_WIDTH (even unused slots), swap nnz_vec -> ELL_WIDTH.
        return;
    }

    // --------------------------
    // Case 2: Dense-tail row
    // --------------------------
    int dr = tail_dense_map[row];
    if (dr < 0) return; // no mapped dense row, nothing to do
    const __nv_bfloat16* a_row = A_dense   + (size_t)dr * N;
    const __nv_bfloat16* b_row = B_dense   + (size_t)dr * N;
    __nv_bfloat16*       o_row = out_dense + (size_t)dr * N;

    // N is multiple of 8, so vec8 loop is exact
    for (int base = lane * VEC; base < N; base += 32 * VEC) {
        const int4 a_raw = *reinterpret_cast<const int4*>(a_row + base);
        const int4 b_raw = *reinterpret_cast<const int4*>(b_row + base);

        // SIMD zero-skip
        if (int4_all_zero(a_raw) || int4_all_zero(b_raw)) {
            *reinterpret_cast<int4*>(o_row + base) = make_int4(0, 0, 0, 0);
            continue;
        }

        int4 o_raw;
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a_raw);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b_raw);
        __nv_bfloat162* o2       = reinterpret_cast<__nv_bfloat162*>(&o_raw);
        __nv_bfloat162 init2      = __bfloat162bfloat162(__float2bfloat16(init));

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            o2[t] = a2[t] * b2[t] + init2;
        }

        *reinterpret_cast<int4*>(o_row + base) = o_raw;
    }
}

// --------------------------------------------------------------------------
// Fused dU ELL kernel: out[i] = bf16(float(A[i]) * float(B[i]) + init)
// Processes ALL ell_stride elements per row (full tensor operation).
// --------------------------------------------------------------------------
__global__ void fused_dU_ell_kernel(
    __nv_bfloat16*       __restrict__ out,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const float* __restrict__ acc_init_ptr,
    int total_elements  // M * ell_stride
) {
    constexpr int VEC = 8;
    float init_val = acc_init_ptr ? *acc_init_ptr : 0.0f;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (idx >= total_elements) return;

    const int4 a_raw = *reinterpret_cast<const int4*>(A + idx);
    const int4 b_raw = *reinterpret_cast<const int4*>(B + idx);

    int4 o_raw;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a_raw);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b_raw);
    __nv_bfloat162*       o2 = reinterpret_cast<__nv_bfloat162*>(&o_raw);

    // Match PyTorch semantics: bf16 multiply, then promote to f32 for add, then cast back
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        __nv_bfloat162 prod = a2[t] * b2[t];  // bf16 multiply
        float2 pf = __bfloat1622float2(prod);  // promote to f32
        o2[t] = __floats2bfloat162_rn(pf.x + init_val, pf.y + init_val);
    }

    *reinterpret_cast<int4*>(out + idx) = o_raw;
}

// --------------------------------------------------------------------------
// Fused dU dense-tail kernel: out = bf16((float(A) * float(B) + init) * (mask > 0))
// Processes dense_active_rows rows of tail_dense.
// --------------------------------------------------------------------------
__global__ void fused_dU_dense_kernel(
    __nv_bfloat16*       __restrict__ out,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ mask,
    const float* __restrict__ acc_init_ptr,
    int total_elements  // dense_active_rows * N
) {
    constexpr int VEC = 8;
    float init_val = acc_init_ptr ? *acc_init_ptr : 0.0f;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (idx >= total_elements) return;

    const int4 a_raw = *reinterpret_cast<const int4*>(A + idx);
    const int4 b_raw = *reinterpret_cast<const int4*>(B + idx);
    const int4 m_raw = *reinterpret_cast<const int4*>(mask + idx);

    if (int4_all_zero(a_raw) || int4_all_zero(b_raw) || int4_all_zero(m_raw)) {
        *reinterpret_cast<int4*>(out + idx) = make_int4(0, 0, 0, 0);
        return;
    }

    int4 o_raw;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a_raw);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b_raw);
    const uint16_t*       mu = reinterpret_cast<const uint16_t*>(&m_raw);
    __nv_bfloat162*       o2 = reinterpret_cast<__nv_bfloat162*>(&o_raw);

    // Match PyTorch: bf16 multiply, promote to f32, add init, apply mask, cast back
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        float2 pf = __bfloat1622float2(a2[t] * b2[t]);  // bf16 mul → f32
        float r0 = pf.x + init_val;
        float r1 = pf.y + init_val;
        if (mu[2*t]   == 0 || (mu[2*t]   & 0x8000u)) r0 = 0.0f;
        if (mu[2*t+1] == 0 || (mu[2*t+1] & 0x8000u)) r1 = 0.0f;
        o2[t] = __floats2bfloat162_rn(r0, r1);
    }

    *reinterpret_cast<int4*>(out + idx) = o_raw;
}

void compute_dU(hybrid_sp_t* dU, hybrid_sp_t* dT, hybrid_sp_t* R, hybrid_sp_t* P,
                const at::Tensor& acc_init, int M, int N, cudaStream_t stream) {
    PERF_START("compute_dU", stream);

    // Break aliasing (dU shares storage with T via copy constructor)
    dU->_ell_values = torch::empty_like(dU->_ell_values);
    dU->_ell_col_indices = P->_ell_col_indices;

    // ELL: out = bf16(float(dT) * float(R) + init), full tensor
    constexpr int VEC = 8;
    constexpr int THREADS = 256;
    {
        int total = dU->_ell_values.numel();
        int blocks = (total / VEC + THREADS - 1) / THREADS;
        fused_dU_ell_kernel<<<blocks, THREADS, 0, stream>>>(
            dU->ell_values(),
            dT->ell_values(),
            R->ell_values(),
            static_cast<const float*>(acc_init.data_ptr()),
            total
        );
    }

    // Dense tail: out = bf16((float(dT) * float(R) + init) * (P > 0))
    int dar = dU->_dense_active_rows;
    if (dar > 0) {
        dU->_tail_dense = torch::empty({dar, N},
            torch::TensorOptions().dtype(torch::kBFloat16).device(dU->_tail_dense.device()));
        int total = dar * N;
        int blocks = (total / VEC + THREADS - 1) / THREADS;
        fused_dU_dense_kernel<<<blocks, THREADS, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dU->_tail_dense.data_ptr()),
            dT->tail_dense(),
            R->tail_dense(),
            P->tail_dense(),
            static_cast<const float*>(acc_init.data_ptr()),
            total
        );
    }

    PERF_STOP("compute_dU");
}

void sparse_elementwise(hybrid_sp_t* out, hybrid_sp_t* A, hybrid_sp_t* B, int M, int N, cudaStream_t stream) {
    PERF_START("sparse_elementwise", stream);

    // Launch config: 4 warps/block (128 threads)
    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;

    dim3 block(THREADS);
    dim3 grid((int)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    // Remove aliasing of storage
    out->_ell_values = torch::empty_like(out->_ell_values);
    if (out->_dense_active_rows > 0) {
        out->_tail_dense = torch::empty_like(out->_tail_dense);
    }
    sparse_elementwise_product<<<grid, block, 0, stream>>>(
        out->ell_values(),
        out->tail_dense(),
        A->ell_values(),
        A->tail_dense(),
        B->ell_values(),
        B->tail_dense(),
        A->row_counters(),
        A->tail_dense_map(),
        B->tail_dense_map(),
        M,
        N,
        nullptr,
        A->_ell_stride,
        A->_ell_stride
    );
#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "Sparse-elemwise-product: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error on transpose");
    }
#endif

    PERF_STOP("sparse_elementwise");
}

void sparse_elementwise(hybrid_sp_t* out, hybrid_sp_t* A, hybrid_sp_t* B, int M, int N, at::Tensor& acc_init, cudaStream_t stream) {
    PERF_START("sparse_elementwise_with_acc", stream);

    // Launch config: 4 warps/block (128 threads)
    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;

    dim3 block(THREADS);
    dim3 grid((int)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    // Remove aliasing of storage
    out->_ell_values = torch::empty_like(out->_ell_values);
    if (out->_dense_active_rows > 0) {
        out->_tail_dense = torch::empty_like(out->_tail_dense);
    }
    sparse_elementwise_product<<<grid, block, 0, stream>>>(
        out->ell_values(),
        out->tail_dense(),
        A->ell_values(),
        A->tail_dense(),
        B->ell_values(),
        B->tail_dense(),
        A->row_counters(),
        A->tail_dense_map(),
        B->tail_dense_map(),
        M,
        N,
        reinterpret_cast<const float*>(acc_init.data_ptr()),
        A->_ell_stride,
        A->_ell_stride
    );
#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "Sparse-elemwise-product: Fatal error: (%s at %s:%d)\n", cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error on transpose");
    }
#endif

    PERF_STOP("sparse_elementwise_with_acc");
}
