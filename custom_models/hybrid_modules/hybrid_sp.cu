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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "hybrid_sp.h"
#include "constants.h"
#include "perf_instrumentation.h"


#define ERROR_CHECK 0

#define VEC_SIZE 8

// Atomic load for visibility under concurrent writes.
__forceinline__ __device__ int atomic_read_i32(const int* p) {
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
    // tail_dense_map state machine: -1 = unmapped, -2 = in-progress, -3 = capacity
    // exhausted (terminal), >=0 = dense row id.
    if (discard == 1) return;

    int dr = atomic_read_i32(&tail_dense_map[g_row]);
    if (dr >= 0 && dr < tail_cap) {
        tail_dense[(size_t)dr * (size_t)dense_ld + (size_t)g_col] = val;
        return;
    }
    if (dr >= tail_cap) return;

    if (dr == -1) {
        int old = atomicCAS(&tail_dense_map[g_row], -1, -2);
        if (old == -1) {
            int new_dr = atomicAdd(tail_dense_counter, 1);
            if (new_dr < tail_cap) {
                atomicExch(&tail_dense_map[g_row], new_dr);
                dr = new_dr;
                tail_dense_map_reverse[dr] = g_row;
            } else {
                atomicExch(&tail_dense_map[g_row], -3);
                return;
            }
            tail_dense[(size_t)dr * (size_t)dense_ld + (size_t)g_col] = val;
            return;
        } else {
            dr = old;
        }
    }

    if (dr == -2) {
        // Bounded spin while the CAS winner finishes its atomicAdd + atomicExch
        // (handful of instructions). If we don't see a publication after the
        // budget elapses, treat the result as if the cap was exhausted: silently
        // drop the write rather than spinning forever and tripping the kernel
        // watchdog -> cudaErrorLaunchFailure.
        int spin_budget = 4096;
        while (dr == -2 && spin_budget-- > 0) {
            dr = atomic_read_i32(&tail_dense_map[g_row]);
        }
        if (dr == -2 || dr < 0 || dr >= tail_cap) {
            return;
        }
    }

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
    const uint16_t*      __restrict__ ell_cols,
    const __nv_bfloat16* __restrict__ ell_vals,
    int*                 __restrict__ row_counters,
    int                  M_rows,
    __nv_bfloat16*       __restrict__ tail_dense,
    int                  dense_ld,
    const int*           __restrict__ tail_dense_map,
    int                  ell_w,
    int                  ell_stride
) {
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int nnz = row_counters[row];
    if (nnz <= ell_w) return;

    int dr = tail_dense_map[row];
    if (dr < 0) return;

    const uint16_t*      r_cols = ell_cols + (size_t)row * ell_stride;
    const __nv_bfloat16* r_vals = ell_vals + (size_t)row * ell_stride;

    // 8 entries per iteration (16B idx + 16B vals); ell_stride must be a multiple of 8.
    int k0 = (threadIdx.x * 8);
    int stride = (blockDim.x * 8);

    for (int k = k0; k < ell_stride; k += stride) {
        int4 idx_raw = *reinterpret_cast<const int4*>(r_cols + k);
        int4 val_raw = *reinterpret_cast<const int4*>(r_vals + k);

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


// Scatters tail-dense rows back into the full output matrix; one block per row.
__global__ void scatter_tail_dense_rows_into_full(
    __nv_bfloat16*       __restrict__ A,
    int                  M_rows,
    int                  N_cols,
    const __nv_bfloat16* __restrict__ B,
    int                  tail_rows,
    const int*           __restrict__ tail_dense_map
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
    const __nv_bfloat16* __restrict__ A,
    __nv_bfloat16*       __restrict__ C,
    const int*           __restrict__ map,
    int M,
    int R,
    int cols
) {
    int sr = blockIdx.x;
    if (sr >= M) return;

    int dr = map[sr];
    if (dr < 0 || dr >= R) return;

    const __nv_bfloat16* __restrict__ src = A + (size_t)sr * cols;
    __nv_bfloat16*       __restrict__ dst = C + (size_t)dr * cols;

    int vec_cols = cols & ~7;
    for (int j = threadIdx.x * 8; j < vec_cols; j += blockDim.x * 8) {
        const int4 v = *reinterpret_cast<const int4*>(src + j);
        *reinterpret_cast<int4*>(dst + j) = v;
    }
    for (int j = vec_cols + threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = src[j];
    }
}

union OutputPack {
    __nv_bfloat16 bf[8];
    int4 vec;
};

__global__ void ell_spmm_rowmajor_b_rowwise_optimized(
    const __nv_bfloat16* __restrict__ A_vals,
    const uint16_t* __restrict__ A_idxs,
    const int* __restrict__ row_counts,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M_rows,
    int K_rows,
    int N_cols,
    int ell_stride,
    int overflow_threshold
) {
    extern __shared__ char _smem[];
    __nv_bfloat16* sh_vals = reinterpret_cast<__nv_bfloat16*>(_smem);
    uint16_t*      sh_idxs = reinterpret_cast<uint16_t*>(_smem + ell_stride * sizeof(__nv_bfloat16));

    for (int row = blockIdx.x; row < M_rows; row += gridDim.x) {
    int nnz = row_counts[row];
    if (nnz <= 0 || nnz > overflow_threshold) continue;

    const __nv_bfloat16* A_row_vals_g = A_vals + (size_t)row * ell_stride;
    const uint16_t* A_row_idxs_g = A_idxs + (size_t)row * ell_stride;

    for (int k = threadIdx.x; k < nnz; k += blockDim.x) {
        sh_idxs[k] = A_row_idxs_g[k];
        sh_vals[k] = A_row_vals_g[k];
    }
    __syncthreads();

    for (int n_out = threadIdx.x * VEC_SIZE; n_out < N_cols; n_out += VEC_SIZE * blockDim.x) {

        float2 acc[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = make_float2(0.f, 0.f);

        for (int k = 0; k < nnz; ++k) {
            __nv_bfloat16 a_val = sh_vals[k];
            uint16_t    col_idx = sh_idxs[k];

            const __nv_bfloat16* B_row_ptr = B + (size_t)col_idx * N_cols + n_out;

            int4 b_vec_raw = *reinterpret_cast<const int4*>(B_row_ptr);
            __nv_bfloat16* b_vec = reinterpret_cast<__nv_bfloat16*>(&b_vec_raw);
            __nv_bfloat162* b_pairs = reinterpret_cast<__nv_bfloat162*>(b_vec);

            float a = __bfloat162float(a_val);

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
    }
}

#define WARPS_PER_BLOCK_WPR 8
#define WARP_SIZE_WPR 32

// Warp-per-row variant: skips rows already covered by the main kernel.
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
    if (row_counts[row] > 0) return;

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
// Persistent grid: each block work-steals rows via atomicAdd on work_counter
// to amortise wave-scheduling overhead at large N.
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

// L2 persistence policy helpers; silently skip on GPUs that don't support the API.
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
    // Don't take ownership of the source's CUDA event: only the original
    // (the one populated by the forward op) is responsible for destroying it.
    this->_counter_copy_ev = nullptr;
}

hybrid_sp_t::~hybrid_sp_t() {
    // Reclaim the CUDA event allocated by the forward op when this object is
    // released without backward ever running (eval/inference, autograd skip,
    // or any path where ff_backward_cuda_gated didn't fire). When backward
    // does run, it destroys the event itself and nulls the handle, so this
    // is a no-op for the normal training path.
    if (_counter_copy_ev != nullptr) {
        cudaEventDestroy(_counter_copy_ev);
        _counter_copy_ev = nullptr;
    }
}

void hybrid_sp_t::reset_vals() {
    auto options_bf16 = at::TensorOptions().dtype(torch::kBFloat16).device(this->_ell_values.device());
    this->_ell_values = at::empty({this->_ell_values.numel()}, options_bf16);
}

// Allocates a tail-dense slot for each overflow row and copies the dense row
// with implicit ReLU (sign-bit mask). One block per row.
__global__ void populate_overflow_tail_kernel(
    const __nv_bfloat16* __restrict__ dense,
    const int*           __restrict__ row_counts,
    int*                 __restrict__ overflow_counter,
    __nv_bfloat16*       __restrict__ tail_dense,
    int*                 __restrict__ tail_dense_map,
    int*                 __restrict__ tail_dense_map_reverse,
    int M_rows, int N_cols,
    int ell_width_threshold,
    int tail_cap,
    int discard)
{
    const int row = (int)blockIdx.x;
    if (row >= M_rows) return;
    if (row_counts[row] <= ell_width_threshold) return;

    __shared__ int smem_dr;
    if (threadIdx.x == 0) {
        int new_dr = atomicAdd(overflow_counter, 1);
        if (discard != 1 && new_dr < tail_cap) {
            tail_dense_map[row]              = new_dr;
            tail_dense_map_reverse[new_dr]   = row;
            smem_dr = new_dr;
        } else {
            smem_dr = -1;
        }
    }
    __syncthreads();
    const int dr = smem_dr;
    if (dr < 0) return;

    const __nv_bfloat16* src = dense     + (size_t)row * N_cols;
    __nv_bfloat16*       dst = tail_dense + (size_t)dr  * N_cols;

    for (int base = threadIdx.x * VEC_SIZE; base < N_cols; base += blockDim.x * VEC_SIZE) {
        int4 raw = *reinterpret_cast<const int4*>(src + base);
        uint16_t* u = reinterpret_cast<uint16_t*>(&raw);
        #pragma unroll
        for (int vi = 0; vi < VEC_SIZE; vi++) {
            if (u[vi] & 0x8000u) u[vi] = 0u;
        }
        *reinterpret_cast<int4*>(dst + base) = raw;
    }
}

template<int WARPS_PER_BLOCK>
__global__ void dense_to_ell_kernel(
    const __nv_bfloat16* __restrict__, uint16_t* __restrict__,
    __nv_bfloat16* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__, int, int, int, int);


// Per-operation ELL overflow thresholds — these also determine buffer allocation size.
// Buffer stride = max(regular, transpose) so both ops fit in the same hybrid_sp_t.
int g_ell_width_regular   = ELL_WIDTH;
int g_ell_width_transpose = 2 * ELL_WIDTH;
// Per-operation dense tail capacity — buffer size = max(regular, transpose).
int g_tail_rows_regular   = TAIL_CAPACITY_ROWS;
int g_tail_rows_transpose = TAIL_CAPACITY_ROWS;
// Discard overflow mode: 1 = drop overflow rows instead of writing to dense tail
int g_discard_overflow    = 0;

void set_ell_width_regular(int v)   { g_ell_width_regular   = (v > 0) ? v : ELL_WIDTH; }
void set_ell_width_transpose(int v) { g_ell_width_transpose = (v > 0) ? v : 2 * ELL_WIDTH; }
void set_tail_rows_regular(int v)   { g_tail_rows_regular   = (v > 0) ? v : TAIL_CAPACITY_ROWS; }
void set_tail_rows_transpose(int v) { g_tail_rows_transpose = (v > 0) ? v : TAIL_CAPACITY_ROWS; }
void set_discard_overflow(int v)    { g_discard_overflow    = v; }

// Builds the gate ELL directly from a dense [M,N] bf16 activation matrix and
// spills overflow rows into the dense tail with implicit ReLU.
void create_hybrid_sparse_from_dense(
    const at::Tensor& dense,
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
    constexpr int WPB = 4;
    size_t d2e_smem = (size_t)WPB * ell_stride * (sizeof(uint16_t) + sizeof(__nv_bfloat16));
    dense_to_ell_kernel<WPB><<<(M + WPB - 1) / WPB, WPB * 32, d2e_smem, stream>>>(
        d_ptr, e_cols, e_vals, e_rcnt, l0_ptr, l1_ptr, M, N, ell_stride, ell_stride);
    PERF_STOP("create_hybrid_sparse_from_dense:dense_to_ell");

#if ERROR_CHECK
    cudaError_t __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "dense_to_ell: Fatal error: (%s at %s:%d)\n",
                cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error in dense_to_ell_kernel");
    }
#endif

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
#if ERROR_CHECK
    __err = cudaDeviceSynchronize();
    if (__err != cudaSuccess) {
        fprintf(stderr, "dense_to_ell: Fatal error: (%s at %s:%d)\n",
                cudaGetErrorString(__err), __FILE__, __LINE__);
        TORCH_CHECK(false, "error in populate_overflow_tail");
    }
#endif
    PERF_STOP("create_hybrid_sparse_from_dense:overflow_tail");

    PERF_STOP("create_hybrid_sparse_from_dense:total");
}


// One block per row: caches A[row] in smem, then each warp computes one ELL
// dot product against the corresponding column of B (rows of B_T).
__global__ void dense_ab_as_sparse_hybrid_unified_optimized(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B_T,
    const uint16_t* __restrict__ ell_cols,
    const int* __restrict__ row_counts,
    __nv_bfloat16* __restrict__ C_ell_vals,
    const float* __restrict__ init_val,
    int M_rows,
    int K,
    int N_cols,
    int ell_stride,
    int overflow_threshold
) {
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int nnz_total = row_counts[row];
    if (nnz_total <= 0) return;
    if (nnz_total > overflow_threshold) return;

    extern __shared__ __nv_bfloat16 sh_A[];

    const __nv_bfloat16* A_row_ptr = A + (size_t)row * K;

    int num_vec_k = K / 8;
    int tail_k    = K % 8;

    int4* sh_A_vec = reinterpret_cast<int4*>(sh_A);
    const int4* A_row_vec = reinterpret_cast<const int4*>(A_row_ptr);

    for (int i = threadIdx.x; i < num_vec_k; i += blockDim.x) {
        sh_A_vec[i] = A_row_vec[i];
    }

    if (tail_k > 0) {
        int tail_start = num_vec_k * 8;
        for (int i = tail_start + threadIdx.x; i < K; i += blockDim.x) {
            sh_A[i] = A_row_ptr[i];
        }
    }

    __syncthreads();

    const int lane_id   = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    float base_init = (init_val) ? *init_val : 0.0f;

    for (int out_idx = warp_id; out_idx < nnz_total; out_idx += num_warps) {
        int col = (int)ell_cols[(size_t)row * ell_stride + out_idx];
        if (col < 0 || col >= N_cols) continue;

        const __nv_bfloat16* B_row_ptr = B_T + (size_t)col * K;

        float acc = base_init;

        int k_vec_idx = lane_id;
        while (k_vec_idx < num_vec_k) {
            int4 a_raw = *reinterpret_cast<const int4*>(&sh_A[k_vec_idx * 8]);
            int4 b_raw = *reinterpret_cast<const int4*>(&B_row_ptr[k_vec_idx * 8]);

            __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&a_raw);
            __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(&b_raw);

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                float2 af = __bfloat1622float2(a2[t]);
                float2 bf = __bfloat1622float2(b2[t]);
                acc = fmaf(af.x, bf.x, acc);
                acc = fmaf(af.y, bf.y, acc);
            }

            k_vec_idx += 32;
        }

        if (tail_k > 0) {
            int k_idx = num_vec_k * 8 + lane_id;
            if (k_idx < K) {
                float a = __bfloat162float(sh_A[k_idx]);
                float b = __bfloat162float(B_row_ptr[k_idx]);
                acc = fmaf(a, b, acc);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }

        if (lane_id == 0) {
            C_ell_vals[(size_t)row * ell_stride + out_idx] = __float2bfloat16(acc);
        }
    }
}

__device__ __forceinline__ bool bf16_nonzero_mask(__nv_bfloat16 v) {
    return (__bfloat16_as_ushort(v) & 0x7FFF) != 0;
}

__global__ void tail_dense_masked_add_inplace(
    __nv_bfloat16*       __restrict__ out_tail,
    const __nv_bfloat16* __restrict__ tail,
    const __nv_bfloat16* __restrict__ dense_out,
    int rows,
    int N,
    const float*         __restrict__ init_val
) {
    const int64_t total_vec = (int64_t)rows * (int64_t)N / 8;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_vec) return;

    float init = init_val ? *init_val : 0.0f;
    int64_t elem0 = tid * 8;

    int4 out_raw  = *reinterpret_cast<const int4*>(out_tail  + elem0);
    int4 tail_raw = *reinterpret_cast<const int4*>(tail      + elem0);
    int4 den_raw  = *reinterpret_cast<const int4*>(dense_out + elem0);

    __nv_bfloat16* out8  = reinterpret_cast<__nv_bfloat16*>(&out_raw);
    __nv_bfloat16* tail8 = reinterpret_cast<__nv_bfloat16*>(&tail_raw);
    __nv_bfloat16* den8  = reinterpret_cast<__nv_bfloat16*>(&den_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (bf16_nonzero_mask(tail8[i])) {
            float d = __bfloat162float(den8[i]);
            out8[i] = __float2bfloat16(init + d);
        } else {
            // Zero else-branch: callers rely on this to clear stale tail values.
            out8[i] = __float2bfloat16(0.0f);
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

    const float* init_ptr = init_val.data_ptr<float>();

    dim3 block(256);
    dim3 grid(M);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);

    // B_T is reused across all rows; pin it in L2 if it fits.
    size_t BT_size = (size_t)N * K * sizeof(__nv_bfloat16);
    if (BT_size < 4 * 1024 * 1024) {
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
        auto options_bf16 = at::TensorOptions().dtype(torch::kBFloat16).device(a.device());
        const int tcap = out->_tail_cap;
        at::Tensor dense = torch::empty({tcap, K}, options_bf16);
        __nv_bfloat16* dense_ptr = reinterpret_cast<__nv_bfloat16*>(dense.data_ptr());

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

        at::Tensor dense_active = dense.narrow(0, 0, out->_dense_active_rows);
        at::Tensor dense_out = torch::matmul(dense_active, b.transpose(0, 1));

        PERF_STOP("new_product_as_sparse:dense_matmul");

        TORCH_CHECK(out->_tail_dense.is_contiguous(), "out_tail_dense must be contiguous");
        TORCH_CHECK(dense_out.is_contiguous(), "dense_out must be contiguous");

        const int vecs_per_row = N / 8;
        const int64_t total_vec = (int64_t)out->_dense_active_rows * (int64_t)vecs_per_row;
        const int threads = 256;
        const int blocks  = (int)((total_vec + threads - 1) / threads);
        // Clone before modifying: _tail_dense may be aliased across P/R/T.
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


// Treats sign-bit zero as "positive". +NaN passes; callers don't produce NaNs.
__device__ __forceinline__ bool bf16_is_pos_nonzero(uint16_t u)
{
    return ((u & 0x8000u) == 0u) && ((u & 0x7fffu) != 0u);
}


// One warp per row, ballot-based prefix scan to fill the ELL buffer with
// the first overflow_threshold positive elements. row_counts gets the true
// NNZ so the overflow tail can promote rows whose count exceeds the buffer.
template<int WARPS_PER_BLOCK>
__global__ void dense_to_ell_kernel(
    const __nv_bfloat16* __restrict__ dense,
    uint16_t*            __restrict__ ell_cols,
    __nv_bfloat16*       __restrict__ ell_vals,
    int*                 __restrict__ row_counts,
    float*               __restrict__ l0_out,
    float*               __restrict__ l1_out,
    int M_rows, int N_cols,
    int ell_stride,
    int overflow_threshold
)
{
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int row     = (int)blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Layout: [WPB * ell_stride * u16] | [WPB * ell_stride * bf16].
    extern __shared__ char _smem_d2e[];
    uint16_t*      smem_cols = reinterpret_cast<uint16_t*>(_smem_d2e);
    __nv_bfloat16* smem_vals = reinterpret_cast<__nv_bfloat16*>(
        _smem_d2e + WARPS_PER_BLOCK * ell_stride * sizeof(uint16_t));

    if (row >= M_rows) return;

    const __nv_bfloat16* row_ptr = dense + (size_t)row * N_cols;
    int   slot_cnt = 0;
    float l0_acc   = 0.0f;
    float l1_acc   = 0.0f;

    for (int base_col = 0; base_col < N_cols; base_col += 32 * VEC_SIZE) {
        const int lane_base = base_col + lane * VEC_SIZE;
        const bool in_bounds = (lane_base + VEC_SIZE - 1) < N_cols;

        __nv_bfloat16 elems[VEC_SIZE];
        if (in_bounds) {
            *reinterpret_cast<int4*>(elems) =
                *reinterpret_cast<const int4*>(row_ptr + lane_base);
        }

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
            }
            slot_cnt += count;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        l0_acc += __shfl_down_sync(0xFFFFFFFF, l0_acc, offset);
        l1_acc += __shfl_down_sync(0xFFFFFFFF, l1_acc, offset);
    }

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

__global__ void transpose_hybrid_ell_dense(
    // Input A (hybrid: ELL + overflow-as-dense-rows)
    const uint16_t*      __restrict__ A_ell_cols,
    const __nv_bfloat16* __restrict__ A_ell_vals,
    const int*           __restrict__ A_row_counts,
    const __nv_bfloat16* __restrict__ A_tail_dense,
    const int*           __restrict__ A_tail_dense_map_reverse,
    int*                 __restrict__ A_tail_dense_rows,
    int                  A_dense_ld,
    int M_rows,
    int N_cols,

    uint16_t*            __restrict__ AT_ell_cols,
    __nv_bfloat16*       __restrict__ AT_ell_vals,
    int*                 __restrict__ AT_row_counts,
    int*                 __restrict__ AT_overflow_counter,
    __nv_bfloat16*       __restrict__ tail_dense,
    int                  dense_ld,
    int*                 __restrict__ tail_dense_map,
    int*                 __restrict__ tail_dense_map_reverse,

    const int*           __restrict__ precomputed_tail_dense_map,
    bool                 use_precomputed_maps,
    int                  ell_w,
    int                  in_ell_stride,
    int                  out_ell_stride,
    int                  tail_cap,
    int                  discard
) {
    // Phase 1: transpose ELL entries of non-overflow rows. Rows whose true NNZ
    // exceeded the input's ELL stride were materialised into A_tail_dense in
    // their entirety (Phase 2 handles them); their ELL data here is a truncated
    // copy and reading past in_ell_stride would be out-of-bounds.
    for (int row = blockIdx.x; row < M_rows; row += gridDim.x) {
        int nnz_row = A_row_counts[row];
        if (nnz_row <= 0) continue;
        if (nnz_row > in_ell_stride) continue;

        const int ell_n = nnz_row;

        for (int k = threadIdx.x; k < ell_n; k += blockDim.x) {
            uint16_t col = A_ell_cols[(size_t)row * in_ell_stride + k];
            __nv_bfloat16 val = A_ell_vals[(size_t)row * in_ell_stride + k];

            const int out_row = (int)col;
            const int out_col = row;
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
    // Phase 2: transpose the materialised dense tail by scanning each row.
    // Iterate over A's column count (A_dense_ld); dense_ld is AT's transposed width.
    int a_dense_rows = *A_tail_dense_rows;
    for (int dense_row = blockIdx.x; dense_row < a_dense_rows; dense_row += gridDim.x) {
        const int src_row = A_tail_dense_map_reverse[dense_row];
        if (src_row < 0) continue;
        if ((unsigned)src_row >= (unsigned)M_rows) continue;

        const __nv_bfloat16* __restrict__ src = A_tail_dense + (size_t)dense_row * A_dense_ld;

        for (int col0 = threadIdx.x * 8; col0 < A_dense_ld; col0 += blockDim.x * 8) {

            if (col0 + 7 < A_dense_ld) {
                int4 raw = *reinterpret_cast<const int4*>(src + col0);

                // bf16 zero is 0x0000, so an all-zero int4 means all 8 lanes are zero.
                if ((raw.x | raw.y | raw.z | raw.w) == 0) {
                    continue;
                }

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
    dim3 grid(min(M_rows, num_sms*4));

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
        AT._ell_stride,
        A._ell_stride,
        AT._ell_stride,
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
    // Spill ELL entries past AT._ell_stride into the dense tail.
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
            AT._ell_stride,
            AT._ell_stride
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


#if ERROR_CHECK
static int op_id = 0;
#endif

void sparse_dense_gemm_hybrid_dense(at::Tensor& out, hybrid_sp_t* A, const at::Tensor& B, int M, int N, int K, cudaStream_t stream) {
    PERF_START("sparse_dense_gemm_total", stream);

    // hitRatio = min(1, L2/B) keeps as much of B resident as the cache allows
    // (~0.89 on H100 for the 56 MB down-projection weight).
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
        at::Tensor A_tail_bf16 = A->_tail_dense.narrow(0, 0, A->_dense_active_rows);
        at::Tensor B_bf16 = B.reshape({K, N});
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

    // Empty rows aren't visited by any of the spmm variants, so zero them up front.
    {
        int zero_grid = ((int)M + WARPS_PER_BLOCK_WPR - 1) / WARPS_PER_BLOCK_WPR;
        zero_empty_rows<<<zero_grid, WARPS_PER_BLOCK_WPR * WARP_SIZE_WPR, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(row_counts.data_ptr()),
            (int)M, (int)N
        );
    }

    // Dispatch heuristic (ORIG_COLS_PER_PASS = 256 * VEC_SIZE = 2048):
    //   N % 2048 != 0  or N < 2048 → warp-per-row
    //   N == 2048                  → rowmajor-optimized (1 pass, all threads busy)
    //   N % 2048 == 0 and > 2048   → persistent (work-steals across SMs)
    const int ORIG_COLS_PER_PASS = 256 * VEC_SIZE;
    const bool n_is_exact_multiple = (N % ORIG_COLS_PER_PASS) == 0;
    if (!n_is_exact_multiple || N < ORIG_COLS_PER_PASS) {
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

__forceinline__ __device__ bool int4_all_zero(const int4& v) {
    return (v.x | v.y | v.z | v.w) == 0;
}

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
    int ell_stride,
    int overflow_threshold
) {
    constexpr int VEC = 8;
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int row = (int)blockIdx.x * warps_per_block + warp;
    if (row >= M) return;

    int nnz_total = row_counts[row];
    if (nnz_total <= 0) return;

    float init = acc_init ? *acc_init : 0.0f;

    if (nnz_total <= overflow_threshold) {
        // ELL-only row.
        const __nv_bfloat16* a_row = A_ell   + (size_t)row * ell_stride;
        const __nv_bfloat16* b_row = B_ell   + (size_t)row * ell_stride;
        __nv_bfloat16*       o_row = out_ell + (size_t)row * ell_stride;

        int nnz_vec = (nnz_total + (VEC - 1)) & ~(VEC - 1);
        if (nnz_vec > overflow_threshold) nnz_vec = overflow_threshold;

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
        return;
    }

    // Dense-tail row.
    int dr = tail_dense_map[row];
    if (dr < 0) return;
    const __nv_bfloat16* a_row = A_dense   + (size_t)dr * N;
    const __nv_bfloat16* b_row = B_dense   + (size_t)dr * N;
    __nv_bfloat16*       o_row = out_dense + (size_t)dr * N;

    for (int base = lane * VEC; base < N; base += 32 * VEC) {
        const int4 a_raw = *reinterpret_cast<const int4*>(a_row + base);
        const int4 b_raw = *reinterpret_cast<const int4*>(b_row + base);

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

// out[i] = bf16(float(A[i]) * float(B[i]) + init), full ELL row.
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

    // Match PyTorch: bf16 multiply, promote to f32, add init, cast back.
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        __nv_bfloat162 prod = a2[t] * b2[t];
        float2 pf = __bfloat1622float2(prod);
        o2[t] = __floats2bfloat162_rn(pf.x + init_val, pf.y + init_val);
    }

    *reinterpret_cast<int4*>(out + idx) = o_raw;
}

// out = bf16((float(A) * float(B) + init) * (mask > 0)), one tail row per group.
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

    // Match PyTorch: bf16 mul, promote to f32, add init, apply mask, cast back.
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        float2 pf = __bfloat1622float2(a2[t] * b2[t]);
        float r0 = pf.x + init_val;
        float r1 = pf.y + init_val;
        if (mu[2*t]   == 0 || (mu[2*t]   & 0x8000u)) r0 = 0.0f;
        if (mu[2*t+1] == 0 || (mu[2*t+1] & 0x8000u)) r1 = 0.0f;
        o2[t] = __floats2bfloat162_rn(r0, r1);
    }

    *reinterpret_cast<int4*>(out + idx) = o_raw;
}

void compute_dU(hybrid_sp_t* dU, hybrid_sp_t* dT, hybrid_sp_t* R, hybrid_sp_t* P,
                const at::Tensor& acc_init, int N, cudaStream_t stream) {
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

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;

    dim3 block(THREADS);
    dim3 grid((int)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    // Break aliasing: out shares storage with P/R/T via the copy constructor.
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

