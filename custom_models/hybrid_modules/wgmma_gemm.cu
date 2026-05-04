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

#include "wgmma_gemm.h"
#include "hybrid_sp.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "perf_instrumentation.h"

// packed.h pulls in Hopper-only declarations and is only safe to include
// when device codegen targets sm_90a or when host-only translation runs.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#include "packed.h"
#endif

// Converts the packed blocked-ELL workspace produced by the WGMMA kernel into
// the canonical hybrid ELL. Per-(row, tile) layout in C_packed:
//   [tile * T_n_comp + 0]            = NNZ count for this tile
//   [tile * T_n_comp + 1 .. cnt]     = (global_col_u16 | bf16_val << 16)
__global__ void blocked_ell_to_ell_packed_kernel(
    const uint32_t*      __restrict__ C_packed,
    __nv_bfloat16*       __restrict__ ell_val,
    int16_t*             __restrict__ ell_col,
    int32_t*             __restrict__ row_nnz,
    float*               __restrict__ l0_out,
    float*               __restrict__ l1_out,
    int M, int N_TILES, int T_n_comp, int ELL_W)
{
    const int row = (int)(blockIdx.x * blockDim.y + threadIdx.y);
    if (row >= M) return;
    const int tid = (int)threadIdx.x;

    const uint32_t* tile_ptr = (tid < N_TILES)
        ? C_packed + (size_t)row * N_TILES * T_n_comp + (size_t)tid * T_n_comp
        : nullptr;

    int cnt = (tid < N_TILES) ? (int)min((uint32_t)tile_ptr[0], (uint32_t)(T_n_comp - 1)) : 0;

    int offset = cnt;
    for (int delta = 1; delta < 32; delta <<= 1) {
        int recv = __shfl_up_sync(0xFFFFFFFF, offset, delta);
        if (tid >= (unsigned)delta) offset += recv;
    }
    int start = offset - cnt;
    // total is the true NNZ; do NOT cap it so overflow rows can be detected.
    int total = __shfl_sync(0xFFFFFFFF, offset, min(N_TILES - 1, 31));

    float l0_acc = 0.0f, l1_acc = 0.0f;
    if (l0_out && cnt > 0) {
        const float inv_M = 1.0f / (float)M;
        l0_acc = (float)cnt * inv_M;
        for (int i = 0; i < cnt; i++) {
            __nv_bfloat16 v = __ushort_as_bfloat16((unsigned short)(tile_ptr[i + 1] >> 16));
            l1_acc += __bfloat162float(v) * inv_M;
        }
    }

    if (cnt > 0 && start < ELL_W) {
        int copy_n = min(cnt, ELL_W - start);
        __nv_bfloat16* dv = ell_val + (size_t)row * ELL_W + start;
        int16_t*       dc = ell_col + (size_t)row * ELL_W + start;
        for (int i = 0; i < copy_n; i++) {
            uint32_t packed = tile_ptr[i + 1];
            dv[i] = __ushort_as_bfloat16((unsigned short)(packed >> 16));
            dc[i] = (int16_t)(packed & 0xFFFFu);
        }
    }
    if (tid == 0) row_nnz[row] = total;

    if (l0_out) {
        for (int s = 16; s > 0; s >>= 1) {
            l0_acc += __shfl_down_sync(0xFFFFFFFF, l0_acc, s);
            l1_acc += __shfl_down_sync(0xFFFFFFFF, l1_acc, s);
        }
        if (tid == 0) {
            atomicAdd(l0_out, l0_acc);
            atomicAdd(l1_out, l1_acc);
        }
    }
}

// One block per row; non-overflow rows exit immediately.
__global__ void overflow_tail_from_packed_kernel(
    const uint32_t*       __restrict__ C_packed,
    const int32_t*        __restrict__ row_nnz,
    int32_t*              __restrict__ overflow_counter,
    __nv_bfloat16*        __restrict__ tail_dense,
    int32_t*              __restrict__ tail_dense_map,
    int32_t*              __restrict__ tail_dense_map_reverse,
    int M, int N, int N_TILES, int T_n_comp, int ELL_W,
    int tail_cap, int discard)
{
    const int row = blockIdx.x;
    if (row >= M || row_nnz[row] <= ELL_W) return;

    __shared__ int smem_dr;
    if (threadIdx.x == 0) {
        int new_dr = atomicAdd(overflow_counter, 1);
        if (discard != 1 && new_dr < tail_cap) {
            tail_dense_map[row]            = new_dr;
            tail_dense_map_reverse[new_dr] = row;
            smem_dr = new_dr;
        } else {
            smem_dr = -1;
        }
    }
    __syncthreads();
    const int dr = smem_dr;
    if (dr < 0) return;

    __nv_bfloat16* dst = tail_dense + (size_t)dr * N;

    // N must be a multiple of 8 bf16 (16 bytes = int4).
    for (int base = (int)threadIdx.x * 8; base < N; base += (int)blockDim.x * 8)
        *reinterpret_cast<int4*>(dst + base) = make_int4(0, 0, 0, 0);
    __syncthreads();

    // Tiles cover disjoint column ranges, so no race when scattering.
    for (int tile = (int)threadIdx.x; tile < N_TILES; tile += (int)blockDim.x) {
        const uint32_t* tile_ptr = C_packed + (size_t)row * N_TILES * T_n_comp + (size_t)tile * T_n_comp;
        int cnt = (int)min((uint32_t)tile_ptr[0], (uint32_t)(T_n_comp - 1));
        for (int i = 0; i < cnt; i++) {
            uint32_t packed = tile_ptr[i + 1];
            int col = (int)(packed & 0xFFFFu);
            if ((unsigned)col < (unsigned)N)
                dst[col] = __ushort_as_bfloat16((unsigned short)(packed >> 16));
        }
    }
}


#ifndef __CUDA_ARCH__

void populate_overflow_tail_from_packed(
    const uint32_t*      C_packed,
    const int32_t*       row_nnz,
    int32_t*             overflow_counter,
    __nv_bfloat16*       tail_dense,
    int32_t*             tail_dense_map,
    int32_t*             tail_dense_map_reverse,
    int M, int N, int N_TILES, int T_n_comp,
    cudaStream_t stream,
    int ell_w, int tail_cap, int discard)
{
    overflow_tail_from_packed_kernel<<<M, 128, 0, stream>>>(
        C_packed, row_nnz,
        overflow_counter, tail_dense, tail_dense_map, tail_dense_map_reverse,
        M, N, N_TILES, T_n_comp, ell_w,
        tail_cap, discard);
}

bool wgmma_gate_gemm_to_ell_packed(
    const at::Tensor& X_flat,
    const at::Tensor& G,
    int M, int N, int K,
    uint16_t*        ell_col,
    __nv_bfloat16*   ell_val,
    int32_t*         row_nnz,
    uint32_t*        C_packed,
    float*           l0_out,
    float*           l1_out,
    cudaStream_t stream)
{
    // __CUDA_ARCH_LIST__ is auto-set by nvcc (CUDA 12+) on both host and device passes.
    // When the build does not target sm_90a+, matmul_d2t.cu is excluded from sources,
    // so calling mm_wgmma_nt_128x256x64TS8 would be a link error. Skip the WGMMA path
    // and let custom_op.cpp take the einsum + create_hybrid_sparse_from_dense fallback.
#if !defined(__CUDA_ARCH_LIST__) || (__CUDA_ARCH_LIST__) < 900
    return false;
#else
    int dev = X_flat.get_device();
    int major;
    cudaError_t err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
    if (major < 9) return false;
    if (M % 256 != 0 || K % 64 != 0 || N % 256 != 0) return false;

    constexpr int T_n      = 256;
    constexpr int T_n_comp = 32;   // T_n / 8
    const int N_TILES = N / T_n;

    at::Tensor X_c = X_flat.contiguous();
    at::Tensor G_c = G.contiguous();

    PERF_START("mmwgmma", stream);
    TWELL_D2T::mm_wgmma_nt_128x256x64TS8(
        reinterpret_cast<at::BFloat16*>(X_c.data_ptr()),
        reinterpret_cast<at::BFloat16*>(G_c.data_ptr()),
        C_packed, M, K, N, stream);
    PERF_STOP("mmwgmma");

    if ((l0_out == nullptr) != (l1_out == nullptr) || (l0_out != nullptr && l0_out == l1_out)) {
        TORCH_CHECK(false, "l0_out and l1_out must be distinct pointers or both null");
    }
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 conv_block(32, ROWS_PER_BLOCK);
    dim3 conv_grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    PERF_START("bell_to_ell", stream);
    blocked_ell_to_ell_packed_kernel<<<conv_grid, conv_block, 0, stream>>>(
        C_packed,
        ell_val, reinterpret_cast<int16_t*>(ell_col), row_nnz,
        l0_out, l1_out,
        M, N_TILES, T_n_comp, g_ell_width_regular);
    PERF_STOP("bell_to_ell");

    return true;
#endif  // __CUDA_ARCH_LIST__ < 900
}

#endif  // !__CUDA_ARCH__
