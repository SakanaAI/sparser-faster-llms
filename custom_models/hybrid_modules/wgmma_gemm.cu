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
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include "perf_instrumentation.h"

// Compile Hopper-specific device code only for sm_90a (and the host pass so
// the function declarations are visible for the kernel launch).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

#include "packed.h"

#endif  // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900


// blocked_ell_to_ell_packed_kernel: convert packed blocked ELL → regular ELL.
// packed format: uint32_t C_packed[M][N_TILES * T_n_comp], layout per (row, tile):
//   [tile*T_n_comp + 0]           = NNZ count
//   [tile*T_n_comp + 1..count]    = packed (global_col_16bit | bf16_val << 16)
__global__ void blocked_ell_to_ell_packed_kernel(
    const uint32_t*      __restrict__ C_packed, // [M, N_TILES*T_n_comp] uint32
    __nv_bfloat16*       __restrict__ ell_val,
    int16_t*             __restrict__ ell_col,
    int32_t*             __restrict__ row_nnz,
    float*               __restrict__ l0_out,   // nullable scalar accumulator
    float*               __restrict__ l1_out,   // nullable scalar accumulator
    int M, int N_TILES, int T_n_comp, int ELL_W)
{
    const int row = (int)(blockIdx.x * blockDim.y + threadIdx.y);
    if (row >= M) return;
    const int tid = (int)threadIdx.x;  // 0..31

    const uint32_t* tile_ptr = (tid < N_TILES)
        ? C_packed + (size_t)row * N_TILES * T_n_comp + (size_t)tid * T_n_comp
        : nullptr;

    // Count is stored at index 0; cap at T_n_comp-1 (max data slots = T_n_comp-1)
    int cnt = (tid < N_TILES) ? (int)min((uint32_t)tile_ptr[0], (uint32_t)(T_n_comp - 1)) : 0;

    // Inclusive warp prefix scan to get write offsets into ELL
    int offset = cnt;
    for (int delta = 1; delta < 32; delta <<= 1) {
        int recv = __shfl_up_sync(0xFFFFFFFF, offset, delta);
        if (tid >= (unsigned)delta) offset += recv;
    }
    int start = offset - cnt;
    int total = __shfl_sync(0xFFFFFFFF, offset, min(N_TILES - 1, 31));
    // Do NOT cap total: store true NNZ so overflow rows are detectable downstream.

    // l0/l1: each active element contributes 1/M and val/M respectively
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
            dc[i] = (int16_t)(packed & 0xFFFFu);  // already global col
        }
    }
    if (tid == 0) row_nnz[row] = total;

    // Warp-reduce l0/l1 and atomicAdd from lane 0
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


// Populate overflow dense tail for rows whose true NNZ exceeds ELL_WIDTH.
// One block per row; only overflow rows do any work.
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
        int new_dr = atomicAdd(overflow_counter, 1);  // always count
        if (discard != 1 && new_dr < tail_cap) {
            tail_dense_map[row]            = new_dr;
            tail_dense_map_reverse[new_dr] = row;
            smem_dr = new_dr;
        } else {
            smem_dr = -1;  // no mapping; tail_dense_map stays -1
        }
    }
    __syncthreads();
    const int dr = smem_dr;
    if (dr < 0) return;

    __nv_bfloat16* dst = tail_dense + (size_t)dr * N;

    // Vectorized zero-out (N must be a multiple of 8 bf16 = 16 bytes = int4)
    for (int base = (int)threadIdx.x * 8; base < N; base += (int)blockDim.x * 8)
        *reinterpret_cast<int4*>(dst + base) = make_int4(0, 0, 0, 0);
    __syncthreads();

    // Scatter all NNZ from packed workspace; tiles cover disjoint column ranges → no races
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
}

#endif  // !__CUDA_ARCH__
