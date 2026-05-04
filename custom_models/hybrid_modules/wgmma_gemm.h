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

#pragma once
#include <torch/all.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "constants.h"

// Populates the dense tail for rows whose true NNZ (written by
// blocked_ell_to_ell_packed_kernel) exceeded the ELL stride.
void populate_overflow_tail_from_packed(
    const uint32_t*      C_packed,
    const int32_t*       row_nnz,
    int32_t*             overflow_counter,
    __nv_bfloat16*       tail_dense,
    int32_t*             tail_dense_map,
    int32_t*             tail_dense_map_reverse,
    int M, int N, int N_TILES, int T_n_comp,
    cudaStream_t stream,
    int ell_w, int tail_cap, int discard);

// Fused gate GEMM (X @ G^T) -> hybrid sparse via a packed blocked-ELL workspace.
// Returns false on non-Hopper builds or unaligned shapes; the caller should then
// take the einsum + create_hybrid_sparse_from_dense fallback.
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
    cudaStream_t stream);
