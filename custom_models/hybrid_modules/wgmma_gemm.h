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

// Populate overflow dense tail for rows whose NNZ exceeded ELL_WIDTH in the packed path.
// Must be called after blocked_ell_to_ell_packed_kernel has written true row_nnz values.
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

// Write GEMM (X @ G^T) directly into ELL format via packed blocked-ELL intermediate.
// Returns true  → WGMMA kernel + convert kernel launched, ELL buffers populated.
// Returns false → prerequisites not met (non-H100 or unaligned shapes);
//                 caller should fall back to einsum + create_hybrid_sparse_from_dense.
bool wgmma_gate_gemm_to_ell_packed(
    const at::Tensor& X_flat,   // [M, K] bf16 row-major
    const at::Tensor& G,        // [N, K] bf16 row-major
    int M, int N, int K,
    uint16_t*        ell_col,   // P->ell_col_indices()     [M, ELL_WIDTH]
    __nv_bfloat16*   ell_val,   // P->ell_values()          [M, ELL_WIDTH]
    int32_t*         row_nnz,   // P->row_counters()        [M]
    uint32_t*        C_packed,  // workspace [M, N_TILES*T_n_comp] uint32
    float*           l0_out,    // nullable scalar accumulator
    float*           l1_out,    // nullable scalar accumulator
    cudaStream_t stream);
