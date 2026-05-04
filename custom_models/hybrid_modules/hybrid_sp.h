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
#include "constants.h"

// Hybrid ELL + dense-tail sparse matrix. Rows whose NNZ fits in _ell_stride
// live in the ELL buffers; the rest are spilled into _tail_dense and indexed
// via _tail_dense_map (sparse_row -> tail_row, -1 = no spill).
struct hybrid_sp_t : torch::CustomClassHolder {
    at::Tensor _ell_col_indices;
    at::Tensor _ell_values;
    at::Tensor _row_counters;
    at::Tensor _overflow_counter;
    at::Tensor _tail_dense;
    at::Tensor _tail_dense_map;
    at::Tensor _tail_dense_map_reverse;
    cudaEvent_t _counter_copy_ev;
    at::Tensor hN;
    int _dense_active_rows;
    int _ell_stride;
    int _tail_cap;
    hybrid_sp_t() {};
    hybrid_sp_t(int M, int N, torch::Device device);
    hybrid_sp_t(int M, int N, torch::Device device, int ell_w, int tcap);
    hybrid_sp_t(const hybrid_sp_t& sp);
    void reset_vals();

    inline uint16_t* ell_col_indices() const {return static_cast<uint16_t*>(_ell_col_indices.data_ptr());}
    inline __nv_bfloat16* ell_values() const {return static_cast<__nv_bfloat16*>(_ell_values.data_ptr());}
    inline int32_t* row_counters() const {return static_cast<int32_t*>(_row_counters.data_ptr());}
    inline int32_t* overflow_counter() const {return static_cast<int32_t*>(_overflow_counter.data_ptr());}
    inline __nv_bfloat16* tail_dense() const {return static_cast<__nv_bfloat16*>(_tail_dense.data_ptr());}
    inline int32_t* tail_dense_map() const {return static_cast<int32_t*>(_tail_dense_map.data_ptr());}
    inline int32_t* tail_dense_map_reverse() const {return static_cast<int32_t*>(_tail_dense_map_reverse.data_ptr());}
    at::Tensor overflow_count_tensor() const { return _overflow_counter; }
};


void new_product_as_sparse_sma(hybrid_sp_t* out, at::Tensor const& a, at::Tensor const& b, at::Tensor const& init_val, int M, int N, int K, cudaStream_t stream);
void transpose_hybrid_dense(const hybrid_sp_t& A, hybrid_sp_t& AT, int M_rows, int N_cols, cudaStream_t stream, const int* precomputed_tail_dense_map = nullptr, const int* precomputed_tail_dense_map_reverse = nullptr);
void sparse_dense_gemm_hybrid_dense(at::Tensor& out, hybrid_sp_t* A, const at::Tensor& B, int M, int N, int K, bool transpose_dense_part, cudaStream_t stream, const at::Tensor& B_fp32_cache = at::Tensor());
void sparse_elementwise(hybrid_sp_t* out, hybrid_sp_t* A, hybrid_sp_t* B, int M, int N, cudaStream_t stream);
void compute_dU(hybrid_sp_t* dU, hybrid_sp_t* dT, hybrid_sp_t* R, hybrid_sp_t* P, const at::Tensor& acc_init, int M, int N, cudaStream_t stream);
void create_hybrid_sparse_from_dense(const at::Tensor& dense, hybrid_sp_t* sp, at::Tensor& l0, at::Tensor& l1, int M, int N, cudaStream_t stream);
extern int g_ell_width_regular;
extern int g_ell_width_transpose;
extern int g_tail_rows_regular;
extern int g_tail_rows_transpose;
extern int g_discard_overflow;
at::Tensor ell_spmm_raw(at::Tensor ell_vals, at::Tensor ell_cols, at::Tensor row_counts, at::Tensor B, int64_t M, int64_t K, int64_t N, int64_t ell_stride, int64_t overflow_threshold);
void set_ell_create_warps_per_row(int v);
void set_ell_width_regular(int v);
void set_ell_width_transpose(int v);
void set_tail_rows_regular(int v);
void set_tail_rows_transpose(int v);
void set_discard_overflow(int v);
