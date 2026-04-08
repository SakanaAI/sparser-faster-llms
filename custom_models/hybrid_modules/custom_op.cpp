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

#include <torch/library.h>
#include<torch/all.h>
#include <ATen/autocast_mode.h>          // cached_cast helpers
#include <c10/core/DispatchKey.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <nvtx3/nvToolsExt.h>
#include "hybrid_sp.h"
#include "perf_instrumentation.h"
#include "wgmma_gemm.h"

using HybridSpPtr = c10::intrusive_ptr<hybrid_sp_t>;

// Holds the overflow_counter tensor from the last transpose in the backward pass.
// Kept alive across calls; user reads it after torch.cuda.synchronize().
static at::Tensor g_last_transpose_overflow_counter;

std::tuple<at::Tensor, at::Tensor, at::Tensor, HybridSpPtr, HybridSpPtr, HybridSpPtr> ff_forward_cuda_gated(const at::Tensor& X, const at::Tensor& G, const at::Tensor& K,  const at::Tensor& V, int64_t out_size)  {
    PERF_START("ff_forward_gated_total", 0);

    int m = X.size(0) * X.size(1);
    int n = K.size(0);
    int k = K.size(1);
    auto device = X.get_device();
    auto stream = at::cuda::getCurrentCUDAStream(device);
    auto options = at::TensorOptions().dtype(torch::kFloat32).device(X.device());
    at::Tensor l0 = at::zeros({}, options);
    at::Tensor l1 = at::zeros({}, options);
    at::Tensor X_flat = X.reshape({m, k});
    PERF_START("wgmma_gate_gemm", stream);
    c10::intrusive_ptr<hybrid_sp_t> P;
    P = c10::make_intrusive<hybrid_sp_t>(m, n, X.device());

    // packed workspace: allocated lazily, reused across calls.
    // T_n_comp=32, T_n=256 must match mm_wgmma_nt_128x256x64TS8 instantiation.
    static at::Tensor bwell_packed_ws;
    static int ws_M = -1, ws_N_TILES = -1;
    constexpr int T_n_comp  = 32;
    constexpr int FUSED_T_n = 256;
    const int N_TILES = n / FUSED_T_n;

    if (!bwell_packed_ws.defined() || ws_M != m || ws_N_TILES != N_TILES) {
        bwell_packed_ws = at::empty({m, N_TILES * T_n_comp}, X.options().dtype(at::kInt));
        ws_M      = m;
        ws_N_TILES = N_TILES;
    }

    bool fused_ok = wgmma_gate_gemm_to_ell_packed(
        X_flat, G, m, n, k,
        P->ell_col_indices(),
        P->ell_values(),
        P->row_counters(),
        reinterpret_cast<uint32_t*>(bwell_packed_ws.data_ptr()),
        static_cast<float*>(l0.data_ptr()),
        static_cast<float*>(l1.data_ptr()),
        stream.stream());

    if (fused_ok) {
        populate_overflow_tail_from_packed(
            reinterpret_cast<const uint32_t*>(bwell_packed_ws.data_ptr()),
            P->row_counters(),
            P->overflow_counter(),
            P->tail_dense(),
            P->tail_dense_map(),
            P->tail_dense_map_reverse(),
            m, n, N_TILES, T_n_comp,
            stream.stream(),
            P->_ell_stride,
            P->_tail_cap,
            g_discard_overflow);
        // In discard mode, overflow_counter was incremented (for counting) but no
        // tail data was written. Zero it so the transpose kernel doesn't iterate
        // over ghost rows and read past tail_dense_map_reverse bounds.
        if (g_discard_overflow == 1) {
            cudaError_t err = cudaMemsetAsync(P->overflow_counter(), 0, sizeof(int), stream.stream());
            TORCH_CHECK(err == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(err));
        }
    } else {
        // Fallback for non-H100 or unaligned shapes
        at::Tensor L = torch::einsum("bmn,kn->bmk", {X, G});
        create_hybrid_sparse_from_dense(L, P.get(), l0, l1, m, n, stream);
    }

    PERF_STOP("wgmma_gate_gemm");


    cudaError_t err = cudaMemcpyAsync(
        P->hN.data_ptr<int>(),
        P->overflow_counter(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
    err = cudaEventCreateWithFlags(&P->_counter_copy_ev, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));
    err = cudaEventRecord(P->_counter_copy_ev, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));

    auto options_bf16 = at::TensorOptions().dtype(torch::kBFloat16).device(X.device());
    auto options_fp32 = at::TensorOptions().dtype(torch::kFloat).device(X.device());
    at::Tensor out2 = at::zeros(X.sizes(), options_bf16);

    auto R = c10::make_intrusive<hybrid_sp_t>(*P);
    R->reset_vals();
    at::Tensor acc_init =  at::zeros({}, options_fp32); // Contribution of L1 to the gradient
    auto T = c10::make_intrusive<hybrid_sp_t>(*P);

    // Do all ELL ops first and the dense ops the last ones

    R->_dense_active_rows = (g_discard_overflow == 1 ? 0 : P->_tail_cap);
    P->_dense_active_rows = (g_discard_overflow == 1 ? 0 : P->_tail_cap);
    T->_dense_active_rows = (g_discard_overflow == 1 ? 0 : P->_tail_cap);
    new_product_as_sparse_sma(R.get(), X, K, acc_init, m, n, k, stream);
    sparse_elementwise(T.get(), R.get(), P.get(), m, n, stream);
    sparse_dense_gemm_hybrid_dense(out2, T.get(), V, m, k, n, false, stream);

    PERF_STOP("ff_forward_gated_total");
    return {out2, l0, l1, P, R, T};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ff_backward_cuda_gated(const at::Tensor& X, const at::Tensor& G, const at::Tensor& K, const at::Tensor& V, HybridSpPtr P, HybridSpPtr R, HybridSpPtr T, const at::Tensor& gR_fp,const at::Tensor& gl0,const at::Tensor& gl1) {
    PERF_START("ff_backward_gated_total", 0);
    int m = X.size(0) * X.size(1);
    int n = K.size(0);
    int k = K.size(1);

    auto device = X.get_device();
    auto stream = at::cuda::getCurrentCUDAStream(device);
    TORCH_CHECK(
      X.scalar_type() == at::kBFloat16,
      "X: Expected a Float (bfloat16) tensor but got ", X.scalar_type());
    TORCH_CHECK(
      K.scalar_type() == at::kBFloat16,
      "K: Expected a Float (bfloat16) tensor but got ", K.scalar_type());
    TORCH_CHECK(
      V.scalar_type() == at::kBFloat16,
      "V: Expected a Float (bfloat16) tensor but got ", V.scalar_type());
    TORCH_CHECK(
      gR_fp.scalar_type() == at::kBFloat16,
      "gR: Expected a Float (bfloat16) tensor but got ", gR_fp.scalar_type());
    auto gR = gR_fp.contiguous();
    auto acc_dtype = torch::kInt32;
    auto options = at::TensorOptions().dtype(acc_dtype).device(X.device());

    acc_dtype = torch::kBFloat16;
    options = at::TensorOptions().dtype(acc_dtype).device(X.device());
    at::Tensor dX_r = at::zeros(X.sizes(), options);
    at::Tensor dX_u = at::zeros(X.sizes(), options);
    at::Tensor dG = at::zeros(G.sizes(), options);
    at::Tensor dK = at::zeros(K.sizes(), options);
    at::Tensor dV = at::zeros({n, k}, options);

    { cudaError_t err = cudaEventSynchronize(P->_counter_copy_ev);
    TORCH_CHECK(err == cudaSuccess, "cudaEventSynchronize failed: ", cudaGetErrorString(err)); }
    int v = *(P->hN.data_ptr<int>()); // Get the current active tail rows
    if (g_discard_overflow == 1) v = 0;    // no tail rows populated in discard mode
    v = (v + 127) & ~int64_t(127);
    if (v > P->_tail_cap) v = P->_tail_cap;  // cap to prevent OOB in narrow()
    R->_dense_active_rows = v;
    P->_dense_active_rows = v;
    T->_dense_active_rows = v;
    hybrid_sp_t T_t(n, m, X.device(), g_ell_width_transpose, g_tail_rows_transpose);
    PERF_START("transpose1_T", stream);
    transpose_hybrid_dense(*T, T_t, m, n, stream);
    PERF_STOP("transpose1_T");
    // Async read of the dense matrix row counters here
    auto hN = at::empty({1}, at::TensorOptions().device(at::kCPU).dtype(at::kInt).pinned_memory(true));
    cudaError_t err = cudaMemcpyAsync(
        hN.data_ptr<int>(),
        T_t.overflow_counter(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
    cudaEvent_t ev;
    err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));
    err = cudaEventRecord(ev, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));

    hybrid_sp_t dT(*T);
    dT.reset_vals();
    // Need to add acc_init here
    auto options_fp32 = at::TensorOptions().dtype(torch::kFloat).device(X.device());
    at::Tensor acc_init = at::zeros({}, options_fp32);
    PERF_START("as_sparse_dT", stream);
    new_product_as_sparse_sma(&dT, gR, V, acc_init, m, n, k, stream);
    PERF_STOP("as_sparse_dT");
    hybrid_sp_t dR(*T);

    dR.reset_vals();
    // dR vals are the dT vals * P
    PERF_START("elemwise_dR", stream);
    sparse_elementwise(&dR, &dT, P.get(),  m, n, stream);
    PERF_STOP("elemwise_dR");

    hybrid_sp_t dU(*T);
    acc_init =  gl1 * (1.0f / m); // Contribution of L1 to the gradient
    compute_dU(&dU, &dT, R.get(), P.get(), acc_init, m, n, stream);

    // Transpose dR and dU for the products
    hybrid_sp_t dR_t(n, m, X.device(), g_ell_width_transpose, g_tail_rows_transpose);
    PERF_START("transpose2_dR", stream);
    // OPTIMIZATION: Reuse tail_maps from T_t (same sparsity pattern)
    transpose_hybrid_dense(dR, dR_t, m, n, stream,
                          T_t.tail_dense_map(),
                          T_t.tail_dense_map_reverse());

    PERF_STOP("transpose2_dR");
    hybrid_sp_t dU_t(n, m, X.device(), g_ell_width_transpose, g_tail_rows_transpose);
    PERF_START("transpose3_dU", stream);
    // OPTIMIZATION: Reuse tail_maps from T_t (same sparsity pattern)
    transpose_hybrid_dense(dU, dU_t, m, n, stream,
                          T_t.tail_dense_map(),
                          T_t.tail_dense_map_reverse());

    PERF_STOP("transpose3_dU");

    PERF_START("gemm_dR_K", stream);
    sparse_dense_gemm_hybrid_dense(dX_r, &dR, K, m, k, n, true, stream);
    PERF_STOP("gemm_dR_K");
    PERF_START("gemm_dU_G", stream);
    sparse_dense_gemm_hybrid_dense(dX_u, &dU, G, m, k, n, true, stream);
    PERF_STOP("gemm_dU_G");
    auto dX = dX_r + dX_u;

    // Wait for the overflow counter to arrive to the host
    { cudaError_t err = cudaEventSynchronize(ev);
    TORCH_CHECK(err == cudaSuccess, "cudaEventSynchronize failed: ", cudaGetErrorString(err)); }
    v = *hN.data_ptr<int>();
    v = (v + 127) & ~int64_t(127);
    if (v > T_t._tail_cap) {
        fprintf(stderr, "Warning: transpose overflow %d exceeds tail_cap %d, capping\n", v, T_t._tail_cap);
        v = T_t._tail_cap;
    }
    dR_t._dense_active_rows = v;
    dU_t._dense_active_rows = v;
    T_t._dense_active_rows = v;
    PERF_START("gemm_dR_t_X", stream);
    sparse_dense_gemm_hybrid_dense(dK, &dR_t, X, n, k, m, false, stream, X);
    PERF_STOP("gemm_dR_t_X");
    PERF_START("gemm_dU_t_X", stream);
    sparse_dense_gemm_hybrid_dense(dG, &dU_t, X, n, k, m, false, stream, X);
    PERF_STOP("gemm_dU_t_X");
    PERF_START("gemm_T_t_gR", stream);
    sparse_dense_gemm_hybrid_dense(dV, &T_t, gR, n, k, m, false, stream, gR);
    PERF_STOP("gemm_T_t_gR");

    // Stash T_t's overflow counter so Python can read it after synchronize().
    g_last_transpose_overflow_counter = T_t._overflow_counter;
    cudaError_t err3 = cudaEventDestroy(P->_counter_copy_ev);
    TORCH_CHECK(err3 == cudaSuccess, "cudaEventDestroy failed: ", cudaGetErrorString(err3));
    err3 = cudaEventDestroy(ev);
    TORCH_CHECK(err3 == cudaSuccess, "cudaEventDestroy failed: ", cudaGetErrorString(err3));
    PERF_STOP("ff_backward_gated_total");
    return {dX, dG, dK, dV};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, HybridSpPtr, HybridSpPtr, HybridSpPtr> ff_forward_meta_gated(const at::Tensor& X, const at::Tensor& G, const at::Tensor& K,  const at::Tensor& V, int64_t out_size)  {
    int b = X.size(0);
    int s = X.size(1);
    int d = X.size(2);
    
    auto acc_dtype = torch::kInt32;
    auto options = at::TensorOptions().dtype(acc_dtype).device(X.device());

    acc_dtype = torch::kFloat;
    options = at::TensorOptions().dtype(acc_dtype).device(X.device());
    at::Tensor out = at::empty({b, s, d}, options);

    acc_dtype = torch::kFloat;
    options = at::TensorOptions().dtype(acc_dtype).device(X.device());
    at::Tensor l0 = at::empty({}, options);
    at::Tensor l1 = at::empty({}, options);
    auto hybrid = c10::make_intrusive<hybrid_sp_t>(b*s, d, X.device());
    return {out, l0, l1, hybrid, hybrid, hybrid};
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ff_backward_meta_gated(const at::Tensor& X, const at::Tensor& G, const at::Tensor& K, const at::Tensor& V, HybridSpPtr P, HybridSpPtr R, HybridSpPtr T, const at::Tensor& gR,const at::Tensor& gl0,const at::Tensor& gl1) {
    auto acc_dtype = torch::kFloat;
    auto options = at::TensorOptions().dtype(acc_dtype).device(X.device());
    at::Tensor dx = at::empty(X.sizes(), options);
    at::Tensor dg = at::empty(G.sizes(), options);
    at::Tensor dk = at::empty(K.sizes(), options);
    at::Tensor dv = at::empty(V.sizes(), options);

    return {dx, dg, dk, dv};
}

TORCH_LIBRARY(sparse_ops, m) {
    m.class_<hybrid_sp_t>("HybridSp")
        .def("overflow_count_tensor", &hybrid_sp_t::overflow_count_tensor);
    m.def("ff_forward_gated(Tensor X, Tensor G, Tensor K, Tensor V, int out_size) -> (Tensor, Tensor, Tensor,  __torch__.torch.classes.sparse_ops.HybridSp, __torch__.torch.classes.sparse_ops.HybridSp, __torch__.torch.classes.sparse_ops.HybridSp)");
    m.def("ff_backward_gated(Tensor X, Tensor G, Tensor K, Tensor V, __torch__.torch.classes.sparse_ops.HybridSp P, __torch__.torch.classes.sparse_ops.HybridSp R, __torch__.torch.classes.sparse_ops.HybridSp T, Tensor gR, Tensor gl0, Tensor gl1) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("ell_spmm_raw(Tensor ell_vals, Tensor ell_cols, Tensor row_counts, Tensor B, int M, int K, int N, int ell_stride, int overflow_threshold) -> Tensor");
    m.def("ell_spmm_raw_orig(Tensor ell_vals, Tensor ell_cols, Tensor row_counts, Tensor B, int M, int K, int N, int ell_stride, int overflow_threshold) -> Tensor");
    m.def("ell_spmm_raw_persistent(Tensor ell_vals, Tensor ell_cols, Tensor row_counts, Tensor B, int M, int K, int N, int ell_stride, int overflow_threshold) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_ops, CUDA, m) {
    m.impl("ff_forward_gated", &ff_forward_cuda_gated);
    m.impl("ff_backward_gated", &ff_backward_cuda_gated);
    m.impl("ell_spmm_raw", &ell_spmm_raw);
    m.impl("ell_spmm_raw_orig", &ell_spmm_raw_orig);
    m.impl("ell_spmm_raw_persistent", &ell_spmm_raw_persistent);
}TORCH_LIBRARY(sparse_ops_config, m) {
    m.def("set_ell_create_warps_per_row(int v) -> ()");
    m.def("set_ell_width_regular(int v) -> ()");
    m.def("set_ell_width_transpose(int v) -> ()");
    m.def("set_tail_rows_regular(int v) -> ()");
    m.def("set_tail_rows_transpose(int v) -> ()");
    m.def("set_discard_overflow(int v) -> ()");
    m.def("get_last_transpose_overflow_count() -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_ops_config, CompositeExplicitAutograd, m) {
    m.impl("set_ell_create_warps_per_row",   [](int64_t v) { set_ell_create_warps_per_row((int)v); });
    m.impl("set_ell_width_regular",         [](int64_t v) { set_ell_width_regular((int)v); });
    m.impl("set_ell_width_transpose",       [](int64_t v) { set_ell_width_transpose((int)v); });
    m.impl("set_tail_rows_regular",         [](int64_t v) { set_tail_rows_regular((int)v); });
    m.impl("set_tail_rows_transpose",       [](int64_t v) { set_tail_rows_transpose((int)v); });
    m.impl("set_discard_overflow",          [](int64_t v) { set_discard_overflow((int)v); });
    m.impl("get_last_transpose_overflow_count", []() -> at::Tensor {
        TORCH_CHECK(g_last_transpose_overflow_counter.defined(),
                    "No backward pass has been run yet");
        return g_last_transpose_overflow_counter;
    });
}

TORCH_LIBRARY_IMPL(sparse_ops, Meta, m) {
    m.impl("ff_forward_gated", &ff_forward_meta_gated);
    m.impl("ff_backward_gated", &ff_backward_meta_gated);
}

// Code for custom backward in C++

class FFSparseGated : public torch::autograd::Function<FFSparseGated> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& X, const at::Tensor& G, const at::Tensor& K, const at::Tensor& V, int64_t out_size) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto ff_forward_op = torch::Dispatcher::singleton()
      .findSchemaOrThrow("sparse_ops::ff_forward_gated", "")
      .typed<decltype(ff_forward_cuda_gated)>();

    auto result = ff_forward_op.call(X, G, K, V, out_size);
    ctx->save_for_backward({X, G, K, V});
    ctx->saved_data["P"] = std::get<3>(result);
    ctx->saved_data["R"] = std::get<4>(result);
    ctx->saved_data["T"] = std::get<5>(result);
    // Must return the same thing as the forward op
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto saved_tensors = ctx->get_saved_variables();
    static auto ff_backward_op = torch::Dispatcher::singleton()
      .findSchemaOrThrow("sparse_ops::ff_backward_gated", "")
      .typed<decltype(ff_backward_cuda_gated)>();
    auto P = ctx->saved_data["P"].toCustomClass<hybrid_sp_t>();
    auto R = ctx->saved_data["R"].toCustomClass<hybrid_sp_t>();
    auto T = ctx->saved_data["T"].toCustomClass<hybrid_sp_t>();
    auto result = ff_backward_op.call(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], P, R, T, grad_output[0], grad_output[1], grad_output[2]);
    at::Tensor undef;
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result), undef}; 
  }
};


std::tuple<at::Tensor, at::Tensor, at::Tensor, HybridSpPtr, HybridSpPtr, HybridSpPtr> ff_forward_autograd_gated(const at::Tensor& X, const at::Tensor& G, const at::Tensor& K,  const at::Tensor& V, int64_t out_size) {
   auto result = FFSparseGated::apply(X, G, K, V, out_size);
    // Must return the same thing as the forward op
   auto hybrid = c10::make_intrusive<hybrid_sp_t>(); //dummy
   return {result[0], result[1], result[2], hybrid, hybrid, hybrid};
}

TORCH_LIBRARY_IMPL(sparse_ops, AutogradCUDA, m) {
    m.impl("ff_forward_gated", &ff_forward_autograd_gated);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, HybridSpPtr, HybridSpPtr, HybridSpPtr> ff_forward_ac_gated(c10::DispatchKeySet ks, const at::Tensor& X, const at::Tensor& G, const at::Tensor& K,  const at::Tensor& V, int64_t out_size)  {
    // 1. Disable further Autocast while we’re inside the wrapper
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Autocast);
    // 2. Pick the dtype AMP is currently using on this thread (fp16 or bf16)
    c10::DispatchKeySet modified_ks = ks.remove(c10::DispatchKey::AutocastCUDA);
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);

    // 3. Cast only if needed (cached_cast is a no-op for non-float / already-cast tensors)
    auto Xc = at::autocast::cached_cast(target_dtype, X);
    auto Gc = at::autocast::cached_cast(target_dtype, G);
    auto Kc = at::autocast::cached_cast(target_dtype, K);
    auto Vc = at::autocast::cached_cast(target_dtype, V);


    static auto op = torch::Dispatcher::singleton()
      .findSchemaOrThrow("sparse_ops::ff_forward_gated", "")
      .typed<decltype(ff_forward_cuda_gated)>();
    // This will automatically skip AutocastCUDA (because we removed it from ks)
    // and pick the next key (CUDA → CompositeImplicitAutograd → Autograd → …)
    return op.redispatch(modified_ks, Xc, Gc, Kc, Vc, out_size);
}

TORCH_LIBRARY_IMPL(sparse_ops, AutocastCUDA, m) {
    m.impl("ff_forward_gated", &ff_forward_ac_gated);
}

// Python-accessible functions for profiling control
void print_perf_report() {
#if ENABLE_PERF_PROFILING
    fprintf(stderr, "[DEBUG] ENABLE_PERF_PROFILING is 1, profiling is enabled\n");
#else
    fprintf(stderr, "[DEBUG] ENABLE_PERF_PROFILING is 0, profiling is DISABLED\n");
#endif
    fprintf(stderr, "[DEBUG] Calling PERF_REPORT()\n");
    PERF_REPORT();
    fprintf(stderr, "[DEBUG] PERF_REPORT() returned\n");
}

void reset_perf_stats() {
    PERF_RESET();
}

TORCH_LIBRARY(sparse_ops_perf, m) {
    m.def("print_report() -> ()");
    m.def("reset_stats() -> ()");
}

TORCH_LIBRARY_IMPL(sparse_ops_perf, CompositeExplicitAutograd, m) {
    m.impl("print_report", &print_perf_report);
    m.impl("reset_stats", &reset_perf_stats);
}


