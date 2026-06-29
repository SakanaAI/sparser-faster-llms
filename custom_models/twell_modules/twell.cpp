#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>
#include <cuda_runtime.h>

#if defined(__GNUC__)
#define TWELL_WEAK __attribute__((weak))
#else
#define TWELL_WEAK
#endif

namespace TWELL_D2T {
void mm_wgmma_nt_128x256x64TS8(
    at::BFloat16* A_d,
    at::BFloat16* B_d,
    uint32_t* C_packed_d,
    const int M,
    const int K,
    const int N,
    cudaStream_t stream = 0
) TWELL_WEAK;

void mm_wgmma_nt_128x256x64TS4(
    at::BFloat16* A_d,
    at::BFloat16* B_d,
    uint32_t* C_packed_d,
    const int M,
    const int K,
    const int N,
    cudaStream_t stream = 0
) TWELL_WEAK;

void mm_wgmma_nt_128x256x64TS2(
    at::BFloat16* A_d,
    at::BFloat16* B_d,
    uint32_t* C_packed_d,
    const int M,
    const int K,
    const int N,
    cudaStream_t stream = 0
) TWELL_WEAK;

void create_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* B_d,
    const int K,
    const int N
) TWELL_WEAK;

void create_d2t_layer_128x256x64TS4(
    const int layer_number,
    at::BFloat16* B_d,
    const int K,
    const int N
) TWELL_WEAK;

void create_d2t_layer_128x256x64TS2(
    const int layer_number,
    at::BFloat16* B_d,
    const int K,
    const int N
) TWELL_WEAK;

void run_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) TWELL_WEAK;

void run_d2t_layer_128x256x64TS4(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) TWELL_WEAK;

void run_d2t_layer_128x256x64TS2(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) TWELL_WEAK;

void ensure_d2t_layer_shape_128x256x64TS8(
    const int layer_number,
    const int M
) TWELL_WEAK;

void ensure_d2t_layer_shape_128x256x64TS4(
    const int layer_number,
    const int M
) TWELL_WEAK;

void ensure_d2t_layer_shape_128x256x64TS2(
    const int layer_number,
    const int M
) TWELL_WEAK;

void destroy_d2t_layer(const int layer_number) TWELL_WEAK;
void destroy_all_d2t_layers() TWELL_WEAK;
}

namespace TWELL_T2D {
void mm_t2d_wid(
    uint32_t* in_packed,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS
) TWELL_WEAK;
}

namespace TWELL_GATED_T2D {
void mm_t2d_wid(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void mm_t2d_wid_flex(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int T_n_compressed
) TWELL_WEAK;

void mm_t2d_wid_inplace(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void mm_t2d_wid_inplace_flex(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int T_n_compressed
) TWELL_WEAK;

void mm_t2d_wid_high_precision(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void mm_t2d_wid_high_precision_flex(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int T_n_compressed
) TWELL_WEAK;

void mm_t2d_wid_high_precision_inplace(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void mm_t2d_wid_high_precision_inplace_flex(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    const int IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int T_n_compressed
) TWELL_WEAK;
}

namespace TWELL_MLP {
void run_mlp_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    at::BFloat16* out_d,
    const int M,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS
) TWELL_WEAK;

void run_gated_mlp_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    at::BFloat16* out_d,
    const int M,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void run_gated_mlp_layer_128x256x64TS8_high_precision(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    at::BFloat16* out_d,
    const int M,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void run_mlp_layer_128x256x64TS8_inplace(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    const int M,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS
) TWELL_WEAK;

void run_gated_mlp_layer_128x256x64TS8_inplace(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    const int M,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;

void run_gated_mlp_layer_128x256x64TS8_high_precision_inplace(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    const int M,
    const int FEATURE_DIM,
    const int OUT_DIM
) TWELL_WEAK;
}

using SparseKernelPackedFn = void (*)(at::BFloat16*, at::BFloat16*, uint32_t*, int, int, int, cudaStream_t);
using CreateD2TLayerFn = void (*)(int, at::BFloat16*, int, int);
using RunD2TLayerFn = void (*)(int, at::BFloat16*, uint32_t*, cudaStream_t);
using EnsureD2TLayerShapeFn = void (*)(int, int);
using DestroyD2TLayerFn = void (*)(int);
using DestroyAllD2TLayersFn = void (*)();
using T2DKernelFn = void (*)(uint32_t*, at::BFloat16*, at::BFloat16*, int, int, int, int);
using GatedT2DKernelFn = void (*)(at::BFloat16*, uint32_t*, at::BFloat16*, at::BFloat16*, at::BFloat16*, int, int, int);
using GatedT2DInplaceKernelFn = void (*)(at::BFloat16*, uint32_t*, at::BFloat16*, at::BFloat16*, int, int, int);
using GatedT2DFlexKernelFn = void (*)(at::BFloat16*, uint32_t*, at::BFloat16*, at::BFloat16*, at::BFloat16*, int, int, int, int);
using GatedT2DInplaceFlexKernelFn = void (*)(at::BFloat16*, uint32_t*, at::BFloat16*, at::BFloat16*, int, int, int, int);
using RunMLPLayerFn = void (*)(int, at::BFloat16*, at::BFloat16*, uint32_t*, at::BFloat16*, int, int, int, int);
using RunGatedMLPLayerFn = void (*)(int, at::BFloat16*, at::BFloat16*, at::BFloat16*, uint32_t*, at::BFloat16*, int, int, int);
using RunMLPLayerInplaceFn = void (*)(int, at::BFloat16*, at::BFloat16*, uint32_t*, int, int, int, int);
using RunGatedMLPLayerInplaceFn = void (*)(int, at::BFloat16*, at::BFloat16*, at::BFloat16*, uint32_t*, int, int, int);

struct D2TLayerMeta {
    int N;
    int K;
    int compression_factor;
};

static std::unordered_map<int, D2TLayerMeta> D2T_LAYER_META;

// returns the sparse algorithms exposed by this extension.
static inline std::vector<std::string> sparse_algorithms() {
    return {"twell_d2t"};
}

// returns the packed sparse kernel.
static inline SparseKernelPackedFn get_sparse_kernel_packed(const std::string& algorithm, int compression_factor) {
    (void)algorithm;
    switch (compression_factor) {
        case 8:
            return TWELL_D2T::mm_wgmma_nt_128x256x64TS8;
        case 4:
            return TWELL_D2T::mm_wgmma_nt_128x256x64TS4;
        case 2:
            return TWELL_D2T::mm_wgmma_nt_128x256x64TS2;
    }
    return TWELL_D2T::mm_wgmma_nt_128x256x64TS8;
}

// returns the cached D2T layer creation entrypoint.
static inline CreateD2TLayerFn get_create_d2t_layer_fn(int compression_factor) {
    switch (compression_factor) {
        case 8:
            return TWELL_D2T::create_d2t_layer_128x256x64TS8;
        case 4:
            return TWELL_D2T::create_d2t_layer_128x256x64TS4;
        case 2:
            return TWELL_D2T::create_d2t_layer_128x256x64TS2;
    }
    return TWELL_D2T::create_d2t_layer_128x256x64TS8;
}

// returns the cached D2T layer execution entrypoint.
static inline RunD2TLayerFn get_run_d2t_layer_fn(int compression_factor) {
    switch (compression_factor) {
        case 8:
            return TWELL_D2T::run_d2t_layer_128x256x64TS8;
        case 4:
            return TWELL_D2T::run_d2t_layer_128x256x64TS4;
        case 2:
            return TWELL_D2T::run_d2t_layer_128x256x64TS2;
    }
    return TWELL_D2T::run_d2t_layer_128x256x64TS8;
}

// returns the cached D2T shape-update entrypoint.
static inline EnsureD2TLayerShapeFn get_ensure_d2t_layer_shape_fn(int compression_factor) {
    switch (compression_factor) {
        case 8:
            return TWELL_D2T::ensure_d2t_layer_shape_128x256x64TS8;
        case 4:
            return TWELL_D2T::ensure_d2t_layer_shape_128x256x64TS4;
        case 2:
            return TWELL_D2T::ensure_d2t_layer_shape_128x256x64TS2;
    }
    return TWELL_D2T::ensure_d2t_layer_shape_128x256x64TS8;
}

// returns the cached D2T layer destruction entrypoint.
static inline DestroyD2TLayerFn get_destroy_d2t_layer_fn() {
    return TWELL_D2T::destroy_d2t_layer;
}

// returns the cached D2T global-destruction entrypoint.
static inline DestroyAllD2TLayersFn get_destroy_all_d2t_layers_fn() {
    return TWELL_D2T::destroy_all_d2t_layers;
}

// returns the packed-to-dense downprojection kernel.
static inline T2DKernelFn get_t2d_kernel_fn() {
    return TWELL_T2D::mm_t2d_wid;
}

// returns the gated packed-to-dense kernel for the requested precision mode.
static inline GatedT2DKernelFn get_gated_t2d_kernel_fn(bool highest_precision) {
    return highest_precision ? TWELL_GATED_T2D::mm_t2d_wid_high_precision : TWELL_GATED_T2D::mm_t2d_wid;
}

// returns the in-place gated packed-to-dense kernel for the requested precision mode.
static inline GatedT2DInplaceKernelFn get_gated_t2d_inplace_kernel_fn(bool highest_precision) {
    return highest_precision ? TWELL_GATED_T2D::mm_t2d_wid_high_precision_inplace : TWELL_GATED_T2D::mm_t2d_wid_inplace;
}

// returns the flex gated packed-to-dense kernel for the requested precision mode.
static inline GatedT2DFlexKernelFn get_gated_t2d_flex_kernel_fn(bool highest_precision) {
    return highest_precision ? TWELL_GATED_T2D::mm_t2d_wid_high_precision_flex : TWELL_GATED_T2D::mm_t2d_wid_flex;
}

// returns the in-place flex gated packed-to-dense kernel for the requested precision mode.
static inline GatedT2DInplaceFlexKernelFn get_gated_t2d_inplace_flex_kernel_fn(bool highest_precision) {
    return highest_precision ? TWELL_GATED_T2D::mm_t2d_wid_high_precision_inplace_flex : TWELL_GATED_T2D::mm_t2d_wid_inplace_flex;
}

// returns the fused non-gated MLP kernel.
static inline RunMLPLayerFn get_run_mlp_layer_fn() {
    return TWELL_MLP::run_mlp_layer_128x256x64TS8;
}

// returns the fused gated MLP kernel for the requested precision mode.
static inline RunGatedMLPLayerFn get_run_gated_mlp_layer_fn(bool highest_precision) {
    return highest_precision
        ? TWELL_MLP::run_gated_mlp_layer_128x256x64TS8_high_precision
        : TWELL_MLP::run_gated_mlp_layer_128x256x64TS8;
}

// returns the in-place fused non-gated MLP kernel.
static inline RunMLPLayerInplaceFn get_run_mlp_layer_inplace_fn() {
    return TWELL_MLP::run_mlp_layer_128x256x64TS8_inplace;
}

// returns the in-place fused gated MLP kernel for the requested precision mode.
static inline RunGatedMLPLayerInplaceFn get_run_gated_mlp_layer_inplace_fn(bool highest_precision) {
    return highest_precision
        ? TWELL_MLP::run_gated_mlp_layer_128x256x64TS8_high_precision_inplace
        : TWELL_MLP::run_gated_mlp_layer_128x256x64TS8_inplace;
}

// runs packed sparse matmul into a caller-provided packed output tensor.
// inputs: A, B, output tensor, algorithm key. output: C_packed filled.
void matmul_sparse_out(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C_packed,
    const std::string& algorithm = "twell_d2t",
    int compression_factor = 8
) {
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    auto* kernel = get_sparse_kernel_packed(algorithm, compression_factor);
    kernel(
        A.data_ptr<at::BFloat16>(),
        B.data_ptr<at::BFloat16>(),
        C_packed.data_ptr<uint32_t>(),
        M,
        K,
        N,
        0
    );
}

// runs packed sparse matmul and returns a newly allocated packed output tensor.
// inputs: A, B, algorithm key. output: packed sparse matmul result.
torch::Tensor matmul_sparse(
    torch::Tensor A,
    torch::Tensor B,
    const std::string& algorithm = "twell_d2t",
    int compression_factor = 8
) {
    const int M = static_cast<int>(A.size(0));
    const int N = static_cast<int>(B.size(0));
    auto C_packed = torch::zeros({M, N / compression_factor}, A.options().dtype(torch::kUInt32));
    matmul_sparse_out(A, B, C_packed, algorithm, compression_factor);
    return C_packed;
}

// runs packed-to-dense downprojection into a caller-provided dense output tensor.
// inputs: packed input, DOWN, output tensor, split count. output: OUT filled.
void matmul_t2d_out(
    torch::Tensor IN_packed,
    torch::Tensor DOWN,
    torch::Tensor OUT,
    int num_splits = 2
) {
    const int M = static_cast<int>(IN_packed.size(0));
    const int FEATURE_DIM = static_cast<int>(DOWN.size(0));
    const int OUT_DIM = static_cast<int>(DOWN.size(1));

    auto* kernel = get_t2d_kernel_fn();
    kernel(
        IN_packed.data_ptr<uint32_t>(),
        DOWN.data_ptr<at::BFloat16>(),
        OUT.data_ptr<at::BFloat16>(),
        M,
        FEATURE_DIM,
        OUT_DIM,
        num_splits
    );
}

// runs packed-to-dense downprojection and returns a newly allocated dense output tensor.
// inputs: packed input, DOWN, split count. output: dense downprojection result.
torch::Tensor matmul_t2d(
    torch::Tensor IN_packed,
    torch::Tensor DOWN,
    int num_splits = 2
) {
    auto OUT = torch::zeros({IN_packed.size(0), DOWN.size(1)}, DOWN.options());
    matmul_t2d_out(IN_packed, DOWN, OUT, num_splits);
    return OUT;
}

// runs gated packed-to-dense projection into a caller-provided dense output tensor.
// inputs: dense input, packed gate, UP, DOWN, output tensor, precision flag. output: OUT filled.
void matmul_gated_t2d_out(
    torch::Tensor IN_dense,
    torch::Tensor GATE_packed,
    torch::Tensor UP,
    torch::Tensor DOWN,
    torch::Tensor OUT,
    bool highest_precision = false,
    int compression_factor = 8,
    bool flex_kernels = false
) {
    const int M = static_cast<int>(IN_dense.size(0));
    const int OUT_DIM = static_cast<int>(IN_dense.size(1));
    const int FEATURE_DIM = static_cast<int>(UP.size(0));

    if (flex_kernels) {
        auto* kernel = get_gated_t2d_flex_kernel_fn(highest_precision);
        kernel(
            IN_dense.data_ptr<at::BFloat16>(),
            GATE_packed.data_ptr<uint32_t>(),
            UP.data_ptr<at::BFloat16>(),
            DOWN.data_ptr<at::BFloat16>(),
            OUT.data_ptr<at::BFloat16>(),
            M,
            FEATURE_DIM,
            OUT_DIM,
            256 / compression_factor
        );
    } else {
        auto* kernel = get_gated_t2d_kernel_fn(highest_precision);
        kernel(
            IN_dense.data_ptr<at::BFloat16>(),
            GATE_packed.data_ptr<uint32_t>(),
            UP.data_ptr<at::BFloat16>(),
            DOWN.data_ptr<at::BFloat16>(),
            OUT.data_ptr<at::BFloat16>(),
            M,
            FEATURE_DIM,
            OUT_DIM
        );
    }
}

// runs gated packed-to-dense projection and returns a newly allocated dense output tensor.
// inputs: dense input, packed gate, UP, DOWN, precision flag. output: dense projection result.
torch::Tensor matmul_gated_t2d(
    torch::Tensor IN_dense,
    torch::Tensor GATE_packed,
    torch::Tensor UP,
    torch::Tensor DOWN,
    bool highest_precision = false,
    int compression_factor = 8,
    bool flex_kernels = false
) {
    auto OUT = torch::zeros({IN_dense.size(0), IN_dense.size(1)}, IN_dense.options());
    matmul_gated_t2d_out(
        IN_dense,
        GATE_packed,
        UP,
        DOWN,
        OUT,
        highest_precision,
        compression_factor,
        flex_kernels
    );
    return OUT;
}

// creates or refreshes cached D2T metadata for one layer id.
// inputs: layer id, B weight tensor. output: layer cache initialized.
void create_d2t_layer(
    int layer_number,
    torch::Tensor B,
    int compression_factor = 8
) {
    const int K = static_cast<int>(B.size(1));
    const int N = static_cast<int>(B.size(0));

    auto* fn = get_create_d2t_layer_fn(compression_factor);
    fn(layer_number, B.data_ptr<at::BFloat16>(), K, N);
    D2T_LAYER_META[layer_number] = D2TLayerMeta{N, K, compression_factor};
}

// runs one cached D2T layer into a caller-provided packed output tensor.
// inputs: layer id, A tensor, packed output tensor. output: C_packed filled.
void run_d2t_layer(
    int layer_number,
    torch::Tensor A,
    torch::Tensor C_packed
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);
    const int N = meta_it->second.N;
    const int K = meta_it->second.K;
    const int compression_factor = meta_it->second.compression_factor;
    const int M = static_cast<int>(A.size(0));

    auto* ensure_fn = get_ensure_d2t_layer_shape_fn(compression_factor);
    ensure_fn(layer_number, M);

    auto* fn = get_run_d2t_layer_fn(compression_factor);
    fn(layer_number, A.data_ptr<at::BFloat16>(), C_packed.data_ptr<uint32_t>(), 0);
    (void)K;
    (void)N;
}

// ensures cached D2T state is ready for the requested row count.
void ensure_d2t_layer_shape(
    int layer_number,
    int M
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);
    auto* fn = get_ensure_d2t_layer_shape_fn(meta_it->second.compression_factor);
    fn(layer_number, M);
}

// runs the fused non-gated MLP using caller-provided packed hidden states.
// inputs: layer id, A, DOWN, packed hidden state buffer, split count. output: dense output tensor.
static torch::Tensor run_mlp_layer_with_hidden_states(
    int layer_number,
    torch::Tensor A,
    torch::Tensor DOWN,
    torch::Tensor C_packed,
    int num_splits
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);

    const int FEATURE_DIM = meta_it->second.N;
    const int OUT_DIM = meta_it->second.K;
    const int M = static_cast<int>(A.size(0));

    auto OUT = torch::empty({M, OUT_DIM}, A.options());

    auto* fn = get_run_mlp_layer_fn();
    fn(
        layer_number,
        A.data_ptr<at::BFloat16>(),
        DOWN.data_ptr<at::BFloat16>(),
        C_packed.data_ptr<uint32_t>(),
        OUT.data_ptr<at::BFloat16>(),
        M,
        FEATURE_DIM,
        OUT_DIM,
        num_splits
    );
    return OUT;
}

// runs the fused non-gated MLP and allocates packed hidden states internally.
// inputs: layer id, A, DOWN, split count. output: dense output tensor.
torch::Tensor run_mlp_layer(
    int layer_number,
    torch::Tensor A,
    torch::Tensor DOWN,
    int num_splits = 2
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);
    const int FEATURE_DIM = meta_it->second.N;
    const int M = static_cast<int>(A.size(0));

    auto* ensure_fn = get_ensure_d2t_layer_shape_fn(meta_it->second.compression_factor);
    ensure_fn(layer_number, M);

    auto C_packed = torch::empty({M, FEATURE_DIM / 8}, A.options().dtype(torch::kUInt32));
    return run_mlp_layer_with_hidden_states(layer_number, A, DOWN, C_packed, num_splits);
}

// runs the fused non-gated MLP in place using caller-provided packed hidden states.
// inputs: layer id, A, DOWN, packed hidden state buffer, split count. output: A overwritten.
static void run_mlp_layer_inplace_with_hidden_states(
    int layer_number,
    torch::Tensor A,
    torch::Tensor DOWN,
    torch::Tensor C_packed,
    int num_splits
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);

    const int FEATURE_DIM = meta_it->second.N;
    const int OUT_DIM = meta_it->second.K;
    const int M = static_cast<int>(A.size(0));

    auto* fn = get_run_mlp_layer_inplace_fn();
    fn(
        layer_number,
        A.data_ptr<at::BFloat16>(),
        DOWN.data_ptr<at::BFloat16>(),
        C_packed.data_ptr<uint32_t>(),
        M,
        FEATURE_DIM,
        OUT_DIM,
        num_splits
    );
}

// runs the fused non-gated MLP in place and allocates packed hidden states internally.
// inputs: layer id, A, DOWN, split count. output: A overwritten.
void run_mlp_layer_inplace(
    int layer_number,
    torch::Tensor A,
    torch::Tensor DOWN,
    int num_splits = 2
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);
    const int FEATURE_DIM = meta_it->second.N;
    const int M = static_cast<int>(A.size(0));

    auto* ensure_fn = get_ensure_d2t_layer_shape_fn(meta_it->second.compression_factor);
    ensure_fn(layer_number, M);

    auto C_packed = torch::empty({M, FEATURE_DIM / 8}, A.options().dtype(torch::kUInt32));
    run_mlp_layer_inplace_with_hidden_states(layer_number, A, DOWN, C_packed, num_splits);
}

// runs the fused gated MLP using caller-provided packed hidden states.
// inputs: layer id, A, UP, DOWN, packed hidden state buffer, precision flag. output: dense output tensor.
static torch::Tensor run_gated_mlp_layer_with_hidden_states(
    int layer_number,
    torch::Tensor A,
    torch::Tensor UP,
    torch::Tensor DOWN,
    torch::Tensor C_packed,
    bool highest_precision,
    int compression_factor = 8,
    bool flex_kernels = false
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);

    const int FEATURE_DIM = meta_it->second.N;
    const int OUT_DIM = meta_it->second.K;
    const int M = static_cast<int>(A.size(0));

    auto OUT = torch::empty({M, OUT_DIM}, A.options());

    if (flex_kernels) {
        run_d2t_layer(layer_number, A, C_packed);
        auto* fn = get_gated_t2d_flex_kernel_fn(highest_precision);
        fn(
            A.data_ptr<at::BFloat16>(),
            C_packed.data_ptr<uint32_t>(),
            UP.data_ptr<at::BFloat16>(),
            DOWN.data_ptr<at::BFloat16>(),
            OUT.data_ptr<at::BFloat16>(),
            M,
            FEATURE_DIM,
            OUT_DIM,
            256 / compression_factor
        );
    } else {
        auto* fn = get_run_gated_mlp_layer_fn(highest_precision);
        fn(
            layer_number,
            A.data_ptr<at::BFloat16>(),
            UP.data_ptr<at::BFloat16>(),
            DOWN.data_ptr<at::BFloat16>(),
            C_packed.data_ptr<uint32_t>(),
            OUT.data_ptr<at::BFloat16>(),
            M,
            FEATURE_DIM,
            OUT_DIM
        );
    }
    return OUT;
}

// runs the fused gated MLP and allocates packed hidden states internally.
// inputs: layer id, A, UP, DOWN, precision flag. output: dense output tensor.
torch::Tensor run_gated_mlp_layer(
    int layer_number,
    torch::Tensor A,
    torch::Tensor UP,
    torch::Tensor DOWN,
    bool highest_precision = false,
    int compression_factor = 8,
    bool flex_kernels = false
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);
    const int FEATURE_DIM = meta_it->second.N;
    const int M = static_cast<int>(A.size(0));

    auto* ensure_fn = get_ensure_d2t_layer_shape_fn(meta_it->second.compression_factor);
    ensure_fn(layer_number, M);

    auto C_packed = torch::empty({M, FEATURE_DIM / compression_factor}, A.options().dtype(torch::kUInt32));
    return run_gated_mlp_layer_with_hidden_states(
        layer_number, A, UP, DOWN, C_packed, highest_precision, compression_factor, flex_kernels
    );
}

// runs the fused gated MLP in place using caller-provided packed hidden states.
// inputs: layer id, A, UP, DOWN, packed hidden state buffer, precision flag. output: A overwritten.
static void run_gated_mlp_layer_inplace_with_hidden_states(
    int layer_number,
    torch::Tensor A,
    torch::Tensor UP,
    torch::Tensor DOWN,
    torch::Tensor C_packed,
    bool highest_precision,
    int compression_factor = 8,
    bool flex_kernels = false
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);

    const int FEATURE_DIM = meta_it->second.N;
    const int OUT_DIM = meta_it->second.K;
    const int M = static_cast<int>(A.size(0));

    if (flex_kernels) {
        run_d2t_layer(layer_number, A, C_packed);
        auto* fn = get_gated_t2d_inplace_flex_kernel_fn(highest_precision);
        fn(
            A.data_ptr<at::BFloat16>(),
            C_packed.data_ptr<uint32_t>(),
            UP.data_ptr<at::BFloat16>(),
            DOWN.data_ptr<at::BFloat16>(),
            M,
            FEATURE_DIM,
            OUT_DIM,
            256 / compression_factor
        );
    } else {
        auto* fn = get_run_gated_mlp_layer_inplace_fn(highest_precision);
        fn(
            layer_number,
            A.data_ptr<at::BFloat16>(),
            UP.data_ptr<at::BFloat16>(),
            DOWN.data_ptr<at::BFloat16>(),
            C_packed.data_ptr<uint32_t>(),
            M,
            FEATURE_DIM,
            OUT_DIM
        );
    }
}

// runs the fused gated MLP in place and allocates packed hidden states internally.
// inputs: layer id, A, UP, DOWN, precision flag. output: A overwritten.
void run_gated_mlp_layer_inplace(
    int layer_number,
    torch::Tensor A,
    torch::Tensor UP,
    torch::Tensor DOWN,
    bool highest_precision = false,
    int compression_factor = 8,
    bool flex_kernels = false
) {
    auto meta_it = D2T_LAYER_META.find(layer_number);
    const int FEATURE_DIM = meta_it->second.N;
    const int M = static_cast<int>(A.size(0));

    auto* ensure_fn = get_ensure_d2t_layer_shape_fn(meta_it->second.compression_factor);
    ensure_fn(layer_number, M);

    auto C_packed = torch::empty({M, FEATURE_DIM / compression_factor}, A.options().dtype(torch::kUInt32));
    run_gated_mlp_layer_inplace_with_hidden_states(
        layer_number, A, UP, DOWN, C_packed, highest_precision, compression_factor, flex_kernels
    );
}

// destroys one cached D2T layer entry and its metadata.
void destroy_d2t_layer(int layer_number) {
    auto* fn = get_destroy_d2t_layer_fn();
    fn(layer_number);
    D2T_LAYER_META.erase(layer_number);
}

// destroys all cached D2T layer entries and metadata.
void destroy_all_d2t_layers() {
    auto* fn = get_destroy_all_d2t_layers_fn();
    fn();
    D2T_LAYER_META.clear();
}

// binds the TwELL extension entrypoints into the Python module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_sparse", &matmul_sparse, "TWELL packed sparse matmul (CUDA)");
    m.def("matmul_sparse_out", &matmul_sparse_out, "TWELL packed sparse matmul out (CUDA)");
    m.def("matmul_t2d", &matmul_t2d, "TWELL packed-to-dense downprojection (CUDA)");
    m.def("matmul_t2d_out", &matmul_t2d_out, "TWELL packed-to-dense downprojection out (CUDA)");
    m.def("matmul_gated_t2d", &matmul_gated_t2d, "TWELL gated packed-to-dense projection (CUDA)");
    m.def("matmul_gated_t2d_out", &matmul_gated_t2d_out, "TWELL gated packed-to-dense projection out (CUDA)");
    m.def("sparse_algorithms", &sparse_algorithms, "Available sparse algorithms");
    m.def("create_d2t_layer", &create_d2t_layer, "Create/cache a D2T layer plan");
    m.def("ensure_d2t_layer_shape", &ensure_d2t_layer_shape, "Ensure cached D2T layer shape state");
    m.def("run_d2t_layer", &run_d2t_layer, "Run a cached D2T layer plan");
    m.def("run_mlp_layer", &run_mlp_layer, "Run fused TwELL non-gated MLP with cached D2T layer");
    m.def("run_gated_mlp_layer", &run_gated_mlp_layer, "Run fused TwELL gated MLP with cached D2T layer");
    m.def("run_mlp_layer_inplace", &run_mlp_layer_inplace, "Run fused TwELL non-gated MLP in-place");
    m.def("run_gated_mlp_layer_inplace", &run_gated_mlp_layer_inplace, "Run fused TwELL gated MLP in-place");
    m.def("_run_mlp_layer_with_hidden_states", &run_mlp_layer_with_hidden_states,
          "Run fused TwELL non-gated MLP with caller-provided packed hidden states");
    m.def("_run_gated_mlp_layer_with_hidden_states", &run_gated_mlp_layer_with_hidden_states,
          "Run fused TwELL gated MLP with caller-provided packed hidden states");
    m.def("_run_mlp_layer_inplace_with_hidden_states", &run_mlp_layer_inplace_with_hidden_states,
          "Run fused TwELL non-gated in-place MLP with caller-provided packed hidden states");
    m.def("_run_gated_mlp_layer_inplace_with_hidden_states", &run_gated_mlp_layer_inplace_with_hidden_states,
          "Run fused TwELL gated in-place MLP with caller-provided packed hidden states");
    m.def("destroy_d2t_layer", &destroy_d2t_layer, "Destroy a cached D2T layer plan");
    m.def("destroy_all_d2t_layers", &destroy_all_d2t_layers, "Destroy all cached D2T layer plans");
}

#undef TWELL_WEAK
