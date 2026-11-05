// Enable the half/BF16 operators required by commondx in this translation unit.
// PyTorch disables them by default for CUDAExtension builds.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <cassert>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include "hilbert.cu"
// Include ATen before cuBLASDx/CuTe to avoid name lookup conflicts.
// TWELL_D2T_SM selects the single architecture-specific target (900 for sm_90a,
// 1000 for sm_100a, 1070 for sm_107a, ...). The build derives it from
// TORCH_CUDA_ARCH_LIST via mathdx_paths.d2t_target_sm.
#ifndef TWELL_D2T_SM
#error "define TWELL_D2T_SM to the target SM version, e.g. -DTWELL_D2T_SM=1000 with -gencode arch=compute_100a,code=sm_100a"
#endif
#if TWELL_D2T_SM < 900
#error "the cuBLASDx D2T implementation requires sm_90a or newer"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != TWELL_D2T_SM
#error "TWELL_D2T_SM does not match the -gencode target: build D2T for exactly one architecture-specific SM"
#endif

// Disable device assertions in library headers, then restore this file's setting.
#ifndef NDEBUG
#define D2T_RESTORE_ASSERTS
#define NDEBUG
#endif
#include <cublasdx.hpp>

#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/tensor.hpp>
#ifdef D2T_RESTORE_ASSERTS
#undef NDEBUG
#undef D2T_RESTORE_ASSERTS
#include <cassert>
#endif

#include <cstdint>

namespace TWELL_D2T {
namespace cublasdx_d2t {

// Baked into the packed format: consumers recover the column-tile as col / TILE_N.
inline constexpr int TILE_N = 256;

// A wrapped write lands at index t_n_comp, one past the last live slot, so the pad
// must be at least one word; 4 also keeps the row's byte pitch a multiple of 16,
// which the bulk store requires of its innermost box extent.
inline constexpr int PACKED_PAD = 4;

// The 3-D output view places column-tiles in its slowest dimension. The store box
// includes the shared-memory row padding; TMA bounds-checking discards those extra
// columns because the tensor's innermost extent is only t_n_comp.
namespace tma_store {

template<int Pitch, int Rows>
inline void encode(CUtensorMap* map, uint32_t* c_packed, int M, int packed_ld, int t_n_comp) {
    auto gtensor = cute::make_tensor(
        cute::make_gmem_ptr(c_packed),
        cute::make_shape(t_n_comp, M, packed_ld / t_n_comp),
        cute::make_stride(cute::_1 {}, packed_ld, t_n_comp));
    auto slayout = cute::make_layout(
        cute::make_shape(cute::Int<Pitch> {}, cute::Int<Rows> {}, cute::_1 {}));

    *map = cute::make_tma_copy(cute::SM90_TMA_STORE {}, gtensor, slayout).tma_desc_;
}

// Single-thread call, and only after a fence publishing smem to the async proxy.
__device__ __forceinline__ void
issue(const CUtensorMap* map, const uint32_t* smem, uint32_t row_offset, uint32_t tile_col) {
    cute::SM90_TMA_STORE::copy(map, smem, 0, static_cast<int32_t>(row_offset), static_cast<int32_t>(tile_col));
    cute::tma_store_arrive();
}

// Waits only until TMA has read the staging tile, not until the store lands, so
// the write-out stays overlapped with the next tile's mainloop.
__device__ __forceinline__ void wait_read() { cute::tma_store_wait<0>(); }

}  // namespace tma_store

// 128x256x64 tiles use 256 compute threads on Hopper (two WGMMA warpgroups) and
// 128 on Blackwell and newer (single-thread UTCMMA issue); cuBLASDx supplies the
// pipeline producer roles.
// Packed-output geometry is independent of the cluster shape.
template<int TnComp>
struct packed_geometry {
    static constexpr int tile_m = 128;
    static constexpr int tile_n = TILE_N;
    static constexpr int tile_k = 64;
    static constexpr int threads = TWELL_D2T_SM >= 1000 ? 128 : 256;
    static constexpr int t_n_comp = TnComp;
    static constexpr int packed_pitch = TnComp + PACKED_PAD;
    static constexpr int packed_elems = tile_m * packed_pitch;

    static_assert((TnComp & (TnComp - 1)) == 0, "T_n_comp must be a power of two: the slot index wraps with a mask");
    static_assert(TnComp <= TILE_N, "a column tile cannot hold more entries than it has columns");
    static_assert(packed_pitch * sizeof(uint32_t) % 16 == 0, "bulk store needs a 16B-multiple box extent");
};

// The library selects the pipeline depth after reserving the packed staging tile.
template<int TnComp, int ClusterM, int ClusterN>
struct gemm_traits: packed_geometry<TnComp> {
    using geom = packed_geometry<TnComp>;
    static constexpr int tile_m = geom::tile_m;
    static constexpr int tile_n = geom::tile_n;
    static constexpr int tile_k = geom::tile_k;
    static constexpr int threads = geom::threads;
    static constexpr int packed_pitch = geom::packed_pitch;

    // N x K row-major B is viewed as K x N column-major without a transpose.
    // Streaming needs no input preprocessing; the compute block size is fixed.
    // A multicasts along cluster N, B along M. The 2x1 cluster shares one B load.
    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::row_major>() +
                          cublasdx::Alignment<16, 16, 16>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<threads>() +
                          cublasdx::WithPipeline() +
                          cublasdx::EnableInputStreaming() +
                          cublasdx::StaticBlockDim() +
                          cublasdx::BlocksPerCluster<ClusterM, ClusterN>() +
                          cublasdx::SM<TWELL_D2T_SM, cublasdx::sm_modifier::arch_specific>());

    // Require the requested multicast geometry; LDGSTS paths ignore clustering.
    static_assert(cublasdx::blocks_per_cluster_of_v_m<BLAS> == ClusterM && cublasdx::blocks_per_cluster_of_v_n<BLAS> == ClusterN,
                  "cuBLASDx did not adopt the requested Cluster extents");
    static_assert(ClusterM > 1 || ClusterN > 1,
                  "a 1x1 cluster carries no multicast; drop BlocksPerCluster<> instead of asking for it");

    // suggest_pipeline<BLAS>() would fill all of shared memory with stages, leaving
    // none for the staging tile, so reserve it here and pick the depth explicitly.
    // The extra 8 columns cover the 128B alignment round-up and the library reserve.
    static constexpr int depth = cublasdx::suggest_max_pipeline_depth<BLAS, __nv_bfloat16, __nv_bfloat16, uint32_t>(
        BLAS::suggest_layout_smem_a(),
        BLAS::suggest_layout_smem_b(),
        cute::make_layout(cute::make_shape(cute::Int<tile_m> {}, cute::Int<packed_pitch + 8> {})));

    static_assert(depth >= 1, "no shared memory left for even one pipeline stage with this T_n_comp");
};

// Each cluster reads a partition of the Hilbert schedule from global memory:
// schedule_size_per_sm words of (cluster_m << 16) | cluster_n, padded with -1.
// Each CTA expands these coordinates by its cluster rank.
template<class Traits, int CLUSTER_DIM_m, int CLUSTER_DIM_n, bool LOOP_OVERFLOW_STORAGE, class DevicePipeline>
__launch_bounds__(DevicePipeline::max_threads_per_block, 1)
__cluster_dims__(CLUSTER_DIM_m * CLUSTER_DIM_n, 1, 1) __global__
void mm_wgmma_nt_kernel(
    const int* schedule_gmem_ptr,
    const int schedule_size_per_sm,
    const CUtensorMap __grid_constant__ c_map,
    DevicePipeline const __grid_constant__ device_pipeline
) {
    constexpr int tile_m = Traits::tile_m;
    constexpr int t_n_comp = Traits::t_n_comp;
    constexpr int packed_pitch = Traits::packed_pitch;

    extern __shared__ __align__(device_pipeline.buffer_alignment()) cublasdx::byte smem[];

    // 128B: the bulk store reads the staging tile straight out of shared memory.
    auto smem_slices = cublasdx::shared_memory::slice_into_pointers<cublasdx::byte, uint32_t>(
        smem,
        device_pipeline.buffer_alignment(),
        device_pipeline.buffer_size(),
        128u,
        static_cast<unsigned>(Traits::packed_elems));
    cublasdx::byte* const pipeline_smem = cute::get<0>(smem_slices);
    uint32_t* const packed = cute::get<1>(smem_slices);

    int cluster_idx;
    asm ("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_idx) :);
    int cluster_ctarank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cluster_ctarank) :);
    // Match cuBLASDx's M-fastest cluster rank layout for operand multicast.
    const int cluster_lane_m = cluster_ctarank % CLUSTER_DIM_m;
    const int cluster_lane_n = (cluster_ctarank / CLUSTER_DIM_m) % CLUSTER_DIM_n;

    // Thread 0 belongs to the compute group and issues the whole-tile store.
    // Bulk-async groups are per-thread, so the issuer must also drain each store.
    const bool issue_tma_store = (threadIdx.x == 0);

    schedule_gmem_ptr += cluster_idx * schedule_size_per_sm;
    int coord = schedule_gmem_ptr[0];
    // An empty partition: legal when there are fewer cluster-tiles than clusters.
    // Nothing was issued, so there is nothing to drain.
    if (coord == -1) return;
    int tile_row = (coord >> 16) * CLUSTER_DIM_m + cluster_lane_m;
    int tile_col = (coord & 0xFFFF) * CLUSTER_DIM_n + cluster_lane_n;

    // reset_tile() preserves pipeline stage/phase across persistent tiles, allowing
    // the producer to load the next tile during the current tile's epilogue.
    auto tile_pipeline = device_pipeline.get_tile(pipeline_smem, tile_row, tile_col);

    // Prefetch one coordinate ahead to overlap the global load with the mainloop.
    // Producer and compute threads traverse the same schedule.
    int coord_next = schedule_size_per_sm > 1 ? schedule_gmem_ptr[1] : -1;

    for (int it = 0;;) {
        const int col_base = tile_col * TILE_N;

        tile_pipeline.execute([&](auto& accumulator) {
            // cuBLASDx invokes the epilogue on BLAS::BlockDim threads. Nothing in
            // this epilogue depends on how an accumulator backend assigns rows to
            // warps or on the shape of its register-memory layout.
            const int tid = static_cast<int>(threadIdx.x);

            // Bulk-async groups are per-thread, so only the previous issuer can have
            // an outstanding group. Synchronize all epilogue threads after that wait
            // before reusing the shared staging tile.
            tma_store::wait_read();
            tile_pipeline.epilogue_sync();
            static_assert(Traits::threads >= tile_m,
                          "counter initialization requires at least one epilogue thread per row");
            if (tid < tile_m) {
                packed[tid * packed_pitch] = 0u;
            }
            tile_pipeline.epilogue_sync();

            // Map fragment elements to (m, n) through the accumulator API.
            auto fragment = accumulator.get_results();

            // Keep each fragment index compile-time so map_fragment_index can fold the
            // backend's per-value mapping without exposing that backend's layout.
            cute::for_each(cute::make_int_sequence<cute::size(fragment)> {}, [&](auto I) {
                constexpr int i = decltype(I)::value;
                auto const coord = accumulator.map_fragment_index(I);
                const int row = cute::get<0>(coord);
                const int col = col_base + cute::get<1>(coord);
                uint32_t* const row_ptr = packed + row * packed_pitch;
                const float value = fragment(i);
                if (value > 0.0f) {
                    // The mapping may assign a row to any epilogue thread. A
                    // block-scoped CUDA atomic is therefore the only ownership
                    // contract needed by the packing algorithm.
                    const uint32_t slot = atomicAdd_block(row_ptr, 1u);
                    // Columns occupy the low 16 bits; shape validation bounds N.
                    const uint32_t entry =
                        static_cast<uint32_t>(col) |
                        (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(value))) << 16);
                    // store the nonzero index and value packed
                    if constexpr (LOOP_OVERFLOW_STORAGE) {
                        // The counter stays uncapped so overflow is detectable; the
                        // write index wraps. Index t_n_comp lands in the pad and is
                        // dropped by the store's bounds check.
                        row_ptr[(slot & (t_n_comp - 1)) + 1] = entry;
                    } else {
                        row_ptr[slot + 1] = entry;
                    }
                }
            });

            // Publish the staging tile to the async proxy before the bulk store
            // reads it.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            // All epilogue threads must have finished writing before the single
            // store reads the whole staging tile. Let the pipeline supply the right
            // synchronization scope and participant count for its backend.
            tile_pipeline.epilogue_sync();
            if (issue_tma_store) {
                tma_store::issue(&c_map,
                                 packed,
                                 static_cast<uint32_t>(tile_row) * tile_m,
                                 static_cast<uint32_t>(tile_col));
            }
            // No trailing barrier: wait_read() at the top of the next iteration
            // guards the staging tile instead.
        });

        ++it;
        if (it == schedule_size_per_sm) break;
        const int c = coord_next;
        if (c == -1) break;
        tile_row = (c >> 16) * CLUSTER_DIM_m + cluster_lane_m;
        tile_col = (c & 0xFFFF) * CLUSTER_DIM_n + cluster_lane_n;
        device_pipeline.reset_tile(tile_pipeline, tile_row, tile_col);
        coord_next = (it + 1 < schedule_size_per_sm) ? schedule_gmem_ptr[it + 1] : -1;
    }

    // The final tile's store is still reading the staging tile; the CTA must not
    // retire and hand that shared memory to the next CTA until TMA has drained.
    // Only the store issuer has an outstanding bulk-async group to drain.
    tma_store::wait_read();
}

// Build the packed-output map without device allocation, including during capture.
template<int TnComp>
void encode_packed_c_map(CUtensorMap* map, uint32_t* c_packed, int M, int N) {
    using geom = packed_geometry<TnComp>;
    tma_store::encode<geom::packed_pitch, geom::tile_m>(
        map, c_packed, M, (N / geom::tile_n) * TnComp, TnComp);
}

template<class Traits, class Pipeline>
unsigned shared_bytes(const Pipeline& pipeline) {
    return cublasdx::make_shared_storage_calculator()
        .add(pipeline.buffer_alignment(), pipeline.buffer_size())
        .add(128, sizeof(uint32_t), Traits::packed_elems)
        .get();
}

// Query once while preparing a layer, outside graph capture. No kernel is launched:
// the weight buffer supplies the addresses for a minimum-height probe descriptor.
// Residency depends on the kernel, block size and shared storage, not matrix M.
template<int TnComp, int ClusterM, int ClusterN>
int resident_clusters(const __nv_bfloat16* weights, int N, int K) {
    using traits = gemm_traits<TnComp, ClusterM, ClusterN>;
    using BLAS = typename traits::BLAS;
    auto a = cublasdx::make_gmem_tensor<cublasdx::row_major>(weights, traits::tile_m * ClusterM, K, K);
    auto b = cublasdx::make_gmem_tensor<cublasdx::col_major>(weights, K, N, K);
    auto pipeline = cublasdx::suggest_pipeline<traits::depth, BLAS>(a, b);
    TORCH_CHECK(pipeline, "D2T pipeline requires N divisible by the cluster tile and enough K stages");
    const unsigned smem = shared_bytes<traits>(*pipeline);
    auto kernel = mm_wgmma_nt_kernel<traits, ClusterM, ClusterN, true, decltype(pipeline->get_device_handle())>;
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "Cannot configure D2T shared memory");
    int device = 0, sms = 0, clusters = 0;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess &&
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess,
                "Cannot query D2T device");
    constexpr int size = ClusterM * ClusterN;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3((sms / size) * size);
    config.blockDim = pipeline->get_block_dim();
    config.dynamicSmemBytes = smem;
    // The kernel has a static one-dimensional cluster declaration.
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) == cudaSuccess && clusters > 0,
                "Cannot query resident D2T clusters");
    return clusters;
}

// Launches the mainloop. `schedule` is a device pointer to n_active_clusters slices of
// schedule_size_per_sm packed cluster coordinates each, padded with -1; one
// CLUSTER_DIM_m x CLUSTER_DIM_n cluster of CTAs is launched per partition. Returns false
// without launching when the shape does not fit the tile or cluster or K is too
// short to fill the pipeline.
template<int TnComp, int CLUSTER_DIM_m, int CLUSTER_DIM_n, bool LOOP_OVERFLOW_STORAGE>
bool launch(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const CUtensorMap& c_map,
    const int* schedule,
    const int schedule_size_per_sm,
    const int n_active_clusters,
    const int M, const int N, const int K,
    cudaStream_t stream
) {
    using traits = gemm_traits<TnComp, CLUSTER_DIM_m, CLUSTER_DIM_n>;
    using BLAS = typename traits::BLAS;

    // cuBLASDx requires complete tiles and enough K tiles to fill the pipeline.
    if (M % traits::tile_m != 0 || N % traits::tile_n != 0 || K % traits::tile_k != 0) return false;
    if (K / traits::tile_k < traits::depth) return false;
    // The schedule packs each coordinate into 16 bits.
    if (M <= 0 || N <= 0 || K <= 0 || M / traits::tile_m > 0xFFFF || N > 65536) return false;
    // Every cluster must expand to real tiles in both dimensions.
    if ((M / traits::tile_m) % CLUSTER_DIM_m != 0 || (N / traits::tile_n) % CLUSTER_DIM_n != 0) return false;

    auto global_a = cublasdx::make_gmem_tensor<cublasdx::row_major>(a, M, K, K);
    auto global_b = cublasdx::make_gmem_tensor<cublasdx::col_major>(b, K, N, K);

    // Descriptors embed A/B pointers, so rebuild them for each invocation.
    auto pipeline = cublasdx::suggest_pipeline<traits::depth, BLAS>(global_a, global_b, stream);
    if (not pipeline) return false;

    const unsigned smem_bytes = shared_bytes<traits>(*pipeline);

    auto kernel = mm_wgmma_nt_kernel<traits, CLUSTER_DIM_m, CLUSTER_DIM_n, LOOP_OVERFLOW_STORAGE, decltype(pipeline->get_device_handle())>;

    // Opt in above the 48 KB dynamic shared-memory default.
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess) {
        return false;
    }

    // __cluster_dims__ makes this a clustered launch: one cluster per schedule
    // partition. CTAs with empty partitions return immediately.
    kernel<<<dim3(n_active_clusters * CLUSTER_DIM_m * CLUSTER_DIM_n), pipeline->get_block_dim(), smem_bytes, stream>>>(
        schedule, schedule_size_per_sm, c_map, pipeline->get_device_handle());

    // Clears the sticky error as well as reporting it, so a refusal here does not
    // surface later as an unrelated failure in the caller.
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cublasdx_d2t
}  // namespace TWELL_D2T

namespace TWELL_D2T {

// Allocate schedules during preparation, outside CUDA graph capture. The public
// 2x1 cluster requires M divisible by 256; partial cluster tiles are unsupported.

struct D2TLayerCache {
    CUtensorMap C_packed_tm{};
    __nv_bfloat16* A_ptr = nullptr;
    __nv_bfloat16* B_ptr = nullptr;
    uint32_t* C_packed_ptr = nullptr;
    bool runtime_maps_initialized = false;
    int M = -1;
    int K = -1;
    int N = -1;
    int T_m = -1;
    int T_n = -1;
    int T_k = -1;
    int T_n_compressed = -1;
    int num_active_clusters = 0;
};

struct D2TSharedScheduleCache {
    int* partitioned_schedule_d = nullptr;
    int schedule_size_per_sm = -1;
    int schedule_num_clusters_m = -1;
    int schedule_num_clusters_n = -1;
    int schedule_num_active_clusters = -1;
};

static std::unordered_map<int, D2TLayerCache> CACHED_D2T_LAYERS;
static D2TSharedScheduleCache SHARED_D2T_SCHEDULE;

// releases the cached per-shape schedule buffer shared across D2T layers.
// inputs: none. output: schedule cache reset and device memory freed if needed.
static void release_shared_schedule_cache() {
    if (SHARED_D2T_SCHEDULE.partitioned_schedule_d != nullptr) {
        cudaError_t status = cudaFree(SHARED_D2T_SCHEDULE.partitioned_schedule_d);
        if (status != cudaSuccess) {
            printf("cudaFree for cached schedule failed: %s\n", cudaGetErrorString(status));
        }
        SHARED_D2T_SCHEDULE.partitioned_schedule_d = nullptr;
    }
    SHARED_D2T_SCHEDULE.schedule_size_per_sm = -1;
    SHARED_D2T_SCHEDULE.schedule_num_clusters_m = -1;
    SHARED_D2T_SCHEDULE.schedule_num_clusters_n = -1;
    SHARED_D2T_SCHEDULE.schedule_num_active_clusters = -1;
}

// initializes the static cache for one D2T layer configuration.
// inputs: layer cache, B pointer, logical K/N. output: layer metadata refreshed.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int T_n_compressed = T_n / 8
>
void prepare_d2t_layer_static_cache(
    D2TLayerCache& cache,
    at::BFloat16* B_d_at,
    const int K, const int N
)
{
    __nv_bfloat16* B_d = reinterpret_cast<__nv_bfloat16*>(B_d_at);
    static_assert(
        (T_k == 64),
        "Only T_k == 64 supported"
    );

    static_assert(
        (T_m == 64 * 2),
        "Only T_m == 128 supported"
    );
    static_assert(
        (T_n == 64 * 4),
        "Only T_n == 256 supported"
    );
    static_assert(
        (CLUSTER_DIM_m * CLUSTER_DIM_n <= 8),
        "Cluster size cannot exceed 8 (CUDA limit)"
    );
    TORCH_CHECK(N > 0 && N <= 65536 && N % (T_n * CLUSTER_DIM_n) == 0,
                "D2T requires positive N <= 65536 divisible by the cluster tile width");
    TORCH_CHECK(K > 0 && K % T_k == 0, "D2T requires positive K divisible by 64");
    // The pipeline depth, and so the minimum K, depends on the target SM's shared memory.
    constexpr int min_k = cublasdx_d2t::gemm_traits<T_n_compressed, CLUSTER_DIM_m, CLUSTER_DIM_n>::depth * T_k;
    TORCH_CHECK(K >= min_k, "D2T requires K >= ", min_k, " to fill the cuBLASDx pipeline on sm_", TWELL_D2T_SM / 10, "a");

    if (cache.B_ptr != B_d ||
        cache.N != N || cache.K != K ||
        cache.T_m != T_m || cache.T_n != T_n || cache.T_k != T_k ||
        cache.T_n_compressed != T_n_compressed)
    {
        cache.num_active_clusters =
            cublasdx_d2t::resident_clusters<T_n_compressed, CLUSTER_DIM_m, CLUSTER_DIM_n>(B_d, N, K);
        cache.B_ptr = B_d;
        cache.M = -1;
        cache.N = N;
        cache.K = K;
        cache.T_m = T_m;
        cache.T_n = T_n;
        cache.T_k = T_k;
        cache.T_n_compressed = T_n_compressed;
        cache.A_ptr = nullptr;
        cache.C_packed_ptr = nullptr;
        cache.runtime_maps_initialized = false;
    }
}

// updates runtime tensor-map addresses for the current A and packed-C buffers.
// inputs: layer cache, A pointer, packed-C pointer. output: runtime tensor maps ready to launch.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int T_n_compressed = T_n / 8
>
void update_d2t_layer_runtime_maps(
    D2TLayerCache& cache,
    at::BFloat16* A_d_at,
    uint32_t* C_packed_d
)
{
    __nv_bfloat16* A_d = reinterpret_cast<__nv_bfloat16*>(A_d_at);
    // cuBLASDx builds A/B descriptors at launch. Cache the packed-output map here.
    if (!cache.runtime_maps_initialized || cache.C_packed_ptr != C_packed_d) {
        cublasdx_d2t::encode_packed_c_map<T_n_compressed>(
            &cache.C_packed_tm,
            C_packed_d,
            cache.M,
            cache.N
        );
        cache.runtime_maps_initialized = true;
    }
    cache.A_ptr = A_d;
    cache.C_packed_ptr = C_packed_d;
}

// ensures the cached schedule and shape-dependent state match the requested M.
// inputs: layer cache, logical M. output: schedule cache and runtime invalidation updated.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int T_n_compressed = T_n / 8
>
void ensure_d2t_layer_shape(
    D2TLayerCache& cache,
    const int M
)
{
    TORCH_CHECK(M > 0 && M % (T_m * CLUSTER_DIM_m) == 0 && M / T_m <= 65535,
                "D2T requires positive M divisible by the cluster tile height and a 16-bit tile index");
    const int NUM_ACTIVE_CLUSTERS = cache.num_active_clusters;
    TORCH_CHECK(NUM_ACTIVE_CLUSTERS > 0, "Prepare the D2T layer before its shape");
    const int num_clusters_m = (M + T_m - 1) / (T_m * CLUSTER_DIM_m);
    const int num_clusters_n = (cache.N + T_n - 1) / (T_n * CLUSTER_DIM_n);
    const int schedule_size_per_sm =
        (num_clusters_m * num_clusters_n + NUM_ACTIVE_CLUSTERS - 1)
        / NUM_ACTIVE_CLUSTERS;
    assert((M / T_m) % CLUSTER_DIM_m == 0);
    const bool reuse_shared_schedule =
        SHARED_D2T_SCHEDULE.partitioned_schedule_d != nullptr &&
        SHARED_D2T_SCHEDULE.schedule_size_per_sm == schedule_size_per_sm &&
        SHARED_D2T_SCHEDULE.schedule_num_clusters_m == num_clusters_m &&
        SHARED_D2T_SCHEDULE.schedule_num_clusters_n == num_clusters_n &&
        SHARED_D2T_SCHEDULE.schedule_num_active_clusters == NUM_ACTIVE_CLUSTERS;

    if (!reuse_shared_schedule) {
        const size_t partitioned_schedule_total_size =
            static_cast<size_t>(NUM_ACTIVE_CLUSTERS) *
            static_cast<size_t>(schedule_size_per_sm);
        const size_t partitioned_schedule_bytes =
            partitioned_schedule_total_size * sizeof(int);
        std::vector<int> schedule_h(partitioned_schedule_total_size);
        build_partitioned_hilbert_schedule_for_grid(
            num_clusters_m,
            num_clusters_n,
            NUM_ACTIVE_CLUSTERS,
            schedule_size_per_sm,
            schedule_h.data()
        );

        release_shared_schedule_cache();
        cudaError_t status = cudaMalloc(
            reinterpret_cast<void**>(&SHARED_D2T_SCHEDULE.partitioned_schedule_d),
            partitioned_schedule_bytes
        );
        if (status != cudaSuccess) {
            printf("cudaMalloc for schedule failed: %s\n", cudaGetErrorString(status));
            exit(-1);
        }

        status = cudaMemcpy(
            SHARED_D2T_SCHEDULE.partitioned_schedule_d,
            schedule_h.data(),
            partitioned_schedule_bytes,
            cudaMemcpyHostToDevice
        );
        if (status != cudaSuccess) {
            printf("cudaMemcpy for schedule failed: %s\n", cudaGetErrorString(status));
            exit(-1);
        }

        SHARED_D2T_SCHEDULE.schedule_size_per_sm = schedule_size_per_sm;
        SHARED_D2T_SCHEDULE.schedule_num_clusters_m = num_clusters_m;
        SHARED_D2T_SCHEDULE.schedule_num_clusters_n = num_clusters_n;
        SHARED_D2T_SCHEDULE.schedule_num_active_clusters = NUM_ACTIVE_CLUSTERS;
    }

    if (cache.M != M) {
        cache.M = M;
        cache.A_ptr = nullptr;
        cache.C_packed_ptr = nullptr;
        cache.runtime_maps_initialized = false;
    }
}

// launches the cached D2T kernel for the current layer state.
// inputs: fully prepared layer cache. output: kernel dispatched on the current CUDA stream.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int T_n_compressed = T_n / 8,
    const bool LOOP_OVERFLOW_STORAGE
>
void run_d2t_layer_cache(
    const D2TLayerCache& cache,
    cudaStream_t stream = 0
)
{
    const bool launched = cublasdx_d2t::launch<T_n_compressed, CLUSTER_DIM_m, CLUSTER_DIM_n, LOOP_OVERFLOW_STORAGE>(
        cache.A_ptr,
        cache.B_ptr,
        cache.C_packed_tm,
        SHARED_D2T_SCHEDULE.partitioned_schedule_d,
        SHARED_D2T_SCHEDULE.schedule_size_per_sm,
        SHARED_D2T_SCHEDULE.schedule_num_active_clusters,
        cache.M, cache.N, cache.K,
        stream
    );
    if (!launched) {
        printf("D2T GEMM declined %dx%dx%d: M, N and K must be multiples of %d, %d and %d, "
               "K must cover the pipeline depth, and the device must be sm_%da\n",
               cache.M, cache.N, cache.K, T_m, T_n, T_k, TWELL_D2T_SM / 10);
        exit(-1);
    }
}

// creates or refreshes the cached 128x256x64 TS8 D2T layer metadata.
// inputs: layer id, B pointer, logical K/N. output: static layer cache prepared.
void create_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* B_d,
    const int K, const int N
) {
    auto& layer_cache = CACHED_D2T_LAYERS[layer_number];
    prepare_d2t_layer_static_cache<128, 256, 64, 2, 1, 32>(
        layer_cache,
        B_d,
        K, N
    );
}

// creates or refreshes the cached 128x256x64 TS4 D2T layer metadata.
// inputs: layer id, B pointer, logical K/N. output: static layer cache prepared.
void create_d2t_layer_128x256x64TS4(
    const int layer_number,
    at::BFloat16* B_d,
    const int K, const int N
) {
    auto& layer_cache = CACHED_D2T_LAYERS[layer_number];
    prepare_d2t_layer_static_cache<128, 256, 64, 2, 1, 64>(
        layer_cache,
        B_d,
        K, N
    );
}

// creates or refreshes the cached 128x256x64 TS2 D2T layer metadata.
// inputs: layer id, B pointer, logical K/N. output: static layer cache prepared.
void create_d2t_layer_128x256x64TS2(
    const int layer_number,
    at::BFloat16* B_d,
    const int K, const int N
) {
    auto& layer_cache = CACHED_D2T_LAYERS[layer_number];
    prepare_d2t_layer_static_cache<128, 256, 64, 2, 1, 128>(
        layer_cache,
        B_d,
        K, N
    );
}

// ensures the cached 128x256x64 TS8 layer shape matches the requested M.
// inputs: layer id, logical M. output: shape-dependent schedule state prepared.
void ensure_d2t_layer_shape_128x256x64TS8(
    const int layer_number,
    const int M
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("ensure_d2t_layer_shape called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    ensure_d2t_layer_shape<128, 256, 64, 2, 1, 32>(
        it->second,
        M
    );
}

// ensures the cached 128x256x64 TS4 layer shape matches the requested M.
// inputs: layer id, logical M. output: shape-dependent schedule state prepared.
void ensure_d2t_layer_shape_128x256x64TS4(
    const int layer_number,
    const int M
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("ensure_d2t_layer_shape called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    ensure_d2t_layer_shape<128, 256, 64, 2, 1, 64>(
        it->second,
        M
    );
}

// ensures the cached 128x256x64 TS2 layer shape matches the requested M.
// inputs: layer id, logical M. output: shape-dependent schedule state prepared.
void ensure_d2t_layer_shape_128x256x64TS2(
    const int layer_number,
    const int M
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("ensure_d2t_layer_shape called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    ensure_d2t_layer_shape<128, 256, 64, 2, 1, 128>(
        it->second,
        M
    );
}

// runs the cached 128x256x64 TS8 D2T layer for one input/output pair.
// inputs: layer id, A pointer, packed-C pointer. output: packed positive activations written.
void run_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("run_d2t_layer called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    update_d2t_layer_runtime_maps<128, 256, 64, 2, 1, 32>(
        it->second,
        A_d,
        C_packed_d
    );
    run_d2t_layer_cache<128, 256, 64, 2, 1, 32, true>(
        it->second,
        stream
    );
}

// runs the cached 128x256x64 TS4 D2T layer for one input/output pair.
// inputs: layer id, A pointer, packed-C pointer. output: packed positive activations written.
void run_d2t_layer_128x256x64TS4(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("run_d2t_layer called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    update_d2t_layer_runtime_maps<128, 256, 64, 2, 1, 64>(
        it->second,
        A_d,
        C_packed_d
    );
    run_d2t_layer_cache<128, 256, 64, 2, 1, 64, true>(
        it->second,
        stream
    );
}

// runs the cached 128x256x64 TS2 D2T layer for one input/output pair.
// inputs: layer id, A pointer, packed-C pointer. output: packed positive activations written.
void run_d2t_layer_128x256x64TS2(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("run_d2t_layer called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    update_d2t_layer_runtime_maps<128, 256, 64, 2, 1, 128>(
        it->second,
        A_d,
        C_packed_d
    );
    run_d2t_layer_cache<128, 256, 64, 2, 1, 128, true>(
        it->second,
        stream
    );
}


// destroys one cached D2T layer entry.
// inputs: layer id. output: layer cache removed and shared schedule freed if now unused.
void destroy_d2t_layer(const int layer_number) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        return;
    }
    CACHED_D2T_LAYERS.erase(it);
    if (CACHED_D2T_LAYERS.empty()) {
        release_shared_schedule_cache();
    }
}

// destroys all cached D2T layer entries and the shared schedule cache.
// inputs: none. output: all cached D2T state released.
void destroy_all_d2t_layers() {
    CACHED_D2T_LAYERS.clear();
    release_shared_schedule_cache();
}

// runs the 128x256x64 TS8 D2T path one-shot without explicit cache management.
// inputs: A pointer, B pointer, packed-C pointer, logical M/K/N. output: packed positive activations written.
void mm_wgmma_nt_128x256x64TS8(
    at::BFloat16* A_d, at::BFloat16* B_d, uint32_t* C_packed_d,
    const int M, const int K, const int N,
    cudaStream_t stream
) {
    constexpr int LEGACY_LAYER_ID = -1;
    create_d2t_layer_128x256x64TS8(
        LEGACY_LAYER_ID,
        B_d,
        K, N
    );
    ensure_d2t_layer_shape_128x256x64TS8(
        LEGACY_LAYER_ID,
        M
    );
    run_d2t_layer_128x256x64TS8(
        LEGACY_LAYER_ID,
        A_d,
        C_packed_d,
        stream
    );
}

// runs the 128x256x64 TS4 D2T path one-shot without explicit cache management.
// inputs: A pointer, B pointer, packed-C pointer, logical M/K/N. output: packed positive activations written.
void mm_wgmma_nt_128x256x64TS4(
    at::BFloat16* A_d, at::BFloat16* B_d, uint32_t* C_packed_d,
    const int M, const int K, const int N,
    cudaStream_t stream
) {
    constexpr int LEGACY_LAYER_ID = -1;
    create_d2t_layer_128x256x64TS4(
        LEGACY_LAYER_ID,
        B_d,
        K, N
    );
    ensure_d2t_layer_shape_128x256x64TS4(
        LEGACY_LAYER_ID,
        M
    );
    run_d2t_layer_128x256x64TS4(
        LEGACY_LAYER_ID,
        A_d,
        C_packed_d,
        stream
    );
}

// runs the 128x256x64 TS2 D2T path one-shot without explicit cache management.
// inputs: A pointer, B pointer, packed-C pointer, logical M/K/N. output: packed positive activations written.
void mm_wgmma_nt_128x256x64TS2(
    at::BFloat16* A_d, at::BFloat16* B_d, uint32_t* C_packed_d,
    const int M, const int K, const int N,
    cudaStream_t stream
) {
    constexpr int LEGACY_LAYER_ID = -1;
    create_d2t_layer_128x256x64TS2(
        LEGACY_LAYER_ID,
        B_d,
        K, N
    );
    ensure_d2t_layer_shape_128x256x64TS2(
        LEGACY_LAYER_ID,
        M
    );
    run_d2t_layer_128x256x64TS2(
        LEGACY_LAYER_ID,
        A_d,
        C_packed_d,
        stream
    );
}

}
