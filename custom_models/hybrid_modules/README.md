# hybrid_modules

Training-time CUDA kernels for sparse gated MLPs (Llama, Qwen) on H100, exposed to PyTorch as `torch.ops.sparse_ops.*`. Activations and weights live in a hybrid ELL + dense-tail format. See `CLAUDE.md` for kernel architecture, `KNOBS.md` for the runtime tuning API.

## Build / loading

JIT-compiled on first call to `load_sparse_ops()`; no manual build step.

```python
from sparse_ops_loader import load_sparse_ops
load_sparse_ops()                      # registers torch.ops.sparse_ops.*
```

The loader honors `TORCH_CUDA_ARCH_LIST` (default `9.0a`). On non-Hopper targets `matmul_d2t.cu` is dropped from the source list and `wgmma_gemm.cu` self-gates the WGMMA call via `__CUDA_ARCH_LIST__`, so `ff_forward_gated` falls back to einsum + `create_hybrid_sparse_from_dense`.

### Pre-built `.so` (no nvcc on the load machine)

```bash
# build machine
cd custom_models/hybrid_modules && python setup.py build

# runtime machine
export SPARSE_OPS_PREBUILT=/path/to/sparse_ops.so
```

When `SPARSE_OPS_PREBUILT` is set, the loader skips JIT and calls `torch.ops.load_library()` on the given path.

### Environment variables

| Var | Purpose |
|---|---|
| `SPARSE_OPS_PREBUILT` | Full path to a prebuilt `sparse_ops.so`. Skips JIT compilation. |
| `TORCH_CUDA_ARCH_LIST` | Standard PyTorch arch list. Default `9.0a`. |

## Quick start

```python
import torch
from sparse_ops_loader import load_sparse_ops
load_sparse_ops()

B_S, D, H = 16384, 2048, 5632

# This is illustrative, matrices should be created such that Relu(X@G) is sparse
X = torch.randn(B_S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
G = torch.randn(H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
K = torch.randn(H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
V = torch.randn(H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

out, l0, l1, _, _, _ = torch.ops.sparse_ops.ff_forward_gated(X, G, K, V)
(out.sum() + 5.7 * l1).backward()      # autograd dispatches ff_backward_gated
```

- Inputs are bf16. Weight shapes are `[H, D]` (out-major), not `[D, H]`.
- `l0`, `l1` are scalar aux losses — fold into the training loss as needed.
- The trailing `(P, R, T)` are opaque `HybridSp` saved-state for backward; ignore.

## Public ops

| Op | Purpose |
|---|---|
| `torch.ops.sparse_ops.ff_forward_gated(X, G, K, V)` | Fused gated MLP. |
| `torch.ops.sparse_ops.ff_backward_gated(...)` | Driven by autograd. |
| `torch.ops.sparse_ops_config.*` | Runtime knobs — see `KNOBS.md`. |

## `down_proj.weight` layout

The kernel expects `down_proj.weight` shape `[intermediate, hidden]`, opposite of `nn.Linear`'s default `[hidden, intermediate]`. Mismatched layout produces silently wrong results, not a shape error.

- **From-scratch** (`SparseLlamaForCausalLM(config)` etc.): `SparseMLP.__init__` transposes at construction when `sparsity_use_hybrid_kernel=True`. No action needed.
- **HF checkpoint**: `hydra_utils.load_model(from_pretrained=True, ...)` re-applies the transpose after weights are loaded. By hand:
  ```python
  for m in model.modules():
      if hasattr(m, "down_proj"):
          m.down_proj.weight.data = m.down_proj.weight.data.t().contiguous()
  ```

State-dict round-trips between HF / sparse / TwELL formats are handled by `_convert_sparse_state_dict` in `../sparse_testing_utils.py`.

## Where things live

- HF model wrappers (`SparseLlamaForCausalLM`, `SparseQwen2ForCausalLM`), `SparseMLP`, `SparsityTracker`: `../sparse_models.py`.
- Hybrid-kernel toggle (`sparsity_use_hybrid_kernel`, `sparsity_l0_cutoff`, …): set via Hydra; see the `cfgs/run_cfg/sparsity_gated_hybrid_*.yaml` configs.
