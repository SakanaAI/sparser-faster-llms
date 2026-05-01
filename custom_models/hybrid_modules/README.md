# hybrid_modules

Training-time CUDA kernels for sparse gated MLPs (Llama, Qwen) on H100. Weights and activations live in a hybrid ELL + dense-tail format and are exposed to PyTorch as `torch.ops.sparse_ops.*`. See `CLAUDE.md` for kernel-level architecture; the runtime tuning API lives in [Runtime configuration](#runtime-configuration) below.

## Build

```bash
cd custom_models/hybrid_modules
python setup.py build
```

- Output: `build/lib.linux-x86_64-3.10/sparse_ops.so`
- GPU target: H100 / sm_90a.

## Loading the extension

No install step. Either:

```python
import torch
torch.ops.load_library("build/lib.linux-x86_64-3.10/sparse_ops.so")
```

or

```python
import sys; sys.path.insert(0, "build/lib.linux-x86_64-3.10")
import sparse_ops  # registers torch.ops.sparse_ops.*
```

## Quick start: gated MLP forward + backward

```python
import torch
torch.ops.load_library("build/lib.linux-x86_64-3.10/sparse_ops.so")

B_S, D, H = 16384, 2048, 5632  # tokens, hidden, intermediate
X = torch.randn(B_S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# G should be picked so that Relu(X @ G) is highly sparse
G = torch.randn(H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
K = torch.randn(H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
V = torch.randn(H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

out, l0, l1, _, _, _ = torch.ops.sparse_ops.ff_forward_gated(X, G, K, V, B_S * D)

loss = out.sum() + 5.7 * l1
loss.backward()  # autograd dispatches ff_backward_gated
```

- Inputs are bf16. Weight shapes are `[H, D]` (out-major), not `[D, H]`.
- `l0`, `l1` are scalar aux losses (sparsity fraction and gate magnitude). Add them to your training loss as needed.
- The three trailing returns are opaque `HybridSp` saved-state objects for backward. Ignore them.

## Public ops

| Op | Purpose |
|---|---|
| `torch.ops.sparse_ops.ff_forward_gated(X, G, K, V)` | Fused gated MLP: gate proj, sparse key proj, sparse elementwise, sparse-dense GEMM down. |
| `torch.ops.sparse_ops.ff_backward_gated(...)` | Driven by autograd. Not called directly. |
| `torch.ops.sparse_ops.ell_spmm_raw(...)` | Low-level ELL x dense GEMM. See `benchmark_spmm.py`. |
| `torch.ops.sparse_ops_config.set_*` | Runtime tunables; see [Runtime configuration](#runtime-configuration). |
| `torch.ops.sparse_ops_perf.{reset_stats,print_report}` | Active only when built with `ENABLE_PROFILING=1`. |

## sparse_models.py

Thin sparse wrappers around HuggingFace causal LMs. Public surface:

- `SparseMLP` - drop-in gated FFN. Uses einsum during training (for gradient correctness) and `ff_forward_gated` for inference paths.
- `SparseLlamaForCausalLM`, `SparseQwen2ForCausalLM` - HF-compatible models built via `_create_sparse_model_class`. They walk the model and replace each MLP with `SparseMLP` through `replace_mlp_modules()`.
- `SparsityTracker`, `NullTracker` - track per-neuron activation, expose L0/L1 penalties for the auxiliary loss.

### `down_proj.weight` layout

The sparse kernels expect `down_proj.weight` shape `[intermediate_size, hidden_size]`, opposite of `nn.Linear`'s default `[hidden_size, intermediate_size]`. Each row maps to one intermediate-dim neuron, which is what the row-wise SpMM iterates over. Mismatched layout produces silently wrong results, not a shape error.

Two cases:

- **From-scratch model** (`SparseLlamaForCausalLM(config)` directly, or `load_model(from_pretrained=False, custom_class=...)`): `SparseMLP.__init__` transposes `down_proj.weight.data` at construction, so the model is already in the right layout. No action needed.
- **HF checkpoint**: state dict arrives in HF layout and must be transposed once after loading. `hydra_utils.load_model(from_pretrained=True, ...)` does this for you (see `hydra_utils.py:56-117`). If you load a checkpoint by hand, apply:

  ```python
  for module in model.modules():
      if hasattr(module, "down_proj"):
          module.down_proj.weight.data = module.down_proj.weight.data.t().contiguous()
  ```

### Use `hydra_utils.load_model`

It loads via `AutoModelForCausalLM.from_pretrained` or instantiates a `custom_class` like `SparseLlamaForCausalLM` from a config, and only applies the HF -> sparse transpose when `from_pretrained=True`.

Hydra config:

```yaml
_target_: custom_models.hybrid_modules.hydra_utils.load_model
from_pretrained: true
model_args:
  model_name_or_path: SakanaAI/SparseLM1.5B
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
custom_class:
  _target_: custom_models.hybrid_modules.sparse_models.SparseLlamaForCausalLM
```

State-dict round-trips between HF, sparse, and TwELL formats are handled by `_convert_sparse_state_dict` in `../sparse_testing_utils.py`, which applies the same transpose for `down_proj.weight` / `down_weight` so checkpoints stay consistent across formats.

## Runtime configuration

All settings live in `torch.ops.sparse_ops_config` and take effect immediately for the **next allocated** sparse object or the next kernel launch. They do **not** retroactively change already-allocated buffers.

Profiling lives in a separate namespace, `torch.ops.sparse_ops_perf`, and only does anything when the extension was built with `ENABLE_PROFILING=1`.

### ELL width

The ELL part of a `hybrid_sp_t` object stores the first N non-zeros per row. Rows whose true NNZ exceeds the ELL width either overflow into the dense tail (default) or are discarded (see [Execution modes](#execution-modes)).

Regular and transposed sparse matrices are separate objects with independently configurable ELL widths.

```python
# Default: 128 for both
torch.ops.sparse_ops_config.set_ell_width_regular(v)    # forward ops: gate, key projection, elementwise
torch.ops.sparse_ops_config.set_ell_width_transpose(v)  # backward ops: transposed ELL (T^T, dR^T, dU^T)
```

- `v` - positive integer, number of non-zeros per row stored in the ELL buffer.
- The value becomes both the buffer stride *and* the overflow threshold for all objects constructed after this call.
- Default: `128` (defined as `ELL_WIDTH` in `constants.h`).

**When to tune:** narrowing the ELL width (e.g. to 64) reduces memory traffic and shared memory usage, at the cost of more rows spilling to the dense tail. Widening it (e.g. to 256) is only useful if the actual NNZ distribution has a long tail above 128.

### Dense tail capacity

Rows that overflow ELL are stored densely. The tail is pre-allocated to hold at most `tail_rows` overflow rows.

```python
# Default: 2048 for both
torch.ops.sparse_ops_config.set_tail_rows_regular(v)    # capacity for regular objects
torch.ops.sparse_ops_config.set_tail_rows_transpose(v)  # capacity for transposed objects
```

- `v` - positive integer, maximum number of overflow rows.
- Default: `2048` (defined as `TAIL_CAPACITY_ROWS` in `constants.h`).
- If more than `v` rows overflow, the extras are **counted but not stored**: they are silently dropped. Check `overflow_count_tensor()` to detect this.

### Execution modes

#### Hybrid ELL + dense (default)

Overflow rows are stored in a dense tail matrix and included in all computations. This is the default and produces correct results regardless of sparsity.

```python
torch.ops.sparse_ops_config.set_discard_overflow(0)  # default
```

#### Discard overflow

Overflow rows are **counted but discarded**: neither stored nor used in matmuls. This sacrifices correctness for rows that exceed the ELL width but avoids all dense-tail GPU work.

```python
torch.ops.sparse_ops_config.set_discard_overflow(1)
```

Use this mode for:

- **Benchmarking** the ELL-only path in isolation.
- **Highly sparse models** where overflow is rare and the accuracy trade-off is acceptable (verify with `overflow_count_tensor()`).

### Checking overflow

#### After the forward pass

```python
out, l0, l1, P, R, T = model(x)   # or direct ff_forward_gated call

# P is the gate sparsity object; synchronize before reading
torch.cuda.synchronize()
overflow_count = P.overflow_count_tensor().item()
print(f"Gate overflow rows: {overflow_count}")
```

`P.overflow_count_tensor()` returns a scalar `int32` CUDA tensor containing the number of rows whose NNZ exceeded the regular ELL width during gate construction. A value of 0 means all rows fit in ELL.

#### After the backward pass

The backward pass transposes the sparse matrices. To check how many rows overflowed during transposition:

```python
loss.backward()

torch.cuda.synchronize()
t_overflow = torch.ops.sparse_ops_config.get_last_transpose_overflow_count().item()
print(f"Transpose overflow rows: {t_overflow}")
```

This returns the overflow counter from the most recent `transpose_hybrid_dense` call in the backward pass (`T^T`). It reflects rows that exceeded the transpose ELL width.

### ELL creation kernel tuning

Controls the dense-to-ELL conversion kernel used in the non-H100 fallback path (`create_hybrid_sparse_from_dense`). On H100 the WGMMA fused path is used instead.

```python
torch.ops.sparse_ops_config.set_ell_create_warps_per_row(N)
# N=0: one warp per row, multiple rows per block (default)
# N=1,2,4,8: N warps cooperate on a single row (faster for wide rows)
```

### Per-object properties (C++)

Each `hybrid_sp_t` object stores its own allocation parameters, set at construction time:

| Field | Type | Description |
|---|---|---|
| `_ell_stride` | `int` | ELL buffer width = overflow threshold for this object |
| `_tail_cap` | `int` | Dense tail capacity for this object |
| `_dense_active_rows` | `int` | How many tail rows are currently populated |

These are fixed at construction and do not change when the global setters are called later.

### Defaults

| Setting | Default | Setter |
|---|---|---|
| ELL width (regular) | 128 | `set_ell_width_regular(v)` |
| ELL width (transpose) | 128 | `set_ell_width_transpose(v)` |
| Tail capacity (regular) | 2048 | `set_tail_rows_regular(v)` |
| Tail capacity (transpose) | 2048 | `set_tail_rows_transpose(v)` |
| Discard overflow | 0 (off) | `set_discard_overflow(0\|1)` |
| Warps per row | 0 (multi-row) | `set_ell_create_warps_per_row(0\|1\|2\|4\|8)` |

### Worked example

```python
import torch, sys
sys.path.insert(0, "build/lib.linux-x86_64-3.10")
import sparse_ops

# Use a tighter ELL for regular ops, larger for transpose
torch.ops.sparse_ops_config.set_ell_width_regular(64)
torch.ops.sparse_ops_config.set_ell_width_transpose(128)

# Smaller tail: we expect low overflow
torch.ops.sparse_ops_config.set_tail_rows_regular(512)
torch.ops.sparse_ops_config.set_tail_rows_transpose(512)

# Run forward
out, l0, l1, P, R, T = torch.ops.sparse_ops.ff_forward_gated(X, G, K, V, out_size)

# Check gate overflow BEFORE backward (P's counter is filled in forward)
torch.cuda.synchronize()
fwd_overflow = P.overflow_count_tensor().item()
print(f"Forward ELL overflow: {fwd_overflow} rows (limit={64})")

# Backward
loss = out.sum()
loss.backward()

torch.cuda.synchronize()
bwd_overflow = torch.ops.sparse_ops_config.get_last_transpose_overflow_count().item()
print(f"Backward transpose overflow: {bwd_overflow} rows (limit={128})")

# Discard mode: benchmark ELL-only throughput (no tail writes)
torch.ops.sparse_ops_config.set_discard_overflow(1)
out_approx, *_ = torch.ops.sparse_ops.ff_forward_gated(X, G, K, V, out_size)
torch.ops.sparse_ops_config.set_discard_overflow(0)  # restore
```

### Notes

1. **Settings take effect at object construction time.** Calling `set_ell_width_regular` after `ff_forward_gated` has been called on a given batch does not affect already-allocated sparse objects (`P`, `R`, `T`). The next forward call will allocate new objects using the updated values.
2. **ELL width constrains column indices.** Column indices are stored as `uint16_t`, so the column space (N) must fit in 16 bits. The ELL width is the per-row NNZ budget, not the column range.
3. **Overflow is not a hard error.** Rows that exceed the ELL width are handled gracefully (moved to tail or discarded); there is no assertion or crash. Use `overflow_count_tensor()` to monitor.
4. **Tail capacity sizing.** Set `tail_rows` to at least the 99th-percentile overflow row count for your data. If the actual overflow count exceeds `tail_rows`, excess rows are silently dropped even in non-discard mode.
