# Runtime knobs

Settings live in `torch.ops.sparse_ops_config` and apply to the **next allocated** sparse object or the next kernel launch — they do not retroactively change already-allocated buffers.

Profiling lives separately in `torch.ops.sparse_ops_perf` and only does anything when the extension was built with `SPARSE_ENABLE_PERF_PROFILING=1`.

## ELL width

The ELL part of a `hybrid_sp_t` stores the first N non-zeros per row. Rows whose true NNZ exceeds the ELL width either overflow into the dense tail (default) or are discarded — see [Execution modes](#execution-modes).

```python
torch.ops.sparse_ops_config.set_ell_width_regular(v)    # forward ops: gate, key proj, elementwise
torch.ops.sparse_ops_config.set_ell_width_transpose(v)  # backward ops: T^T, dR^T, dU^T
```

- `v` is the per-row NNZ budget; the value becomes both the buffer stride *and* the overflow threshold for objects constructed after the call.
- Default `128` (defined as `ELL_WIDTH` in `constants.h`).
- Narrowing (e.g. 64) cuts memory traffic and smem at the cost of more rows spilling to the tail. Widening (256) is only worth it when the actual NNZ distribution has a long tail above 128.

## Dense tail capacity

Rows that overflow ELL are stored densely; the tail is pre-allocated for at most `tail_rows` rows.

```python
torch.ops.sparse_ops_config.set_tail_rows_regular(v)
torch.ops.sparse_ops_config.set_tail_rows_transpose(v)
```

- Default `2048` (`TAIL_CAPACITY_ROWS` in `constants.h`).
- Excess overflow rows are **counted but not stored** — silently dropped. Check `overflow_count_tensor()` to detect.

## Execution modes

```python
torch.ops.sparse_ops_config.set_discard_overflow(0)  # default: hybrid ELL + dense tail
torch.ops.sparse_ops_config.set_discard_overflow(1)  # discard: count overflow but don't store
```

Discard mode trades correctness for throughput on rows that exceed the ELL width. Use it for benchmarking the ELL-only path or for highly sparse models where overflow is rare (verify with `overflow_count_tensor()`).

## Checking overflow

After the forward:
```python
out, l0, l1, P, R, T = model(x)
torch.cuda.synchronize()
print("gate overflow rows:", P.overflow_count_tensor().item())
```

After the backward (transposed sparse matrices):
```python
loss.backward()
torch.cuda.synchronize()
print("transpose overflow rows:",
      torch.ops.sparse_ops_config.get_last_transpose_overflow_count().item())
```

A value of 0 means all rows fit within the ELL width.

## Per-object properties (C++)

Each `hybrid_sp_t` snapshots its allocation parameters at construction:

| Field | Type | Description |
|---|---|---|
| `_ell_stride` | `int` | ELL buffer width = overflow threshold for this object |
| `_tail_cap` | `int` | Dense tail capacity for this object |
| `_dense_active_rows` | `int` | How many tail rows are currently populated |

These are fixed at construction and do not change when the global setters fire later.

## Defaults

| Setting | Default | Setter |
|---|---|---|
| ELL width (regular) | 128 | `set_ell_width_regular(v)` |
| ELL width (transpose) | 128 | `set_ell_width_transpose(v)` |
| Tail capacity (regular) | 2048 | `set_tail_rows_regular(v)` |
| Tail capacity (transpose) | 2048 | `set_tail_rows_transpose(v)` |
| Discard overflow | 0 (off) | `set_discard_overflow(0\|1)` |

## Worked example

```python
import torch
from sparse_ops_loader import load_sparse_ops
load_sparse_ops()

torch.ops.sparse_ops_config.set_ell_width_regular(64)
torch.ops.sparse_ops_config.set_ell_width_transpose(128)
torch.ops.sparse_ops_config.set_tail_rows_regular(512)
torch.ops.sparse_ops_config.set_tail_rows_transpose(512)

out, l0, l1, P, R, T = torch.ops.sparse_ops.ff_forward_gated(X, G, K, V)
torch.cuda.synchronize()
print("forward ELL overflow:", P.overflow_count_tensor().item(), "(limit=64)")

loss = out.sum()
loss.backward()
torch.cuda.synchronize()
print("backward transpose overflow:",
      torch.ops.sparse_ops_config.get_last_transpose_overflow_count().item(),
      "(limit=128)")

# Benchmark ELL-only throughput (no tail writes)
torch.ops.sparse_ops_config.set_discard_overflow(1)
out_approx, *_ = torch.ops.sparse_ops.ff_forward_gated(X, G, K, V)
torch.ops.sparse_ops_config.set_discard_overflow(0)
```

## Notes

1. **Settings take effect at object construction.** Calling a setter after `ff_forward_gated` does not change already-allocated `P`/`R`/`T`. The next forward picks up the new value.
2. **Column indices are `uint16_t`.** The column space (N) must fit in 16 bits. ELL width is the per-row NNZ budget, not the column range.
3. **Overflow is not a hard error.** Excess rows go to the tail or are discarded — no crash. Use `overflow_count_tensor()` to monitor.
4. **Sizing the tail.** Pick `tail_rows` >= the 99th-percentile overflow row count for your data. Excess overflow is silently dropped even outside discard mode.
