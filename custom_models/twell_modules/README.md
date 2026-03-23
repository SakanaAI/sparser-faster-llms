# TwELL Kernels

TwELL provides Hopper-optimized sparse feed-forward kernels for LLM MLP blocks.

## Main Python APIs

- `D2TLinear`: dense -> packed sparse hidden.
- `T2DLinear`: packed sparse hidden -> dense down projection.
- `GatedT2DLinear`: dense input + packed gate -> fused gated up/down projection.
- `TwELLMLP`: fused non-gated feed-forward block.
- `TwELLGatedMLP`: fused gated feed-forward block.

## LLM usage

Use `TwELLMLP` for non-gated FFNs of the form:

```python
hidden = relu(x @ up_weight.T)
out = hidden @ down_weight
```

Use `TwELLGatedMLP` for gated FFNs of the form:

```python
gate = relu(x @ gate_weight.T)
up = x @ up_weight.T
out = (gate * up) @ down_weight
```

If you want to split the block manually, use:

- `D2TLinear` + `T2DLinear` for non-gated blocks.
- `D2TLinear` + `GatedT2DLinear` for gated blocks.

## Precision and in-place modes

`TwELLGatedMLP` and `GatedT2DLinear` support `highest_precision=True`. This uses a
full fp32 path for the fused gated up/down projection; the default
`False` path is the regular mixed-precision bf16-style path, which is more consistent with a dense Pytorch implementation.

`TwELLMLP` and `TwELLGatedMLP` also support `inplace=True` to overwrite the input
activation and reduce memory traffic. 


