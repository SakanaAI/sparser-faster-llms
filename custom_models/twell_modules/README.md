# TwELL Kernels

TwELL provides Hopper-optimized sparse feed-forward kernels for LLM MLP blocks.

`matmul_d2t.cu` builds against cuBLASDx from a released MathDx package
(26.10 or newer). Point `MATHDX_HOME` at the versioned package root:

```bash
export MATHDX_HOME=/path/to/nvidia-mathdx-26.10.0-cuda13/nvidia/mathdx/26.10
```

The build uses `$MATHDX_HOME/include` and the CUTLASS headers bundled under
`$MATHDX_HOME/external/cutlass/include`.

The kernel targets `sm_90a` only. Set `TORCH_CUDA_ARCH_LIST=9.0a` when building
D2T; mixed architecture lists are unsupported. The shipped 2x1 cluster requires
`M % 256 == 0`, `N % 256 == 0`, and `K % 64 == 0`, with positive dimensions,
`N <= 65536`, and at least four K stages for TS8 or three for TS4/TS2.

The persistent grid is selected with CUDA's cluster-occupancy query during layer
preparation. Prepare layers and their shapes before graph capture.

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

## Compression factor and flex kernels

Flex kernels are a variants to the default TwELL kernels that are expected to 
be ~0.5% more performant in cases of non-uniform sparsity patterns. In terms of
the overall gated MLP block, the performance difference between flex and 
non-flex kernels is expected to be less than 0.1% in most cases.

Packed sparse outputs use `compression_factor=8` by default, which is a 
conservative recommended value for the model trained with $L_1=2\times 10^5$.
Gated TwELL paths also support `compression_factor=4` and `compression_factor=2` when
`flex_kernels=True`.

## Precision and in-place modes

`TwELLGatedMLP` and `GatedT2DLinear` support `highest_precision=True`. This uses a
full fp32 path for the fused gated up/down projection; the default
`False` path is the regular mixed-precision bf16-style path, which is more consistent with a dense Pytorch implementation.

`TwELLMLP` and `TwELLGatedMLP` also support `inplace=True` to overwrite the input
activation and reduce memory traffic. 
