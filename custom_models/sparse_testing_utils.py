"""Helpers for converting sparse checkpoints to torch or TwELL inference."""

from typing import Dict, Mapping

import torch

from custom_models.twell_modules.twell import TwELLGatedMLP, TwELLMLP


def convert_sparse_mlp_to_twell_fused(
    up_linear,
    down_linear,
    layer_idx,
    config,
    gate_linear=None,
    num_splits: int = 2,
    inplace: bool = False,
    highest_precision: bool = False,
    preallocate_shared_hidden_state: bool = False,
    compression_factor: int = 8,
    flex_kernels: bool = False,
):
    if gate_linear is not None:
        return TwELLGatedMLP(
            layer_number=layer_idx,
            GATE=gate_linear.weight.detach().clone().contiguous(),
            UP=up_linear.weight.detach().clone().contiguous(),
            DOWN=down_linear.weight.detach().clone().T.contiguous(),
            inplace=inplace,
            preallocate_shared_hidden_state=preallocate_shared_hidden_state,
            highest_precision=highest_precision,
            compression_factor=compression_factor,
            flex_kernels=flex_kernels,
        )
    return TwELLMLP(
        layer_number=layer_idx,
        UP=up_linear.weight.detach().clone().contiguous(),
        DOWN=down_linear.weight.detach().clone().T.contiguous(),
        num_splits=num_splits,
        inplace=inplace,
        preallocate_shared_hidden_state=preallocate_shared_hidden_state,
        compression_factor=compression_factor,
    )


def get_sparse_mlp_to_twell_fused_conversion_fn(
    num_splits: int = 2,
    inplace: bool = False,
    highest_precision: bool = False,
    preallocate_shared_hidden_state: bool = False,
    compression_factor: int = 8,
    flex_kernels: bool = False,
):
    def conversion_fn(
        up_linear,
        down_linear,
        layer_idx,
        config,
        gate_linear=None,
    ):
        return convert_sparse_mlp_to_twell_fused(
            up_linear=up_linear,
            down_linear=down_linear,
            layer_idx=layer_idx,
            config=config,
            gate_linear=gate_linear,
            num_splits=num_splits,
            inplace=inplace,
            highest_precision=highest_precision,
            preallocate_shared_hidden_state=preallocate_shared_hidden_state,
            compression_factor=compression_factor,
            flex_kernels=flex_kernels,
        )

    return conversion_fn


def _convert_sparse_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    target: str,
    gated: bool = True,
    drop_auxiliary: bool = True,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if drop_auxiliary and (
            key.startswith("tracker.")
            or ".tracker." in key
            or key.startswith("metrics_accumulator.")
            or ".metrics_accumulator." in key
        ):
            continue

        if key.endswith("gate_proj.weight"):
            if not gated:
                raise ValueError(
                    "Attempting to convert gate_proj.weight but gated is False"
                )
            prefix = key[: -len("gate_proj.weight")]
            if target == "hf":
                out[key] = value.contiguous()
            else:
                out[prefix + "GATE"] = value.contiguous()
            continue

        if key.endswith("up_proj.weight"):
            prefix = key[: -len("up_proj.weight")]
            if target == "hf":
                out[key] = value.contiguous()
            else:
                out[prefix + "UP"] = value.contiguous()
            continue

        if key.endswith("down_proj.weight"):
            prefix = key[: -len("down_proj.weight")]
            if target == "hf":
                out[key] = value.contiguous()
            else:
                out[prefix + "DOWN"] = value.T.contiguous()
            continue

        if key.endswith("gate_weight"):
            if not gated:
                raise ValueError(
                    "Attempting to convert gate_weight but gated is False"
                )
            prefix = key[: -len("gate_weight")]
            if target == "hf":
                out[prefix + "gate_proj.weight"] = value.T.contiguous()
            else:
                out[prefix + "GATE"] = value.T.contiguous()
            continue

        if key.endswith("up_weight"):
            prefix = key[: -len("up_weight")]
            if target == "hf":
                out[prefix + "up_proj.weight"] = (
                    value.contiguous() if gated else value.T.contiguous()
                )
            else:
                out[prefix + "UP"] = (
                    value.contiguous() if gated else value.T.contiguous()
                )
            continue

        if key.endswith("down_weight"):
            prefix = key[: -len("down_weight")]
            if target == "hf":
                out[prefix + "down_proj.weight"] = value.T.contiguous()
            else:
                out[prefix + "DOWN"] = value.contiguous()
            continue

        if key.endswith("GATE"):
            if not gated:
                raise ValueError("Attempting to convert GATE but gated is False")
            prefix = key[: -len("GATE")]
            if target == "hf":
                out[prefix + "gate_proj.weight"] = value.contiguous()
            else:
                out[key] = value.contiguous()
            continue

        if key.endswith("UP"):
            prefix = key[: -len("UP")]
            if target == "hf":
                out[prefix + "up_proj.weight"] = value.contiguous()
            else:
                out[key] = value.contiguous()
            continue

        if key.endswith("DOWN"):
            prefix = key[: -len("DOWN")]
            if target == "hf":
                out[prefix + "down_proj.weight"] = value.T.contiguous()
            else:
                out[key] = value.contiguous()
            continue

        out[key] = value

    return out


def sparse_to_hf_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    drop_auxiliary: bool = True,
    gated: bool = True,
) -> Dict[str, torch.Tensor]:
    return _convert_sparse_state_dict(
        state_dict=state_dict,
        target="hf",
        gated=gated,
        drop_auxiliary=drop_auxiliary,
    )


def sparse_to_twell_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    gated: bool = True,
) -> Dict[str, torch.Tensor]:
    return _convert_sparse_state_dict(
        state_dict=state_dict,
        target="twell",
        gated=gated,
    )
