"""Small helpers shared by the public inference benchmark."""

import gc

import torch


def parse_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def free_cuda_memory() -> None:
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()


def get_core_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        return model.model
    return model
