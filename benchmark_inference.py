#!/usr/bin/env python3

"""Benchmark sparse checkpoints with standard torch and TwELL inference."""

import argparse
import copy
import csv
import gc
import json
import os
import timeit
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_extension
from tqdm.auto import tqdm
from transformers.modeling_utils import load_state_dict
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.utils import cached_file

from benchmark_base import free_cuda_memory, get_core_model, parse_dtype
from custom_models.sparse_models import SparseLlamaConfig, SparseLlamaForCausalLM
from custom_models.sparse_testing_utils import (
    convert_sparse_mlp_to_twell_fused,
    sparse_to_hf_state_dict,
    sparse_to_twell_state_dict,
)
from custom_models.twell_modules import twell as twell_module
from energy_utils import GPUEnergyMonitor


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sparse checkpoints with torch and TwELL inference."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup-reps", type=int, default=5)
    parser.add_argument("--measure-energy", action="store_true", default=False)
    return parser.parse_args()


def configure_twell_cuda_home() -> None:
    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda is None:
        return

    preferred_cuda_home = Path(f"/home/common_modules/cuda/cuda-{torch_cuda}")
    if not preferred_cuda_home.exists():
        return

    os.environ["CUDA_HOME"] = str(preferred_cuda_home)
    os.environ["CUDA_PATH"] = str(preferred_cuda_home)
    os.environ["PATH"] = f"{preferred_cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{preferred_cuda_home / 'lib64'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    cpp_extension.CUDA_HOME = str(preferred_cuda_home)
    twell_module.CUDA_HOME = str(preferred_cuda_home)


def patch_twell_cuda_12_1_build() -> None:
    cuda_home = str(getattr(twell_module, "CUDA_HOME", "") or "")
    if "12.1" not in cuda_home:
        return

    def load_ext_with_compat(
        name=twell_module._DEFAULT_EXT_NAME,
        algorithms=None,
        verbose=False,
    ):
        selected = twell_module._resolve_algorithms_with_deps(
            twell_module._validate_algorithms(algorithms or ["twell_d2t"])
        )
        sources = [
            str(twell_module._BASE_DIR / twell_module.ALGORITHM_SPECS[key]["source"])
            for key in selected
        ]
        sources.append(str(twell_module._BASE_DIR / "twell.cpp"))

        cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-D__NV_ATOMIC_RELAXED=0",
            "-D__NV_THREAD_SCOPE_BLOCK=2",
        ]

        extra_ldflags = []
        if twell_module.CUDA_HOME is not None:
            extra_ldflags.append(f"-L{Path(twell_module.CUDA_HOME) / 'lib64' / 'stubs'}")
        extra_ldflags.append("-lcuda")

        return twell_module.load(
            name=name,
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=extra_ldflags,
            verbose=verbose,
        )

    twell_module.load_ext = load_ext_with_compat
    twell_module._EXT_CACHE.clear()


def resolve_artifact_file(model_path: str, filename: str) -> Path:
    local_path = Path(model_path)
    if local_path.exists():
        if local_path.is_dir():
            return local_path / filename
        return local_path.parent / filename
    return Path(cached_file(model_path, filename))


def resolve_state_dict_path(model_path: str) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            candidate = resolve_artifact_file(model_path, filename)
        except Exception:
            continue
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve model weights for {model_path}")


def load_sparse_config(model_path: str, device: torch.device) -> SparseLlamaConfig:
    print(f"[setup] Loading sparse config from {model_path}", flush=True)
    config_path = resolve_artifact_file(model_path, "config.json")
    config = SparseLlamaConfig(**json.loads(config_path.read_text()))
    config._attn_implementation = (
        "flash_attention_2" if device.type == "cuda" else "eager"
    )
    config.use_cache = False
    return config


def build_inputs(config: SparseLlamaConfig, batch_size: int, seq_len: int, device: torch.device):
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(0)
    input_ids = torch.randint(
        low=0,
        high=int(config.vocab_size),
        size=(int(batch_size), int(seq_len)),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones(
        (int(batch_size), int(seq_len)),
        device=device,
        dtype=torch.long,
    )
    return input_ids, attention_mask


def validate_twell_shape(batch_size: int, seq_len: int) -> None:
    tokens = int(batch_size) * int(seq_len)
    if tokens % 256 != 0:
        raise ValueError(
            "TwELL currently requires batch_size * seq_len to be divisible by 256. "
            f"Got batch_size={batch_size}, seq_len={seq_len}, tokens={tokens}."
        )


def raise_on_incompatible_keys(implementation_name: str, incompatible) -> None:
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if not missing and not unexpected:
        return

    def format_keys(name: str, values: list[str]) -> str:
        preview = values[:10]
        suffix = ""
        if len(values) > 10:
            suffix = f", ... (+{len(values) - 10} more)"
        return f"{name}={preview}{suffix}"

    details = []
    if missing:
        details.append(format_keys("missing_keys", missing))
    if unexpected:
        details.append(format_keys("unexpected_keys", unexpected))
    joined = "; ".join(details)
    raise RuntimeError(f"[{implementation_name}] state_dict load mismatch: {joined}")


def get_gpu_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def finalize_model(model: torch.nn.Module, device: torch.device, dtype: torch.dtype):
    model = model.eval().to(device).to(dtype)
    core = get_core_model(model)
    if hasattr(core, "config") and hasattr(core.config, "use_cache"):
        core.config.use_cache = False
    return model


def load_torch_model(
    sparse_config: SparseLlamaConfig,
    state_dict_file: Path,
    dtype: torch.dtype,
    device: torch.device,
):
    print("[torch] Loading model and converting sparse checkpoint to HF format", flush=True)
    model = LlamaForCausalLM(copy.deepcopy(sparse_config))
    state = load_state_dict(str(state_dict_file))
    state = sparse_to_hf_state_dict(
        state,
        gated=bool(getattr(sparse_config, "sparsity_gated_mlp", True)),
    )
    incompatible = model.load_state_dict(state, strict=False)
    raise_on_incompatible_keys("torch", incompatible)
    del state
    gc.collect()
    model = finalize_model(model, device=device, dtype=dtype)
    return model


def load_twell_model(
    sparse_config: SparseLlamaConfig,
    state_dict_file: Path,
    dtype: torch.dtype,
    device: torch.device,
):
    print("[twell] Loading model and converting sparse checkpoint to TwELL format", flush=True)
    model = SparseLlamaForCausalLM(copy.deepcopy(sparse_config))
    state = load_state_dict(str(state_dict_file))
    state = sparse_to_hf_state_dict(
        state,
        gated=bool(getattr(sparse_config, "sparsity_gated_mlp", True)),
    )
    incompatible = model.load_state_dict(state, strict=False)
    raise_on_incompatible_keys("twell", incompatible)
    del state
    gc.collect()
    model.replace_mlp_modules(convert_sparse_mlp_to_twell_fused)
    model = finalize_model(model, device=device, dtype=dtype)
    return model


def benchmark_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reps: int,
    warmup_reps: int,
    implementation_name: str,
    measure_energy: bool = False,
) -> dict[str, float]:
    core = get_core_model(model)
    device = input_ids.device
    num_tokens = int(input_ids.shape[0]) * int(input_ids.shape[1])
    energy_monitor = None

    if measure_energy:
        if device.type != "cuda":
            raise RuntimeError("Energy measurement requires a CUDA device.")
        energy_monitor = GPUEnergyMonitor(
            gpu_index=get_gpu_index(device),
            poll_interval_s=0.05,
        )
        if not energy_monitor.enabled:
            raise RuntimeError(
                "Energy measurement requested, but no GPU power backend is available. "
                "Install pynvml or ensure nvidia-smi is available."
            )

    def run_once():
        with torch.inference_mode():
            _ = core(input_ids=input_ids, attention_mask=attention_mask)

    print(f"[{implementation_name}] Starting warmup ({warmup_reps} reps)", flush=True)
    for _ in tqdm(
        range(int(warmup_reps)),
        desc=f"{implementation_name} warmup",
        leave=False,
    ):
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_times_ms = []
    print(
        f"[{implementation_name}] Starting timed benchmark ({reps} reps)",
        flush=True,
    )
    if energy_monitor is not None:
        energy_monitor.start()
    for _ in tqdm(
        range(int(reps)),
        desc=f"{implementation_name} reps",
        leave=False,
    ):
        start = timeit.default_timer()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        total_times_ms.append((end - start) * 1000.0)

    energy_stats = None
    if energy_monitor is not None:
        energy_monitor.stop()
        energy_stats = energy_monitor.results()
        if energy_stats is None:
            raise RuntimeError(
                f"[{implementation_name}] Energy measurement failed to collect samples."
            )

    result = {
        "avg_total_ms": float(sum(total_times_ms) / len(total_times_ms)),
    }
    if energy_stats is not None:
        total_energy_per_fwd = float(energy_stats.energy_joules / float(reps))
        result["total_energy_per_fwd"] = total_energy_per_fwd
        result["total_energy_per_token"] = float(total_energy_per_fwd / float(num_tokens))
    return result


def default_out_csv(model_path: str, batch_size: int, seq_len: int) -> Path:
    safe_name = model_path.rstrip("/").split("/")[-1].replace(":", "_")
    return Path("results/benchmark_inference") / (
        f"{safe_name}_bs{batch_size}_sl{seq_len}.csv"
    )


def write_rows(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "implementation",
        "avg_total_ms",
        "batch_size",
        "seq_len",
        "dtype",
        "device",
        "model_path",
        "state_dict_file",
        "reps",
        "warmup_reps",
    ]
    if rows and "total_energy_per_fwd" in rows[0]:
        fieldnames.extend(
            [
                "total_energy_per_fwd",
                "total_energy_per_token",
            ]
        )
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def colorize_if_needed(text: str, improvement_value: float | None) -> str:
    if improvement_value is None:
        return text
    if improvement_value > 0:
        return f"{GREEN}{text}{RESET}"
    if improvement_value < 0:
        return f"{RED}{text}{RESET}"
    return text


def format_speed_improvement(
    baseline_tokens_per_s: float | None,
    current_tokens_per_s: float | None,
) -> tuple[str, float | None]:
    if baseline_tokens_per_s is None or current_tokens_per_s is None:
        return "--", None
    if baseline_tokens_per_s <= 0.0:
        return "--", None
    improvement = ((current_tokens_per_s - baseline_tokens_per_s) / baseline_tokens_per_s) * 100.0
    return f"{improvement:+.1f}%", improvement


def print_summary_table(rows: list[dict]) -> None:
    baseline_row = next((row for row in rows if row["implementation"] == "torch"), None)
    baseline_tokens_per_s = None
    if baseline_row is not None:
        baseline_num_tokens = int(baseline_row["batch_size"]) * int(baseline_row["seq_len"])
        baseline_avg_total_ms = float(baseline_row["avg_total_ms"])
        if baseline_avg_total_ms > 0.0:
            baseline_tokens_per_s = baseline_num_tokens * 1000.0 / baseline_avg_total_ms

    display_rows = []
    for row in rows:
        num_tokens = int(row["batch_size"]) * int(row["seq_len"])
        avg_total_ms = float(row["avg_total_ms"])
        tokens_per_s = (num_tokens * 1000.0 / avg_total_ms) if avg_total_ms > 0.0 else float("nan")
        if row["implementation"] == "torch":
            improvement_text, improvement_value = "baseline", None
        else:
            improvement_text, improvement_value = format_speed_improvement(
                baseline_tokens_per_s,
                tokens_per_s,
            )

        formatted = {
            "implementation": str(row["implementation"]),
            "avg_total_ms": f"{avg_total_ms:.3f}",
            "tokens_per_s": f"{tokens_per_s:,.1f}",
            "improvement_vs_torch": improvement_text,
            "_improvement_color": improvement_value,
        }
        if "total_energy_per_fwd" in row:
            formatted["total_energy_per_fwd"] = f"{float(row['total_energy_per_fwd']):.6f}"
        display_rows.append(formatted)

    columns = [
        ("implementation", "Implementation"),
        ("avg_total_ms", "Avg Total ms"),
        ("tokens_per_s", "Input Tokens/s"),
    ]
    if display_rows and "total_energy_per_fwd" in display_rows[0]:
        columns.extend(
            [
                ("total_energy_per_fwd", "Total Energy/fwd"),
            ]
        )
    columns.append(("improvement_vs_torch", "Input Tokens/s vs torch"))
    widths = {}
    for key, title in columns:
        widths[key] = max(len(title), *(len(row[key]) for row in display_rows))

    def render_cell(row: dict, key: str) -> str:
        text = row[key]
        if key in {"implementation"}:
            padded = text.ljust(widths[key])
        else:
            padded = text.rjust(widths[key])
        if key == "improvement_vs_torch":
            return colorize_if_needed(padded, row.get("_improvement_color"))
        if key == "implementation" and row["implementation"] == "torch":
            return f"{DIM}{padded}{RESET}"
        return padded

    header = " | ".join(
        (title.ljust(widths[key]) if key == "implementation" else title.rjust(widths[key]))
        for key, title in columns
    )
    separator = "-+-".join("-" * widths[key] for key, _ in columns)

    print("\nBenchmark Summary", flush=True)
    print(header, flush=True)
    print(separator, flush=True)
    for row in display_rows:
        print(" | ".join(render_cell(row, key) for key, _ in columns), flush=True)


def main():
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)
    configure_twell_cuda_home()
    patch_twell_cuda_12_1_build()
    if device.type != "cuda":
        raise RuntimeError("benchmark_inference.py requires --device cuda for TwELL.")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda device requested but cuda is unavailable")
    validate_twell_shape(args.batch_size, args.seq_len)
    if args.measure_energy:
        print("[setup] Energy measurement enabled", flush=True)

    sparse_config = load_sparse_config(args.model_path, device=device)
    print("[setup] Resolving checkpoint weights", flush=True)
    state_dict_file = resolve_state_dict_path(args.model_path)
    print(
        f"[setup] Building synthetic inputs for batch_size={args.batch_size}, "
        f"seq_len={args.seq_len}",
        flush=True,
    )
    input_ids, attention_mask = build_inputs(
        sparse_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    implementations = [
        ("twell", load_twell_model),
        ("torch", load_torch_model),
    ]
    rows = []

    for implementation_name, load_fn in implementations:
        print(f"[{implementation_name}] Preparing benchmark run", flush=True)
        free_cuda_memory()
        model = load_fn(
            sparse_config=sparse_config,
            state_dict_file=state_dict_file,
            dtype=dtype,
            device=device,
        )
        benchmark_stats = benchmark_model(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reps=args.reps,
            warmup_reps=args.warmup_reps,
            implementation_name=implementation_name,
            measure_energy=args.measure_energy,
        )
        row = {
            "implementation": implementation_name,
            "avg_total_ms": float(benchmark_stats["avg_total_ms"]),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "dtype": args.dtype,
            "device": str(device),
            "model_path": args.model_path,
            "state_dict_file": str(state_dict_file),
            "reps": int(args.reps),
            "warmup_reps": int(args.warmup_reps),
        }
        if args.measure_energy:
            row["total_energy_per_fwd"] = float(benchmark_stats["total_energy_per_fwd"])
            row["total_energy_per_token"] = float(benchmark_stats["total_energy_per_token"])
        rows.append(row)
        del model
        gc.collect()
        free_cuda_memory()

    out_csv = Path(args.out_csv) if args.out_csv is not None else default_out_csv(
        model_path=args.model_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    print(f"[done] Writing results to {out_csv}", flush=True)
    write_rows(rows, out_csv)
    print_summary_table(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
