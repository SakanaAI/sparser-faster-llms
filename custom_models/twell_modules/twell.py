import abc
import hashlib
import os
from typing import Final
from pathlib import Path

import torch
from torch import nn
from torch.utils.cpp_extension import CUDA_HOME, load

# Hopper kernels: compile for sm90a by default.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_EXT_NAME = "twell_prod_ext"
_SHARED_PACKED_HIDDEN_STATE: torch.Tensor | None = None
_SHARED_PACKED_HIDDEN_STATE_SPEC: tuple[str, int, int, int] | None = None

ALGORITHM_SPECS = {
    "twell_d2t": {
        "source": "matmul_d2t.cu",
        "deps": [],
    },
    "twell_t2d": {
        "source": "matmul_t2d.cu",
        "deps": [],
    },
    "twell_gated_t2d": {
        "source": "matmul_gated_t2d.cu",
        "deps": [],
    },
    "twell_mlp": {
        "source": "mlp_twell.cu",
        "deps": ["twell_d2t", "twell_t2d", "twell_gated_t2d"],
    },
}


def sparse_algorithms() -> list[str]:
    """Returns the supported sparse algorithm keys."""
    return sorted(ALGORITHM_SPECS.keys())


def _validate_algorithms(algorithms: list[str]) -> list[str]:
    """Validates user-selected algorithm keys."""
    selected = [a.strip().lower() for a in algorithms if a.strip()]
    if not selected:
        raise ValueError("No algorithms selected")
    unknown = [a for a in selected if a not in ALGORITHM_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown algorithm(s): {unknown}. "
            f"Available: {', '.join(sparse_algorithms())}"
        )
    return selected


def _resolve_algorithms_with_deps(selected: list[str]) -> list[str]:
    """Expands algorithm keys to include their declared dependencies."""
    resolved: list[str] = []
    seen: set[str] = set()

    def add_with_deps(key: str) -> None:
        if key in seen:
            return
        for dep in ALGORITHM_SPECS[key].get("deps", []):
            add_with_deps(dep)
        seen.add(key)
        resolved.append(key)

    for key in selected:
        add_with_deps(key)
    return resolved


def _ext_name_for_algorithms(selected: list[str]) -> str:
    """Builds a deterministic extension name for one algorithm set."""
    digest = hashlib.sha1(",".join(selected).encode("utf-8")).hexdigest()[:10]
    return f"{_DEFAULT_EXT_NAME}_{digest}"


def load_ext(
    name: str = _DEFAULT_EXT_NAME,
    algorithms: list[str] | None = None,
    verbose: bool = False,
):
    """Builds and loads the requested TwELL extension variant."""
    selected = _resolve_algorithms_with_deps(
        _validate_algorithms(algorithms or ["twell_d2t"])
    )
    sources = [str(_BASE_DIR / ALGORITHM_SPECS[key]["source"]) for key in selected]
    sources.append(str(_BASE_DIR / "twell.cpp"))

    cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    extra_ldflags = []
    if CUDA_HOME is not None:
        extra_ldflags.append(f"-L{Path(CUDA_HOME) / 'lib64' / 'stubs'}")
    extra_ldflags.append("-lcuda")

    return load(
        name=name,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        verbose=verbose,
    )


_EXT_CACHE: dict[tuple[str, ...], object] = {}


def get_ext(
    algorithms: list[str] | None = None,
    verbose: bool = False,
):
    """Returns a cached extension module for the requested algorithms."""
    selected = _resolve_algorithms_with_deps(
        _validate_algorithms(algorithms or ["twell_d2t"])
    )
    cache_key = tuple(selected)
    module = _EXT_CACHE.get(cache_key)
    if module is None:
        module = load_ext(
            name=_ext_name_for_algorithms(selected),
            algorithms=selected,
            verbose=verbose,
        )
        _EXT_CACHE[cache_key] = module
    return module


def _module_for(ext, *, algorithms: list[str]):
    """Returns the provided extension or loads one for the requested algorithms."""
    return ext if ext is not None else get_ext(algorithms=algorithms)


def packed_width(output_n: int, tile_n: int = 256) -> int:
    """Returns the packed width used by TwELL sparse outputs."""
    if output_n % tile_n != 0:
        raise ValueError(f"output_n must be divisible by {tile_n}")
    if output_n % 8 != 0:
        raise ValueError("output_n must be divisible by 8")
    return output_n // 8


def allocate_packed_output(
    m: int,
    n: int,
    *,
    device: torch.device | str,
    tile_n: int = 256,
) -> torch.Tensor:
    """Allocates a zeroed packed sparse output buffer."""
    return torch.zeros((m, packed_width(n, tile_n=tile_n)), device=device, dtype=torch.uint32)


def _shared_packed_hidden_state(
    m: int,
    n: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Returns a shared packed hidden-state workspace for the requested shape."""
    global _SHARED_PACKED_HIDDEN_STATE
    global _SHARED_PACKED_HIDDEN_STATE_SPEC

    device_index: Final[int] = -1 if device.index is None else int(device.index)
    cache_key = (device.type, device_index, int(m), int(n))
    if (
        _SHARED_PACKED_HIDDEN_STATE is None
        or _SHARED_PACKED_HIDDEN_STATE_SPEC != cache_key
        or _SHARED_PACKED_HIDDEN_STATE.device != device
    ):
        _SHARED_PACKED_HIDDEN_STATE = allocate_packed_output(m, n, device=device)
        _SHARED_PACKED_HIDDEN_STATE_SPEC = cache_key
    return _SHARED_PACKED_HIDDEN_STATE


def _clear_shared_packed_hidden_state() -> None:
    """Clears the cached shared packed hidden-state workspace."""
    global _SHARED_PACKED_HIDDEN_STATE
    global _SHARED_PACKED_HIDDEN_STATE_SPEC
    _SHARED_PACKED_HIDDEN_STATE = None
    _SHARED_PACKED_HIDDEN_STATE_SPEC = None


def packed_to_dense(
    packed: torch.Tensor,
    output_n: int,
    tile_n: int = 256,
) -> torch.Tensor:
    """Converts a packed TwELL tensor back to a dense bf16 tensor."""
    m, compressed_n = packed.shape
    expected_compressed_n = packed_width(output_n, tile_n=tile_n)
    if packed.dtype != torch.uint32:
        raise ValueError(f"packed tensor must be uint32, got {packed.dtype}")
    if compressed_n != expected_compressed_n:
        raise ValueError(
            f"packed width mismatch: got {compressed_n}, expected {expected_compressed_n}"
        )

    compressed_per_tile = tile_n // 8
    payload_slots = compressed_per_tile - 1
    out = torch.zeros((m, output_n), device=packed.device, dtype=torch.bfloat16)
    row_ids = torch.arange(m, device=packed.device, dtype=torch.int64)[:, None]
    slot_ids = torch.arange(payload_slots, device=packed.device, dtype=torch.int64)[None, :]

    for tile_n_idx in range(output_n // tile_n):
        comp_col_start = tile_n_idx * compressed_per_tile
        comp_col_end = comp_col_start + compressed_per_tile
        packed_tile = packed[:, comp_col_start:comp_col_end]

        counts = packed_tile[:, 0].to(torch.int64)
        if bool((counts > payload_slots).any()):
            raise ValueError(
                f"Embedded count exceeds payload slots in tile {tile_n_idx}: "
                f"max={int(counts.max().item())}, max_allowed={payload_slots}"
            )

        payload_i64 = packed_tile[:, 1:].to(torch.int64)
        idx_global = payload_i64 % 65536
        vals = (payload_i64 // 65536).to(torch.uint16).contiguous().view(torch.bfloat16)
        valid_mask = slot_ids < counts[:, None]
        if not bool(valid_mask.any()):
            continue

        row_grid = row_ids.expand(-1, payload_slots)
        out[row_grid[valid_mask], idx_global[valid_mask]] = vals[valid_mask]

    return out


def matmul_sparse(
    A: torch.Tensor,
    B: torch.Tensor,
    algorithm: str = "twell_d2t",
    ext=None,
) -> torch.Tensor:
    """Runs packed sparse matmul and returns the packed output."""
    module = _module_for(ext, algorithms=[algorithm])
    return module.matmul_sparse(A, B, algorithm)


def matmul_sparse_out(
    A: torch.Tensor,
    B: torch.Tensor,
    C_packed: torch.Tensor,
    algorithm: str = "twell_d2t",
    ext=None,
) -> None:
    """Runs packed sparse matmul into a caller-provided packed output tensor."""
    module = _module_for(ext, algorithms=[algorithm])
    module.matmul_sparse_out(A, B, C_packed, algorithm)


def matmul_t2d(
    IN_packed: torch.Tensor,
    DOWN: torch.Tensor,
    num_splits: int = 2,
    ext=None,
) -> torch.Tensor:
    """Runs packed-to-dense downprojection and returns the dense output."""
    module = _module_for(ext, algorithms=["twell_t2d"])
    return module.matmul_t2d(IN_packed, DOWN, int(num_splits))


def matmul_t2d_out(
    IN_packed: torch.Tensor,
    DOWN: torch.Tensor,
    OUT: torch.Tensor,
    num_splits: int = 2,
    ext=None,
) -> None:
    """Runs packed-to-dense downprojection into a caller-provided output tensor."""
    module = _module_for(ext, algorithms=["twell_t2d"])
    module.matmul_t2d_out(IN_packed, DOWN, OUT, int(num_splits))


def matmul_gated_t2d(
    IN_dense: torch.Tensor,
    GATE_packed: torch.Tensor,
    UP: torch.Tensor,
    DOWN: torch.Tensor,
    highest_precision: bool = False,
    ext=None,
) -> torch.Tensor:
    """Runs gated packed-to-dense projection and returns the dense output."""
    module = _module_for(ext, algorithms=["twell_gated_t2d"])
    return module.matmul_gated_t2d(IN_dense, GATE_packed, UP, DOWN, bool(highest_precision))


def matmul_gated_t2d_out(
    IN_dense: torch.Tensor,
    GATE_packed: torch.Tensor,
    UP: torch.Tensor,
    DOWN: torch.Tensor,
    OUT: torch.Tensor,
    highest_precision: bool = False,
    ext=None,
) -> None:
    """Runs gated packed-to-dense projection into a caller-provided output tensor."""
    module = _module_for(ext, algorithms=["twell_gated_t2d"])
    module.matmul_gated_t2d_out(IN_dense, GATE_packed, UP, DOWN, OUT, bool(highest_precision))


def create_d2t_layer(
    layer_number: int,
    B: torch.Tensor,
    ext=None,
) -> None:
    """Creates or refreshes cached D2T state for one layer id."""
    module = _module_for(ext, algorithms=["twell_d2t"])
    module.create_d2t_layer(int(layer_number), B)


def run_d2t_layer(
    layer_number: int,
    A: torch.Tensor,
    C_packed: torch.Tensor,
    ext=None,
) -> None:
    """Runs one cached D2T layer into a packed output tensor."""
    module = _module_for(ext, algorithms=["twell_d2t"])
    module.run_d2t_layer(int(layer_number), A, C_packed)


def ensure_d2t_layer_shape(
    layer_number: int,
    M: int,
    ext=None,
) -> None:
    """Ensures cached D2T state is ready for the requested row count."""
    module = _module_for(ext, algorithms=["twell_d2t"])
    module.ensure_d2t_layer_shape(int(layer_number), int(M))


def run_mlp_layer(
    layer_number: int,
    A: torch.Tensor,
    DOWN: torch.Tensor,
    num_splits: int = 2,
    ext=None,
) -> torch.Tensor:
    """Runs the fused non-gated TwELL MLP."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    return module.run_mlp_layer(int(layer_number), A, DOWN, int(num_splits))


def run_gated_mlp_layer(
    layer_number: int,
    A: torch.Tensor,
    UP: torch.Tensor,
    DOWN: torch.Tensor,
    highest_precision: bool = False,
    ext=None,
) -> torch.Tensor:
    """Runs the fused gated TwELL MLP."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    return module.run_gated_mlp_layer(int(layer_number), A, UP, DOWN, bool(highest_precision))


def _run_mlp_layer_with_hidden_states(
    layer_number: int,
    A: torch.Tensor,
    DOWN: torch.Tensor,
    C_packed: torch.Tensor,
    num_splits: int = 2,
    ext=None,
) -> torch.Tensor:
    """Runs the fused non-gated TwELL MLP with caller-provided packed hidden states."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    return module._run_mlp_layer_with_hidden_states(
        int(layer_number), A, DOWN, C_packed, int(num_splits)
    )


def _run_gated_mlp_layer_with_hidden_states(
    layer_number: int,
    A: torch.Tensor,
    UP: torch.Tensor,
    DOWN: torch.Tensor,
    C_packed: torch.Tensor,
    highest_precision: bool = False,
    ext=None,
) -> torch.Tensor:
    """Runs the fused gated TwELL MLP with caller-provided packed hidden states."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    return module._run_gated_mlp_layer_with_hidden_states(
        int(layer_number), A, UP, DOWN, C_packed, bool(highest_precision)
    )


def run_mlp_layer_inplace(
    layer_number: int,
    A: torch.Tensor,
    DOWN: torch.Tensor,
    num_splits: int = 2,
    ext=None,
) -> None:
    """Runs the fused non-gated TwELL MLP in place."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    module.run_mlp_layer_inplace(int(layer_number), A, DOWN, int(num_splits))


def run_gated_mlp_layer_inplace(
    layer_number: int,
    A: torch.Tensor,
    UP: torch.Tensor,
    DOWN: torch.Tensor,
    highest_precision: bool = False,
    ext=None,
) -> None:
    """Runs the fused gated TwELL MLP in place."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    module.run_gated_mlp_layer_inplace(int(layer_number), A, UP, DOWN, bool(highest_precision))


def _run_mlp_layer_inplace_with_hidden_states(
    layer_number: int,
    A: torch.Tensor,
    DOWN: torch.Tensor,
    C_packed: torch.Tensor,
    num_splits: int = 2,
    ext=None,
) -> None:
    """Runs the fused non-gated TwELL MLP in place with caller-provided packed hidden states."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    module._run_mlp_layer_inplace_with_hidden_states(
        int(layer_number), A, DOWN, C_packed, int(num_splits)
    )


def _run_gated_mlp_layer_inplace_with_hidden_states(
    layer_number: int,
    A: torch.Tensor,
    UP: torch.Tensor,
    DOWN: torch.Tensor,
    C_packed: torch.Tensor,
    highest_precision: bool = False,
    ext=None,
) -> None:
    """Runs the fused gated TwELL MLP in place with caller-provided packed hidden states."""
    module = _module_for(ext, algorithms=["twell_mlp"])
    module._run_gated_mlp_layer_inplace_with_hidden_states(
        int(layer_number), A, UP, DOWN, C_packed, bool(highest_precision)
    )


def destroy_d2t_layer(
    layer_number: int,
    ext=None,
) -> None:
    """Destroys one cached D2T layer entry."""
    module = _module_for(ext, algorithms=["twell_d2t"])
    module.destroy_d2t_layer(int(layer_number))


def destroy_all_d2t_layers(ext=None) -> None:
    """Destroys all cached D2T layer entries."""
    module = _module_for(ext, algorithms=["twell_d2t"])
    module.destroy_all_d2t_layers()


class D2TLayer:
    """Caches and runs one D2T layer configuration."""
    def __init__(
        self,
        layer_number: int,
        B: torch.Tensor,
        ext=None,
    ):
        self.layer_number = int(layer_number)
        self.ext = _module_for(ext, algorithms=["twell_d2t"])
        self.m: int | None = None
        self.n = int(B.size(0))
        self.c_packed: torch.Tensor | None = None
        create_d2t_layer(
            self.layer_number,
            B,
            ext=self.ext,
        )

    def prepare(
        self,
        A: torch.Tensor,
        C_packed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = int(A.size(0))
        if self.m != m:
            ensure_d2t_layer_shape(self.layer_number, m, ext=self.ext)
            self.m = m
            self.c_packed = None
        if C_packed is not None:
            return C_packed
        if (
            self.c_packed is None
            or self.c_packed.size(0) != A.size(0)
            or self.c_packed.device != A.device
        ):
            self.c_packed = allocate_packed_output(A.size(0), self.n, device=A.device)
        return self.c_packed

    def prepare_shared_hidden_state(self, A: torch.Tensor) -> torch.Tensor:
        m = int(A.size(0))
        if self.m != m:
            ensure_d2t_layer_shape(self.layer_number, m, ext=self.ext)
            self.m = m
        return _shared_packed_hidden_state(m, self.n, device=A.device)

    def run(
        self,
        A: torch.Tensor,
        C_packed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed = self.prepare(A, C_packed=C_packed)
        run_d2t_layer(self.layer_number, A, packed, ext=self.ext)
        return packed

    def destroy(self) -> None:
        destroy_d2t_layer(self.layer_number, ext=self.ext)
        self.m = None
        self.c_packed = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


class D2TLinear(nn.Module):
    """nn.Module wrapper around one cached D2T layer."""
    def __init__(
        self,
        layer_number: int,
        B: torch.Tensor,
        ext=None,
    ):
        super().__init__()
        self.register_buffer("B", B.contiguous())
        self.layer = D2TLayer(layer_number=layer_number, B=self.B, ext=ext)

    def forward(self, A: torch.Tensor, C_packed: torch.Tensor | None = None) -> torch.Tensor:
        return self.layer.run(A, C_packed=C_packed)

    def destroy(self) -> None:
        self.layer.destroy()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


class T2DLinear(nn.Module):
    """nn.Module wrapper around the packed-to-dense downprojection."""
    def __init__(
        self,
        DOWN: torch.Tensor,
        num_splits: int = 2,
        ext=None,
    ):
        super().__init__()
        self.register_buffer("DOWN", DOWN.contiguous())
        self.num_splits = int(num_splits)
        self.ext = _module_for(ext, algorithms=["twell_t2d"])
        self._out: torch.Tensor | None = None

    def forward(self, IN_packed: torch.Tensor, OUT: torch.Tensor | None = None) -> torch.Tensor:
        if OUT is None:
            if self._out is None or self._out.size(0) != IN_packed.size(0) or self._out.device != IN_packed.device:
                self._out = torch.empty(
                    (IN_packed.size(0), self.DOWN.size(1)),
                    device=IN_packed.device,
                    dtype=self.DOWN.dtype,
                )
            OUT = self._out
        matmul_t2d_out(IN_packed, self.DOWN, OUT, num_splits=self.num_splits, ext=self.ext)
        return OUT


class GatedT2DLinear(nn.Module):
    """nn.Module wrapper around the gated packed-to-dense projection."""
    def __init__(
        self,
        UP: torch.Tensor,
        DOWN: torch.Tensor,
        highest_precision: bool = False,
        ext=None,
    ):
        super().__init__()
        self.register_buffer("UP", UP.contiguous())
        self.register_buffer("DOWN", DOWN.contiguous())
        self.ext = _module_for(ext, algorithms=["twell_gated_t2d"])
        # When enabled, the fused gated up+down projection uses fp32
        # instead of the regular mixed-precision bf16-style matmul path.
        self.highest_precision = bool(highest_precision)
        self._out: torch.Tensor | None = None

    def forward(
        self,
        IN_dense: torch.Tensor,
        GATE_packed: torch.Tensor,
        OUT: torch.Tensor | None = None,
        highest_precision: bool | None = None,
    ) -> torch.Tensor:
        use_highest_precision = (
            self.highest_precision if highest_precision is None else bool(highest_precision)
        )
        if OUT is None:
            if self._out is None or self._out.size(0) != IN_dense.size(0) or self._out.device != IN_dense.device:
                self._out = torch.empty(
                    (IN_dense.size(0), IN_dense.size(1)),
                    device=IN_dense.device,
                    dtype=IN_dense.dtype,
                )
            OUT = self._out
        matmul_gated_t2d_out(
            IN_dense,
            GATE_packed,
            self.UP,
            self.DOWN,
            OUT,
            highest_precision=use_highest_precision,
            ext=self.ext,
        )
        return OUT


class _BaseTwELLMLP(nn.Module, abc.ABC):
    """Shared base class for fused TwELL MLP wrappers."""
    def __init__(
        self,
        layer_number: int,
        B: torch.Tensor,
        *,
        inplace: bool = False,
        preallocate_shared_hidden_state: bool = False,
        ext=None,
    ):
        super().__init__()
        self.layer_number = int(layer_number)
        self.ext = _module_for(ext, algorithms=["twell_mlp"])
        self.inplace = bool(inplace)
        self.preallocate_shared_hidden_state = bool(preallocate_shared_hidden_state)
        if B.is_cuda:
            self.layer: D2TLayer | None = D2TLayer(
                layer_number=self.layer_number,
                B=B,
                ext=self.ext,
            )
        else:
            self.layer = None
        self._forward_impl = None
        self._runtime_forward = None

    @abc.abstractmethod
    def _select_forward_impl(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_d2t_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def _finalize_runtime_forward(self) -> None:
        self._runtime_forward = self._forward_impl if self.layer is not None else self._initialize_and_forward

    def _initialize_and_forward(self, A: torch.Tensor) -> torch.Tensor:
        B = self._get_d2t_weight()
        if not B.is_cuda:
            raise RuntimeError("TwELL MLP weights must be moved to CUDA before first forward")
        self.layer = D2TLayer(
            layer_number=self.layer_number,
            B=B,
            ext=self.ext,
        )
        self._runtime_forward = self._forward_impl
        return self._forward_impl(A)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A_shape = A.shape
        return self._runtime_forward(A.view(-1, A_shape[-1])).view(A_shape)

    def destroy(self) -> None:
        if self.layer is not None:
            self.layer.destroy()
            self.layer = None
        if self.preallocate_shared_hidden_state:
            _clear_shared_packed_hidden_state()
        self._runtime_forward = self._initialize_and_forward

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


class TwELLMLP(_BaseTwELLMLP):
    """Fused non-gated TwELL MLP wrapper."""
    def __init__(
        self,
        layer_number: int,
        UP: torch.Tensor,
        DOWN: torch.Tensor,
        num_splits: int = 2,
        inplace: bool = False,
        preallocate_shared_hidden_state: bool = False,
        ext=None,
    ):
        up = UP.contiguous()
        down = DOWN.contiguous()
        super().__init__(
            layer_number=layer_number,
            B=up,
            inplace=inplace,
            preallocate_shared_hidden_state=preallocate_shared_hidden_state,
            ext=ext,
        )
        self.register_buffer("UP", up)
        self.register_buffer("DOWN", down)
        self.num_splits = int(num_splits)
        self._forward_impl = self._select_forward_impl()
        self._finalize_runtime_forward()

    def _get_d2t_weight(self) -> torch.Tensor:
        return self.UP

    def _select_forward_impl(self):
        if self.preallocate_shared_hidden_state:
            return self._forward_shared_inplace if self.inplace else self._forward_shared
        return self._forward_temporary_inplace if self.inplace else self._forward_temporary

    def _forward_temporary(self, A: torch.Tensor) -> torch.Tensor:
        return run_mlp_layer(
            self.layer.layer_number,
            A,
            self.DOWN,
            num_splits=self.num_splits,
            ext=self.ext,
        )

    def _forward_temporary_inplace(self, A: torch.Tensor) -> torch.Tensor:
        run_mlp_layer_inplace(
            self.layer.layer_number,
            A,
            self.DOWN,
            num_splits=self.num_splits,
            ext=self.ext,
        )
        return A

    def _forward_shared(self, A: torch.Tensor) -> torch.Tensor:
        return _run_mlp_layer_with_hidden_states(
            self.layer.layer_number,
            A,
            self.DOWN,
            self.layer.prepare_shared_hidden_state(A),
            num_splits=self.num_splits,
            ext=self.ext,
        )

    def _forward_shared_inplace(self, A: torch.Tensor) -> torch.Tensor:
        _run_mlp_layer_inplace_with_hidden_states(
            self.layer.layer_number,
            A,
            self.DOWN,
            self.layer.prepare_shared_hidden_state(A),
            num_splits=self.num_splits,
            ext=self.ext,
        )
        return A


class TwELLGatedMLP(_BaseTwELLMLP):
    """Fused gated TwELL MLP wrapper."""
    def __init__(
        self,
        layer_number: int,
        GATE: torch.Tensor,
        UP: torch.Tensor,
        DOWN: torch.Tensor,
        inplace: bool = False,
        preallocate_shared_hidden_state: bool = False,
        highest_precision: bool = False,
        ext=None,
    ):
        gate = GATE.contiguous()
        up = UP.contiguous()
        down = DOWN.contiguous()
        super().__init__(
            layer_number=layer_number,
            B=gate,
            inplace=inplace,
            preallocate_shared_hidden_state=preallocate_shared_hidden_state,
            ext=ext,
        )
        # When enabled, the fused gated up+down projection uses fp32
        # instead of the regular mixed-precision bf16-style matmul path.
        self.highest_precision = bool(highest_precision)
        self.register_buffer("GATE", gate)
        self.register_buffer("UP", up)
        self.register_buffer("DOWN", down)
        self._forward_impl = self._select_forward_impl()
        self._finalize_runtime_forward()

    def _get_d2t_weight(self) -> torch.Tensor:
        return self.GATE

    def _select_forward_impl(self):
        if self.preallocate_shared_hidden_state:
            if self.inplace:
                return (
                    self._forward_shared_high_precision_inplace
                    if self.highest_precision
                    else self._forward_shared_inplace
                )
            return self._forward_shared_high_precision if self.highest_precision else self._forward_shared
        if self.inplace:
            return (
                self._forward_temporary_high_precision_inplace
                if self.highest_precision
                else self._forward_temporary_inplace
            )
        return self._forward_temporary_high_precision if self.highest_precision else self._forward_temporary

    def _forward_temporary(self, A: torch.Tensor) -> torch.Tensor:
        return run_gated_mlp_layer(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            highest_precision=False,
            ext=self.ext,
        )

    def _forward_temporary_high_precision(self, A: torch.Tensor) -> torch.Tensor:
        return run_gated_mlp_layer(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            highest_precision=True,
            ext=self.ext,
        )

    def _forward_temporary_inplace(self, A: torch.Tensor) -> torch.Tensor:
        run_gated_mlp_layer_inplace(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            highest_precision=False,
            ext=self.ext,
        )
        return A

    def _forward_temporary_high_precision_inplace(self, A: torch.Tensor) -> torch.Tensor:
        run_gated_mlp_layer_inplace(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            highest_precision=True,
            ext=self.ext,
        )
        return A

    def _forward_shared(self, A: torch.Tensor) -> torch.Tensor:
        return _run_gated_mlp_layer_with_hidden_states(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            self.layer.prepare_shared_hidden_state(A),
            highest_precision=False,
            ext=self.ext,
        )

    def _forward_shared_high_precision(self, A: torch.Tensor) -> torch.Tensor:
        return _run_gated_mlp_layer_with_hidden_states(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            self.layer.prepare_shared_hidden_state(A),
            highest_precision=True,
            ext=self.ext,
        )

    def _forward_shared_inplace(self, A: torch.Tensor) -> torch.Tensor:
        _run_gated_mlp_layer_inplace_with_hidden_states(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            self.layer.prepare_shared_hidden_state(A),
            highest_precision=False,
            ext=self.ext,
        )
        return A

    def _forward_shared_high_precision_inplace(self, A: torch.Tensor) -> torch.Tensor:
        _run_gated_mlp_layer_inplace_with_hidden_states(
            self.layer.layer_number,
            A,
            self.UP,
            self.DOWN,
            self.layer.prepare_shared_hidden_state(A),
            highest_precision=True,
            ext=self.ext,
        )
        return A
