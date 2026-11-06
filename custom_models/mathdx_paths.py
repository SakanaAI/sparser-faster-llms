"""Build paths for cuBLASDx from a released MathDx package."""

import os
import re
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


_ARCH_SPECIFIC = re.compile(r"(\d+)\.(\d+)a(?:\+PTX)?")


def d2t_target_sm(arch_list: str) -> int | None:
    """SM version D2T compiles for (e.g. 900, 1000, 1070), or None to skip D2T.

    D2T is built for exactly one architecture-specific target of sm_90a or newer,
    e.g. "9.0a" (Hopper), "10.0a" (Blackwell) or "10.7a" (Rubin). Lists without
    such a target select the non-D2T fallback; mixed lists are rejected.
    """
    targets = [target for target in re.split(r"[\s;,]+", arch_list.strip()) if target]
    selected = set()
    for target in targets:
        match = _ARCH_SPECIFIC.fullmatch(target)
        if match and (int(match[1]), int(match[2])) >= (9, 0):
            selected.add(int(match[1]) * 100 + int(match[2]) * 10)
    if not selected:
        return None
    if len(selected) != 1 or any(not _ARCH_SPECIFIC.fullmatch(target) for target in targets):
        raise ValueError("D2T requires a single architecture-specific target such as 9.0a, "
                         "10.0a or 10.7a; split mixed TORCH_CUDA_ARCH_LIST builds")
    return selected.pop()


def d2t_gencode(sm: int) -> str:
    """NVCC -gencode value for an SM version returned by d2t_target_sm."""
    return f"arch=compute_{sm // 10}a,code=sm_{sm // 10}a"


def default_arch_list() -> str:
    """Architecture-specific target of the current GPU, e.g. "10.0a".

    Falls back to "9.0a" when no sm_90+ GPU is visible (e.g. build machines).
    """
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 9:
                return f"{major}.{minor}a"
    except Exception:
        pass
    return "9.0a"


def cublasdx_include_dirs() -> list[str]:
    """Header paths for cuBLASDx from a released MathDx package (26.10 or newer).

    MATHDX_HOME is the versioned package root, e.g.
    nvidia-mathdx-<ver>-cuda13/nvidia/mathdx/<ver>, which provides include/
    (cuBLASDx, CommonDx) and the bundled CUTLASS/CuTe in external/cutlass/include.

    Use extra_include_paths/include_dirs: torch suppresses automatic architecture
    flags if an NVCC flag contains "arch", which can occur in an include path.
    """
    mathdx = _env_path("MATHDX_HOME")
    if mathdx is None:
        raise RuntimeError("Set MATHDX_HOME to a MathDx package root, "
                           "e.g. nvidia-mathdx-26.10.0-cuda13/nvidia/mathdx/26.10.")
    include = mathdx / "include"
    cutlass = mathdx / "external" / "cutlass" / "include"
    for required in (include / "cublasdx.hpp", include / "commondx", cutlass / "cute/tensor.hpp"):
        if not required.exists():
            raise RuntimeError(f"{required} not found; check MATHDX_HOME.")
    return [str(include), str(cutlass)]
