"""Build paths for cuBLASDx from a released MathDx package."""

import os
import re
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def d2t_sm90a_target(arch_list: str) -> bool:
    """Whether a Torch architecture list can compile the Hopper-only D2T source."""
    targets = [target for target in re.split(r"[\s;,]+", arch_list.strip()) if target]
    sm90a = {"9.0a", "9.0a+PTX"}
    selected = [target for target in targets if target in sm90a]
    if selected and len(selected) != len(targets):
        raise ValueError("D2T requires only sm_90a targets; split mixed TORCH_CUDA_ARCH_LIST builds")
    return bool(selected)


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
