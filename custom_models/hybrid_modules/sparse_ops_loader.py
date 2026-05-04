import os
import re
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

# Env vars (all prefixed with SPARSE_):
#   SPARSE_OPS_PREBUILT          - full path to a prebuilt sparse_ops.so. When set,
#                                  JIT compilation is skipped and the .so is loaded
#                                  directly. Use this on machines without nvcc.
#   SPARSE_ENABLE_PERF_PROFILING - "1" to enable perf instrumentation in the JIT
#                                  build. Read on each call so distinct values
#                                  produce distinct cached builds.
# TORCH_CUDA_ARCH_LIST is honored as-is (set by torch / user) and drives nvcc.

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

_BASE_DIR = Path(__file__).resolve().parent
_EXT_NAME = "sparse_ops"

_BASE_SOURCES = [
    str(_BASE_DIR / "custom_op.cpp"),
    str(_BASE_DIR / "hybrid_sp.cu"),
    str(_BASE_DIR / "wgmma_gemm.cu"),
]
_HOPPER_SOURCES = [
    str(_BASE_DIR.parent / "twell_modules" / "matmul_d2t.cu"),
]

_loaded = False


def _profiling_flag() -> str:
    val = os.environ.get("SPARSE_ENABLE_PERF_PROFILING", "0")
    enabled = val not in ("0", "", "false", "False")
    return f"-DENABLE_PERF_PROFILING={1 if enabled else 0}"


def _has_hopper_target() -> bool:
    """Returns True if TORCH_CUDA_ARCH_LIST targets sm_90 or newer."""
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    for tok in re.split(r"[\s;,]+", arch_list.strip()):
        m = re.match(r"(\d+)\.(\d+)", tok)
        if m and (int(m.group(1)), int(m.group(2))) >= (9, 0):
            return True
    return False


def load_sparse_ops(verbose: bool = False):
    """Loads sparse_ops. Uses SPARSE_OPS_PREBUILT if set, else JIT-compiles."""
    global _loaded
    if _loaded:
        return

    prebuilt = os.environ.get("SPARSE_OPS_PREBUILT")
    if prebuilt:
        torch.ops.load_library(prebuilt)
        _loaded = True
        return

    profiling_flag = _profiling_flag()
    cuda_cflags = [
        "-DNDEBUG",
        "-O3",
        "-Xcompiler=-Wno-psabi",
        "-Xcompiler=-fno-strict-aliasing",
        "--resource-usage",
        "--expt-relaxed-constexpr",
        profiling_flag,
    ]
    cflags = ["-O3", profiling_flag]

    # matmul_d2t.cu (twell_modules) uses Hopper-only PTX and only compiles for sm_90a+.
    # Drop it for non-Hopper builds; wgmma_gemm.cu self-gates via __CUDA_ARCH_LIST__.
    sources = list(_BASE_SOURCES)
    if _has_hopper_target():
        sources += _HOPPER_SOURCES

    extra_ldflags = []
    if CUDA_HOME is not None:
        extra_ldflags.append(f"-L{Path(CUDA_HOME) / 'lib64' / 'stubs'}")
    extra_ldflags.append("-lcuda")

    load(
        name=_EXT_NAME,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        verbose=verbose,
        is_python_module=False,
    )
    _loaded = True
