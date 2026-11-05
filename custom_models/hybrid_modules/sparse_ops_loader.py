import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathdx_paths import cublasdx_include_dirs, d2t_target_sm, default_arch_list  # noqa: E402

# Environment variables:
#   SPARSE_OPS_PREBUILT          - full path to a prebuilt sparse_ops.so. When set,
#                                  JIT compilation is skipped and the .so is loaded
#                                  directly. Use this on machines without nvcc.
#   SPARSE_ENABLE_PERF_PROFILING - "1" to enable perf instrumentation in the JIT
#                                  build. Read on each call so distinct values
#                                  produce distinct cached builds.
#   MATHDX_HOME                  - MathDx package root (26.10+) providing cuBLASDx.
#                                  See ../twell_modules/README.md.
# TORCH_CUDA_ARCH_LIST drives nvcc (default: the current GPU, e.g. 10.0a). D2T is
# built for a single architecture-specific target such as 9.0a, 10.0a or 10.7a.

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", default_arch_list())

_BASE_DIR = Path(__file__).resolve().parent
_EXT_NAME = "sparse_ops"

_BASE_SOURCES = [
    str(_BASE_DIR / "custom_op.cpp"),
    str(_BASE_DIR / "hybrid_sp.cu"),
    str(_BASE_DIR / "wgmma_gemm.cu"),
]
_D2T_SOURCES = [
    str(_BASE_DIR.parent / "twell_modules" / "matmul_d2t.cu"),
]

_loaded = False


def _profiling_flag() -> str:
    val = os.environ.get("SPARSE_ENABLE_PERF_PROFILING", "0")
    enabled = val not in ("0", "", "false", "False")
    return f"-DENABLE_PERF_PROFILING={1 if enabled else 0}"


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

    # matmul_d2t.cu compiles for one architecture-specific SM; tell the hybrid
    # wrapper whether its D2T symbol is linked and which SM it runs on.
    d2t_sm = d2t_target_sm(os.environ.get("TORCH_CUDA_ARCH_LIST", ""))
    compile_d2t = d2t_sm is not None
    cuda_cflags.append(f"-DSPARSE_OPS_HAS_D2T={int(compile_d2t)}")
    sources = list(_BASE_SOURCES)
    if compile_d2t:
        cuda_cflags.append(f"-DTWELL_D2T_SM={d2t_sm}")
        sources += _D2T_SOURCES

    extra_ldflags = []
    if CUDA_HOME is not None:
        extra_ldflags.append(f"-L{Path(CUDA_HOME) / 'lib64' / 'stubs'}")
    extra_ldflags.append("-lcuda")

    load(
        name=_EXT_NAME,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        # Resolve cuBLASDx only when compiling the D2T source.
        extra_include_paths=cublasdx_include_dirs() if compile_d2t else [],
        extra_ldflags=extra_ldflags,
        verbose=verbose,
        is_python_module=False,
    )
    _loaded = True
