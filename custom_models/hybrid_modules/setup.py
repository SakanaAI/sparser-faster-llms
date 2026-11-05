import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathdx_paths import cublasdx_include_dirs, d2t_gencode, d2t_target_sm, default_arch_list  # noqa: E402

# Performance profiling toggle
# Set to 0 for maximum performance (disables all profiling overhead)
# Set to 1 for detailed performance profiling
ENABLE_PROFILING = 0

# Build configuration
profiling_flag = f'-DENABLE_PERF_PROFILING={ENABLE_PROFILING}'

# Target one architecture-specific SM, e.g. TORCH_CUDA_ARCH_LIST=9.0a (Hopper),
# 10.0a (Blackwell) or 10.7a (Rubin). Defaults to the current GPU.
D2T_SM = d2t_target_sm(os.environ.get('TORCH_CUDA_ARCH_LIST') or default_arch_list())
if D2T_SM is None:
    raise RuntimeError('sparse_ops requires an architecture-specific TORCH_CUDA_ARCH_LIST such as 10.0a')

setup(
    name='sparse_ops',
    ext_modules=[
        CUDAExtension(
            name='sparse_ops',
            sources=['custom_op.cpp', 'hybrid_sp.cu', 'wgmma_gemm.cu', '../twell_modules/matmul_d2t.cu'],
            # cuBLASDx headers for matmul_d2t.cu. Must go here rather than as -I in
            # extra_compile_args: see mathdx_paths.cublasdx_include_dirs.
            include_dirs=cublasdx_include_dirs(),
            extra_compile_args={
                'nvcc': [
                    '-DNDEBUG',
                    '-O3',
                    '-Xcompiler=-Wno-psabi',
                    '-Xcompiler=-fno-strict-aliasing',
                    '--resource-usage',
                    '--expt-relaxed-constexpr',
                    '-DSPARSE_OPS_HAS_D2T=1',
                    f'-DTWELL_D2T_SM={D2T_SM}',
                    f'-gencode={d2t_gencode(D2T_SM)}',
                    profiling_flag,
                ],
                'cxx': ['-DPy_LIMITED_API=0x03090000', profiling_flag],
            },
            libraries=['cuda'],
            library_dirs=['/usr/local/cuda/lib64'],
            py_limited_api=True,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    options={'bdist_wheel': {'py_limited_api': 'cp39'}}
)
