from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Performance profiling toggle
# Set to 0 for maximum performance (disables all profiling overhead)
# Set to 1 for detailed performance profiling
ENABLE_PROFILING = 0

# Build configuration
profiling_flag = f'-DENABLE_PERF_PROFILING={ENABLE_PROFILING}'

setup(
    name='sparse_ops',
    ext_modules=[
        CUDAExtension(
            name='sparse_ops',
            sources=['custom_op.cpp', 'hybrid_sp.cu', 'wgmma_gemm.cu', '../twell_modules/matmul_d2t.cu'],
            extra_compile_args={
                'nvcc': [
                    '-DNDEBUG',
                    '-O3',
                    '-Xcompiler=-Wno-psabi',
                    '-Xcompiler=-fno-strict-aliasing',
                    '--resource-usage',
                    '--expt-relaxed-constexpr',
                    #'-gencode=arch=compute_80,code=sm_80',     # Ampere A100
                    #'-gencode=arch=compute_86,code=sm_86',     # Ampere RTX 30xx
                    '-gencode=arch=compute_90a,code=sm_90a',   # Hopper H100
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
