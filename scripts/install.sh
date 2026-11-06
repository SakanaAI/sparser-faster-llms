#!/usr/bin/env bash
set -e

if [[ "${1-}" == "--uv" ]]; then
PIP=(uv pip)
shift
else
PIP=(python -m pip)
fi

# CUDA 13.0 builds of PyTorch and flash-attn (MathDx 26.10 requires CUDA 13.0+).
"${PIP[@]}" install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
"${PIP[@]}" install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.7.4+cu130torch2.9-cp312-cp312-linux_x86_64.whl
"${PIP[@]}" install -r requirements.txt