#!/usr/bin/env bash
set -e

if [[ "${1-}" == "--uv" ]]; then
PIP=(uv pip)
shift
else
PIP=(python -m pip)
fi

"${PIP[@]}" install torch==2.9.1
"${PIP[@]}" install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl
"${PIP[@]}" install -r requirements.txt