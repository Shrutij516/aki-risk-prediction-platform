#!/usr/bin/env bash
# Build the C++ inference executable against the LibTorch that ships with the venv's PyTorch wheel.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x .venv/bin/cmake ] || uv pip install --quiet --python "$PY" cmake
PREFIX="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
.venv/bin/cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build cpp/build -j 8
echo "built cpp/build/aki_infer"
