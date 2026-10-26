#!/usr/bin/env bash
# Configure and build ConvoX into ./build. Picks up CUDA and MPI if present.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
[ -d /usr/local/cuda-12.6/bin ] && export PATH="/usr/local/cuda-12.6/bin:$PATH"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build -j"$(nproc)"
