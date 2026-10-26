#!/usr/bin/env bash
# Visual correctness check: run the Gaussian, Sobel and box filters on one image with the
# optimised CUDA kernel (serial if no GPU) and compose original + outputs side by side.
# usage: scripts/visual_check.sh [IMAGE] [K]
set -euo pipefail
cd "$(dirname "$0")/.."

image=${1:-data/images/img02_4k.png}
k=${2:-9}
out=results/plots/visual
mkdir -p "$out"
impl=cuda-tiled
./build/convox info | grep -q 'GPU:' || impl=serial

for f in gaussian sobel box; do
  mode=clamp; [ "$f" = sobel ] && mode=abs
  ./build/convox run --impl $impl -i "$image" -k "$k" --filter $f --reps 1 --no-csv \
    --save-image "$out/$f.png" --save-mode $mode
done
python3 analysis/montage.py "$image" "$out" "results/plots/visual_check.png" "$k"
