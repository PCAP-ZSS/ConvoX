#!/usr/bin/env bash
# Reproduce everything: build, test, full benchmark sweep, tables and plots.
# usage: scripts/run_all.sh [--quick]
#   --quick  small sweep (one 4K and one 8K image, K = 3, 7, 15, 3 reps) into results/quick/
set -euo pipefail
cd "$(dirname "$0")/.."

raw=results/raw
sweep_args=()
if [ "${1:-}" = "--quick" ]; then
  raw=results/quick
  sweep_args=(--images img01_4k,img03_8k --ks 3,7,15 --reps 3)
fi

echo "== build"
bash scripts/build.sh >/dev/null
echo "== tests"
bash scripts/run_tests.sh
echo "== images"
[ -f data/images/img01_4k.png ] || python3 scripts/gen_images.py
echo "== sweep -> $raw"
./build/convox sweep --out "$raw/results.csv" "${sweep_args[@]}"
echo "== summary"
python3 scripts/summary.py --raw "$raw"
echo "== plots"
python3 analysis/plots.py --raw "$raw" --out "$(dirname "$raw")/plots"
python3 analysis/amdahl.py --raw "$raw" --out "$(dirname "$raw")/plots"
echo "== visual check"
bash scripts/visual_check.sh
echo "done. plots in $(dirname "$raw")/plots"
