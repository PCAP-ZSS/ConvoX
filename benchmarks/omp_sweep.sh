#!/usr/bin/env bash
# OpenMP thread-count sweep (task 2.4). Appends to the given results CSV.
# usage: benchmarks/omp_sweep.sh IMAGE [OUT_CSV] [REPS] [K ...]
set -euo pipefail
cd "$(dirname "$0")/.."

image=${1:?usage: omp_sweep.sh IMAGE [OUT_CSV] [REPS] [K ...]}
out=${2:-results/raw/results.csv}
reps=${3:-10}
shift $(( $# < 3 ? $# : 3 ))
ks=${*:-3 7 15}
threads="1 2 4 6 8 12 16"

for k in $ks; do
  ./build/convox run --impl serial -i "$image" -k "$k" --reps "$reps" --out "$out"
  for t in $threads; do
    ./build/convox run --impl omp --threads "$t" --schedule static -i "$image" -k "$k" --reps "$reps" --out "$out"
  done
done
