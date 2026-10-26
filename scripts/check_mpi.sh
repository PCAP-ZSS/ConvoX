#!/usr/bin/env bash
# Verify the MPI path: every rank count and local kernel must reproduce the serial result
# (convox_mpi exits 1 if validation fails). Uses a small crop so it finishes quickly.
cd "$(dirname "$0")/.."
[ -x build/convox_mpi ] || { echo "build/convox_mpi missing (MPI not found at configure time?)"; exit 3; }

img=data/images/img01_4k.png
fail=0
for local in serial omp cuda-tiled; do
  for np in 1 2 3 4 7; do
    for k in 3 15; do
      out=$(mpirun --oversubscribe -np $np ./build/convox_mpi -i $img -k $k --reps 1 --local $local --no-csv 2>&1)
      rc=$?
      if [ $rc -eq 0 ]; then echo "ok   np=$np local=$local K=$k"; else echo "FAIL np=$np local=$local K=$k (rc=$rc): $out"; fail=1; fi
    done
  done
done
exit $fail
