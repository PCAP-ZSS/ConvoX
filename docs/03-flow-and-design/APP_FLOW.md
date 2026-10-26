# ConvoX: Application Flow

ConvoX has no screens. Its "flow" is the path a benchmark run takes from input to result, and the path a user takes through the CLI.

## 1. End-to-end pipeline

```
Image file (data/images)
   |
   v
Image loader (stb) -> float buffer
   |
Filter generator (gaussian | sobel | box, K x K)
   |
   +--> Serial ------------------+
   +--> OpenMP (threads, sched) -+
   +--> CUDA naive --------------+--> Output buffer --> Validator (vs serial)
   +--> CUDA +const -------------+                         |
   +--> CUDA +const +shared -----+                         v
   +--> MPI (strips + halo) -----+                  Timing harness
                                                    (1 warm-up + 10 reps, median)
                                                          |
                                                          v
                                           results/raw/results.csv, reps.csv
                                                          |
                                                          v
                                         analysis: plots, Amdahl, summary tables
                                                          |
                                                          v
                                         results/plots/*.png  +  terminal summary
```

## 2. Single-run flow (`convox run`)

1. Parse and validate arguments (odd K from 3 to 15, a supported variant, an existing image).
2. Load the image and convert it to a single-channel float buffer.
3. Build the filter.
4. Run the serial reference once and keep its output.
5. Warm-up run of the requested implementation, discarded.
6. 10 measured runs. For GPU, record H2D, kernel and D2H separately, plus total.
7. Validate the last output against the serial reference. Compute max-abs and RMS error.
8. Take the median time, append a row to `results.csv` and all repetitions to `reps.csv`.
9. Print a one-line summary (impl, K, time, speedup, pass/fail).
10. If `--save-image` was given, write the output image.

Failure branches:
- No CUDA device: print which variants are unavailable and exit with a clear code. CPU variants still work.
- Validation fails: record `pass=0`, print the error values, and exit non-zero.
- Image missing or unreadable: exit before any timing starts.

## 3. Sweep flow (`convox sweep`)

1. Read `benchmarks/sweep.yaml`. It lists images, K values, thread counts, schedules, block sizes and variants.
2. Expand to the full run matrix (Table I of the synopsis).
3. Skip combinations that don't apply (for example, block size for CPU variants).
4. For each combination, run the single-run flow. Reuse the loaded image and serial reference between runs on the same image and K.
5. Show progress as `n / total` with an estimated time remaining.
6. Write a `sweep_manifest.json` with the environment info (CPU, GPU, compiler, commit).

## 4. MPI flow (optional)

1. `mpirun -np N convox_mpi ...` starts N ranks.
2. Rank 0 loads the image and computes strip boundaries, each `H/N + 2R` rows with halo overlap.
3. `MPI_Scatterv` sends the strips.
4. Each rank runs its local OpenMP or CUDA kernel.
5. `MPI_Gatherv` returns the interior rows (halo excluded) to rank 0.
6. Rank 0 validates against serial, then records timing and the row.

## 5. User flow (the team and the evaluator)

| Step | Command | Result |
|---|---|---|
| 1. Check the machine | `convox info` | CPU, GPU, compilers, OpenMP and CUDA versions |
| 2. Build | `scripts/build.sh` | `build/convox`, plus CUDA and MPI targets if found |
| 3. Smoke test | `convox run --impl serial --image ... --k 5 --save-image out.png` | A blurred image to check by eye |
| 4. Full sweep | `scripts/run_all.sh` | `results.csv`, `reps.csv` |
| 5. See numbers | `python scripts/summary.py` | Tables in the terminal |
| 6. See plots | `python analysis/plots.py --show` | PNGs in `results/plots/`, opened in the viewer |
| 7. Amdahl check | `python analysis/amdahl.py` | Predicted ceiling vs observed plateau |

## 6. Data flow between modules

| From | To | Data |
|---|---|---|
| Loader | Every implementation | `Image` (float, row-major) |
| Filter generator | Every implementation | `Filter` (K x K floats) |
| Implementation | Validator | Output `Image` |
| Implementation | Harness | `Timing` (kernel, H2D, D2H, total) |
| Validator | CSV writer | max-abs, RMS, pass flag |
| CSV writer | Analysis | `results.csv`, `reps.csv` |
