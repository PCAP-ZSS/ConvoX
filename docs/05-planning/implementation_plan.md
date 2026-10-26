# ConvoX: Implementation Plan

Work through the tasks in order, one at a time. Mark a task `[x]` when its "done when" check passes. The phases follow the six-week timeline in the synopsis. Phase 6 (MPI) is optional and is the first thing cut if time runs short.

Each phase is a separable module (code, data and write-up), so the project can later be divided into three equal parts for the presentation. Division is done at the end, in `docs/06-presentation`.

## Phase 0: Environment and skeleton (Week 1)

- [x] 0.1 Confirm the toolchain: WSL2 Ubuntu 22.04, GCC 11+, OpenMP, CMake. Done when `convox info` or an equivalent `gcc --version` and `cmake --version` work.
- [x] 0.2 Confirm GPU access: `nvidia-smi` and `nvcc --version` in WSL2, or choose Colab or a lab machine. Done when a trivial CUDA kernel runs.
- [x] 0.3 CMake skeleton with optional CUDA and MPI detection and a `scripts/build.sh`. Done when a CPU-only build succeeds.
- [x] 0.4 Vendor `stb_image` and `stb_image_write` under `third_party/`. Done when an image loads and saves.
- [x] 0.5 Generate 4 synthetic test images (2x 4K, 2x 8K) via scripts/gen_images.py and fill `data/images/images.csv`. Done when the files exist with sources recorded.
- [x] 0.6 Initial commit and push of the skeleton.

## Phase 1: Common core and serial baseline (Week 1)

- [x] 1.1 `Image` and `Filter` types, image loader, grayscale float conversion.
- [x] 1.2 Filter generator: Gaussian, Sobel, Box for odd K from 3 to 15.
- [x] 1.3 Serial convolution with zero-padding, built `-O2` with no vectorisation. Border mode switch for clamp-to-edge.
- [x] 1.4 Validator: max-abs and RMS error, pass at < 1e-4.
- [x] 1.5 Timing harness: warm-up plus 10 reps, median, `steady_clock`.
- [x] 1.6 CSV writer for `results.csv` and `reps.csv`, and environment capture.
- [x] 1.7 CLI: `convox run` and `convox info` for the serial path, with `--save-image`.
- [x] 1.8 Unit checks: delta filter returns the input, box filter on a constant image returns that constant.
- Done when: serial blur and Sobel outputs look right, and a row lands in `results.csv`.

## Phase 2: OpenMP (Week 2)

- [x] 2.1 Row-parallel `parallel for` variant, validated against serial.
- [x] 2.2 `collapse(2)` variant.
- [x] 2.3 Runtime-selectable schedule: static, dynamic(1, 8, 32), guided.
- [x] 2.4 Thread sweep driver {1, 2, 4, 6, 8, 12, 16}.
- [x] 2.5 Run on 4K and 8K for K in {3, 7, 15} and inspect the scaling shape.
- Done when: all variants pass validation and the thread-sweep data is in the CSV.

## Phase 3: Naive CUDA (Week 3)

- [ ] 3.1 Error-check macro, device query, buffer allocation and transfers.
- [ ] 3.2 Naive kernel with a bounds guard, validated against serial.
- [ ] 3.3 Event-based timing: H2D, kernel and D2H separately, plus host-side total.
- [ ] 3.4 Block sweep: 8x8, 16x16, 32x32.
- Done when: naive CUDA passes validation and the kernel-vs-transfer split is recorded.

## Phase 4: Optimised CUDA (Week 4)

- [ ] 4.1 `cuda-const`: filter in `__constant__` memory. Validate, then measure on its own.
- [ ] 4.2 `cuda-tiled`: shared-memory tile with halo, `__syncthreads()`, compute from shared memory. Validate.
- [ ] 4.3 Edge cases: image sizes not divisible by the block, K=15 at T=32 shared-memory footprint.
- [ ] 4.4 Nsight Compute runs for naive vs tiled: DRAM throughput, occupancy. Record in the CSV or report.
- Done when: three CUDA variants all pass validation and the profiler numbers are saved.

## Phase 5: Sweep and analysis (Weeks 5-6)

- [ ] 5.1 `convox sweep` and `benchmarks/sweep.yaml` expanding the full matrix.
- [ ] 5.2 `scripts/run_all.sh`: build, sweep and plots in one command.
- [ ] 5.3 Full benchmark run to produce the final `results.csv`.
- [ ] 5.4 `scripts/summary.py` terminal tables.
- [ ] 5.5 `analysis/plots.py` for the five required plots, with `--show`.
- [ ] 5.6 `analysis/amdahl.py`: measured serial fraction, predicted ceiling vs observed plateau.
- [ ] 5.7 Visual-check images (original, Gaussian, Sobel, box side by side).
- Done when: one command regenerates every table and plot from a clean checkout.

## Phase 6: MPI extension (Week 5, optional)

- [ ] 6.1 Strip decomposition with `H/N + 2R` rows and halo handling at the edges.
- [ ] 6.2 `MPI_Scatterv` and `MPI_Gatherv` driver around the OpenMP kernel.
- [ ] 6.3 Hook the CUDA kernel in as the local kernel.
- [ ] 6.4 Validate against serial and record timing for 2 and 4 ranks.
- Done when: gathered output matches serial. If time is short, drop this phase and note it as future work.

## Phase 7: Report and presentation (Week 6)

- [ ] 7.1 Report sections: methodology, setup, results, discussion, conclusion.
- [ ] 7.2 Interpret each plot, covering the bandwidth plateau, the block-size sweet spot, the tiling gain growing with K, the transfer cost and the Amdahl gap.
- [ ] 7.3 Final README with build and run instructions.
- [ ] 7.4 Divide the work three ways and write `docs/06-presentation/work_split.md`.
- [ ] 7.5 Slides and a rehearsal.

## Risks to watch

| Risk | Trigger | Response |
|---|---|---|
| No usable GPU | Task 0.2 fails | Switch GPU runs to Colab T4 and record that in the report |
| Serial baseline gets vectorised | Speedups look implausibly low | Check the compiler report and add `-fno-tree-vectorize` |
| Timing noise | High variance in the 10 reps | Close other apps, fix the power mode, report min-max bands |
| Behind schedule | End of Week 4 without Phase 4 done | Drop Phase 6 |
