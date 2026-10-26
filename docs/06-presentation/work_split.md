# ConvoX: Three-Way Work Split

The project divides into three parts of similar size, each with its own code, tests, results and a share of the synopsis objectives. Each part can be presented on its own and hands a clean interface to the next (`Image`, `Filter`, `Config`, `Timing` in `src/common`). Assign names to Part 1, 2 and 3 as you prefer; nothing below depends on who takes which.

| | Part 1: Foundation, serial, OpenMP and the build and test system | Part 2: GPU, profiling, plots and the demo page | Part 3: Benchmark driver, MPI, demo server and report |
|---|---|---|---|
| Synopsis objectives | 1, 2 | 3, 4 | 5, 6 |
| Lines (code, tests, scripts, docs text) | about 1,410 | about 1,720 | about 1,580, of which about 260 is README and report text |

Counts are lines in the files listed for each part (excluding third-party headers and planning documents). The GPU kernels are the densest code, which is why Part 2 has the most lines but the page and plotting code is simple.

## Part 1: Foundation, serial baseline, OpenMP, build and tests

**Question answered:** how fast is the algorithm on one core, how far do threads take it, and how is everything checked?

- **Code:**
  - `src/common/image.*`, `filter.*`, `validator.*`, `timing.*`, `config.h`, `csv.*`, `env.*`, `profile.*`: image loading, Gaussian/Sobel/box filters, the error check, warm-up and median timing, CSV rows, machine description, timing of the non-parallel parts.
  - `src/serial/`: the four-loop reference, with vectorisation disabled.
  - `src/openmp/`: row and `collapse(2)` variants, runtime schedule selection.
  - `src/common/runner.*`: picks an implementation by name, times it, validates it.
  - `CMakeLists.txt`, `scripts/build.sh`, `run_tests.sh`, `check_cli.sh`, `gen_images.py`, `benchmarks/omp_sweep.sh`.
  - `analysis/amdahl.py`.
  - Tests: `test_image`, `test_filter`, `test_validator`, `test_serial`, `test_timing`, `test_omp` (432 variants).
- **Results to present:** `omp_scaling.png`, `omp_schedules.png`, `amdahl.png`.
- **Key findings:** near-linear scaling to 4 threads, then 6.7x to 8.5x at 16 threads (efficiency 0.38 to 0.53). Dynamic and guided scheduling edge out static on this hybrid CPU, and `collapse(2)` with chunk 1 is 5x slower. The serial fraction (image I/O) caps whole-job speedup at 1.1x to 2.6x.
- **Likely questions:** why the serial baseline is compiled without vectorisation; why 1e-4 is the tolerance; why scaling stops (bandwidth, and the hybrid core mix); why static was expected to win and what happened instead.

## Part 2: GPU (CUDA), profiling, plots and the demo page

**Question answered:** what does each GPU memory optimisation buy, what does the GPU really cost end to end, and how are the results shown?

- **Code:**
  - `src/cuda/conv_cuda.cu` and `.h`: naive, constant-memory and tiled kernels, event timing, launch checks, occupancy query.
  - `src/cuda/device_info.*`: device query and theoretical peaks.
  - `analysis/ncu_profile.py` (Nsight Compute driver).
  - `analysis/plots.py`, `common.py`, `montage.py`, `scripts/summary.py`, `scripts/visual_check.sh`: loading results, terminal tables, all plots, the visual check.
  - `ui/static/index.html`, `app.js`, `style.css`: the demo page and its charts.
  - Test: `test_cuda` (360 variants, odd image sizes, bad launches).
- **Results to present:** `block_sensitivity.png`, `transfer_breakdown.png`, `roofline.png`, `runtime_vs_k.png`, `visual_check.png`, `results/raw/ncu_metrics.csv`, and the live demo page.
- **Key findings:** 16x16 blocks are fastest everywhere. Tiling gain grows from none at K=3 to 1.94x at K=15. Nsight shows DRAM traffic is the same for all three kernels, while L1 global-load traffic drops 86x at K=15 and occupancy rises from 71-88% to 94-97%. Transfers are 70% to 93% of whole-job time, so 389x compute-only becomes 104x whole-job.
- **Likely questions:** why constant memory suits a filter (all threads in a warp read the same coefficient); how the halo is loaded and why `__syncthreads()` is needed; why tiling did not reduce DRAM traffic; what limits the kernel in the roofline; pinned versus pageable memory.

## Part 3: Benchmark driver, MPI, demo server and report

**Question answered:** how were all the numbers produced reproducibly, how does the work scale across processes, and how does the demo run the real program?

- **Code:**
  - `src/common/main.cpp` (command line: `info`, `run`, `sweep`, `profile`, `--json`), `sweep.*` (matrix expansion, resume, manifest), `benchmarks/sweep.conf`.
  - `src/mpi/convox_mpi.cpp` (strip scatter, halo rows, gather) and `scripts/check_mpi.sh`.
  - `ui/server.py` (job runner for the demo page), `scripts/start_ui.sh`, `check_ui.sh`, `start_demo.bat`.
  - `scripts/run_all.sh`, `README.md`, `report/REPORT.md`.
  - Test: `test_sweep`.
- **Results to present:** the 728-run `results.csv`, the MPI table, a live run from the demo page, and the report's discussion and limitations.
- **Key findings:** one command reproduces everything. Every run is validated, and the sweep resumes after interruption. MPI matches serial for 1, 2, 3, 4 and 7 ranks, including uneven row counts. Splitting threads across ranks gives no gain, because they were already saturating memory. The demo server runs one job at a time so timings stay clean and rejects bad input.
- **Likely questions:** how halo rows are computed and why edge ranks pad them; why median instead of mean; how the schedule sweep was reduced and why; what the limits of a single-node MPI test are; why the demo never fakes results (it calls the same program as the benchmark).

## Balance and hand-over

| Aspect | Part 1 | Part 2 | Part 3 |
|---|---|---|---|
| Implementations | serial, OpenMP | 3 CUDA kernels | MPI driver, sweep driver, demo server |
| Tests | 6 programs | 1 program | 1 program plus check scripts for CLI, MPI and the demo |
| Plots | OpenMP scaling and schedules, Amdahl | 5 plots, visual check, demo charts | none; owns the measurements and the report |
| Core concepts | threads, scheduling, Amdahl, validation | memory hierarchy, occupancy, transfers, roofline | decomposition, halo exchange, reproducible measurement |

Each part needs to understand the common interface in `src/common/config.h`, `image.h` and `timing.h`, and the validation rule, since every result depends on them.

## Suggested 15-minute order

1. Problem, algorithm and measurement design (all, 2 min).
2. Part 1 (4 min), then Part 2 (4 min, including the live demo), then Part 3 (4 min).
3. Combined conclusion and limitations (1 min).
