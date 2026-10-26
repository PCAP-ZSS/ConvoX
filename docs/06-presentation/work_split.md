# ConvoX: Three-Way Work Split

The project divides into three parts of similar size, each with its own code, tests, results and a part of the synopsis objectives. Each part can be presented on its own, and each hands a clean interface to the next (`Image`, `Filter`, `Config`, `Timing` in `src/common`). Assign names to Part 1, 2 and 3 as you prefer; nothing below depends on who takes which.

| | Part 1: Foundation, serial and shared memory | Part 2: GPU | Part 3: Benchmark framework, distributed memory and analysis |
|---|---|---|---|
| Synopsis objectives | 1, 2 | 3, 4 | 5, 6 |
| Lines (code, tests, scripts) | about 1,130 | about 910 | about 1,340, of which about 260 is README and report text |

## Part 1: Foundation, serial baseline and OpenMP

**Question answered:** how fast is the algorithm on one core, and how far do threads take it?

- **Code:**
  - `src/common/image.*`, `filter.*`, `validator.*`, `timing.*`, `csv.*`, `env.*`, `config.h`: image loading, Gaussian/Sobel/box filters, error check, warm-up and median timing, CSV rows, machine description.
  - `src/serial/`: the four-loop reference, with vectorisation disabled.
  - `src/openmp/`: row and `collapse(2)` variants, runtime schedule selection.
  - `tests/test_image.cpp`, `test_filter.cpp`, `test_validator.cpp`, `test_serial.cpp`, `test_timing.cpp`, `test_omp.cpp` (432 variants).
  - `benchmarks/omp_sweep.sh`, `analysis/amdahl.py`, `scripts/gen_images.py`.
- **Results to present:** `omp_scaling.png`, `omp_schedules.png`, and the serial-fraction measurement behind `amdahl.png` (`src/common/profile.*`, `serial_profile.csv`).
- **Key findings:** near-linear scaling to 4 threads, then 6.7x to 8.5x at 16 threads (efficiency 0.38 to 0.53). Dynamic and guided scheduling edge out static on this hybrid CPU, and `collapse(2)` with chunk 1 is 5x slower. The serial fraction (image I/O) caps end-to-end speedup at 1.1x to 2.6x.
- **Likely questions:** why the serial baseline is compiled without vectorisation; why 1e-4 is the tolerance; why scaling stops (bandwidth, and the hybrid core mix); why static was expected to win and what happened instead.

## Part 2: GPU (CUDA)

**Question answered:** what does each GPU memory optimisation buy, and what does the GPU really cost end to end?

- **Code:**
  - `src/cuda/conv_cuda.cu` and `.h`: naive, constant-memory and tiled kernels, event timing, launch checks, occupancy query.
  - `src/cuda/device_info.*`: device query and theoretical peaks.
  - `tests/test_cuda.cpp` (360 variants, odd image sizes, bad launches).
  - `analysis/ncu_profile.py` (Nsight Compute driver).
  - `analysis/plots.py`, `common.py`, `montage.py`, `scripts/summary.py`, `scripts/visual_check.sh`: loading results, terminal tables, all plots, the visual check.
- **Results to present:** `block_sensitivity.png`, `transfer_breakdown.png`, `roofline.png`, `runtime_vs_k.png`, `visual_check.png`, `results/raw/ncu_metrics.csv`.
- **Key findings:** 16x16 blocks are fastest everywhere. Tiling gain grows from none at K=3 to 1.94x at K=15. Nsight shows DRAM traffic is the same for all three kernels, while L1 global-load traffic drops 86x at K=15 and occupancy rises from 71-88% to 94-97%. Transfers are 70% to 93% of end-to-end time, so 389x kernel-only becomes 104x end-to-end.
- **Likely questions:** why constant memory suits a filter (all threads in a warp read the same coefficient); how the halo is loaded and why `__syncthreads()` is needed; why tiling did not reduce DRAM traffic; what limits the kernel in the roofline; pinned versus pageable memory.

## Part 3: Benchmark framework, MPI and analysis

**Question answered:** how were all the numbers produced reproducibly, and how does the work scale across processes?

- **Code:**
  - `src/common/main.cpp` (CLI), `runner.*` (dispatch), `sweep.*` (matrix expansion, resume, manifest).
  - `src/mpi/convox_mpi.cpp` (strip scatter, halo rows, gather).
  - `scripts/build.sh`, `run_tests.sh`, `run_all.sh`, `check_cli.sh`, `check_mpi.sh`; `benchmarks/sweep.conf`; `CMakeLists.txt`; `README.md`.
  - `tests/test_sweep.cpp`.
  - `report/REPORT.md`.
- **Results to present:** the 728-run `results.csv`, the MPI table (5x with 16 ranks, 25 to 35 ms scatter/gather cost), and the report's discussion and limitations.
- **Key findings:** one command reproduces everything. Every run is validated, and the sweep resumes after interruption. MPI matches serial for 1, 2, 3, 4 and 7 ranks, including uneven row counts. Hybrid MPI + OpenMP gives no gain because the threads were already saturating memory.
- **Likely questions:** how halo rows are computed and why edge ranks pad them; why median instead of mean; how the schedule sweep was reduced and why; what the limits of a single-node MPI test are.

## Balance and hand-over

| Aspect | Part 1 | Part 2 | Part 3 |
|---|---|---|---|
| Implementations | 2 | 3 kernels | 1 distributed driver plus the framework |
| Tests | 6 programs | 1 program | 1 program plus 2 check scripts |
| Plots | 2 (OpenMP scaling and schedules) plus the Amdahl plot | 5 plus the visual check | none; owns the measurements and the report |
| Core concepts | threads, scheduling, Amdahl, serial fraction | memory hierarchy, occupancy, transfers, roofline | decomposition, halo exchange, reproducible measurement |

Each part needs to understand the common interface in `src/common/config.h`, `image.h` and `timing.h`, and the validation rule, since every result depends on them.

## Suggested 15-minute order

1. Problem, algorithm and measurement design (all, 2 min).
2. Part 1 (4 min), then Part 2 (4 min), then Part 3 (4 min).
3. Combined conclusion and limitations (1 min).
