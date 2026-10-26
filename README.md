# ConvoX

A comparative study of shared-memory, GPU and distributed parallelisation strategies for 2D image
convolution. One algorithm (convolution with an odd K x K filter, K = 3..15) is implemented as:

| Name | What it is |
|---|---|
| `serial` | Plain C++ reference. The correctness oracle and the speedup baseline. |
| `omp` | OpenMP: row or `collapse(2)` decomposition, static / dynamic / guided schedules, 1-16 threads |
| `cuda-naive` | One thread per output pixel, everything from global memory |
| `cuda-const` | Filter coefficients in `__constant__` memory |
| `cuda-tiled` | Constant-memory filter plus a shared-memory input tile with halo |
| `mpi-*` | Optional: horizontal strips with halo rows scattered across MPI ranks |

Every output is validated against the serial reference (max absolute error < 1e-4). Results are written
to CSV, plotted with matplotlib, and compared with Amdahl's law and a roofline view.

Course project for Parallel Computing Architecture and Programming (PCAP), Semester V, Manipal
Institute of Technology, Bengaluru. Team: Saatvik Das, Saket Rama, Zaid Khan. Guide: Prof. Anil Kumar.
The synopsis is in `references/`; planning documents are in `docs/`.

## Requirements

Ubuntu 22.04 or newer (native or WSL2), GCC 11+, CMake 3.20+, Python 3.10+ with pandas, matplotlib and
numpy. Optional: CUDA Toolkit 12.x with an NVIDIA GPU (compute capability 6.0+), OpenMPI 4.1+.
CUDA and MPI are detected at configure time; without them the CPU parts still build and run.

## Build and test

```bash
bash scripts/build.sh          # builds ./build/convox
bash scripts/run_tests.sh      # builds and runs every test in tests/
bash scripts/check_cli.sh      # CLI exit-code smoke checks
```

Test images are generated, not downloaded:

```bash
python3 scripts/gen_images.py  # data/images/img01_4k ... img04_8k plus images.csv
```

## Use

```bash
./build/convox info
./build/convox run --impl serial     -i data/images/img01_4k.png -k 7
./build/convox run --impl omp        -i data/images/img03_8k.png -k 7 --threads 8 --schedule static
./build/convox run --impl cuda-tiled -i data/images/img03_8k.png -k 7 --block 16x16
./build/convox sweep --dry-run                      # size of the full benchmark matrix
./build/convox sweep                                # run it (long); appends to results/raw/; add --resume after an interruption
```

`run` prints one line per configuration and appends a row to `results/raw/results.csv`. Exit codes:
0 ok, 1 validation failed, 2 bad arguments, 3 implementation or device unavailable.
Add `--save-image out.png` to write the result (`--save-mode abs` for Sobel output).

## Demo page

For a live demonstration, double-click `start_demo.bat` (Windows) or run `bash scripts/start_ui.sh` inside Ubuntu, then open http://localhost:8000. Pick a sample image or upload your own, choose the effect and filter size, and press Run. The page runs serial, OpenMP and the three CUDA versions on your image, checks each against the serial result, and draws the images, timings, speedups and charts as they arrive. Shrinking the image (setting 4) makes runs much faster. `bash scripts/check_ui.sh` tests the page's server end to end.

## Reproduce all results

```bash
bash scripts/run_all.sh          # build, test, full sweep, tables, plots
bash scripts/run_all.sh --quick  # small sweep into results/quick/
```

Or step by step:

```bash
python3 scripts/summary.py --image img03_8k --k 7   # terminal table
python3 analysis/plots.py --show                    # PNG + PDF plots in results/plots/
python3 analysis/amdahl.py                          # Amdahl prediction vs measurement
bash scripts/visual_check.sh                        # original vs Gaussian / Sobel / box
```

MPI (single node; `--bind-to none` stops Open MPI pinning each rank to one core):

```bash
mpirun --oversubscribe --bind-to none -np 4 ./build/convox_mpi -i data/images/img03_8k.png -k 7 --local omp
bash scripts/check_mpi.sh                       # every rank count and local kernel vs serial
python3 analysis/ncu_profile.py                 # DRAM / L2 / L1 traffic per CUDA kernel (needs GPU counters)
```

Border handling is zero-padding by default. For clamp-to-edge, configure with
`bash scripts/build.sh -DCONVOX_CLAMP_EDGE=ON`.

## Layout

```
src/common/    image I/O, filters, validator, timing, CSV, CLI, sweep driver
src/serial/    reference convolution
src/openmp/    OpenMP implementation
src/cuda/      CUDA kernels and device queries
src/mpi/       MPI strip decomposition (optional)
ui/            demo page and its server
benchmarks/    sweep definition and helper sweeps
analysis/      plotting and Amdahl scripts
scripts/       build, test, data generation, summaries
tests/         unit and cross-implementation checks
docs/          PRD, TRD, flows, schema, implementation plan
results/       CSV and plots (git-ignored)
```
