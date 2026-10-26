# ConvoX: Technical Requirements Document

Companion to `docs/01-product/PRD.md`. Requirement IDs (FR1...) refer to that file.

## 1. Stack

| Concern | Choice |
|---|---|
| Language | C++17 for host code, CUDA C++ for kernels, Python 3.10+ for analysis |
| Compilers | GCC 11+ (or Clang 14+) with `-fopenmp`, `nvcc` from CUDA 12.x |
| Build | CMake 3.20+. CUDA and MPI targets are optional and detected at configure time. |
| Image I/O | `stb_image` / `stb_image_write`, vendored under `third_party/` (single headers). OpenCV is not used. |
| Parallel | OpenMP 4.5+, CUDA, optional OpenMPI 4.1+ or MPICH |
| Profiling | Nsight Compute (`ncu`), Nsight Systems (`nsys`), `nvidia-smi` |
| Analysis | pandas, numpy, matplotlib |
| OS | Ubuntu 22.04, native or WSL2 |

## 2. Repository layout

```
src/common/      image I/O, filter generation, validator, timing harness, CSV writer, CLI
src/serial/      reference convolution
src/openmp/      OpenMP variants
src/cuda/        naive, const, const+shared kernels and host wrappers
src/mpi/         strip scatter/gather driver (optional)
benchmarks/      sweep driver configs and scripts that run the full matrix
analysis/        plotting and Amdahl scripts
scripts/         build helpers, run_all, summary
data/images/     input images (git-ignored)
results/raw/     results.csv (git-ignored)
results/plots/   PNG output (git-ignored)
report/          written report
references/      synopsis and cited material
third_party/     stb headers
```

## 3. Module design

### 3.1 Common interface

Every implementation exposes the same function shape so the harness can call any of them:

```cpp
struct Image  { int w, h; std::vector<float> px; };      // single channel, row-major
struct Filter { int k; std::vector<float> w; };           // k odd, row-major
struct Config { /* impl, variant, threads, schedule, chunk, collapse, block_x, block_y, border */ };
struct Timing { double kernel_ms, h2d_ms, d2h_ms, total_ms; };

Timing convolve_<impl>(const Image& in, const Filter& f, const Config& cfg, Image& out);
```

CPU implementations report `kernel_ms == total_ms` and zero transfer times.

### 3.2 Algorithm (all versions)

`O[y][x] = sum_{i=-R..R} sum_{j=-R..R} I[y+i][x+j] * F[i+R][j+R]`, R = (K-1)/2, float32. Out-of-range reads return 0 (zero-pad) or the clamped edge pixel when built with `CONVOX_CLAMP_EDGE`.

### 3.3 Serial (FR4)

Loops over output rows, output columns, filter rows, filter columns. Built at `-O2`. No pragmas and no `restrict`. Compiled in its own translation unit with `-fno-tree-vectorize` so the oracle is not auto-optimised.

### 3.4 OpenMP (FR5)

- Row variant: `#pragma omp parallel for` over `y`, filter loops serial.
- Collapse variant: `collapse(2)` over `y, x`.
- Schedule is set at run time with `schedule(runtime)` and `omp_set_schedule`, so one binary covers static, dynamic(1/8/32) and guided.
- Thread count via `omp_set_num_threads`. Sweep {1, 2, 4, 6, 8, 12, 16}.

### 3.5 CUDA (FR6, FR7)

Three kernels, kept separate so each optimisation is attributable:

| Variant | Filter | Input |
|---|---|---|
| `naive` | global memory | global memory |
| `const` | `__constant__ float d_filter[MAX_K*MAX_K]` via `cudaMemcpyToSymbol` | global memory |
| `tiled` | `__constant__` | `__shared__` tile of `(T+K-1)^2` floats, loaded cooperatively with halo, then `__syncthreads()` |

- `MAX_K = 15`, so the filter takes 900 bytes.
- Grid is `ceil(W/BX) x ceil(H/BY)` with a bounds guard.
- Tile size T equals the block dimension. At T=32, K=15 the tile is 46x46 floats, about 8.5 KB per block.
- Block sizes 8x8, 16x16, 32x32. Each is a template or runtime parameter, not a rebuild.
- Timing: `cudaEvent` pairs around H2D, kernel and D2H for the split, plus a host `steady_clock` span around the whole path, with `cudaDeviceSynchronize()` before stopping it.
- Every CUDA call is wrapped in an error-check macro.

### 3.6 MPI (FR15, optional)

Rank 0 loads the image and scatters strips of `H/N + 2R` rows (halo overlap) with `MPI_Scatterv`. Each rank runs the local OpenMP or CUDA kernel. Interior rows (halo excluded) are returned with `MPI_Gatherv`. Edge ranks pad their outer halo according to the border mode. Built only when MPI is found.

### 3.7 Validator (FR8)

Compares an output against the serial result element by element. Reports max absolute error and RMS error. Pass condition is max-abs < 1e-4. A failing run is still written to the CSV with `pass=0`. The harness exits non-zero if any run fails.

### 3.8 Timing harness (FR9, FR10)

For each configuration: 1 discarded warm-up run, then 10 measured runs. Store every repetition in `results/raw/reps.csv` and the median in `results.csv`. CPU clock is `std::chrono::steady_clock`. GPU kernel time is from CUDA events.

### 3.9 CLI

One executable, `convox`:

```
convox run   --impl {serial|omp|cuda} --variant ... --image PATH --k 7 --filter gaussian
             [--threads N --schedule static|dynamic|guided --chunk N --collapse]
             [--block 16x16] [--reps 10] [--out results/raw/results.csv] [--save-image PATH]
convox sweep --config benchmarks/sweep.conf
convox info                      # CPU, GPU, compiler, OpenMP and CUDA versions
```

The MPI driver is a separate binary, launched with `mpirun -np N convox_mpi ...`.

### 3.10 Analysis (FR12, FR13)

- `analysis/plots.py` reads `results.csv` and writes the five plots: speedup vs threads, runtime vs K per implementation, runtime vs block size, stacked kernel-vs-transfer, roofline-style intensity.
- `analysis/amdahl.py` estimates the serial fraction `f` and plots `S(N) = 1 / (f + (1-f)/N)` against measured speedup.
- `scripts/summary.py` prints aligned terminal tables. `--show` opens plots in the default viewer.

### 3.11 Demo interface

`ui/server.py` (standard library plus Pillow, no network access needed) serves `ui/static/` and a small JSON API:

| Endpoint | Purpose |
|---|---|
| `GET /api/info` | CPU, GPU, available samples and saved plots |
| `POST /api/run?filter=&k=&reps=&threads=&maxdim=&omp_sweep=&block_sweep=&sample=` | start a job; the body is the uploaded image, or `sample` names a generated image |
| `GET /api/job/<id>` | progress and results so far |
| `GET /files/<id>/original.jpg`, `output.jpg` | display images |
| `GET /plots/<name>.png` | saved benchmark plots |

A job converts the image to 8-bit grayscale, then calls `build/convox run --json` once per version (serial, OpenMP, three CUDA kernels, plus optional OpenMP thread and CUDA block sweeps), so each result is validated against the serial output by the same code as the benchmark. Only one job runs at a time (HTTP 409 otherwise) because timings need the machine to themselves. The server listens on 127.0.0.1 only, limits uploads to 200 MB and 60 megapixels, validates every parameter, and serves files only by fixed name pattern. `scripts/check_ui.sh` tests the API end to end.

## 4. Analytical model

- Naive kernel: 2K^2 FLOPs per output pixel and 4K^2 bytes loaded, so arithmetic intensity is about 0.5 FLOP/byte for any K.
- Tiled kernel: loads `(T+K-1)^2 / T^2` input elements per output pixel. For T=16, K=5 this falls from 25 to about 1.56 (about 16x reuse), raising intensity to about 8 FLOP/byte.
- Serial fraction `f` is measured from image I/O, allocation, filter setup and (for GPU) PCIe transfer, and substituted into Amdahl's law.

## 5. Non-functional requirements

| Area | Requirement |
|---|---|
| Reproducibility | One command (`scripts/run_all.sh`) rebuilds, runs the sweep and regenerates plots. Seeds and image list are fixed. |
| Fairness | Same images, filters and validator for every implementation. Same compiler flags documented. |
| Environment capture | Every CSV row records CPU, GPU, compiler, CUDA version and git commit. |
| Memory | An 8K float image is about 133 MB. Peak host RAM stays under 2 GB for single-node runs. |
| Portability | CPU-only build must work without CUDA or MPI present. |
| Code size | Core implementations stay within the synopsis budget (about 400-600 lines plus about 100 lines of Python). |

## 6. Testing

- Unit check: a 1-pixel delta filter returns the input, and a box filter on a constant image returns that constant.
- Correctness: every implementation against serial on a small image, then on full-size images in the sweep.
- Edge cases: K=3 and K=15, image dimensions not divisible by the block size, and image smaller than the filter.
- Border modes: zero-pad and clamp, built both ways.

## 7. Resolved questions

- CUDA runs through WSL2 on the development laptop (RTX 3050 Laptop GPU, CUDA 12.6), so no Colab or lab machine was needed.
- No MPI cluster is available. MPI is demonstrated with several ranks on one node (`mpirun --oversubscribe --bind-to none`).
- Test images are synthetic and generated by `scripts/gen_images.py`, since convolution time does not depend on pixel values.
- `convox sweep` reads a plain `key = values` file (`benchmarks/sweep.conf`) instead of YAML, to avoid a parser dependency.
- The serial fraction for Amdahl's law is measured by `convox profile` (image decode, filter build, allocation, image encode) and stored in `serial_profile.csv`.
