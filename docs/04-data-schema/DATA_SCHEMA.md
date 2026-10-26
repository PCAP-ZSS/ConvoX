# ConvoX: Data Schema

There is no backend or database. All data lives in flat files, so every table here is a CSV or JSON file. Files are written by the C++ harness and read by the Python analysis.

## 1. Files

| File | Written by | Purpose |
|---|---|---|
| `results/raw/results.csv` | `convox` | One row per configuration, with the median timing and validation result |
| `results/raw/reps.csv` | `convox` | One row per repetition (warm-up excluded), for variance and error bars |
| `results/raw/sweep_manifest.json` | `convox sweep` | Environment and sweep definition |
| `benchmarks/sweep.yaml` | people | Input: what to sweep |
| `data/images/images.csv` | people | Input: image catalogue |
| `results/plots/*.png`, `*.pdf` | `analysis/plots.py` | Derived figures |

## 2. `results.csv`

One row per unique configuration. Append only. Never edit by hand.

| Column | Type | Description |
|---|---|---|
| `run_id` | string | Unique per row (timestamp plus counter) |
| `sweep_id` | string | Groups the rows of one sweep |
| `timestamp` | ISO 8601 | When the configuration finished |
| `impl` | enum | `serial`, `omp`, `cuda-naive`, `cuda-const`, `cuda-tiled`, `mpi-omp`, `mpi-cuda` |
| `image_id` | string | Key into `images.csv` |
| `width`, `height` | int | Image size in pixels |
| `filter_type` | enum | `gaussian`, `sobel`, `box` |
| `k` | int | Filter size, odd, 3 to 15 |
| `border` | enum | `zero`, `clamp` |
| `threads` | int | OpenMP threads. Empty for non-OpenMP. |
| `schedule` | enum | `static`, `dynamic`, `guided`. Empty for non-OpenMP. |
| `chunk` | int | Schedule chunk size. Empty if not applicable. |
| `collapse` | int | 1 (rows) or 2 (`collapse(2)`). Empty for non-OpenMP. |
| `block_x`, `block_y` | int | CUDA block dimensions. Empty for non-CUDA. |
| `mpi_ranks` | int | Rank count. Empty if not MPI. |
| `reps` | int | Number of measured repetitions (10) |
| `kernel_ms` | float | Median kernel time. For CPU, equals `total_ms`. |
| `h2d_ms`, `d2h_ms` | float | Median transfer times. Empty for CPU. |
| `total_ms` | float | Median end-to-end time (H2D + kernel + D2H for GPU) |
| `max_abs_err` | float | Versus serial reference |
| `rms_err` | float | Versus serial reference |
| `pass` | 0/1 | 1 if `max_abs_err < 1e-4` |
| `smem_bytes` | int | Shared memory per block (tiled only) |
| `occupancy` | float | Achieved occupancy if profiled, else empty |
| `dram_gbps` | float | Profiler-measured DRAM throughput if profiled, else empty |
| `cpu_model`, `gpu_model` | string | Hardware identifiers |
| `compiler`, `cuda_version` | string | Toolchain identifiers |
| `git_commit` | string | Short hash of the code that produced the row |

Primary key: `run_id`. Unique configuration: (`impl`, `image_id`, `filter_type`, `k`, `border`, `threads`, `schedule`, `chunk`, `collapse`, `block_x`, `block_y`, `mpi_ranks`, `sweep_id`).

Speedup is not stored. It is derived in analysis as `T_serial / T` for the same `image_id`, `filter_type`, `k` and `border`, so the baseline can't drift out of sync.

## 3. `reps.csv`

| Column | Type | Description |
|---|---|---|
| `run_id` | string | Foreign key to `results.csv` |
| `rep` | int | 1 to 10 |
| `kernel_ms`, `h2d_ms`, `d2h_ms`, `total_ms` | float | Timings for this repetition |

## 4. `images.csv`

| Column | Type | Description |
|---|---|---|
| `image_id` | string | Short key, for example `img01_4k` |
| `path` | string | Relative path under `data/images/` |
| `width`, `height` | int | Pixels |
| `class` | enum | `4k` (3840x2160) or `8k` (7680x4320) |
| `source` | string | Where the image came from, and its licence |

## 5. `sweep.yaml`

```yaml
images: [img01_4k, img02_4k, img03_8k, img04_8k]
filter: gaussian
k: [3, 5, 7, 9, 11, 13, 15]
reps: 10
omp:
  threads: [1, 2, 4, 6, 8, 12, 16]
  schedule: [static, "dynamic:1", "dynamic:8", "dynamic:32", guided]
  collapse: [1, 2]
cuda:
  variants: [naive, const, tiled]
  block: [8x8, 16x16, 32x32]
mpi:
  ranks: [2, 4]
  local: [omp, cuda]
```

Tables I and II in the synopsis define the values. Combinations that don't apply to an implementation are skipped.

## 6. `sweep_manifest.json`

```json
{
  "sweep_id": "2026-11-02T10-15-00",
  "git_commit": "abc1234",
  "cpu": {"model": "", "physical_cores": 0, "logical_cores": 0},
  "gpu": {"model": "", "compute_capability": "", "vram_mb": 0, "driver": ""},
  "compiler": "", "cuda_version": "", "openmp_version": "",
  "sweep_config": "benchmarks/sweep.yaml"
}
```

## 7. Validation rules

- `k` is odd and between 3 and 15.
- `threads` is empty unless `impl` is `omp` or an MPI variant using OpenMP.
- `block_x * block_y` is at most 1024.
- `pass` is 1 only if `max_abs_err < 1e-4`.
- `total_ms >= kernel_ms`.
- Rows with `pass = 0` are kept in the file and flagged in every summary. They are never dropped silently.

## 8. Derived quantities (analysis only)

| Quantity | Formula |
|---|---|
| Speedup | `T_serial / T` |
| Parallel efficiency | `speedup / threads` |
| Arithmetic intensity (naive) | about 0.5 FLOP/byte |
| Arithmetic intensity (tiled) | `2K^2 / (4 * (T+K-1)^2 / T^2)` FLOP/byte |
| Serial fraction `f` | I/O + allocation + filter setup + PCIe, divided by the serial total |
| Amdahl ceiling | `S(N) = 1 / (f + (1-f)/N)` |
