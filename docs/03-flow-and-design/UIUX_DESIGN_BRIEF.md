# ConvoX: UI/UX Design Brief

ConvoX has no graphical interface. Its user experience is the command line, the terminal output, the plots and the report. This brief sets the rules for each so that the output is clear in a viva and in a screenshot.

## 1. Principles

1. **One obvious command per job.** Build, run one config, run the sweep, see results.
2. **Output that can be read in a screenshot.** Aligned columns, units on every number, no wrapped lines at 100 columns.
3. **Comparable at a glance.** Every table and plot uses the same implementation names, order and colours.
4. **Honest numbers.** Always show kernel-only and end-to-end for GPU. Never show a speedup without saying what it is relative to.
5. **Fail loudly and helpfully.** An error says what went wrong and what to do about it.

## 2. Naming

| Concept | Label used everywhere |
|---|---|
| Serial baseline | `serial` |
| OpenMP | `omp` (with `threads`, `schedule`, `collapse` as attributes) |
| Naive CUDA | `cuda-naive` |
| CUDA constant memory | `cuda-const` |
| CUDA constant + shared tiling | `cuda-tiled` |
| MPI | `mpi-omp`, `mpi-cuda` |

## 3. CLI design

- Subcommands: `run`, `sweep`, `info`. Long flags only, with short aliases for `--image`, `--k` and `--impl`.
- Every flag has a default, and `--help` shows an example for each subcommand.
- Invalid input is rejected before any work starts. Example: `error: K must be odd and between 3 and 15 (got 8)`.
- Exit codes: 0 success, 1 validation failure, 2 bad arguments, 3 missing device or library.

## 4. Terminal output

Single-run line:

```
cuda-tiled  4K  K=7  block=16x16  kernel 3.21 ms  total 12.84 ms  speedup 41.2x  err 2.1e-6  PASS
```

Sweep progress: one line per completed run, then `[ 37 / 420 ]  ~6 min left`.

Summary table (`scripts/summary.py`), one row per implementation for a chosen image and K:

```
impl         threads  kernel_ms  total_ms  speedup  eff   err      pass
serial            1     812.4     812.4      1.00  1.00  0         yes
omp               8     148.9     148.9      5.46  0.68  1.8e-7    yes
cuda-naive        -      21.7      33.1     24.5   -     1.2e-7    yes
cuda-tiled        -       3.2      14.8     54.9   -     2.1e-6    yes
```

Rules: right-align numbers, state units in headers, show speedup to 2 decimals, mark failures in plain text (`FAIL`), and use colour only as a second cue, never the only one.

## 5. Plots

Five required plots, saved as PNG at 200 dpi and also as vector PDF for the report.

| # | Plot | X | Y | Notes |
|---|---|---|---|---|
| 1 | OpenMP scaling | threads | speedup | Include the ideal-linear line and the Amdahl ceiling. One line per K (3, 7, 15). |
| 2 | Runtime vs filter size | K | runtime (log scale) | One line per implementation. |
| 3 | Block-dimension sensitivity | block size | kernel ms | Grouped bars for naive and tiled, per K. |
| 4 | Transfer vs kernel | implementation | ms (stacked) | H2D, kernel, D2H segments. |
| 5 | Roofline-style | arithmetic intensity | GFLOP/s | Naive and tiled points per K, with memory and compute roofs. |

Style rules:
- Fixed colour per implementation across all plots. Use a colour-blind-safe palette.
- Titles state the finding, not just the variables (for example, "Speedup plateaus below core count").
- Axis labels carry units. Log scale only where the range needs it, and say so.
- Show the hardware (CPU and GPU model) in a footer line on every plot.
- Error bars or a min-max band for the 10 repetitions where it is readable.
- Legible when printed in greyscale: vary line style and markers as well as colour.

## 6. Visual correctness check

For the demo, write three output images from one sample photo: Gaussian blur, Sobel edges and box blur, saved side by side with the original. Evaluators see correctness directly, not only a pass flag.

## 7. Report layout

`report/` follows the IEEE structure of the synopsis: abstract, introduction, methodology, experimental setup, results (one subsection per plot), discussion (bandwidth saturation, Amdahl gap, transfer cost), conclusion, references. Each plot sits next to the paragraph that interprets it.

## 8. Presentation

Slides mirror the pipeline: the problem, the four implementations, the measurement design, the five results, the conclusion. Use the same colours and labels as the plots. The three-way split of the speaking parts is in `docs/06-presentation`.
