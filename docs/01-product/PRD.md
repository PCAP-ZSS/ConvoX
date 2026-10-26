# ConvoX: Product Requirements Document

| | |
|---|---|
| Course | Parallel Computing Architecture and Programming (PCAP), Semester V |
| Institute | Manipal Institute of Technology, Bengaluru, CSE |
| Guide | Prof. Anil Kumar |
| Team | Saatvik Das, Saket Rama, Zaid Khan |
| Source | `references/PCAP_Synopsis_IEEE_Format.pdf` |

## 1. Summary

ConvoX is a command-line benchmark suite that implements one algorithm, 2D image convolution with an arbitrary odd-sized filter, in four progressively parallel forms (serial C++, OpenMP, naive CUDA, optimised CUDA) plus an optional MPI layer. It measures all of them under identical conditions, validates every output against the serial reference, and produces a results table, plots and an analytical report.

It is a study tool, not a consumer product. There is no web UI. Results are delivered as a CSV file, PNG plots and a terminal summary.

## 2. Problem

Production convolution libraries (cuDNN, OpenCV, oneDNN) are tuned black boxes. They report a timing but do not show which mechanism produced the speedup: thread parallelism, cache blocking, constant-memory broadcast or shared-memory reuse. Published comparisons often skip host-to-device transfer cost and the effect of launch configuration. ConvoX builds the same algorithm four times and attributes each performance gain to a named hardware mechanism.

## 3. Goals

| ID | Goal |
|---|---|
| G1 | A correct serial C++ reference, which is the correctness oracle and the speedup denominator |
| G2 | An OpenMP version with row-wise and `collapse(2)` decomposition and static, dynamic and guided scheduling, measured from 1 to 2x physical cores |
| G3 | A naive CUDA kernel (one thread per output pixel, global memory only) with a block-dimension sweep |
| G4 | An optimised CUDA kernel using `__constant__` filter memory and shared-memory tiles with halo, with the two optimisations measurable separately |
| G5 | A full sweep over image size and filter size, reporting kernel-only and end-to-end times, and an Amdahl's-law comparison of predicted and observed speedup |
| G6 | (Extension) An MPI layer that scatters horizontal strips with halo rows and gathers results at rank 0 |

## 4. Non-goals

- A hosted web service. A local demo page that runs the same program is provided (`ui/`, see section 8).
- Beating cuDNN or OpenCV, or implementing FFT, Winograd or im2col methods.
- Separable-filter optimisation, multi-channel batching or video pipelines.
- Communication-optimal MPI distribution.
- Code volume. The synopsis targets about 400-600 lines of C++/CUDA and about 100 lines of Python.

## 5. Users

| User | Need |
|---|---|
| Team (builders) | Run one command to reproduce the whole sweep |
| Faculty evaluator | See correctness evidence, clear plots and an explanation of the results |
| A future student | Read the code and report as a worked example of memory-hierarchy reasoning |

## 6. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | Load PNG/JPG images via stb_image and convert to a single-channel float buffer. The image library is used for file I/O only, never for convolution. | Must |
| FR2 | Generate K x K filters: normalised Gaussian (default), Sobel and Box. K is odd, 3 to 15. | Must |
| FR3 | Zero-padding at borders by default. Clamp-to-edge available as a compile-time option. | Must |
| FR4 | Serial implementation with four nested loops, built at `-O2`, no vectorisation hints. | Must |
| FR5 | OpenMP implementation. Variants: rows vs `collapse(2)`, schedules static / dynamic(1, 8, 32) / guided, threads {1, 2, 4, 6, 8, 12, 16}. | Must |
| FR6 | Naive CUDA kernel, block sizes 8x8, 16x16, 32x32, with a bounds guard for non-divisible sizes. | Must |
| FR7 | CUDA +constant variant, then CUDA +constant +shared-tiled variant. | Must |
| FR8 | Validator compares each output to the serial reference. Pass means max-abs error below 1e-4. RMS is also reported. | Must |
| FR9 | Timing harness: one discarded warm-up run and 10 measured runs, median reported. CPU uses `steady_clock`. GPU kernels use CUDA events. | Must |
| FR10 | GPU runs record kernel-only, H2D, D2H and end-to-end times. | Must |
| FR11 | Every run appends rows to `results/raw/results.csv` using the schema in `docs/04-data-schema`. | Must |
| FR12 | `analysis/` produces the plots and the Amdahl comparison from the CSV. | Must |
| FR13 | Terminal summary tables and an option to write or open the plots. | Should |
| FR14 | Write the convolved image to disk for visual checks (Gaussian blur, Sobel edges). | Should |
| FR15 | MPI strip decomposition with halo rows (`MPI_Scatterv` / `MPI_Gatherv`), each rank running the OpenMP or CUDA kernel. | Could |
| FR16 | Nsight Compute profiling of DRAM throughput for naive vs tiled kernels, with the numbers recorded in the report. | Should |

## 7. Success criteria

- All implementations pass validation (max-abs error < 1e-4) across the full image and filter sweep.
- One scripted command reproduces `results.csv` and every plot.
- The OpenMP scaling curve shows a plateau, and the report explains it with the bandwidth argument.
- The naive-vs-tiled CUDA gap is shown to grow with K, with the arithmetic-intensity explanation.
- A stacked kernel-vs-transfer chart shows the real end-to-end cost of GPU offload.
- Amdahl's predicted ceiling is plotted against the observed plateau, with a written explanation of the gap.
- The report and a presentation can be split into three equal parts (see `docs/06-presentation`).

## 8. How results are delivered

| Output | Form |
|---|---|
| Raw measurements | `results/raw/results.csv` |
| Per-run console summary | Aligned table printed by `scripts/summary.py` |
| Plots | PNG files in `results/plots/`, optionally opened with `--show` |
| Visual correctness check | Output images from FR14 |
| Analysis | Written report in `report/` |
| Live demo | Local web page (`start_demo.bat` or `scripts/start_ui.sh`): pick or upload an image, run every version, see images, timings, speedups and charts. It calls the real `convox` program; nothing is simulated. |

## 9. Constraints and assumptions

- Hardware: x86-64 CPU with at least 4 physical cores, 16 GB RAM, NVIDIA GPU with compute capability 6.0 or higher and 4 GB VRAM or more. A Colab T4 is an acceptable substitute.
- Software: Ubuntu 22.04 (native or WSL2), GCC 11+, CUDA 12.x, optional OpenMPI, Python 3.10+.
- The development machine runs Windows 11, so the toolchain probably runs under WSL2. This is unconfirmed.
- Timeline in the synopsis is six weeks. See `docs/05-planning`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| No local NVIDIA GPU or no WSL2 CUDA support | Use Colab T4 or a lab machine. Keep the CUDA code buildable in a standalone notebook-free way. |
| Timing noise (OS scheduling, DVFS) | Median of 10 runs after warm-up. Fix power mode and note the CPU model in results. |
| Compiler auto-parallelises or vectorises the serial baseline | Build at `-O2`, inspect the assembly or vectorisation report once |
| MPI is unavailable or cluster access is missing | MPI is optional. Demonstrate multiple ranks on one node. |
| Scope creep | The synopsis is the scope. The extension (MPI) is dropped first if time runs short. |
