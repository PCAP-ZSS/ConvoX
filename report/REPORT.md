# A Comparative Study of Shared-Memory, GPU and Distributed Parallelisation Strategies for 2D Image Convolution

Saatvik Das, Saket Rama, Zaid Khan. Department of Computer Science and Engineering, Manipal Institute of Technology, Bengaluru. Guide: Prof. Anil Kumar. PCAP, Semester V.

All numbers below come from `results/raw/results.csv` (728 configurations, every one validated against the serial reference, maximum absolute error 4.8e-07 against a 1e-4 limit) and `results/raw/ncu_metrics.csv`. Plots are in `results/plots/`. Unless stated, figures are for the 8K image (7680x4320) with a Gaussian filter and the median of 10 runs after one warm-up.

## 1. Abstract

Two-dimensional convolution is implemented as a serial C++ baseline, an OpenMP version, a naive CUDA kernel, and a CUDA kernel with constant-memory filter and shared-memory tiling, plus an MPI strip decomposition. Measured on an Intel i7-12650H and an RTX 3050 Laptop GPU, the OpenMP version reaches 6.7x to 8.5x over serial at 16 threads and stops scaling near 4 threads. The tiled CUDA kernel is 0.9x to 1.9x faster than the naive kernel, with the gain growing with filter size. Nsight Compute shows why: DRAM traffic is identical for all three kernels because the caches already capture the reuse, while tiling cuts global-load traffic through L1 by up to 86x. Host-device transfers take 70% to 93% of GPU end-to-end time, so an end-to-end speedup of 8x to 110x corresponds to a kernel-only speedup of 165x to 389x. Measured serial fractions bound end-to-end OpenMP speedup to between 1.1x and 2.6x, and the measured speedups sit within 2% to 16% of that bound.

## 2. Setup

| Item | Value |
|---|---|
| CPU | 12th Gen Intel Core i7-12650H (6 performance + 4 efficiency cores, 16 hardware threads) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB, compute capability 8.6, peak about 188 GB/s and 6.1 TFLOP/s |
| Software | WSL2 Ubuntu 24.04, GCC 13.3 (-O2), CUDA 12.6, OpenMPI 4.1.6 |
| Images | 2 images at 3840x2160 and 2 at 7680x4320, synthetic (smooth regions plus sharp edges) |
| Filter | Gaussian, K = 3, 5, 7, 9, 11, 13, 15; zero-padded borders; float32 |
| Protocol | one discarded warm-up, 10 timed runs, median; CPU `steady_clock`, GPU CUDA events plus a host clock around transfers and kernel |

Convolution cost does not depend on pixel values, which is why synthetic images are valid benchmarks. The serial baseline is compiled with vectorisation disabled so that it stays a plain scalar loop.

## 3. Results

### 3.1 OpenMP scaling (`omp_scaling.png`)

Speedup over serial, static schedule, row parallelism:

| Threads | K=3 | K=7 | K=15 |
|---|---|---|---|
| 1 | 1.2x | 1.0x | 1.1x |
| 4 | 3.8x | 3.9x | 4.1x |
| 8 | 4.8x | 5.3x | 6.0x |
| 16 | 6.7x | 6.0x | 8.5x |

Scaling is close to linear up to 4 threads, then flattens. Efficiency at 16 threads is 0.38 to 0.53, lower for small filters (K=3: 0.42) than for large ones (K=15: 0.53). Small filters do little arithmetic per byte, so threads compete for the same memory bandwidth sooner. K=7 even gets slower between 12 and 16 threads (216 to 238 ms). The CPU is a hybrid design, so threads beyond 6 land on efficiency cores or on hyper-threads of busy cores, which adds to the shortfall. This test machine cannot separate bandwidth saturation from the hybrid-core effect.

### 3.2 Schedules and collapse (`omp_schedules.png`)

At 8 threads, K=7, 8K image:

| Schedule | rows (ms) | collapse(2) (ms) |
|---|---|---|
| static | 271 | 288 |
| dynamic, chunk 1 | 238 | 1209 |
| dynamic, chunk 8 | 244 | 324 |
| dynamic, chunk 32 | 247 | 266 |
| guided | 246 | 255 |

The synopsis expected static to win on uniform rows. Here dynamic and guided are 9% to 12% faster than static, which is plausible on a hybrid CPU where some cores are slower, so equal static shares finish unevenly; this was not isolated in a separate experiment. The clear result is the cost of fine-grained distribution: `collapse(2)` with chunk 1 hands out single pixels and is 5x slower than the same schedule over rows.

### 3.3 CUDA block dimensions (`block_sensitivity.png`)

Kernel time in ms, 8K image:

| K | Kernel | 8x8 | 16x16 | 32x32 |
|---|---|---|---|---|
| 3 | naive | 3.88 | 2.50 | 4.10 |
| 3 | tiled | 3.51 | 2.80 | 4.48 |
| 9 | naive | 14.57 | 11.72 | 15.92 |
| 9 | tiled | 7.38 | 6.68 | 9.21 |
| 15 | naive | 40.76 | 31.20 | 36.83 |
| 15 | tiled | 17.73 | 16.09 | 19.73 |

16x16 is fastest for every kernel and every K shown, as the synopsis predicted. 8x8 blocks have only two warps and, for the tiled kernel, a large halo relative to the tile. 32x32 uses 1024 threads, the hardware maximum, and a larger shared tile, which limits resident blocks per SM.

### 3.4 Constant memory and tiling

Kernel-only time ratio naive / tiled at 16x16: 0.89 (K=3), 1.14 (K=5), 1.39 (K=7), 1.75 (K=9), 1.84 (K=11), 1.91 (K=13), 1.94 (K=15). The gain is absent at K=3, where tiling is slightly slower, and grows with K, as predicted. Constant memory alone helps by 3% to 12% at the sizes in the table above.

Nsight Compute at 16x16, 8K image:

| K | Kernel | DRAM read (MB) | DRAM write (MB) | L1 global loads (MB) | Warps active (%) | Kernel time (ms) |
|---|---|---|---|---|---|---|
| 3 | naive | 132.8 | 132.1 | 1889 | 70.9 | 3.2 |
| 3 | tiled | 132.9 | 132.1 | 322 | 93.6 | 3.6 |
| 7 | naive | 133.1 | 132.2 | 10905 | 81.9 | 11.3 |
| 7 | tiled | 133.6 | 132.1 | 408 | 96.3 | 8.2 |
| 15 | naive | 137.9 | 132.5 | 51192 | 88.3 | 47.7 |
| 15 | tiled | 147.6 | 132.3 | 597 | 97.4 | 25.0 |

This corrects an expectation in the synopsis. The 8K image is 133 MB, and every kernel reads about that much and writes the same amount: each pixel crosses DRAM once, because the L1 and L2 caches already serve the K^2 reuse of the naive kernel. Tiling does not reduce DRAM traffic. What it removes is traffic between the cache and the SM, 51 GB down to 0.6 GB at K=15 (86x). The speedup then comes from fewer load instructions and higher occupancy, not from saved DRAM bandwidth. Profiler kernel times are somewhat higher than the harness times because the profiler serialises and fixes clocks.

### 3.5 Roofline (`roofline.png`)

Analytical intensity is about 0.5 FLOP/byte for the naive kernel and `2K^2 / (4 (T+K-1)^2 / T^2)` for the tiled kernel, 3.6 to 32 FLOP/byte for T=16. Naive points plot above the memory roof because the real DRAM intensity is much higher than the analytical value, for the cache reason in 3.4. Tiled points move right, and at K=15 reach about 930 GFLOP/s, 15% of the 6.1 TFLOP/s peak. They remain well below the roofs, so something other than DRAM bandwidth or arithmetic peak limits them; load and shared-memory instruction throughput are the likely candidates, but this was not measured.

### 3.6 Transfers (`transfer_breakdown.png`)

For the tiled kernel at 16x16, host-device copies are 93% of the end-to-end time at K=3 and 70% at K=15. Kernel-only versus end-to-end speedup over serial:

| K | Naive kernel only | Naive end-to-end | Tiled kernel only | Tiled end-to-end |
|---|---|---|---|---|
| 3 | 184x | 7.8x | 165x | 9.1x |
| 7 | 199x | 35.0x | 277x | 36.9x |
| 15 | 201x | 86.3x | 389x | 103.7x |

Reporting kernel time alone overstates the practical gain by roughly 20x at K=3. Even end-to-end, the GPU beats 16 OpenMP threads at every K (9.1x vs 6.7x at K=3, 110x vs 8.5x at K=15), but at small K the margin is small. Offload pays when arithmetic per transferred byte is high: large filters, batches, or pipelines that keep data on the device. Transfers use pageable memory, the plain `cudaMemcpy` path of the synopsis; pinned memory would shorten them.

### 3.7 Amdahl's law (`amdahl.png`)

The serial fraction is measured per filter size from the parts that never parallelise: image decode (about 0.8 to 1.4 s for 8K), filter build, output allocation (about 20 ms) and PNG encode (2.5 to 3.2 s). Against the serial convolution time this gives:

| K | f | Amdahl ceiling 1/f | Best observed end-to-end | Best observed kernel-only |
|---|---|---|---|---|
| 3 | 0.902 | 1.11x | 1.09x | 6.74x |
| 7 | 0.726 | 1.38x | 1.30x | 6.62x |
| 15 | 0.387 | 2.58x | 2.18x | 8.52x |

Image I/O dominates a single convolution of one image, so end-to-end OpenMP speedup is capped at 1.1x to 2.6x whatever the thread count. The observed end-to-end values are 2% to 16% below the ceiling. The kernel-only speedup, which Amdahl's law with f=0 would put at 16x for 16 threads, reaches only 6.6x to 8.5x. The law does not model that gap: memory-bandwidth contention, hybrid cores, SMT and synchronisation. It also does not model PCIe transfer for the GPU versions, which Section 3.6 measures directly.

### 3.8 MPI extension

Single node, 8K image, K=7, ranks bound to no particular core. Strips carry R halo rows on each side and are scattered and gathered at rank 0 (`MPI_Scatterv`, `MPI_Gatherv`). Every output matched the serial reference for 1, 2, 3, 4 and 7 ranks with all three local kernels, at K=3 and K=15, including row counts that do not divide evenly.

| Ranks | Local kernel | Compute (ms) | Total with scatter/gather (ms) |
|---|---|---|---|
| 1 | serial | 1550 | 1580 |
| 2 | serial | 763 | 790 |
| 4 | serial | 391 | 427 |
| 8 | serial | 321 | 345 |
| 16 | serial | 306 | 341 |
| 1, 2, 4 | OpenMP, 16 threads total | 194, 211, 199 | 227, 248, 228 |
| 1, 2, 4 | CUDA tiled | 6.8, 9.6, 3.5 | 84, 85, 65 |

Pure MPI scales almost linearly to 4 ranks, then flattens at about 5x, the same bandwidth and core-mix limit as OpenMP. Scatter and gather add 25 to 35 ms. Splitting OpenMP threads across ranks gives no gain, since the threads were already saturating memory. With CUDA all ranks share one GPU, so the ranks serialise on the device; this demonstrates decomposition and halo handling, not distributed GPU scaling. No cluster was available, so these are not network measurements.

## 4. Discussion

1. Memory behaviour, not parallelism alone, sets performance. The CPU stops scaling after 4 threads, and the tiled GPU kernel reaches 15% of arithmetic peak.
2. The caches matter more than the synopsis anticipated. Naive CUDA is already close to one DRAM pass, so the tiling benefit is in cache-to-SM traffic and occupancy, and it only appears once K is large enough.
3. Transfers change the conclusion. A kernel-only 389x becomes 104x end-to-end, and 8x at small K.
4. Amdahl's law is a useful first-order bound, here dominated by image I/O. Its failure to predict the kernel-only plateau is the interesting part.

## 5. Limitations

- One machine, with a hybrid CPU, a 4 GB laptop GPU that can throttle, and WSL2 on Windows. CPU and GPU results are not portable to other hardware.
- Thread counts above 6 mix efficiency cores and hyper-threads, so OpenMP plateau causes are not isolated.
- The schedule x collapse sweep runs at 8 threads only. Static scheduling being slower than dynamic is a measurement on this CPU, not a general rule.
- Images are synthetic and grayscale. Border handling is zero-padding only in the reported runs (clamp-to-edge builds and tests exist).
- A GPU driver reset interrupted the sweep once and one run hung once; the sweep was resumed from the recorded runs, so it spans a restart of WSL.
- MPI was measured on one node, so it has no network cost.
- Profiler-measured DRAM throughput comes from one 8K image at K=3, 7 and 15 with 16x16 blocks.

## 6. Reproduce

```bash
bash scripts/build.sh && bash scripts/run_tests.sh
python3 scripts/gen_images.py
./build/convox sweep                          # long; add --resume after an interruption
python3 scripts/summary.py && python3 analysis/plots.py && python3 analysis/amdahl.py
python3 analysis/ncu_profile.py               # needs GPU performance-counter access
mpirun --bind-to none -np 4 ./build/convox_mpi -i data/images/img03_8k.png -k 7 --local omp
```

## References

The reference list of the project synopsis (`references/PCAP_Synopsis_IEEE_Format.pdf`) applies.
