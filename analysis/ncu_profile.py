#!/usr/bin/env python3
"""Profile the three CUDA kernels with Nsight Compute and write results/raw/ncu_metrics.csv.

usage: ncu_profile.py [--image ID] [--k K ...] [--block 16x16] [--raw DIR]

For every (variant, K) it profiles the second kernel launch (the first is the harness warm-up) and
records DRAM bytes read and written, L2 bytes, global-load bytes through L1, kernel time, DRAM
throughput and achieved occupancy (active warps as a percentage of the SM maximum). Needs GPU performance-counter access; on
Windows/WSL enable it in NVIDIA Control Panel > Developer > Manage GPU Performance Counters.
Run it while nothing else uses the GPU.
"""
import argparse
import csv
import io
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_bytes.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
]
BYTES = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}
SECONDS = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def ncu_path():
    for cand in (shutil.which("ncu"), "/usr/local/cuda-12.6/bin/ncu", "/usr/local/cuda/bin/ncu"):
        if cand and pathlib.Path(cand).exists():
            return cand
    sys.exit("ncu not found")


def profile(ncu, image, k, impl, block):
    cmd = [
        ncu, "--csv", "--metrics", ",".join(METRICS), "-k", "regex:k_", "--launch-skip", "1", "--launch-count", "1",
        str(ROOT / "build" / "convox"), "run", "--impl", impl, "-i", str(ROOT / "data" / "images" / f"{image}.png"),
        "-k", str(k), "--reps", "1", "--block", block, "--no-csv",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if "ERR_NVGPUCTRPERM" in out.stdout + out.stderr:
        sys.exit("GPU performance counters are not accessible (ERR_NVGPUCTRPERM); see the docstring")
    start = next((i for i, l in enumerate(out.stdout.splitlines()) if l.startswith('"ID"')), None)
    if start is None:
        sys.exit(f"no profiler output for {impl} K={k}:\n{out.stdout}\n{out.stderr}")
    rows = list(csv.DictReader(io.StringIO("\n".join(out.stdout.splitlines()[start:]))))
    vals = {}
    for r in rows:
        v = float(r["Metric Value"].replace(",", ""))
        vals[r["Metric Name"]] = (v, r["Metric Unit"])
    read = vals["dram__bytes_read.sum"][0] * BYTES[vals["dram__bytes_read.sum"][1]]
    write = vals["dram__bytes_write.sum"][0] * BYTES[vals["dram__bytes_write.sum"][1]]
    secs = vals["gpu__time_duration.sum"][0] * SECONDS[vals["gpu__time_duration.sum"][1]]
    return {
        "dram_read_mb": read / 1e6,
        "dram_write_mb": write / 1e6,
        "kernel_ms": secs * 1e3,
        "dram_gbps": (read + write) / secs / 1e9,
        "warps_active_pct": vals["sm__warps_active.avg.pct_of_peak_sustained_active"][0],
        "l2_mb": vals["lts__t_bytes.sum"][0] * BYTES[vals["lts__t_bytes.sum"][1]] / 1e6,
        "l1_global_ld_mb": vals["l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"][0]
        * BYTES[vals["l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"][1]] / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="img03_8k")
    ap.add_argument("--k", type=int, nargs="*", default=[3, 7, 15])
    ap.add_argument("--block", default="16x16")
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    a = ap.parse_args()

    ncu = ncu_path()
    out = pathlib.Path(a.raw) / "ncu_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["impl", "image_id", "k", "block", "dram_read_mb", "dram_write_mb", "kernel_ms", "dram_gbps", "warps_active_pct", "l2_mb", "l1_global_ld_mb"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        print(f"{'impl':<11} {'K':>3} {'DRAM read MB':>13} {'write MB':>9} {'kernel ms':>10} {'GB/s':>7} {'warps active %':>15} {'L2 MB':>9} {'L1 global ld MB':>16}")
        for k in a.k:
            for impl in ("cuda-naive", "cuda-const", "cuda-tiled"):
                m = profile(ncu, a.image, k, impl, a.block)
                w.writerow([impl, a.image, k, a.block] + [f"{m[c]:.4f}" for c in cols[4:]])
                print(f"{impl:<11} {k:>3} {m['dram_read_mb']:13.1f} {m['dram_write_mb']:9.1f} {m['kernel_ms']:10.3f} {m['dram_gbps']:7.1f} {m['warps_active_pct']:15.1f} {m['l2_mb']:9.0f} {m['l1_global_ld_mb']:16.0f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
