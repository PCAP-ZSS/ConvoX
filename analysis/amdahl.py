#!/usr/bin/env python3
"""Amdahl's-law check: predicted speedup ceiling vs what the OpenMP runs actually reach.

usage: amdahl.py [--raw DIR] [--out DIR] [--image ID] [--k K ...] [--show]

Serial fraction f is measured from serial_profile.csv (image load, filter setup, output
allocation, image save) against the serial convolution time of the same configuration:

    s = load + filter + alloc + save          (the parts that never get parallelised)
    c = serial convolution time               (the parallel region on one thread)
    f = s / (s + c)
    predicted  S(N) = 1 / (f + (1 - f) / N)
    observed   S_e2e(N) = (s + c) / (s + c_N)     c_N = OpenMP convolution time at N threads
    observed kernel-only speedup = c / c_N         (what the earlier plots show)

Amdahl's law bounds S_e2e. The plot also shows the kernel-only speedup against the ideal line,
because the gap between that and linear scaling is memory-bandwidth saturation, which the
law does not model.
"""
import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from plots import finish, open_file  # noqa: E402


def amdahl(f, n):
    return 1.0 / (f + (1.0 - f) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=str(common.RAW.parent / "plots"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--k", type=int, nargs="*", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    df, raw = common.load(a.raw)
    prof_path = pathlib.Path(raw) / "serial_profile.csv"
    if not prof_path.exists():
        sys.exit(f"missing {prof_path}; run `convox sweep` (or `convox profile`) first")
    prof = pd.read_csv(prof_path).drop_duplicates(["image_id", "filter_type", "k"], keep="last")

    image = a.image or common.pick_image(df)
    omp = common.omp_thread_rows(df[df["image_id"] == image])
    ks = a.k or sorted(omp["k"].unique())
    ks = [k for k in ks if k in set(omp["k"])]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(ks)))
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    print(f"\nAmdahl check, {image} (filter {omp['filter_type'].iloc[0]})\n")
    print(f"{'K':>3} {'serial s (ms)':>14} {'conv c (ms)':>12} {'f':>7} {'ceiling 1/f':>12} {'best kernel x':>14} {'best e2e x':>11}")
    for ax, k, color in zip(axes, ks, shades):
        p = prof[(prof["image_id"] == image) & (prof["k"] == k)]
        if p.empty:
            continue
        p = p.iloc[0]
        s = p["load_ms"] + p["filter_ms"] + p["alloc_ms"] + p["save_ms"]
        rows = omp[omp["k"] == k].sort_values("threads")
        c = rows["serial_ms"].iloc[0]
        f = s / (s + c)
        n = rows["threads"].values
        kernel = c / rows["total_ms"].values
        e2e = (s + c) / (s + rows["total_ms"].values)
        grid = np.linspace(1, n.max(), 100)
        ax.plot(grid, grid, "--", color="#999999", label="ideal (linear)")
        ax.plot(grid, amdahl(f, grid), color="#D55E00", label=f"Amdahl, f={f:.3f}")
        ax.plot(n, e2e, marker="s", color=color, label="observed end-to-end")
        ax.plot(n, kernel, marker="o", ls=":", color="#0072B2", label="observed kernel only")
        ax.set_title(f"K={int(k)}")
        ax.set_xlabel("OpenMP threads")
        print(f"{int(k):3d} {s:14.1f} {c:12.1f} {f:7.4f} {1 / f:12.2f} {kernel.max():14.2f} {e2e.max():11.2f}")
    axes[0].set_ylabel("speedup (x)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Amdahl's law vs measured speedup ({image}): serial I/O caps end-to-end speedup, "
                 "and kernel-only speedup saturates below linear")
    written = []
    finish(fig, "amdahl", out, raw, written)
    print(f"\nwrote {written[0]}")
    if a.show:
        open_file(written[0])


if __name__ == "__main__":
    main()
