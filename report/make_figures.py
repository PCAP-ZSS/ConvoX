#!/usr/bin/env python3
"""Compact single-column figures for report/ConvoX_Project_Report.pdf, drawn from results/raw.

usage: python report/make_figures.py      (writes report/figures/*.png)
Values for the MPI and Amdahl figures are the measured numbers listed in report/REPORT.md
(MPI runs were made by hand, so they are not rows of the sweep file).
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import common  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
C = common.COLORS
IMG = "img03_8k"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "figure.dpi": 100, "savefig.dpi": 300,
})
W = 3.45

df, raw = common.load()
d8 = df[df["image_id"] == IMG]


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def fig_omp():
    d = common.omp_thread_rows(d8)
    fig, ax = plt.subplots(figsize=(W, 2.4))
    th = sorted(d["threads"].unique())
    ax.plot(th, th, "--", color="#999999", lw=0.9, label="ideal")
    for k, c in ((3, "#9ecae1"), (7, "#4292c6"), (15, "#08306b")):
        s = d[d["k"] == k].sort_values("threads")
        ax.plot(s["threads"], s["speedup"], marker="s", ms=3, lw=1.1, color=c, label=f"K={k}")
    ax.set_xticks(th)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speedup over serial (x)")
    ax.legend(loc="upper left")
    save(fig, "omp_scaling")


def fig_block():
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.9))
    blocks = [(8, 8), (16, 16), (32, 32)]
    for ax, k in zip(axes, (3, 9, 15)):
        for off, impl in ((-0.2, "cuda-naive"), (0.2, "cuda-tiled")):
            v = []
            for bx, by in blocks:
                r = d8[(d8["impl"] == impl) & (d8["k"] == k) & (d8["block_x"] == bx) & (d8["block_y"] == by)]
                v.append(r["kernel_ms"].iloc[0] if len(r) else np.nan)
            ax.bar(np.arange(3) + off, v, 0.4, color=C[impl], label=impl.replace("cuda-", ""))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["8x8", "16x16", "32x32"], fontsize=6)
        ax.set_title(f"K={k}")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
        if k == 3:
            ax.set_ylabel("kernel time (ms)")
            ax.legend(loc="upper left")
    save(fig, "block_sensitivity")


def fig_ratio():
    d = d8[(d8["block_x"] == 16) & (d8["block_y"] == 16)]
    ks = sorted(d["k"].unique())
    g = lambda impl: d[d["impl"] == impl].set_index("k")["kernel_ms"].reindex(ks)
    fig, ax = plt.subplots(figsize=(W, 2.1))
    ax.axhline(1, color="#999999", lw=0.8, ls="--")
    ax.plot(ks, g("cuda-naive") / g("cuda-const"), marker="v", ms=3, color=C["cuda-const"], label="naive / constant")
    ax.plot(ks, g("cuda-naive") / g("cuda-tiled"), marker="D", ms=3, color=C["cuda-tiled"], label="naive / tiled")
    ax.set_xticks(ks)
    ax.set_xlabel("filter size K")
    ax.set_ylabel("kernel-time ratio (x)")
    ax.legend(loc="upper left")
    save(fig, "tiling_ratio")


def fig_split():
    d = d8[(d8["block_x"] == 16) & (d8["block_y"] == 16)]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    labels, x = [], 0
    for k in (3, 9, 15):
        for impl, name in (("cuda-naive", "naive"), ("cuda-const", "const"), ("cuda-tiled", "tiled")):
            r = d[(d["impl"] == impl) & (d["k"] == k)].iloc[0]
            ax.bar(x, r["h2d_ms"], 0.75, color="#56B4E9", label="host to device" if x == 0 else None)
            ax.bar(x, r["kernel_ms"], 0.75, bottom=r["h2d_ms"], color="#D55E00", label="kernel" if x == 0 else None)
            ax.bar(x, r["d2h_ms"], 0.75, bottom=r["h2d_ms"] + r["kernel_ms"], color="#009E73", label="device to host" if x == 0 else None)
            labels.append(f"{name}\nK={k}" if impl == "cuda-naive" else name)
            x += 1
        x += 0.5
    pos = [i + (i // 3) * 0.5 for i in range(9)]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("time (ms)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    save(fig, "transfer_split")


def fig_k():
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ks = sorted(d8["k"].unique())
    rows = {
        "serial": d8[d8["impl"] == "serial"],
        "OpenMP, 16 threads": common.omp_thread_rows(d8)[lambda t: t["threads"] == 16],
        "CUDA naive": d8[(d8["impl"] == "cuda-naive") & (d8["block_x"] == 16)],
        "CUDA constant": d8[(d8["impl"] == "cuda-const") & (d8["block_x"] == 16)],
        "CUDA tiled": d8[(d8["impl"] == "cuda-tiled") & (d8["block_x"] == 16)],
    }
    cols = [C["serial"], C["omp"], C["cuda-naive"], C["cuda-const"], C["cuda-tiled"]]
    for (name, r), c in zip(rows.items(), cols):
        s = r.groupby("k")["kernel_ms"].min().reindex(ks)
        ax.plot(ks, s, marker="o", ms=2.8, lw=1.1, color=c, label=name)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("filter size K")
    ax.set_ylabel("compute time (ms, log scale)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    save(fig, "runtime_vs_k")


def fig_amdahl():
    ks = [3, 5, 7, 9, 11, 13, 15]
    ceil = [1.11, 1.22, 1.38, 1.67, 1.91, 2.36, 2.58]
    obs = [1.09, 1.19, 1.30, 1.54, 1.71, 2.01, 2.18]
    kern = [6.74, 6.74, 6.62, 8.13, 7.83, 7.78, 8.52]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(ks))
    ax.bar(x - 0.27, ceil, 0.27, color="#999999", label="Amdahl ceiling 1/f")
    ax.bar(x, obs, 0.27, color=C["omp"], label="observed end-to-end")
    ax.bar(x + 0.27, kern, 0.27, color=C["cuda-tiled"], label="observed kernel only")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_xlabel("filter size K")
    ax.set_ylabel("best OpenMP speedup (x)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 11)
    save(fig, "amdahl")


def fig_mpi():
    ranks = [1, 2, 4, 8, 16]
    total = [1580, 790, 427, 345, 341]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    ax.bar(range(5), total, 0.6, color="#7A5195")
    for i, t in enumerate(total):
        ax.text(i, t + 25, f"{t}\n({total[0] / t:.1f}x)" if i else f"{t}", ha="center", fontsize=6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(ranks)
    ax.set_xlabel("MPI ranks (serial local kernel)")
    ax.set_ylabel("total time (ms)")
    ax.set_ylim(0, 1900)
    save(fig, "mpi_scaling")


def fig_arch():
    fig, ax = plt.subplots(figsize=(W, 3.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fa", ec="#34495e", fs=6.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.7, color="#34495e"))

    box(1.5, 13.0, 7, 1.3, "Image loader (stb_image: decode / encode only)\nFilter generator (Gaussian, Sobel, box; K = 3..15)", "#fdf3e1")
    arrow(5, 13.0, 5, 12.35)
    box(0.1, 8.7, 1.85, 3.5, "Serial\n4 loops\n-O2\n(oracle)", "#e8e8e8")
    box(2.1, 8.7, 1.85, 3.5, "OpenMP\nrow-parallel\nschedule,\ncollapse", "#dcebf7")
    box(4.1, 8.7, 1.85, 3.5, "CUDA\nnaive\n(global\nmemory)", "#fcefcf")
    box(6.1, 8.7, 1.85, 3.5, "CUDA\nconstant\nmemory", "#d8f0e8")
    box(8.1, 8.7, 1.8, 3.5, "CUDA\ntiled\n(shared\nmemory)", "#f9dccb")
    box(0.1, 6.55, 9.8, 1.4, "MPI extension: strips with halo rows, Scatterv / Gatherv,\neach rank runs OpenMP or CUDA locally", "#e9e1f2")
    for xc in (1.0, 3.0, 5.0, 7.0, 9.0):
        arrow(xc, 8.7, xc, 7.95) if False else None
    arrow(5, 8.7, 5, 7.98)
    arrow(5, 6.55, 5, 5.9)
    box(1.5, 4.5, 7, 1.35, "Validator: max |error| < 1e-4 against the serial output\nTiming harness: warm-up + 10 runs, median", "#fdf3e1")
    arrow(5, 4.5, 5, 3.85)
    box(1.5, 2.5, 7, 1.3, "results.csv (33 columns), reps.csv, Nsight metrics", "#eef3fa")
    arrow(5, 2.5, 5, 1.85)
    box(1.5, 0.3, 7, 1.5, "Python analysis: plots, Amdahl check\nLocal demo page (runs the same program)", "#fdf3e1")
    save(fig, "architecture")


def fig_visual():
    im = plt.imread(HERE.parent / "results" / "plots" / "visual_check.png")
    fig, ax = plt.subplots(figsize=(W, W * im.shape[0] / im.shape[1]))
    ax.imshow(im)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "visual_check.png")
    plt.close(fig)


for f in (fig_omp, fig_block, fig_ratio, fig_split, fig_k, fig_amdahl, fig_mpi, fig_arch, fig_visual):
    f()
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
