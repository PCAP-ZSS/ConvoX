#!/usr/bin/env python3
"""Generate the benchmark plots from results/raw/results.csv.

usage: plots.py [--raw DIR] [--out DIR] [--image ID] [--show]
Writes PNG (200 dpi) and PDF into the output directory:
  omp_scaling, omp_schedules, runtime_vs_k, block_sensitivity, transfer_breakdown, roofline
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})


def finish(fig, name, out, raw, written):
    fig.text(0.01, 0.005, common.footer(raw), fontsize=7, color="#666666", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext, dpi in (("png", 200), ("pdf", None)):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi) if dpi else fig.savefig(p)
        written.append(p)
    plt.close(fig)


def best_per_k(df, impl, col="total_ms"):
    d = df[df["impl"] == impl]
    return d.loc[d.groupby("k")[col].idxmin()].sort_values("k")


def plot_omp_scaling(df, image, out, raw, written):
    d = common.omp_thread_rows(df[df["image_id"] == image])
    if d.empty:
        return
    ks = sorted(d["k"].unique())
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(ks)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    threads = sorted(d["threads"].unique())
    ax.plot(threads, threads, "--", color="#999999", label="ideal (linear)")
    peaks = []
    for k, c in zip(ks, shades):
        s = d[d["k"] == k].sort_values("threads")
        ax.plot(s["threads"], s["speedup"], marker="s", color=c, label=f"K={int(k)}")
        peaks.append((s["speedup"].max(), int(s.loc[s["speedup"].idxmax(), "threads"]), int(k)))
    lo = min(peaks)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speedup over serial (x)")
    ax.set_xticks(threads)
    ax.set_title(f"OpenMP scaling saturates well below ideal ({image}); K={lo[2]} peaks at {lo[0]:.1f}x with {lo[1]} threads")
    ax.legend(ncol=2, fontsize=8)
    finish(fig, "omp_scaling", out, raw, written)


def plot_omp_schedules(df, image, out, raw, written):
    d = df[(df["image_id"] == image) & (df["impl"] == "omp")]
    thr = d.groupby("threads").size().idxmax()  # the thread count the schedule sweep ran at
    d = d[d["threads"] == thr].copy()
    if d["schedule"].nunique() < 2:
        return
    d["label"] = d.apply(lambda r: r["schedule"] + (f"({int(r['chunk'])})" if r["chunk"] == r["chunk"] and r["schedule"] != "static" else ""), axis=1)
    ks = sorted(d["k"].unique())
    pick = [ks[0], ks[len(ks) // 2], ks[-1]]
    labels = list(dict.fromkeys(d.sort_values(["schedule", "chunk"])["label"]))
    fig, axes = plt.subplots(1, len(pick), figsize=(12, 4), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), pick):
        x = np.arange(len(labels))
        for off, col, name in ((-0.2, 1, "rows"), (0.2, 2, "collapse(2)")):
            s = d[(d["k"] == k) & (d["collapse"] == col)].set_index("label")["total_ms"].reindex(labels)
            ax.bar(x + off, s.values, 0.4, label=name, color=common.COLORS["omp"], alpha=0.95 if col == 1 else 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"K={int(k)}")
        ax.set_ylabel("runtime (ms)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"OpenMP schedule and collapse at {int(thr)} threads ({image}): dynamic overhead is not recovered on uniform rows")
    finish(fig, "omp_schedules", out, raw, written)


def plot_runtime_vs_k(df, image, out, raw, written):
    d = df[df["image_id"] == image]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in common.IMPLS:
        s = best_per_k(d, impl)
        if s.empty:
            continue
        ax.plot(s["k"], s["total_ms"], marker=common.MARKERS[impl], color=common.COLORS[impl], label=f"{impl} (end-to-end)")
        if impl.startswith("cuda") and impl != "cuda-const":
            sk = best_per_k(d, impl, "kernel_ms")
            ax.plot(sk["k"], sk["kernel_ms"], ls="--", marker=common.MARKERS[impl], mfc="none", color=common.COLORS[impl], label=f"{impl} (kernel only)")
    ax.set_yscale("log")
    ax.set_xlabel("filter size K")
    ax.set_ylabel("runtime (ms, log scale)")
    ax.set_xticks(sorted(d["k"].unique()))
    ax.set_title(f"Runtime vs filter size, best configuration per implementation ({image})")
    ax.legend(fontsize=7, ncol=2)
    finish(fig, "runtime_vs_k", out, raw, written)


def plot_block_sensitivity(df, image, out, raw, written):
    d = df[(df["image_id"] == image) & (df["impl"].str.startswith("cuda"))]
    if d.empty:
        return
    ks = sorted(d["k"].unique())
    pick = [ks[0], ks[len(ks) // 2], ks[-1]]
    d = d.assign(block=d["block_x"].astype(int).astype(str) + "x" + d["block_y"].astype(int).astype(str))
    blocks = sorted(d["block"].unique(), key=lambda b: int(b.split("x")[0]))
    impls = [i for i in ("cuda-naive", "cuda-const", "cuda-tiled") if i in d["impl"].unique()]
    fig, axes = plt.subplots(1, len(pick), figsize=(12, 4))
    for ax, k in zip(np.atleast_1d(axes), pick):
        x = np.arange(len(blocks))
        w = 0.8 / len(impls)
        for i, impl in enumerate(impls):
            s = d[(d["k"] == k) & (d["impl"] == impl)].set_index("block")["kernel_ms"].reindex(blocks)
            ax.bar(x + (i - (len(impls) - 1) / 2) * w, s.values, w, label=impl, color=common.COLORS[impl])
        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.set_xlabel("block dimensions")
        ax.set_ylabel("kernel time (ms)")
        ax.set_title(f"K={int(k)}")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"CUDA block-dimension sensitivity, kernel only ({image})")
    finish(fig, "block_sensitivity", out, raw, written)


def plot_transfer_breakdown(df, image, out, raw, written):
    d = df[(df["image_id"] == image) & (df["impl"].str.startswith("cuda"))]
    if d.empty:
        return
    block = 16 if (d["block_x"] == 16).any() else int(d["block_x"].iloc[0])
    d = d[(d["block_x"] == block) & (d["block_y"] == block)]
    ks = sorted(d["k"].unique())
    pick = [k for k in (ks[0], ks[len(ks) // 2], ks[-1])]
    impls = [i for i in ("cuda-naive", "cuda-const", "cuda-tiled") if i in d["impl"].unique()]
    labels, h2d, ker, d2h = [], [], [], []
    for k in pick:
        for impl in impls:
            r = d[(d["k"] == k) & (d["impl"] == impl)]
            if r.empty:
                continue
            r = r.iloc[0]
            labels.append(f"K={int(k)}\n{impl.replace('cuda-', '')}")
            h2d.append(r["h2d_ms"]); ker.append(r["kernel_ms"]); d2h.append(r["d2h_ms"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, h2d, label="host to device copy", color="#56B4E9")
    ax.bar(x, ker, bottom=h2d, label="kernel", color="#D55E00")
    ax.bar(x, d2h, bottom=np.array(h2d) + np.array(ker), label="device to host copy", color="#009E73")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("time (ms)")
    share = 100 * (np.array(h2d) + np.array(d2h)) / (np.array(h2d) + np.array(ker) + np.array(d2h))
    ax.set_title(f"GPU time split ({image}, {block}x{block} blocks): transfers are {share.min():.0f}-{share.max():.0f}% of the total")
    ax.legend(fontsize=8)
    finish(fig, "transfer_breakdown", out, raw, written)


def plot_roofline(df, image, out, raw, written):
    d = df[(df["image_id"] == image) & df["impl"].isin(["cuda-naive", "cuda-tiled"])]
    m = common.manifest(raw)
    bw, peak = m.get("peak_gpu_gbps", 0), m.get("peak_gpu_gflops", 0)
    if d.empty or not bw or not peak:
        return
    block = 16 if (d["block_x"] == 16).any() else int(d["block_x"].iloc[0])
    d = d[(d["block_x"] == block) & (d["block_y"] == block)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ai = np.logspace(-1.3, 2.2, 200)
    ax.plot(ai, np.minimum(peak, bw * ai), color="#333333", lw=2, label=f"roof: {bw:.0f} GB/s, {peak:.0f} GFLOP/s")
    for impl in ("cuda-naive", "cuda-tiled"):
        s = d[d["impl"] == impl].sort_values("k")
        k = s["k"].values
        flops = 2.0 * k**2 * s["width"].values * s["height"].values
        gflops = flops / (s["kernel_ms"].values * 1e-3) / 1e9
        if impl == "cuda-naive":
            intensity = np.full_like(k, 0.5, dtype=float)
        else:
            t = float(block)
            intensity = 2.0 * k**2 / (4.0 * (t + k - 1) ** 2 / t**2)
        ax.scatter(intensity, gflops, color=common.COLORS[impl], marker=common.MARKERS[impl], s=45, label=impl, zorder=3)
        for xi, yi, ki in zip(intensity, gflops, k):
            ax.annotate(f"K={int(ki)}", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte, analytical)")
    ax.set_ylabel("achieved GFLOP/s (kernel only)")
    ax.set_title(f"Roofline view: tiling moves kernels right, toward compute-bound ({image})")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.98, 0.04, "Naive points sit above the roof: L1/L2 caches serve much of the K^2 reuse,\n"
            "so their real DRAM intensity is higher than the analytical 0.5 FLOP/byte.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
    finish(fig, "roofline", out, raw, written)


def open_file(path):
    for cmd in (["wslview"], ["explorer.exe"], ["open"], ["xdg-open"]):
        if shutil.which(cmd[0]):
            arg = str(path)
            if cmd[0] == "explorer.exe" and shutil.which("wslpath"):
                arg = subprocess.run(["wslpath", "-w", arg], capture_output=True, text=True).stdout.strip()
            subprocess.Popen(cmd + [arg])
            return
    print(f"(no viewer found; open {path} manually)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=str(common.RAW.parent / "plots"))
    ap.add_argument("--image", default=None, help="image_id to plot (default: first 8K image)")
    ap.add_argument("--show", action="store_true", help="open the PNGs in the default viewer")
    a = ap.parse_args()

    df, raw = common.load(a.raw)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    image = a.image or common.pick_image(df)
    written = []
    for fn in (plot_omp_scaling, plot_omp_schedules, plot_runtime_vs_k, plot_block_sensitivity, plot_transfer_breakdown, plot_roofline):
        fn(df, image, out, raw, written)
    for p in written:
        if p.suffix == ".png":
            print(f"wrote {p}")
    if a.show:
        for p in written:
            if p.suffix == ".png":
                open_file(p)


if __name__ == "__main__":
    main()
