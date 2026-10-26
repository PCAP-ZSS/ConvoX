"""Shared loading and derived quantities for the analysis scripts.

Speedup is derived here, never stored: T_serial / T for the same image, filter, K and border,
using the serial row of the same sweep (see docs/04-data-schema/DATA_SCHEMA.md).
"""
import json
import pathlib

import pandas as pd

RAW = pathlib.Path(__file__).resolve().parent.parent / "results" / "raw"

# One fixed colour and order per implementation, used by every plot.
IMPLS = ["serial", "omp", "cuda-naive", "cuda-const", "cuda-tiled"]
COLORS = {
    "serial": "#555555",
    "omp": "#0072B2",
    "cuda-naive": "#E69F00",
    "cuda-const": "#009E73",
    "cuda-tiled": "#D55E00",
}
MARKERS = {"serial": "o", "omp": "s", "cuda-naive": "^", "cuda-const": "v", "cuda-tiled": "D"}


def load(raw_dir=None, sweep=None):
    """Load results.csv for the latest sweep (or `sweep`), with `speedup` added on total_ms and kernel_ms."""
    raw = pathlib.Path(raw_dir) if raw_dir else RAW
    df = pd.read_csv(raw / "results.csv")
    if sweep is None:
        sweep = df["sweep_id"].dropna().iloc[-1] if df["sweep_id"].notna().any() else None
    if sweep is not None:
        df = df[df["sweep_id"] == sweep].copy()
    keys = ["image_id", "filter_type", "k", "border"]
    base = (
        df[df["impl"] == "serial"]
        .set_index(keys)[["total_ms"]]
        .rename(columns={"total_ms": "serial_ms"})
    )
    df = df.join(base, on=keys)
    df["speedup"] = df["serial_ms"] / df["total_ms"]
    df["kernel_speedup"] = df["serial_ms"] / df["kernel_ms"]
    df["class"] = df["width"].map({3840: "4K", 7680: "8K"}).fillna(df["width"].astype(str))
    return df, raw


def manifest(raw):
    p = pathlib.Path(raw) / "sweep_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def footer(raw):
    m = manifest(raw)
    cpu, gpu = m.get("cpu", "CPU"), m.get("gpu", "")
    return f"{cpu}" + (f"  |  {gpu}" if gpu else "") + (f"  |  commit {m['git_commit']}" if "git_commit" in m else "")


def pick_image(df, preferred_class="8K"):
    """First image of the preferred resolution class, falling back to any image."""
    ids = df[df["class"] == preferred_class]["image_id"].unique()
    return ids[0] if len(ids) else df["image_id"].iloc[0]


def omp_thread_rows(df):
    """OpenMP rows of the thread sweep: static schedule, row-parallel."""
    return df[(df["impl"] == "omp") & (df["schedule"] == "static") & (df["collapse"] == 1)]
