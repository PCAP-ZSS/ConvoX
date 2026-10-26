#!/usr/bin/env python3
"""Print terminal summary tables from results/raw/results.csv.

usage: summary.py [--raw DIR] [--image ID] [--k K] [--all]
Default: for one image and K, the best configuration of each implementation, plus a
validation summary over the whole sweep.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
import common  # noqa: E402


def label(r):
    if r["impl"] == "omp":
        sched = r["schedule"] + (f":{int(r['chunk'])}" if r["chunk"] == r["chunk"] else "")
        return f"thr={int(r['threads'])} {sched} col={int(r['collapse'])}"
    if str(r["impl"]).startswith("cuda"):
        return f"block={int(r['block_x'])}x{int(r['block_y'])}"
    return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--all", action="store_true", help="list every configuration, not only the best per implementation")
    a = ap.parse_args()

    df, raw = common.load(a.raw)
    image = a.image or common.pick_image(df)
    sub = df[(df["image_id"] == image) & (df["k"] == a.k)]
    if sub.empty:
        sys.exit(f"no rows for image={image} K={a.k}")
    if not a.all:
        sub = sub.loc[sub.groupby("impl")["total_ms"].idxmin()]
    sub = sub.assign(_o=sub["impl"].map({n: i for i, n in enumerate(common.IMPLS)})).sort_values(["_o", "total_ms"])

    print(f"\n{image}  K={a.k}  ({sub['width'].iloc[0]}x{sub['height'].iloc[0]})  filter={sub['filter_type'].iloc[0]}\n")
    print(f"{'impl':<11} {'config':<22} {'kernel_ms':>10} {'total_ms':>10} {'speedup':>8} {'eff':>6} {'max_err':>9}  pass")
    for _, r in sub.iterrows():
        eff = r["speedup"] / r["threads"] if r["impl"] == "omp" else float("nan")
        eff_s = f"{eff:6.2f}" if eff == eff else f"{'-':>6}"
        print(
            f"{r['impl']:<11} {label(r):<22} {r['kernel_ms']:10.3f} {r['total_ms']:10.3f} "
            f"{r['speedup']:8.2f} {eff_s} {r['max_abs_err']:9.1e}  {'yes' if r['pass'] == 1 else 'FAIL'}"
        )

    bad = df[df["pass"] != 1]
    print(f"\nvalidation over the sweep: {len(df)} runs, {len(bad)} failed" + ("" if bad.empty else " (listed below)"))
    if not bad.empty:
        print(bad[["impl", "image_id", "k", "threads", "block_x", "max_abs_err"]].to_string(index=False))


if __name__ == "__main__":
    main()
