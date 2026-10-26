#!/usr/bin/env python3
"""Compose the original image and the Gaussian / Sobel / box outputs side by side.

usage: montage.py ORIGINAL_PNG OUTPUT_DIR RESULT_PNG K
Shows the same crop of every image so edge detail is visible.
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    original, out_dir, result, k = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    panels = [("Original", original)] + [
        (f"{name} (K={k})", out_dir / f"{key}.png")
        for name, key in (("Gaussian blur", "gaussian"), ("Sobel edges", "sobel"), ("Box blur", "box"))
    ]
    first = mpimg.imread(panels[0][1])
    h, w = first.shape[:2]
    ch, cw = min(h, 540), min(w, 960)
    y0, x0 = (h - ch) // 2, (w - cw) // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (title, path) in zip(axes.flat, panels):
        img = mpimg.imread(path)
        ax.imshow(img[y0 : y0 + ch, x0 : x0 + cw], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Visual correctness check: same crop, four images")
    fig.tight_layout()
    fig.savefig(result, dpi=150)
    print(f"wrote {result}")


if __name__ == "__main__":
    main()
