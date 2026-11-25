"""Exploratory data analysis for FER-2013. Writes reports/eda.json and figures to reports/figures/.

Usage: python scripts/eda.py [--data-root data]
Everything reported here is computed from the data; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fer.constants import CLASSES
from fer.data.loading import load_fer2013
from fer.data.splits import build_splits, hash_images
from fer.evaluation.plots import plot_class_distribution


def sample_grid(data, path, per_class=10, seed=0):
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.1, len(CLASSES) * 1.2))
    for c, name in enumerate(CLASSES):
        idx = rng.choice(np.where(data.labels == c)[0], per_class, replace=False)
        for j, i in enumerate(idx):
            axs[c, j].imshow(data.images[i], cmap="gray", vmin=0, vmax=255); axs[c, j].axis("off")
        axs[c, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def mean_images(data, path):
    fig, axs = plt.subplots(1, len(CLASSES), figsize=(len(CLASSES) * 1.8, 2.2))
    for c, name in enumerate(CLASSES):
        axs[c].imshow(data.images[data.labels == c].mean(0), cmap="gray"); axs[c].set_title(name, fontsize=8); axs[c].axis("off")
    fig.suptitle("Per-class mean face"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--cache-dir", default="data/processed")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    data = load_fer2013(a.data_root, a.cache_dir)
    print(f"Loaded {len(data)} images from {data.source}")
    splits = build_splits(data)
    r = splits.report

    hashes = hash_images(data.images)
    std = data.images.reshape(len(data), -1).std(1)
    mean = data.images.reshape(len(data), -1).mean(1)
    suspicious = {
        "near_constant_images(std<5)": int((std < 5).sum()),
        "very_dark_images(mean<20)": int((mean < 20).sum()),
        "very_bright_images(mean>235)": int((mean > 235).sum()),
        "unique_images": int(len(set(hashes.tolist()))),
        "total_images": int(len(data)),
    }
    counts = np.bincount(data.labels, minlength=len(CLASSES))
    stats = {
        "pixel_mean": float(data.images.mean() / 255), "pixel_std": float(data.images.std() / 255),
        "overall_class_counts": {c: int(n) for c, n in zip(CLASSES, counts)},
        "imbalance_ratio_max_over_min": float(counts.max() / counts.min()),
        "split_report": r, "quality": suspicious,
    }
    (out / "eda.json").write_text(json.dumps(stats, indent=1))
    plot_class_distribution(r["class_counts"], out / "figures" / "class_distribution.png")
    sample_grid(data, out / "figures" / "samples_per_class.png")
    mean_images(data, out / "figures" / "mean_faces.png")

    # Most extreme suspicious images for manual inspection
    order = np.argsort(std)[:32]
    fig, axs = plt.subplots(4, 8, figsize=(12, 6.5))
    for ax, i in zip(axs.ravel(), order):
        ax.imshow(data.images[i], cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.set_title(f"{CLASSES[data.labels[i]]} sd={std[i]:.1f}", fontsize=7)
    fig.suptitle("Lowest-contrast images (candidates for non-faces / blanks)"); fig.tight_layout()
    fig.savefig(out / "figures" / "lowest_contrast.png", dpi=120); plt.close(fig)

    print(json.dumps({k: stats[k] for k in ("overall_class_counts", "imbalance_ratio_max_over_min", "quality")}, indent=1))
    print("Split report:", json.dumps({k: r[k] for k in ("raw_sizes", "cross_split_overlap_unique_images",
          "within_split_duplicates", "images_with_conflicting_labels", "removed_for_leakage", "final_sizes")}, indent=1))
    print(f"Wrote {out/'eda.json'} and figures in {out/'figures'}")


if __name__ == "__main__":
    main()
