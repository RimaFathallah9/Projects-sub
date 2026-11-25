"""Matplotlib figures (Agg backend, saved to disk)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fer.constants import CLASSES  # noqa: E402


def plot_confusion_matrix(cm, path, class_names=CLASSES, title="Confusion matrix"):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title + " (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_training_curves(history: list[dict], path, title="Training curves"):
    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(ep, [h["train_loss"] for h in history], label="train"); axs[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(ep, [h["train_acc"] for h in history], label="train"); axs[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axs[1].set_title("Accuracy"); axs[1].legend()
    axs[2].plot(ep, [h["val_macro_f1"] for h in history], color="tab:green"); axs[2].set_title("Val macro-F1")
    for a in axs:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_class_distribution(counts_by_split: dict[str, dict], path):
    classes = list(next(iter(counts_by_split.values())).keys())
    x = np.arange(len(classes)); w = 0.8 / len(counts_by_split)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (split, counts) in enumerate(counts_by_split.items()):
        tot = sum(counts.values())
        vals = [100 * counts[c] / tot for c in classes]
        ax.bar(x + i * w, vals, w, label=f"{split} (n={tot})")
    ax.set_xticks(x + w * (len(counts_by_split) - 1) / 2, classes)
    ax.set_ylabel("% of split"); ax.set_title("Class distribution"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_metric_bars(df, metrics: list[str], path, title="Model comparison"):
    """df: index = model name, columns like '<metric>_mean' and '<metric>_std'."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(metrics); x = np.arange(len(df))
    for i, m in enumerate(metrics):
        std = df[f"{m}_std"].fillna(0) if f"{m}_std" in df else None
        ax.bar(x + i * w, df[f"{m}_mean"], w, yerr=std, capsize=3, label=m)
    ax.set_xticks(x + w * (len(metrics) - 1) / 2, df.index, rotation=15)
    ax.set_ylim(0, 1); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_class_f1(per_class_by_model: dict[str, dict], path, class_names=CLASSES):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n = len(per_class_by_model); w = 0.8 / n; x = np.arange(len(class_names))
    for i, (name, pc) in enumerate(per_class_by_model.items()):
        ax.bar(x + i * w, [pc[c] for c in class_names], w, label=name)
    ax.set_xticks(x + w * (n - 1) / 2, class_names); ax.set_ylim(0, 1)
    ax.set_ylabel("F1"); ax.set_title("Per-class F1"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_dir(path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
