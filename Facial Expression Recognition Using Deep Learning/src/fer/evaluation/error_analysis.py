"""Error analysis: top confusions, confident mistakes, imbalance-vs-performance correlation."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from fer.constants import CLASSES  # noqa: E402


def top_confusions(cm, class_names=CLASSES, k: int = 10) -> list[dict]:
    cm = np.asarray(cm)
    rows = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                rows.append({"true": class_names[i], "pred": class_names[j], "count": int(cm[i, j]),
                             "rate_of_true_class": float(cm[i, j] / max(cm[i].sum(), 1))})
    return sorted(rows, key=lambda r: r["count"], reverse=True)[:k]


def imbalance_correlation(per_class: dict, train_counts, class_names=CLASSES) -> dict:
    """Spearman correlation between training-set class size and per-class recall / F1.
    With only 7 classes this has very little statistical power; report it as descriptive only."""
    n = np.asarray(train_counts, dtype=float)
    out = {}
    for key in ("recall", "precision", "f1"):
        vals = [per_class[c][key] for c in class_names]
        rho, pval = spearmanr(n, vals)
        out[key] = {"spearman_rho": float(rho), "p_value": float(pval)}
    out["note"] = "n=7 classes: descriptive only, not evidence of causality"
    return out


def save_misclassified_grid(images: np.ndarray, y_true, y_pred, probs, path, n: int = 32,
                            class_names=CLASSES, confident_first: bool = True):
    """Grid of misclassified faces, most confident mistakes first (often label noise or ambiguous faces)."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    conf = probs[wrong, y_pred[wrong]]
    order = wrong[np.argsort(-conf)] if confident_first else wrong[:n]
    order = order[:n]
    cols = 8
    rows = int(np.ceil(len(order) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axs).ravel(), order):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"T:{class_names[y_true[i]]}\nP:{class_names[y_pred[i]]} {probs[i, y_pred[i]]:.2f}",
                     fontsize=7, color="crimson")
    fig.suptitle("Misclassified samples (most confident errors first)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
