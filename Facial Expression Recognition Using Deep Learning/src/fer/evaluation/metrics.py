"""Classification metrics (all computed with scikit-learn, zero_division=0)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from fer.constants import CLASSES


def compute_metrics(y_true, y_pred, class_names=CLASSES) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "f1_macro": float(f.mean()),
    }
    w = s / max(s.sum(), 1)
    out.update({"precision_weighted": float((p * w).sum()), "recall_weighted": float((r * w).sum()),
                "f1_weighted": float((f * w).sum())})
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, c in enumerate(class_names)}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


@torch.no_grad()
def predict_logits(model, loader, device, amp: bool = False):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted mean |accuracy - confidence| over equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
