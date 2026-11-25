"""Evaluate a trained run on a split and write metrics, figures and (optionally) benchmarks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fer.constants import CLASSES
from fer.data.dataset import build_dataloaders
from fer.evaluation.benchmark import benchmark_model
from fer.evaluation.error_analysis import imbalance_correlation, save_misclassified_grid, top_confusions
from fer.evaluation.metrics import compute_metrics, expected_calibration_error, predict_logits
from fer.evaluation.plots import plot_confusion_matrix
from fer.inference import load_checkpoint
from fer.pipeline import get_device, prepare_data


def evaluate_run(run_dir: str | Path, split: str = "val", benchmark: bool = False, device: str = "auto") -> dict:
    run_dir = Path(run_dir)
    dev = get_device(device)
    model, cfg, _ = load_checkpoint(run_dir / "best.pt", dev)
    data, splits = prepare_data(cfg)
    loaders, counts = build_dataloaders(cfg["data"], data, splits.indices, cfg["seed"], which=(split,))
    logits, y = predict_logits(model, loaders[split], dev)
    probs = torch.softmax(logits, 1).numpy()
    y_np, pred = y.numpy(), probs.argmax(1)

    m = compute_metrics(y_np, pred)
    m["ece"] = expected_calibration_error(probs, y_np)
    m["top_confusions"] = top_confusions(m["confusion_matrix"])
    m["imbalance_correlation"] = imbalance_correlation(m["per_class"], counts)
    m["split"] = split
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(m, indent=1))

    idx = splits.indices[split]
    np.savez_compressed(run_dir / f"{split}_predictions.npz", probs=probs, labels=y_np, dataset_indices=idx)
    plot_confusion_matrix(m["confusion_matrix"], run_dir / f"{split}_confusion.png",
                          title=f"{cfg['name']} / {split}")
    save_misclassified_grid(data.images[idx], y_np, pred, probs, run_dir / f"{split}_misclassified.png")

    if benchmark:
        size = int(cfg["data"].get("image_size", 48))
        b = benchmark_model(model, (int(cfg["data"]["channels"]), size, size))
        summ = run_dir / "summary.json"
        if summ.is_file():
            s = json.loads(summ.read_text())
            b["train_time_s"], b["epochs_run"] = s.get("train_time_s"), s.get("epochs_run")
        (run_dir / "benchmark.json").write_text(json.dumps(b, indent=1))
        m["benchmark"] = b
    return m


def format_summary(m: dict) -> str:
    lines = [f"split={m['split']} n={m['n_samples']} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
             f"macroF1={m['f1_macro']:.4f} weightedF1={m['f1_weighted']:.4f} ECE={m['ece']:.4f}"]
    for c in CLASSES:
        p = m["per_class"][c]
        lines.append(f"  {c:9s} P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f} n={p['support']}")
    return "\n".join(lines)
