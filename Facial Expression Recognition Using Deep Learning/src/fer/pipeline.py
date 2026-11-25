"""End-to-end experiment pipeline shared by scripts/train.py and scripts/run_sweep.py."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from fer.data.dataset import build_dataloaders
from fer.data.loading import FERData, load_fer2013
from fer.data.splits import SplitResult, build_splits
from fer.models.factory import build_model, count_parameters
from fer.training.trainer import Trainer
from fer.utils.config import save_config
from fer.utils.seed import seed_everything


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu" or (prefer == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")


def prepare_data(cfg: dict) -> tuple[FERData, SplitResult]:
    d = cfg["data"]
    data = load_fer2013(d["root"], d.get("cache_dir", "data/processed"))
    splits = build_splits(data, d.get("val_fraction", 0.1), d.get("split_seed", 42),
                          d.get("remove_cross_split_duplicates", True))
    return data, splits


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def run_dir_for(cfg: dict) -> Path:
    return Path(cfg.get("output_dir", "outputs/runs")) / cfg["name"] / f"seed{cfg['seed']}"


def run_training(cfg: dict) -> dict:
    seed = int(cfg["seed"])
    seed_everything(seed, cfg.get("deterministic", True))
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    device = get_device(cfg.get("device", "auto"))

    data, splits = prepare_data(cfg)
    (run_dir / "split_report.json").write_text(json.dumps(splits.report, indent=1))
    loaders, counts = build_dataloaders(cfg["data"], data, splits.indices, seed, which=("train", "val"))

    model = build_model(cfg["model"], cfg["data"]["channels"]).to(device)
    params = count_parameters(model)
    print(f"Run {cfg['name']} seed {seed} | device {device} | params {params['total']:,} "
          f"(trainable {params['trainable']:,}) | train/val {len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}")

    trainer = Trainer(model, loaders, cfg, run_dir, counts, device)
    summary = trainer.fit()
    summary.update({
        "name": cfg["name"], "seed": seed, "params": params, "train_class_counts": counts.tolist(),
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "git": _git_hash(), "numpy": np.__version__},
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    from fer.evaluation.plots import plot_training_curves
    plot_training_curves(trainer.history, run_dir / "curves.png", f"{cfg['name']} (seed {seed})")
    return summary
