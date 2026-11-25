"""Train one configuration.

  python scripts/train.py --config configs/custom_cnn.yaml
  python scripts/train.py --config configs/custom_cnn.yaml --seed 1 --set train.lr=0.003 name=cnn_lr3e-3

Trains on `train`, selects the best epoch on `val`. The test split is never loaded here.
"""
from __future__ import annotations

import argparse
import json

from fer.pipeline import run_training
from fer.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=0.01")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    s = run_training(cfg)
    vm = s["val_metrics"]
    print(json.dumps({"best_epoch": s["best_epoch"], "train_time_s": round(s["train_time_s"], 1),
                      "val_acc": round(vm["accuracy"], 4), "val_macro_f1": round(vm["f1_macro"], 4),
                      "val_weighted_f1": round(vm["f1_weighted"], 4)}, indent=1))


if __name__ == "__main__":
    main()
