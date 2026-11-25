"""Evaluate trained run(s) on a split.

  python scripts/evaluate.py --run outputs/runs/custom_cnn/seed42 --split val
  python scripts/evaluate.py --run outputs/runs/custom_cnn/seed42 --split test --benchmark

METHODOLOGY: use --split val for all model/hyper-parameter selection. Run --split test only once per
final configuration, after every choice has been frozen. Repeatedly peeking at test turns it into a
second validation set and makes the reported numbers optimistic.
"""
from __future__ import annotations

import argparse

from fer.evaluation.run_eval import evaluate_run, format_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="run directories (contain best.pt)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--benchmark", action="store_true", help="also measure params/MACs/latency")
    a = ap.parse_args()
    if a.split == "test":
        print("NOTE: evaluating on the TEST split. Do not use these numbers to make further tuning decisions.")
    for r in a.run:
        print(f"\n=== {r} ===")
        print(format_summary(evaluate_run(r, a.split, a.benchmark)))


if __name__ == "__main__":
    main()
