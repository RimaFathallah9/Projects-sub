"""Aggregate final runs (multiple seeds) into comparison tables and figures.

  python scripts/compare.py --split test --groups custom_cnn=outputs/runs/custom_cnn \
      resnet18=outputs/runs/resnet18_finetune mobilenetv2=outputs/runs/mobilenetv2

Each group directory holds seed*/ sub-folders produced by train.py + evaluate.py (<split>_metrics.json,
benchmark.json). Reports mean +- std across seeds. Only numbers present on disk are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fer.constants import CLASSES
from fer.evaluation.plots import plot_metric_bars, plot_per_class_f1

METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "balanced_accuracy", "ece"]


def collect(group_dir: Path, split: str):
    rows, pcs, bench = [], [], []
    for sd in sorted(group_dir.glob("seed*")):
        f = sd / f"{split}_metrics.json"
        if not f.is_file():
            continue
        m = json.loads(f.read_text())
        rows.append({k: m[k] for k in METRICS})
        pcs.append({c: m["per_class"][c]["f1"] for c in CLASSES})
        b = sd / "benchmark.json"
        if b.is_file():
            bench.append(json.loads(b.read_text()))
    return rows, pcs, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--groups", nargs="+", required=True, help="name=path/to/group_dir")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)

    table, per_class = {}, {}
    for g in a.groups:
        name, path = g.split("=", 1)
        rows, pcs, bench = collect(Path(path), a.split)
        if not rows:
            print(f"[warn] no {a.split}_metrics.json under {path}; run scripts/evaluate.py first"); continue
        df = pd.DataFrame(rows)
        rec = {"n_seeds": len(df)}
        for m in METRICS:
            rec[f"{m}_mean"], rec[f"{m}_std"] = df[m].mean(), df[m].std(ddof=1) if len(df) > 1 else np.nan
        if bench:
            b = bench[0]
            rec.update({"params_M": b["params"]["total"] / 1e6, "size_MB": b["model_size_mb"],
                        "GMACs": (b["macs"] or np.nan) / 1e9,
                        "lat_cpu_bs1_ms": b["latency_cpu_bs1"]["median_ms"],
                        "lat_gpu_bs1_ms": b.get("latency_gpu_bs1", {}).get("median_ms", np.nan),
                        "train_time_min": np.mean([x.get("train_time_s") or np.nan for x in bench]) / 60})
        table[name] = rec
        per_class[name] = pd.DataFrame(pcs).mean().to_dict()
    if not table:
        return
    T = pd.DataFrame(table).T
    T.to_csv(out / "tables" / f"comparison_{a.split}.csv")
    lines = ["| model | seeds | acc | macro-F1 | weighted-F1 | params (M) | GMACs | CPU ms/img | GPU ms/img |", "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda r, m: f"{r[m + '_mean']:.4f}" + (f" ± {r[m + '_std']:.4f}" if not np.isnan(r[m + "_std"]) else "")
    for n, r in T.iterrows():
        g = lambda k: f"{r[k]:.3f}" if k in r and not pd.isna(r[k]) else "n/a"
        lines.append(f"| {n} | {int(r['n_seeds'])} | {fmt(r, 'accuracy')} | {fmt(r, 'f1_macro')} | {fmt(r, 'f1_weighted')} | "
                     f"{g('params_M')} | {g('GMACs')} | {g('lat_cpu_bs1_ms')} | {g('lat_gpu_bs1_ms')} |")
    (out / "tables" / f"comparison_{a.split}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    plot_metric_bars(T, ["accuracy", "f1_macro", "f1_weighted"], out / "figures" / f"comparison_{a.split}.png",
                     f"Model comparison ({a.split}, mean ± std over seeds)")
    plot_per_class_f1(per_class, out / "figures" / f"per_class_f1_{a.split}.png")


if __name__ == "__main__":
    main()
