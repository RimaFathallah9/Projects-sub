"""Run a controlled sweep defined in configs/sweeps/*.yaml (validation metrics only).

  python scripts/run_sweep.py --sweep configs/sweeps/cnn_lr.yaml
  python scripts/run_sweep.py --sweep configs/sweeps/cnn_lr.yaml --dry-run

Sweep file keys: name, base, seeds, and EITHER
  factors + mode (ofat | grid): dotted-key -> list of values
  variants: {tag: {dotted-key: value, ...}}   (explicit override sets)
Finished runs (summary.json present) are skipped, so an interrupted sweep can be resumed.
Results -> outputs/sweeps/<name>/results.csv, ranked by validation macro-F1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from fer.pipeline import run_dir_for, run_training
from fer.utils.config import apply_overrides, load_config


def _tag(d: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", "__".join(f"{k.split('.')[-1]}={v}" for k, v in d.items())) or "base"


def expand(sw: dict) -> list[tuple[str, dict]]:
    if "variants" in sw:
        return list(sw["variants"].items())
    f = sw["factors"]
    if sw.get("mode", "ofat") == "grid":
        keys = list(f)
        return [(_tag(dict(zip(keys, v))), dict(zip(keys, v))) for v in itertools.product(*f.values())]
    return [(_tag({k: v}), {k: v}) for k, vals in f.items() for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sweep_path = Path(a.sweep)
    sw = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    out_root = Path("outputs/sweeps") / sw["name"]
    base_spec = Path(sw["base"])
    if not base_spec.is_absolute():
        beside = (sweep_path.parent / base_spec).resolve()
        base_spec = beside if beside.is_file() else (Path.cwd() / base_spec).resolve()
    base = load_config(base_spec)
    jobs = [(tag, ov, seed) for tag, ov in expand(sw) for seed in sw.get("seeds", [42])]
    print(f"Sweep '{sw['name']}': {len(jobs)} runs")
    rows = []
    for tag, ov, seed in jobs:
        cfg = apply_overrides(base, ov)
        cfg.update({"name": tag, "seed": seed, "output_dir": str(out_root)})
        if a.dry_run:
            print(f"  {tag} seed{seed}: {ov}")
            continue
        summ_path = run_dir_for(cfg) / "summary.json"
        if summ_path.is_file():
            print(f"[skip] {tag} seed{seed} (already done)")
            s = json.loads(summ_path.read_text())
        else:
            print(f"\n##### {tag} seed{seed}: {ov}")
            s = run_training(cfg)
        vm = s["val_metrics"]
        rows.append({"variant": tag, "seed": seed, "overrides": json.dumps(ov), "best_epoch": s["best_epoch"],
                     "val_acc": vm["accuracy"], "val_f1_macro": vm["f1_macro"], "val_f1_weighted": vm["f1_weighted"],
                     "params": s["params"]["total"], "train_time_s": s["train_time_s"]})
    if rows:
        df = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
        out_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_root / "results.csv", index=False)
        print("\n" + df.drop(columns="overrides").to_string(index=False))
        print(f"\nSaved {out_root/'results.csv'}  (selection is on validation macro-F1 only)")


if __name__ == "__main__":
    main()
