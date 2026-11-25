# Experiment protocol

Goal: every claim in the report traces to a file under `outputs/` or `reports/`.

Data on disk: `data/train` (28,709, official Training) and `data/test` (7,178, PublicTest and PrivateTest combined). Validation is carved from train. Test metrics from this protocol are not PrivateTest scores.

## Phase A - sanity
1. `pytest -q` passes.
2. `scripts/eda.py`; read `reports/eda.json` (duplicates, conflicting labels, imbalance).
3. `scripts/train.py --config configs/smoke.yaml` runs end to end.
4. Overfit check: train the custom CNN on a tiny subset (or `augmentation=none`, `weight_decay=0`) and confirm train accuracy -> ~100%. If not, there is a bug.

## Phase B - ablations (validation split only, seed 42, one factor at a time)
Order matters; freeze the winner of each step into the base config before the next:
1. `cnn_lr` -> `cnn_batch` -> `cnn_optimizer` -> `cnn_architecture` -> `cnn_dropout` -> `cnn_augmentation` -> `cnn_imbalance`
2. `resnet18_input_protocol` -> `resnet18_freezing`
3. `mobilenetv2_input_protocol` -> `mobilenetv2_freezing`

`python scripts/run_sweep.py --sweep configs/sweeps/<name>.yaml` -> `outputs/sweeps/<name>/results.csv`.
Single-seed differences of ~1 point are within seed noise on a 3.5k-image validation set; confirm any
close call with 3 seeds (set `seeds: [42, 43, 44]`) before adopting it. Record every decision in `docs/PROJECT_LOG.md`.

## Phase C - final multi-seed training
Update the three configs with the frozen choices, then for seeds 42, 43, 44:
```
python scripts/train.py --config configs/<model>.yaml --seed <s>
```

## Phase D - one-time test evaluation
```
python scripts/evaluate.py --run outputs/runs/<model>/seed<s> --split test --benchmark
python scripts/compare.py --split test --groups custom_cnn=outputs/runs/custom_cnn resnet18=outputs/runs/resnet18_finetune mobilenetv2=outputs/runs/mobilenetv2
```
No hyper-parameter changes after this point.

## Phase E - analysis
`scripts/gradcam_run.py --run <dir> --split test`; inspect `*_misclassified.png`, `*_confusion.png`, `top_confusions`
and `imbalance_correlation` in `test_metrics.json`.
