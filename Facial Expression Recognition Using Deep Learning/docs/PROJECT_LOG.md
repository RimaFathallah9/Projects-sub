# Project log

## Completed
- [x] Stage 0: repo skeleton, packaging, requirements, config system
- [x] Stage 1-2: loader (csv/folders), hash-based leakage-safe splits, transforms, datasets
- [x] Stage 3: trainer (AMP, schedulers, early stopping, checkpoints, TensorBoard)
- [x] Stages 4-6: Custom CNN, ResNet18, MobileNetV2 + factory
- [x] Evaluation: metrics, ECE, confusions, benchmark (params/MACs/latency), error analysis
- [x] Grad-CAM + randomisation and border-mass checks
- [x] FastAPI demo with server-side consent
- [x] Unit tests, sweeps, docs scaffolding

## Pending (needs data / compute)
- [x] Image folders are in `data/train/` and `data/test/` (28,709 + 7,178). Run `scripts/eda.py` and record findings below
- [ ] Phase B ablations (docs/EXPERIMENT_PROTOCOL.md)
- [ ] Phase C-E final runs, comparison, Grad-CAM
- [ ] Fill docs/MODEL_CARD.md from real outputs
- [ ] Write report (reports/REPORT_OUTLINE.md); verify every reference before citing

## Decisions (with rationale)
| Date | Decision | Why | Evidence |
|---|---|---|---|
| 2026-11-25 | Hold out `data/test` untouched; carve validation from `data/train` with hash groups | the folder export has no Usage column, and test (7,178) is PublicTest+PrivateTest combined | class counts match the official Training set (28,709) |
| 2026-11-25 | Select on val macro-F1, not accuracy | class imbalance (Disgust is tiny) | - |
| 2026-11-25 | Fixed split seed shared by all training seeds | seed variance should reflect training, not data | - |
| 2026-11-25 | Pretrained default `small_input=true` (stride-1 stem, no max-pool) | stock stem leaves 2x2 maps at 48px | **to be validated** in resnet18_input_protocol sweep |

## Experiments
_(append: sweep name, date, winner, val macro-F1, seed noise observed, decision taken)_

## EDA findings
_(fill from reports/eda.json after running scripts/eda.py)_

## Known issues / notes
- Weighted-loss runs report a weighted `val_loss`, not comparable with unweighted runs; compare on macro-F1.
- Deterministic cuDNN is on by default; disable with `deterministic: false` for speed.
