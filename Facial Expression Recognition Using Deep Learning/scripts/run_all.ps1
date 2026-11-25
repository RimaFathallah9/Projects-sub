# Full protocol, Phases A-E (long-running). Run after activating .venv. Edit the model configs with your
# frozen ablation choices BEFORE phase C. Sweeps are resumable.
$ErrorActionPreference = "Stop"
python -m pytest -q
python scripts/eda.py

# Phase B: ablations (validation only)
foreach ($s in "cnn_lr","cnn_batch","cnn_optimizer","cnn_architecture","cnn_dropout","cnn_augmentation","cnn_imbalance",
               "resnet18_input_protocol","resnet18_freezing","mobilenetv2_input_protocol","mobilenetv2_freezing") {
    python scripts/run_sweep.py --sweep "configs/sweeps/$s.yaml"
}

# Phase C: final multi-seed runs
$models = "custom_cnn","resnet18_finetune","mobilenetv2"
foreach ($m in $models) { foreach ($seed in 42,43,44) { python scripts/train.py --config "configs/$m.yaml" --seed $seed } }

# Phase D: one-time test evaluation + comparison
foreach ($m in $models) { foreach ($seed in 42,43,44) { python scripts/evaluate.py --run "outputs/runs/$m/seed$seed" --split test --benchmark } }
python scripts/compare.py --split test --groups custom_cnn=outputs/runs/custom_cnn resnet18=outputs/runs/resnet18_finetune mobilenetv2=outputs/runs/mobilenetv2

# Phase E: explainability
foreach ($m in $models) { python scripts/gradcam_run.py --run "outputs/runs/$m/seed42" --split test }
