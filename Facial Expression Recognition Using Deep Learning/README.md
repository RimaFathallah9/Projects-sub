# Facial Expression Recognition Using Deep Learning

A comparative study of a custom convolutional network and ImageNet transfer learning (ResNet18, MobileNetV2) on FER-2013.

The code trains, evaluates, and explains seven-way facial-expression classifiers. Reported numbers in a paper must come from `outputs/` and `reports/` after the scripts below are run. This README does not contain accuracy figures, because those experiments have not been run yet.

## Abstract

Recognising a facial expression from a single low-resolution image is a standard test of whether a vision model learns facial structure or only memorizes dataset artifacts. FER-2013 provides 48×48 grayscale faces labelled as angry, disgust, fear, happy, sad, surprise, or neutral. The labels are noisy, the classes are heavily imbalanced (disgust is rare; happy is common), and published scores are easy to inflate by tuning on the test set or by letting duplicate images leak across splits.

This repository compares three models under one protocol: a VGG-style CNN trained from scratch, ResNet18 fine-tuned from ImageNet, and MobileNetV2 fine-tuned from ImageNet. Validation is carved from the training images with a stratified, hash-grouped split. The folder named `data/test` is held out and is not used for training or for choosing hyperparameters. Model selection uses validation macro-F1, which does not let the large happy class dominate the decision. Final claims are meant to be mean ± standard deviation over at least three seeds. Grad-CAM maps, with a weight-randomisation check, are used to see whether the networks attend to the face.

## 1. Introduction

Facial expression recognition (FER) asks a model to map a face image to one of a small set of expression categories. On FER-2013 the input is a 48×48 grayscale crop, so the task is less about high-frequency texture than about the layout of eyes, brows, and mouth. That makes it a useful comparison between a small network trained only on this dataset and larger networks whose early filters were learned on ImageNet.

Three properties of the dataset shape the study:

- **Label noise.** Many faces are ambiguous, mislabelled, or not faces. A confident error is sometimes a bad label.
- **Imbalance.** Happy has roughly sixteen times as many training images as disgust. Accuracy alone hides failure on the rare classes.
- **Split discipline.** Tuning on the test folder, or training on an image that also appears in test, makes a result look better than it is.

The study makes three contributions in the experimental design, not in claimed accuracy:

1. A leakage check that hashes pixels and removes cross-split duplicates from train and validation only. The test folder is never edited.
2. A controlled ablation path (one factor at a time) so a gain can be attributed to a learning rate, an augmentation, or a freezing schedule.
3. An explainability pass (Grad-CAM plus a parameter-randomisation sanity check) so a high score can be questioned if the map ignores the face.

The models predict expression **categories**. They do not measure a person’s internal emotional state.

## 2. Related work

The dataset comes from the FER-2013 challenge in Goodfellow et al. (2013). Later work showed that a large share of the error is label noise: Barsoum et al. (2016) replaced the single labels with crowd distributions (FER+). Khaireddin and Chen (2021) survey how far modern CNNs get on the original labels.

The transfer-learning models are ResNet18 (He et al., 2016) and MobileNetV2 (Sandler et al., 2018). The question for a 48×48 face is not whether those architectures are strong on ImageNet, but whether their features survive the domain gap and the aggressive downsampling of a 48-pixel stem. Yosinski et al. (2014) motivate the freezing sweeps: early layers are more generic, later layers are more task-specific.

Training choices that the sweeps are allowed to test, rather than assume, include AdamW (Loshchilov and Hutter, 2019), focal loss (Lin et al., 2017), and class-balanced weights based on the effective number of samples (Cui et al., 2019). Explanations use Grad-CAM (Selvaraju et al., 2017) and the randomisation sanity check of Adebayo et al. (2018). The ethical limit of the task is stated by Barrett et al. (2019): facial movements are not a reliable readout of emotion across people and cultures.

Bibliographic details should be checked against the papers before they are copied into a thesis. A longer outline is in `reports/REPORT_OUTLINE.md`.

## 3. Dataset

### 3.1 What is on disk

Images live in class folders. Counts were taken from the files currently in this project:

| Class | `data/train` | `data/test` | Total |
|---|---:|---:|---:|
| angry | 3,995 | 958 | 4,953 |
| disgust | 436 | 111 | 547 |
| fear | 4,097 | 1,024 | 5,121 |
| happy | 7,215 | 1,774 | 8,989 |
| sad | 4,830 | 1,247 | 6,077 |
| surprise | 3,171 | 831 | 4,002 |
| neutral | 4,965 | 1,233 | 6,198 |
| **Total** | **28,709** | **7,178** | **35,887** |

The training counts match the official FER-2013 **Training** partition exactly (28,709). The test folder has 7,178 images, which is PublicTest (3,589) plus PrivateTest (3,589). The folder export does not keep the `Usage` column, so those two official test sets cannot be separated again.

Class order inside the code follows the original FER-2013 ids, not alphabetical folder order:

`0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral`

The loader maps by folder name, so a folder called `neutral` still receives id 6.

### 3.2 How the splits are built

| Role | Source | Used for |
|---|---|---|
| Train | ~90% of `data/train`, after leakage removal | gradient updates |
| Validation | ~10% of `data/train`, stratified and grouped by image hash | early stopping, ablations, checkpoint selection |
| Test | all of `data/test`, unmodified | one evaluation after every choice is frozen |

The same `split_seed` (42) is used for every training seed. Seed-to-seed differences then come from optimisation, not from a new split.

An image that appears in test is dropped from validation and from train. An image that appears in validation is dropped from train. `assert_no_leakage` runs on every build and raises if any hash remains in two splits. Within-split duplicates are counted in `split_report.json` and kept, because they are part of the original collection.

### 3.3 What this means for published FER-2013 numbers

Papers that quote a single FER-2013 test accuracy almost always mean **PrivateTest** (3,589 images), with the model trained on all 28,709 Training images and tuned on PublicTest. This project cannot reproduce that exact partition from the folders. The held-out set here is the **union** of PublicTest and PrivateTest (7,178 images), and about 10% of Training is spent on validation. Scores from this repo are therefore not directly comparable to a PrivateTest number from a paper. They are comparable to each other, because every model in the study uses the same split.

A CSV with an `emotion,pixels,Usage` header (`fer2013.csv` or `icml_face_data.csv`) is still accepted. If that file is placed at `data/fer2013.csv`, the loader keeps the official Training / PublicTest / PrivateTest assignment and does not invent a validation split.

## 4. Methodology

### 4.1 Preprocessing

- Training images are augmented. Validation and test images are not.
- `basic` augmentation: horizontal flip and a small affine (rotation, shift, scale).
- `strong` augmentation: a wider affine, brightness and contrast jitter, and random erasing.
- Grayscale pixels are mapped with either a simple mean/std of 0.5 or ImageNet statistics. ResNet18 and MobileNetV2 configs repeat the gray image to three channels so the pretrained stem can be used.
- For a 48×48 input, the pretrained configs set `small_input: true` (stride-1 stem, and no stem max-pool on ResNet). The stock stem would leave the last ResNet map at 2×2.

### 4.2 Models

**Custom CNN.** Four VGG-style stages. Each stage is 3×3 convolutions, batch normalisation, ReLU, 2×2 max-pooling, and spatial dropout. Widths double while the map halves (48→24→12→6→3). Global average pooling replaces a large fully connected layer. This is the from-scratch baseline.

**ResNet18.** ImageNet-1K weights. The 1000-way layer is replaced by dropout and a 7-way linear layer. `trainable_from` selects the first block that is allowed to learn (`fc`, `layer4`, `layer3`, `layer2`, `layer1`, or `all`). Frozen batch-normalisation layers stay in eval mode so their running statistics do not drift.

**MobileNetV2.** ImageNet-1K weights, inverted residual blocks, a 7-way classifier. `freeze_blocks` freezes the first *k* feature blocks (0 means fine-tune everything; 19 trains only the classifier). It is the efficiency comparison against ResNet18.

### 4.3 Training and selection

- Optimiser, learning rate, weight decay, scheduler, loss, and class weights are config fields. Defaults are AdamW, cosine decay, and unweighted cross-entropy. Sweeps are allowed to change them; the defaults are not treated as proven.
- Mixed precision is on when CUDA is available.
- The checkpoint is the epoch with the best validation macro-F1 (early stopping patience 12 in the base config).
- `scripts/train.py` builds loaders for train and validation only. The test split is not loaded during training.
- Each run stores `config.yaml`, `split_report.json`, `history.json`, `summary.json`, `best.pt`, and TensorBoard logs.

### 4.4 Metrics

Accuracy, balanced accuracy, macro and weighted precision / recall / F1, a full confusion matrix, per-class scores, and expected calibration error. Macro-F1 is the selection metric because disgust would otherwise be invisible. Efficiency is reported as parameter count, model size, MACs, and CPU/GPU latency.

### 4.5 Explainability

Grad-CAM is taken from the last spatial map (the last ReLU of the custom CNN, ResNet `layer4`, or the last MobileNet feature block). A second pass randomises the weights. If the maps stay almost the same, the explanation does not depend on what was learned and should not be shown as evidence. Border mass compares how much of the map sits on the outer rim of the 48×48 image with the mass expected from a uniform map.

## 5. Experimental protocol

The full sequence is `docs/EXPERIMENT_PROTOCOL.md`. In short:

1. **Sanity.** `pytest`, then `scripts/eda.py`, then a 2-epoch smoke run.
2. **Ablations, validation only, one factor at a time.** Learning rate, batch size, optimiser, depth/width, dropout, augmentation, and imbalance handling for the custom CNN; input protocol and freezing for the two pretrained models. A one-point gap on a single seed is seed noise until it is repeated.
3. **Final training** of the frozen configs on seeds 42, 43, and 44.
4. **One test evaluation** per frozen run. After this step, hyperparameters stay frozen.
5. **Error analysis and Grad-CAM** on that test output.

`scripts/run_all.ps1` runs phases A–E. It is long (many full trainings on a laptop GPU). Run the smoke test before it.

## 6. Results

No test or validation accuracy is reported here. After the final runs:

```powershell
python scripts/compare.py --split test --groups custom_cnn=outputs/runs/custom_cnn resnet18=outputs/runs/resnet18_finetune mobilenetv2=outputs/runs/mobilenetv2
```

writes `reports/tables/comparison_test.md` and the figures under `reports/figures/`. Those files are the results section of the report. Fill `docs/MODEL_CARD.md` from the same files.

## 7. Limitations and ethics

- The test folder merges the two official FER-2013 test sets, so headline numbers are not PrivateTest scores.
- Labels are noisy. Disgust has 436 training images, so its F1 will move a lot between seeds.
- FER-2013 has no demographic labels. This project cannot support a fairness claim about age, skin tone, or gender presentation.
- A webcam photo is a different distribution from these 48×48 web crops. The demo will be wrong on many real photos.
- Expression category is not emotion, intent, or a trait. The demo requires a consent flag on the server, keeps the upload in memory, and does not write the image to disk. It is a research interface, not a tool for decisions about people.

## 8. How to reproduce

Windows, PowerShell, from this folder. The machine used for the protocol notes is an RTX 4050 Laptop (6 GB) with driver 565.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
pytest -q
python scripts/eda.py
python scripts/train.py --config configs/smoke.yaml
```

The image folders are already in `data/train/<class>/` and `data/test/<class>/`. The first load builds `data/processed/fer2013_cache.npz` so later runs skip the JPEG decode. Configs point at `data` (`configs/base.yaml`).

```powershell
python scripts/train.py --config configs/custom_cnn.yaml
python scripts/train.py --config configs/resnet18_finetune.yaml
python scripts/train.py --config configs/mobilenetv2.yaml
python scripts/run_sweep.py --sweep configs/sweeps/cnn_lr.yaml
python scripts/evaluate.py --run outputs/runs/custom_cnn/seed42 --split val
tensorboard --logdir outputs
```

Test evaluation, only after the config is frozen:

```powershell
python scripts/evaluate.py --run outputs/runs/custom_cnn/seed42 --split test --benchmark
python scripts/gradcam_run.py --run outputs/runs/custom_cnn/seed42 --split test
```

Demo, after a checkpoint exists:

```powershell
$env:FER_CHECKPOINT = "outputs/runs/resnet18_finetune/seed42/best.pt"
uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000`. The page will not call the model until the consent box is checked.

`data.num_workers` is 0 because DataLoader workers often hang on Windows when the project lives under OneDrive. On Linux it can be raised in the config.

## 9. Repository layout

```
data/train, data/test     class folders (angry … surprise)
src/fer/data              loading, leakage-safe splits, transforms, loaders
src/fer/models            custom CNN, ResNet18, MobileNetV2
src/fer/training          loss, early stopping, trainer
src/fer/evaluation        metrics, plots, benchmark, error analysis
src/fer/explain           Grad-CAM and sanity checks
configs/                  one YAML per model; configs/sweeps/ for ablations
scripts/                  eda, train, sweep, evaluate, compare, gradcam
tests/                    split, metric, model, and Grad-CAM tests
app/                      FastAPI demo
docs/                     protocol, project log, model card
reports/                  EDA, figures, tables, report outline
outputs/                  created by training (gitignored)
```

## References

- Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. *arXiv:1307.0414*.
- Barsoum, E., Zhang, C., Canton Ferrer, C., & Zhang, Z. (2016). Training deep networks for facial expression recognition with crowd-sourced label distribution. *ICMI*. *arXiv:1608.01041*.
- Khaireddin, Y., & Chen, Z. (2021). Facial emotion recognition: State of the art performance on FER2013. *arXiv:2105.03588*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *NeurIPS*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. *CVPR*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Adebayo, J., et al. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements. *Psychological Science in the Public Interest*.
