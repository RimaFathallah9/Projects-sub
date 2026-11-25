"""Grad-CAM analysis for a trained run: per-class galleries (correct vs wrong), border-mass artifact
check, and a model-randomisation sanity check.

  python scripts/gradcam_run.py --run outputs/runs/custom_cnn/seed42 --split val

Use --split val while developing; use test only for the final reported figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fer.constants import CLASSES
from fer.data.transforms import build_transform
from fer.explain.gradcam import GradCAM, border_mass, cam_similarity, overlay_cam, randomized_copy
from fer.inference import load_checkpoint
from fer.pipeline import get_device, prepare_data
from PIL import Image


def batch_cams(cam: GradCAM, tf, imgs, device, bs=128):
    cams, preds, probs = [], [], []
    for i in range(0, len(imgs), bs):
        x = torch.stack([tf(Image.fromarray(im, mode="L")) for im in imgs[i:i + bs]]).to(device)
        c, logits, idx = cam(x)
        cams.append(c); preds.append(idx.cpu().numpy()); probs.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
    return np.concatenate(cams), np.concatenate(preds), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--sanity-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run); dev = get_device()
    model, cfg, _ = load_checkpoint(run / "best.pt", dev)
    data, splits = prepare_data(cfg)
    idx = splits.indices[a.split]
    imgs, labels = data.images[idx], data.labels[idx]
    tf = build_transform(cfg["data"], train=False)
    gc = GradCAM(model, model.gradcam_layer())
    cams, preds, conf = batch_cams(gc, tf, imgs, dev)
    rng = np.random.default_rng(a.seed)

    # --- galleries: for each class, k correct and k wrong (true class = c), explaining the PREDICTED class
    fig, axs = plt.subplots(len(CLASSES), 2 * a.per_class, figsize=(2 * a.per_class * 1.3, len(CLASSES) * 1.5))
    for c, name in enumerate(CLASSES):
        for half, ok in enumerate((True, False)):
            pool = np.where((labels == c) & ((preds == c) == ok))[0]
            pick = rng.choice(pool, min(a.per_class, len(pool)), replace=False) if len(pool) else []
            for j in range(a.per_class):
                ax = axs[c, half * a.per_class + j]; ax.axis("off")
                if j < len(pick):
                    i = pick[j]
                    ax.imshow(overlay_cam(imgs[i], cams[i], out_size=96))
                    ax.set_title(f"{name}→{CLASSES[preds[i]]}\n{conf[i]:.2f}", fontsize=6, color="green" if ok else "crimson")
    fig.suptitle(f"Grad-CAM ({run.name}, {a.split}): left = correct, right = misclassified (true→pred, conf)", fontsize=9)
    fig.tight_layout(); fig.savefig(run / f"gradcam_{a.split}.png", dpi=140); plt.close(fig)

    # --- artifact check: CAM mass on the outer border vs the uniform expectation
    bm = np.array([border_mass(c)[0] for c in cams]); exp = border_mass(cams[0])[1]
    per_class_bm = {n: {"correct": float(bm[(labels == c) & (preds == c)].mean()) if ((labels == c) & (preds == c)).any() else None,
                        "wrong": float(bm[(labels == c) & (preds != c)].mean()) if ((labels == c) & (preds != c)).any() else None}
                    for c, n in enumerate(CLASSES)}

    # --- sanity: explanations should change when the weights are randomised
    sub = rng.choice(len(imgs), min(a.sanity_n, len(imgs)), replace=False)
    rmodel = randomized_copy(model).to(dev)
    gc_r = GradCAM(rmodel, rmodel.gradcam_layer())
    cams_r, _, _ = batch_cams(gc_r, tf, imgs[sub], dev)
    sims = [cam_similarity(cams[i], cams_r[k]) for k, i in enumerate(sub)]
    out = {"split": a.split, "n": int(len(imgs)),
           "border_mass_mean": float(bm.mean()), "border_mass_expected_if_uniform": exp,
           "border_mass_by_class": per_class_bm,
           "randomisation_spearman_mean": float(np.nanmean(sims)), "randomisation_spearman_median": float(np.nanmedian(sims)),
           "interpretation": "Spearman near 0 after randomisation = explanation depends on learned weights (pass). "
                             "Near 1 = explanation largely independent of the model (fail). border_mass >> expected suggests "
                             "reliance on image borders/background."}
    (run / f"gradcam_{a.split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "border_mass_by_class"}, indent=1))


if __name__ == "__main__":
    main()
