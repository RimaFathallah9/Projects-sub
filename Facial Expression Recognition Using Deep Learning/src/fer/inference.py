"""Load trained checkpoints and run single-image inference (used by evaluation, Grad-CAM and the web app)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from fer.constants import CLASSES, IMG_SIZE
from fer.data.transforms import build_transform
from fer.explain.gradcam import GradCAM, overlay_cam
from fer.models.factory import build_model


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    device = torch.device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["data"]["channels"], pretrained=False)  # weights come from ckpt
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt


class Predictor:
    """Wraps model + the exact eval preprocessing used in training."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.cfg, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.transform = build_transform(self.cfg["data"], train=False)
        self.cam = GradCAM(self.model, self.model.gradcam_layer())

    def preprocess(self, gray_u8: np.ndarray) -> torch.Tensor:
        if gray_u8.shape != (IMG_SIZE, IMG_SIZE):
            gray_u8 = cv2.resize(gray_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return self.transform(Image.fromarray(gray_u8, mode="L")).unsqueeze(0).to(self.device)

    def predict(self, gray_u8: np.ndarray, with_cam: bool = True) -> dict:
        x = self.preprocess(gray_u8)
        cam, logits, idx = self.cam(x)
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        res = {"probs": {c: float(p) for c, p in zip(CLASSES, probs)},
               "prediction": CLASSES[int(probs.argmax())], "confidence": float(probs.max())}
        if with_cam:
            face48 = cv2.resize(gray_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            res["cam_overlay"] = overlay_cam(face48, cam[0])
        return res
