"""Grad-CAM (Selvaraju et al., 2017) plus quantitative sanity checks.

CAM_c = ReLU( sum_k alpha_k^c * A^k ),  alpha_k^c = spatial mean of d y_c / d A^k
where A is the activation map of the chosen conv layer and y_c the pre-softmax score of class c.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.acts = output.detach()
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: (B,C,H,W). Returns (cams (B,H,W) in [0,1] numpy, logits, class_idx used)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)  # guarantees a graph even if every parameter is frozen
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                idx = logits.argmax(1)
            elif isinstance(class_idx, int):
                idx = torch.full((x.size(0),), class_idx, dtype=torch.long, device=x.device)
            else:
                idx = torch.as_tensor(class_idx, dtype=torch.long, device=x.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx[:, None]).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        mn = cam.amin(dim=(1, 2), keepdim=True)
        mx = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.cpu().numpy(), logits.detach(), idx

    def remove(self) -> None:
        self._handle.remove()


def overlay_cam(gray_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.45, out_size: int = 192) -> np.ndarray:
    """Blend a jet heatmap over the grayscale face. Returns RGB uint8 (out_size, out_size, 3)."""
    base = cv2.resize(gray_u8, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap((cv2.resize(cam, (out_size, out_size)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1 - alpha, heat, alpha, 0)


# ------------------------------------------------------------------ sanity / artifact checks
def border_mass(cam: np.ndarray, margin: float = 0.1) -> tuple[float, float]:
    """Fraction of CAM mass in the outer ring (width = margin * side) vs the ring's area fraction.
    Observed >> expected suggests reliance on borders/background rather than the face interior."""
    h, w = cam.shape
    mh, mw = int(round(h * margin)), int(round(w * margin))
    inner = cam[mh:h - mh, mw:w - mw].sum()
    total = cam.sum() + 1e-12
    expected = 1.0 - ((h - 2 * mh) * (w - 2 * mw)) / (h * w)
    return float(1.0 - inner / total), float(expected)


def randomized_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    """Model with all parameters re-initialised (Adebayo et al., 2018 'model parameter randomisation')."""
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    for mod in m.modules():
        if hasattr(mod, "reset_parameters") and mod is not m:
            mod.reset_parameters()
    return m.eval()


def cam_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two maps. High value after randomising the weights means
    the explanation does not depend on what the model learned (a failed sanity check)."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(spearmanr(a.ravel(), b.ravel())[0])
