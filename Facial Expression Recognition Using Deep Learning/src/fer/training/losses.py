"""Loss functions and class-imbalance weighting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(counts, scheme: str = "none", beta: float = 0.999) -> np.ndarray | None:
    """Per-class loss weights, normalised so they sum to the number of classes.

    inverse        : w_c ~ 1/n_c (strong correction)
    sqrt_inverse   : w_c ~ 1/sqrt(n_c) (milder)
    effective_num  : w_c ~ (1-beta)/(1-beta^n_c)  (Cui et al., 2019, class-balanced loss)
    """
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    if scheme == "none":
        return None
    if scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "effective_num":
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown class_weight scheme '{scheme}'")
    return w / w.sum() * len(counts)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): -(1-p_t)^gamma * log p_t, optionally class-weighted."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logpt.exp()) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_loss(tcfg: dict, class_counts, device: torch.device) -> nn.Module:
    w = class_weights(class_counts, tcfg.get("class_weight", "none"), tcfg.get("cb_beta", 0.999))
    wt = torch.as_tensor(w, dtype=torch.float32, device=device) if w is not None else None
    ls = float(tcfg.get("label_smoothing", 0.0))
    kind = tcfg.get("loss", "ce")
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=wt, label_smoothing=ls)
    if kind == "focal":
        if ls > 0:
            raise ValueError("label_smoothing is not supported with focal loss")
        return FocalLoss(float(tcfg.get("focal_gamma", 2.0)), wt).to(device)
    raise ValueError(f"Unknown loss '{kind}'")
