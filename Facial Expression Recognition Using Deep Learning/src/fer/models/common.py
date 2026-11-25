"""Shared model utilities."""
from __future__ import annotations

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class FreezeAwareModule(nn.Module):
    """nn.Module whose frozen BatchNorm layers stay in eval mode during training.

    Setting requires_grad=False on a BN layer only freezes gamma/beta; the running mean/var would
    still update in train mode and silently drift the "frozen" backbone. We keep such BN layers in
    eval mode so a frozen layer is truly frozen.
    """

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
