"""Model 3: MobileNetV2 with ImageNet weights (Sandler et al., 2018).

Why it is interesting here: inverted residual blocks with depthwise-separable convolutions give
~3.4M params / ~0.3 GMACs at 224x224 vs ~11.7M / ~1.8 GMACs for ResNet18, so we can quantify the
accuracy-vs-efficiency trade-off on FER-2013.

Adaptations: same as ResNet18 (classifier swap, 1-channel stem option, small_input stride change).
freeze_blocks=k freezes features[:k] (0 = fine-tune everything, 19 = only the classifier trains).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from fer.models.common import FreezeAwareModule


class MobileNetV2FER(FreezeAwareModule):
    def __init__(self, num_classes: int = 7, in_channels: int = 3, pretrained: bool = True,
                 freeze_blocks: int = 0, dropout: float = 0.2, small_input: bool = False):
        super().__init__()
        net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        conv = net.features[0][0]
        if in_channels != 3 or small_input:
            new = nn.Conv2d(in_channels, 32, 3, stride=1 if small_input else 2, padding=1, bias=False)
            if pretrained:
                w = conv.weight.data
                new.weight.data = w.clone() if in_channels == 3 else w.sum(dim=1, keepdim=True).repeat(
                    1, in_channels, 1, 1) / in_channels
            net.features[0][0] = new
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        self.net = net
        n = len(net.features)
        if not 0 <= freeze_blocks <= n:
            raise ValueError(f"freeze_blocks must be in [0, {n}]")
        for i, blk in enumerate(net.features):
            for p in blk.parameters():
                p.requires_grad = i >= freeze_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def backbone_parameters(self):
        return [p for p in self.net.features.parameters() if p.requires_grad]

    def head_parameters(self):
        return [p for p in self.net.classifier.parameters() if p.requires_grad]

    def gradcam_layer(self) -> nn.Module:
        return self.net.features[-1]
