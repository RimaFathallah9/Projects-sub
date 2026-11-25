"""Model 1: Custom CNN trained from scratch.

Design (VGG-style, adapted to 48x48 inputs):
  [Conv3x3 -> BN -> ReLU] x convs_per_block -> MaxPool2 -> Dropout2d     (repeated per stage)
  GlobalAvgPool -> Dropout -> FC(hidden) -> ReLU -> Dropout -> FC(classes)

Reasoning:
  * 3x3 convs stacked give large receptive fields with fewer params than 5x5/7x7 kernels.
  * BN right after conv stabilises training, allows higher LR, acts as mild regulariser. The conv
    bias is disabled since BN's beta subsumes it.
  * Widths double per stage (64..512) while resolution halves (48->24->12->6->3), keeping per-stage
    compute roughly constant.
  * Dropout2d (spatial) drops whole feature maps, which regularises better than element-wise dropout
    on strongly spatially-correlated conv activations. Standard dropout is used in the FC head.
  * Global average pooling instead of flatten removes ~half a million FC parameters, reducing
    overfitting on a ~29k-image dataset, and makes Grad-CAM maps well-defined.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fer.models.common import init_weights


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_convs):
            layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False)]
        self.convs = nn.Sequential(*layers)
        self.pool = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout))

    def forward(self, x):
        return self.pool(self.convs(x))


class CustomCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 7, widths=(64, 128, 256, 512),
                 convs_per_block: int = 2, dropout: float = 0.25, head_dropout: float = 0.5,
                 hidden: int = 256):
        super().__init__()
        blocks, prev = [], in_channels
        for w in widths:
            blocks.append(ConvBlock(prev, w, convs_per_block, dropout))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(head_dropout),
            nn.Linear(prev, hidden), nn.ReLU(inplace=True), nn.Dropout(head_dropout),
            nn.Linear(hidden, num_classes),
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def gradcam_layer(self) -> nn.Module:
        """Last ReLU before the final pooling (spatial resolution 6x6 for a 48x48 input)."""
        return self.features[-1].convs[-1]
