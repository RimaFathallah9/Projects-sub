"""Model 2: ResNet18 with ImageNet weights (He et al., 2016).

Adaptations:
  * Classifier: fc(512->1000) replaced by Dropout + Linear(512->7).
  * Input: in_channels=3 keeps the pretrained stem; in_channels=1 sums the pretrained RGB filters
    into one channel (preserves the response to a gray image replicated over 3 channels).
  * small_input=True sets conv1 stride 1 and removes the stem max-pool. For a 48x48 input the stock
    stem would shrink layer4 to 2x2; this keeps layer4 at 6x6 (pretrained weights are all reused,
    only the stride changes). Costs ~16x more compute than the stock stem on the same input.
  * trainable_from selects what is fine-tuned: 'fc' (linear probe) < 'layer4' < 'layer3' < 'layer2'
    < 'layer1' < 'all'. Everything before it is frozen (BN statistics frozen too).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from fer.models.common import FreezeAwareModule

_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]


class ResNet18FER(FreezeAwareModule):
    def __init__(self, num_classes: int = 7, in_channels: int = 3, pretrained: bool = True,
                 trainable_from: str = "all", dropout: float = 0.0, small_input: bool = False):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels != 3:
            old = net.conv1
            new = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            if pretrained:
                new.weight.data = old.weight.data.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / (
                    in_channels if in_channels > 1 else 1)
            net.conv1 = new
        if small_input:
            net.conv1.stride = (1, 1)
            net.maxpool = nn.Identity()
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
        self.net = net
        self.set_trainable_from(trainable_from)

    def set_trainable_from(self, name: str) -> None:
        if name == "all":
            name = "stem"
        if name not in _ORDER:
            raise ValueError(f"trainable_from must be one of {_ORDER + ['all']}, got '{name}'")
        k = _ORDER.index(name)
        groups = {"stem": [self.net.conv1, self.net.bn1], "layer1": [self.net.layer1],
                  "layer2": [self.net.layer2], "layer3": [self.net.layer3],
                  "layer4": [self.net.layer4], "fc": [self.net.fc]}
        for i, g in enumerate(_ORDER):
            for m in groups[g]:
                for p in m.parameters():
                    p.requires_grad = i >= k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def backbone_parameters(self):
        return [p for n, p in self.net.named_parameters() if not n.startswith("fc.") and p.requires_grad]

    def head_parameters(self):
        return [p for p in self.net.fc.parameters() if p.requires_grad]

    def gradcam_layer(self) -> nn.Module:
        return self.net.layer4[-1]
