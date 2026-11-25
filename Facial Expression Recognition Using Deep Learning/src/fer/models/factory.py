"""Build a model from the `model:` section of a config."""
from __future__ import annotations

import torch.nn as nn

from fer.constants import NUM_CLASSES
from fer.models.custom_cnn import CustomCNN
from fer.models.mobilenetv2 import MobileNetV2FER
from fer.models.resnet18 import ResNet18FER


def build_model(mcfg: dict, in_channels: int, num_classes: int = NUM_CLASSES,
                pretrained: bool | None = None) -> nn.Module:
    """`pretrained` overrides the config (used when loading a checkpoint: no weight download needed)."""
    name = mcfg["name"].lower()
    pre = mcfg.get("pretrained", True) if pretrained is None else pretrained
    if name == "custom_cnn":
        return CustomCNN(
            in_channels=in_channels, num_classes=num_classes,
            widths=tuple(mcfg.get("widths", (64, 128, 256, 512))),
            convs_per_block=mcfg.get("convs_per_block", 2),
            dropout=mcfg.get("dropout", 0.25), head_dropout=mcfg.get("head_dropout", 0.5),
            hidden=mcfg.get("hidden", 256),
        )
    if name == "resnet18":
        return ResNet18FER(num_classes, in_channels, pre, mcfg.get("trainable_from", "all"),
                           mcfg.get("dropout", 0.0), mcfg.get("small_input", False))
    if name == "mobilenetv2":
        return MobileNetV2FER(num_classes, in_channels, pre, mcfg.get("freeze_blocks", 0),
                              mcfg.get("dropout", 0.2), mcfg.get("small_input", False))
    raise ValueError(f"Unknown model '{name}'")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
