"""Image transforms. Augmentation is applied on the training split ONLY.

augmentation levels:
  none   : no augmentation
  basic  : horizontal flip + small affine (rotation 10deg, 10% shift, 0.9-1.1 scale)
  strong : basic (rotation 15, shear 10) + brightness/contrast jitter + random erasing
"""
from __future__ import annotations

from torchvision import transforms as T

from fer.constants import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def _norm_stats(channels: int, kind: str):
    if kind == "imagenet":
        if channels == 3:
            return IMAGENET_MEAN, IMAGENET_STD
        return (sum(IMAGENET_MEAN) / 3,), (sum(IMAGENET_STD) / 3,)
    if kind == "simple":
        return (0.5,) * channels, (0.5,) * channels
    raise ValueError(f"Unknown normalize '{kind}'")


def build_transform(dcfg: dict, train: bool) -> T.Compose:
    size = int(dcfg.get("image_size", IMG_SIZE))
    ch = int(dcfg.get("channels", 1))
    aug = dcfg.get("augmentation", "none") if train else "none"
    if aug not in {"none", "basic", "strong"}:
        raise ValueError(f"Unknown augmentation '{aug}'")

    ops: list = []
    if size != IMG_SIZE:
        ops.append(T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC))
    if aug == "basic":
        ops += [T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))]
    elif aug == "strong":
        ops += [T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
                T.ColorJitter(brightness=0.3, contrast=0.3)]
    if ch == 3:
        ops.append(T.Grayscale(num_output_channels=3))
    mean, std = _norm_stats(ch, dcfg.get("normalize", "simple"))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if aug == "strong":
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0))
    return T.Compose(ops)
