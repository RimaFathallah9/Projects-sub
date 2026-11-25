"""Dataset and DataLoader construction."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fer.constants import NUM_CLASSES
from fer.data.loading import FERData
from fer.data.transforms import build_transform
from fer.utils.seed import make_generator, seed_worker


class FERDataset(Dataset):
    """In-memory dataset of 48x48 uint8 grayscale faces."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels)
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = Image.fromarray(self.images[i], mode="L")
        x = self.transform(img) if self.transform is not None else torch.from_numpy(self.images[i]).float()[None] / 255
        return x, int(self.labels[i])


def build_dataloaders(dcfg: dict, data: FERData, indices: dict[str, np.ndarray], seed: int,
                      which: tuple[str, ...] = ("train", "val")):
    """Returns (loaders dict, train class counts). Only the requested splits get a loader, so the
    test set cannot be touched by accident during training."""
    bs = int(dcfg.get("batch_size", 128))
    nw = int(dcfg.get("num_workers", 0))
    counts = np.bincount(data.labels[indices["train"]], minlength=NUM_CLASSES)
    pin = torch.cuda.is_available()
    loaders = {}
    for split in which:
        idx = indices[split]
        train = split == "train"
        ds = FERDataset(data.images[idx], data.labels[idx], build_transform(dcfg, train=train))
        sampler = None
        if train and dcfg.get("weighted_sampler", False):
            w = 1.0 / np.maximum(counts, 1)
            sample_w = torch.as_tensor(w[data.labels[idx]], dtype=torch.double)
            sampler = WeightedRandomSampler(sample_w, num_samples=len(ds), replacement=True,
                                            generator=make_generator(seed))
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(train and sampler is None), sampler=sampler,
            num_workers=nw, pin_memory=pin, drop_last=False,
            persistent_workers=nw > 0, worker_init_fn=seed_worker if nw > 0 else None,
            generator=make_generator(seed + 1),
        )
    return loaders, counts
