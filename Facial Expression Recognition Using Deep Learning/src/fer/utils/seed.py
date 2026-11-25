"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU+CUDA).

    deterministic=True forces cuDNN to pick deterministic kernels and disables
    the autotuner. This costs some speed. Note that some CUDA ops are still
    non-deterministic (e.g. atomicAdd based ops); bit-exact reproducibility
    across different GPUs/driver versions is not guaranteed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
