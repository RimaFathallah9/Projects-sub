import numpy as np
import pytest

from fer.data.loading import FERData


def make_data(n_per_split=(200, 40, 40), seed=0, with_val=True) -> FERData:
    rng = np.random.default_rng(seed)
    n_tr, n_va, n_te = n_per_split
    n = n_tr + n_va + n_te
    images = rng.integers(0, 256, size=(n, 48, 48), dtype=np.uint8)
    labels = rng.integers(0, 7, size=n).astype(np.int64)
    labels[:7] = np.arange(7)  # ensure every class exists
    split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te, dtype="<U5")
    if not with_val:
        split[split == "val"] = "train"
    return FERData(images, labels, split, "synthetic")


@pytest.fixture
def synthetic():
    return make_data()
