import numpy as np
import pytest

from conftest import make_data
from fer.data.splits import SPLITS, assert_no_leakage, build_splits


def test_clean_data_has_no_leakage(synthetic):
    r = build_splits(synthetic)
    assert_no_leakage(r.hashes, r.indices)
    assert sum(len(v) for v in r.indices.values()) == len(synthetic)


def test_cross_split_duplicates_removed_from_lower_priority_split():
    d = make_data()
    tr = np.where(d.split == "train")[0]; va = np.where(d.split == "val")[0]; te = np.where(d.split == "test")[0]
    d.images[va[0]] = d.images[te[0]]      # val duplicates a test image
    d.images[tr[0]] = d.images[te[1]]      # train duplicates a test image
    d.images[tr[1]] = d.images[va[1]]      # train duplicates a val image
    r = build_splits(d)
    assert r.report["removed_for_leakage"]["val"] == 1
    assert r.report["removed_for_leakage"]["train"] == 2
    assert len(r.indices["test"]) == len(te), "test set must never lose samples"
    assert_no_leakage(r.hashes, r.indices)


def test_leakage_detector_fires_when_removal_disabled():
    d = make_data()
    d.images[0] = d.images[-1]  # train image identical to a test image
    with pytest.raises(AssertionError):
        build_splits(d, remove_cross_split_duplicates=False)


def test_val_created_from_train_is_group_aware_and_deterministic():
    d = make_data((300, 0, 50), with_val=False)
    d.images[5] = d.images[6]  # within-train duplicate pair must stay together
    r1, r2 = build_splits(d, 0.2, 1), build_splits(d, 0.2, 1)
    assert r1.report["val_created_from_train"]
    assert all(np.array_equal(r1.indices[s], r2.indices[s]) for s in SPLITS)
    assert_no_leakage(r1.hashes, r1.indices)
    assert 0.1 < len(r1.indices["val"]) / 300 < 0.3
