"""Leakage-safe train/val/test splits.

Policy (priority test > val > train):
  * Test indices are never modified (official PrivateTest, or the folder export's test/).
  * If the source has no validation split, one is carved out of train with StratifiedGroupKFold,
    grouping by image hash so identical images never straddle train/val.
  * Any val image whose pixels also occur in test is dropped from val; any train image whose pixels
    occur in val or test is dropped from train. All overlaps are counted in the report.
  * `assert_no_leakage` re-verifies the final result and is executed on every call.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from fer.constants import CLASSES
from fer.data.loading import FERData

SPLITS = ("train", "val", "test")


@dataclass
class SplitResult:
    indices: dict[str, np.ndarray]
    hashes: np.ndarray
    report: dict


def hash_images(images: np.ndarray) -> np.ndarray:
    return np.array([hashlib.md5(im.tobytes()).hexdigest() for im in images])


def assert_no_leakage(hashes: np.ndarray, indices: dict[str, np.ndarray]) -> None:
    sets = {s: set(hashes[indices[s]].tolist()) for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        if inter:
            raise AssertionError(f"Data leakage: {len(inter)} identical images in both {a} and {b}")
    all_idx = np.concatenate([indices[s] for s in SPLITS])
    if len(all_idx) != len(np.unique(all_idx)):
        raise AssertionError("An index appears in more than one split")


def build_splits(
    data: FERData,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    remove_cross_split_duplicates: bool = True,
) -> SplitResult:
    hashes = hash_images(data.images)
    idx = {s: np.where(data.split == s)[0] for s in SPLITS}
    report: dict = {"source": data.source, "val_created_from_train": False}

    if len(idx["val"]) == 0:
        n_splits = max(2, int(round(1.0 / val_fraction)))
        tr = idx["train"]
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        _, val_rel = next(sgkf.split(tr, data.labels[tr], groups=hashes[tr]))
        mask = np.zeros(len(tr), dtype=bool)
        mask[val_rel] = True
        idx["val"], idx["train"] = tr[mask], tr[~mask]
        report["val_created_from_train"] = True
        report["protocol_note"] = (
            "No validation folder was provided. A stratified, hash-grouped validation split "
            f"(target fraction {val_fraction}) was taken from train. The test folder is unchanged "
            "and is not used for training or model selection."
        )

    sets = {s: set(hashes[idx[s]].tolist()) for s in SPLITS}
    report["raw_sizes"] = {s: int(len(idx[s])) for s in SPLITS}
    report["cross_split_overlap_unique_images"] = {
        "train&val": len(sets["train"] & sets["val"]),
        "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
    }
    report["within_split_duplicates"] = {s: int(len(idx[s]) - len(sets[s])) for s in SPLITS}

    # Same pixels carrying different labels anywhere in the dataset (label noise indicator).
    lab_per_hash: dict[str, set] = {}
    for h, y in zip(hashes.tolist(), data.labels.tolist()):
        lab_per_hash.setdefault(h, set()).add(y)
    report["images_with_conflicting_labels"] = int(sum(len(v) > 1 for v in lab_per_hash.values()))

    removed = {"val": 0, "train": 0}
    if remove_cross_split_duplicates:
        test_h = np.array(list(sets["test"]))
        keep = ~np.isin(hashes[idx["val"]], test_h)
        removed["val"] = int((~keep).sum())
        idx["val"] = idx["val"][keep]
        held_out = np.array(list(set(hashes[idx["val"]].tolist()) | sets["test"]))
        keep = ~np.isin(hashes[idx["train"]], held_out)
        removed["train"] = int((~keep).sum())
        idx["train"] = idx["train"][keep]
    report["removed_for_leakage"] = removed

    assert_no_leakage(hashes, idx)  # raises if anything slipped through (or if removal was disabled)

    report["final_sizes"] = {s: int(len(idx[s])) for s in SPLITS}
    report["class_counts"] = {
        s: {CLASSES[c]: int(n) for c, n in enumerate(np.bincount(data.labels[idx[s]], minlength=len(CLASSES)))}
        for s in SPLITS
    }
    return SplitResult(indices=idx, hashes=hashes, report=report)
