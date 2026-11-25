"""NTNU-HARChildren wearable recordings (thigh + lower-back accelerometers).

Files are one session per child, 50 Hz, with frame-level activity labels.
This is not a longitudinal visit table: there is no age column. Windows from
the same child stay in the same CV fold.

Cohort tags follow the Dataverse file-name note (numeric ids = CP, TD*/PM* =
typically developing). The companion paper reports 16 CP and 63 TD children,
which does not match the 32 numeric files in this folder, so the tag is stored
and is not used as a model feature or as a training target.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

AXES = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
FS = 50
# Transition (9) is excluded in the source experiments. 0 is unlabeled.
DROP_LABELS = {0, 9}
ACTIVITY = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    10: "bending",
    11: "jumping",
    12: "cycling_stand",
    13: "cycling_sit",
    14: "cycling",
}


def cohort_tag(stem: str) -> str:
    if stem.startswith(("TD", "PM")):
        return "typically_developing_filename"
    if stem.isdigit():
        return "cerebral_palsy_filename"
    return "unknown"


def _windows(frame: pd.DataFrame, max_per_label: int, rng: np.random.Generator):
    x = frame[AXES].to_numpy(dtype=np.float32)
    lab = pd.to_numeric(frame["label"], errors="coerce").to_numpy()
    nwin = len(frame) // FS
    if nwin == 0:
        return None, None
    x = x[: nwin * FS].reshape(nwin, FS, len(AXES))
    lab = lab[: nwin * FS].reshape(nwin, FS)
    finite = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(lab).all(axis=1)
    pure = np.all(lab == lab[:, :1], axis=1)
    y = lab[:, 0].astype(np.int32)
    keep = finite & pure & ~np.isin(y, list(DROP_LABELS))
    x, y = x[keep], y[keep]
    if len(y) == 0:
        return None, None
    chosen = []
    for code in np.unique(y):
        idx = np.flatnonzero(y == code)
        if len(idx) > max_per_label:
            idx = rng.choice(idx, size=max_per_label, replace=False)
        chosen.append(idx)
    idx = np.sort(np.concatenate(chosen))
    w = x[idx]
    feat = np.concatenate([w.mean(1), w.std(1), w.min(1), w.max(1)], axis=1)
    return feat, y[idx]


def load_windows(csv_dir: Path | str, max_per_label: int = 25, seed: int = 0) -> pd.DataFrame:
    """1-second non-overlapping windows, capped per child per activity."""
    csv_dir = Path(csv_dir)
    rng = np.random.default_rng(seed)
    rows = []
    files = sorted(p for p in csv_dir.glob("*.csv") if p.stem[:1].isdigit() or p.stem.startswith(("TD", "PM")))
    for path in files:
        frame = pd.read_csv(path, usecols=AXES + ["label"])
        feat, y = _windows(frame, max_per_label, rng)
        if feat is None:
            continue
        part = pd.DataFrame(feat, columns=[f"{ax}_{stat}" for stat in ("mean", "std", "min", "max") for ax in AXES])
        part.insert(0, "subject_id", path.stem)
        part.insert(1, "cohort_filename", cohort_tag(path.stem))
        part.insert(2, "y", y)
        rows.append(part)
    if not rows:
        raise FileNotFoundError(f"no wearable csv files in {csv_dir}")
    return pd.concat(rows, ignore_index=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("subject_id", "cohort_filename", "y")]
