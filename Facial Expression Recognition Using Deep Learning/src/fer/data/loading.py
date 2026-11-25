"""Load FER-2013 from either the original CSV or an image-folder export.

Supported layouts under `root` (auto-detected):
  1. fer2013.csv (columns: emotion, pixels, Usage)   -> official train/val/test split preserved
     (also icml_face_data.csv, whose column names have leading spaces)
  2. train/<class>/*.png|jpg and test/<class>/*  -> no validation split; one is created in splits.py
     (optional val/ or validation/ folder is honoured)
Parsed arrays are cached as .npz for fast reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from fer.constants import CLASS_TO_IDX, IMG_SIZE

_USAGE = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class FERData:
    images: np.ndarray  # (N, 48, 48) uint8
    labels: np.ndarray  # (N,) int64
    split: np.ndarray   # (N,) '<U5' in {'train','val','test'} as provided by the source
    source: str

    def __len__(self) -> int:
        return len(self.labels)


def _find_source(root: Path) -> tuple[str, Path]:
    for c in (root / "fer2013.csv", root / "fer2013" / "fer2013.csv", root / "icml_face_data.csv"):
        if c.is_file():
            return "csv", c
    for base in (root, root / "fer2013"):
        if (base / "train").is_dir() and (base / "test").is_dir():
            return "folders", base
    raise FileNotFoundError(
        f"No FER-2013 data found under '{root}'. Expected fer2013.csv or train/ + test/ class folders."
    )


def _signature(kind: str, path: Path) -> str:
    if kind == "csv":
        st = path.stat()
        return f"csv|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    n = sum(1 for p in path.rglob("*") if p.suffix.lower() in _IMG_EXT)
    return f"folders|{path.resolve()}|{n}"


def _load_csv(path: Path) -> FERData:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    usage = df["Usage"].map(_USAGE)
    if usage.isna().any():
        raise ValueError(f"Unknown Usage values: {sorted(df['Usage'][usage.isna()].unique())}")
    images = np.stack([np.array(s.split(), dtype=np.uint8) for s in df["pixels"]])
    if images.shape[1] != IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Expected {IMG_SIZE * IMG_SIZE} pixels per row, got {images.shape[1]}")
    return FERData(
        images=images.reshape(-1, IMG_SIZE, IMG_SIZE),
        labels=df["emotion"].to_numpy(np.int64),
        split=usage.to_numpy(dtype="<U5"),
        source=str(path),
    )


def _image_records(base: Path) -> list[tuple[Path, int, str]]:
    split_dirs = {"train": "train", "test": "test", "val": "val", "validation": "val"}
    records: list[tuple[Path, int, str]] = []
    for dname, sname in split_dirs.items():
        d = base / dname
        if not d.is_dir():
            continue
        for cdir in sorted(p for p in d.iterdir() if p.is_dir()):
            cname = cdir.name.lower()
            if cname not in CLASS_TO_IDX:
                raise ValueError(f"Unexpected class folder '{cdir.name}' in {d}")
            for f in sorted(cdir.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMG_EXT:
                    records.append((f, CLASS_TO_IDX[cname], sname))
    return records


def _read_gray(path: Path) -> np.ndarray:
    """Load one face as a 48x48 uint8 array and close the file (Windows handle limit)."""
    with Image.open(path) as im:
        gray = im.convert("L")
        if gray.size != (IMG_SIZE, IMG_SIZE):
            gray = gray.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        return np.asarray(gray, dtype=np.uint8).copy()


def _load_folders(base: Path) -> FERData:
    records = _image_records(base)
    if not records:
        raise FileNotFoundError(f"No images found under {base}")
    images = np.empty((len(records), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    labels = np.empty(len(records), dtype=np.int64)
    splits = np.empty(len(records), dtype="<U5")
    for i, (path, label, split) in enumerate(tqdm(records, desc="Loading images", unit="img")):
        images[i] = _read_gray(path)
        labels[i] = label
        splits[i] = split
    return FERData(images, labels, splits, str(base))


def load_fer2013(root: str | Path, cache_dir: str | Path | None = None, use_cache: bool = True) -> FERData:
    root = Path(root)
    kind, path = _find_source(root)
    sig = _signature(kind, path)
    cache_file = Path(cache_dir) / "fer2013_cache.npz" if cache_dir else None
    if use_cache and cache_file is not None and cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as z:
            if str(z["signature"]) == sig:
                return FERData(z["images"], z["labels"], z["split"], str(z["source"]))
    data = _load_csv(path) if kind == "csv" else _load_folders(path)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, images=data.images, labels=data.labels, split=data.split,
                            signature=np.array(sig), source=np.array(data.source))
    return data
