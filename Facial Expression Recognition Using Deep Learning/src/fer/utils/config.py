"""YAML config loading with inheritance (`base:`) and dotted CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str] | dict | None) -> dict:
    """Overrides: ["train.lr=0.01", "data.augmentation=strong"] or {"train.lr": 0.01}."""
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    items = overrides.items() if isinstance(overrides, dict) else (o.split("=", 1) for o in overrides)
    for k, v in items:
        set_by_path(cfg, k.strip(), yaml.safe_load(v) if isinstance(v, str) else v)
    return cfg


def resolve_base(config_path: Path, base: str) -> Path:
    """Resolve `base:` relative to the config file, then relative to the working directory."""
    raw = Path(base)
    if raw.is_absolute():
        return raw
    beside = (config_path.parent / raw).resolve()
    if beside.is_file():
        return beside
    from_cwd = (Path.cwd() / raw).resolve()
    if from_cwd.is_file():
        return from_cwd
    return beside


def load_config(path: str | Path, overrides: list[str] | dict | None = None) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(load_config(resolve_base(path, str(base))), cfg)
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
