"""Canonical long-format schema + validation.

One row = one subject x one visit. Real datasets (CP-NET, CPOP, ...) must be
mapped to this schema by a dataset-specific adapter. Raw patient data never
enters the repo (see .gitignore).
"""
from __future__ import annotations
import pandas as pd

REQUIRED = ["subject_id", "age_months"]
# Column-prefix convention that defines modalities (used for modality ablations)
MODALITY_PREFIXES = {
    "clinical": "clin_",     # GMFCS, CP subtype, gestational age, ...
    "imaging": "img_",       # DERIVED measures (FA, volumes); NOT 'plasticity'
    "cognitive": "cog_",     # cognitive / developmental scores
    "motor": "mot_",         # motor outcomes (e.g. GMFM, PEDI)
}


class SchemaError(ValueError):
    pass


def modality_columns(df: pd.DataFrame, modality: str) -> list[str]:
    p = MODALITY_PREFIXES[modality]
    return [c for c in df.columns if c.startswith(p)]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise SchemaError on structural problems; return df sorted by subject/age."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    if df["subject_id"].isna().any():
        raise SchemaError("null subject_id")
    if df["age_months"].isna().any() or (df["age_months"] < 0).any():
        raise SchemaError("age_months must be non-null and >= 0")
    if df.duplicated(["subject_id", "age_months"]).any():
        raise SchemaError("duplicate (subject_id, age_months) rows")
    return df.sort_values(["subject_id", "age_months"]).reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> dict:
    """Cohort + missingness summary; the first thing to inspect on any dataset."""
    g = df.groupby("subject_id")
    visits = g.size()
    miss = df.drop(columns=["subject_id"]).isna().mean().sort_values(ascending=False)
    return {
        "n_subjects": int(g.ngroups),
        "n_rows": int(len(df)),
        "visits_per_subject": visits.describe().to_dict(),
        "age_range_months": [float(df.age_months.min()), float(df.age_months.max())],
        "missing_fraction": miss.round(3).to_dict(),
        "subjects_single_visit": int((visits == 1).sum()),
    }
