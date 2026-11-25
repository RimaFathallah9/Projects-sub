"""Landmark-style feature construction.

Task: using only observations at age <= landmark_months, predict the outcome at
the visit closest to horizon_months. Features come strictly from the past of the
landmark -> no temporal leakage. One row per subject.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PREFIXES = ("clin_", "img_", "cog_", "mot_")


def build_landmark_table(df: pd.DataFrame, outcome: str, landmark_months: float,
                         horizon_months: float, tol: float = 3.0) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("subject_id"):
        past = g[g.age_months <= landmark_months]
        fut = g[(g.age_months - horizon_months).abs() <= tol]
        if past.empty or fut.empty:
            continue
        tgt = fut.iloc[int((fut.age_months - horizon_months).abs().argmin())]
        if pd.isna(tgt[outcome]):
            continue
        feat = {"subject_id": sid, "y": float(tgt[outcome])}
        for c in g.columns:
            if not c.startswith(PREFIXES):
                continue
            s = past[["age_months", c]].dropna()
            feat[c] = s[c].iloc[-1] if len(s) else np.nan  # last observed value
            if c.startswith(("cog_", "mot_")):
                feat[c + "_slope"] = (np.polyfit(s.age_months, s[c], 1)[0]
                                      if len(s) >= 2 else np.nan)
        rows.append(feat)
    return pd.DataFrame(rows)


def feature_sets(table: pd.DataFrame, outcome: str | None = None) -> dict[str, list[str]]:
    """Modality ablation sets; 'fusion' = all (early/concatenation fusion).

    If `outcome` is given, adds 'persistence' (the outcome's own history only).
    Any model that cannot beat it has not learned anything beyond 'children
    stay on their trajectory'.
    """
    cols = [c for c in table.columns if c not in ("subject_id", "y")]
    by = lambda *p: [c for c in cols if c.startswith(p)]
    sets = {"clinical": by("clin_"), "imaging": by("img_"),
            "cognitive_motor": by("cog_", "mot_"), "fusion": cols}
    if outcome:
        sets = {"persistence": [c for c in cols if c in (outcome, outcome + "_slope")], **sets}
    return {k: v for k, v in sets.items() if v}
