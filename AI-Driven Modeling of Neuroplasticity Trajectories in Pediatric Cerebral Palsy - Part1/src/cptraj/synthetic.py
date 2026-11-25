"""SYNTHETIC DATA FOR PIPELINE TESTING ONLY.

These are NOT patients and carry NO scientific information about cerebral palsy.
Latent classes, effect sizes and noise are arbitrary choices so the code can be
unit-tested against a known ground truth. Never report results on this data as
findings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def simulate(n_subjects=150, ages=(6, 12, 18, 24, 36, 48), seed=0,
             signal=True, missing_rate=0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_subjects):
        cls = int(rng.integers(0, 3))  # latent trajectory class (known ground truth)
        gmfcs = int(rng.integers(1, 6))
        ga = rng.normal(34, 4)
        img_fa = rng.normal(0.4 - (0.02 * gmfcs if signal else 0), 0.05)
        slope = [0.8, 0.5, 0.25][cls] + (-0.04 * gmfcs if signal else 0)
        intercept = rng.normal(20, 4)
        for a in ages:
            y = intercept + slope * a + rng.normal(0, 2.0)
            rows.append(dict(subject_id=f"S{sid:04d}", age_months=a,
                             clin_gmfcs=gmfcs, clin_ga_weeks=ga, img_fa=img_fa,
                             cog_score=y + rng.normal(0, 3), mot_score=y,
                             true_class=cls))
    df = pd.DataFrame(rows)
    # MCAR missingness here; real data are likely MAR/MNAR (sicker kids drop out)
    for c in ["img_fa", "cog_score"]:
        df.loc[rng.random(len(df)) < missing_rate, c] = np.nan
    return df.rename(columns={"img_fa": "img_fa"})
