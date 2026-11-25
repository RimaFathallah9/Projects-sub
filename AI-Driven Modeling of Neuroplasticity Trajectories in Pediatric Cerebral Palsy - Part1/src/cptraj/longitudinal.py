"""Longitudinal models. LME is the primary inferential baseline.

- fit_lme: linear mixed-effects (random intercept + slope by subject).
- two_stage_classes: EXPLORATORY approximation of growth mixture modeling
  (per-subject growth coefficients -> Gaussian mixture, BIC selection). It
  ignores uncertainty in per-subject coefficients; use true LCGA/GMM (R `lcmm`,
  Mplus) for confirmatory claims, and report this limitation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.mixture import GaussianMixture


def fit_lme(df: pd.DataFrame, outcome: str, fixed: str = "age_months"):
    d = df.dropna(subset=[outcome]).copy()
    model = smf.mixedlm(f"{outcome} ~ {fixed}", d, groups=d["subject_id"],
                        re_formula="~age_months")
    return model.fit(reml=True, method="lbfgs")


def subject_growth_coefs(df: pd.DataFrame, outcome: str, min_obs: int = 3) -> pd.DataFrame:
    out = []
    for sid, g in df.dropna(subset=[outcome]).groupby("subject_id"):
        if len(g) >= min_obs:
            b1, b0 = np.polyfit(g.age_months, g[outcome], 1)
            out.append((sid, b0, b1))
    return pd.DataFrame(out, columns=["subject_id", "intercept", "slope"]).set_index("subject_id")


def two_stage_classes(coefs: pd.DataFrame, k_range=range(1, 6), seed=0):
    X = (coefs - coefs.mean()) / coefs.std()
    fits = {k: GaussianMixture(k, n_init=5, random_state=seed).fit(X) for k in k_range}
    bic = {k: m.bic(X) for k, m in fits.items()}
    best = min(bic, key=bic.get)
    return pd.Series(fits[best].predict(X), index=coefs.index, name="class"), bic
