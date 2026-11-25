"""Leakage-safe evaluation.

 * splits are by SUBJECT (GroupKFold): repeated measures never straddle folds;
 * imputation/scaling live INSIDE the sklearn Pipeline (fit on train folds only);
 * every model is compared with a mean-predictor baseline and a label-permutation null.
For small n use repeated group k-fold and report spread, not a single split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def make_models(seed=0):
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    return {
        "mean_baseline": DummyRegressor(),
        "ridge": make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "random_forest": make_pipeline(imp(), RandomForestRegressor(
            300, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
        "xgboost": make_pipeline(imp(), XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=seed, n_jobs=1)),
    }


def group_cv(table, cols, model, n_splits=5, n_repeats=5, seed=0):
    """Repeated subject-level CV. Returns array of (R2, MAE) per repeat."""
    X, y, g = table[cols], table["y"].values, table["subject_id"].values
    ids = np.unique(g)
    res = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        fold_of = dict(zip(ids, rng.permutation(len(ids))))  # reshuffle subject order
        grp = np.array([fold_of[s] for s in g])
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(X, y, grp):
            m = clone(model).fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
        res.append((r2_score(y, pred), mean_absolute_error(y, pred)))
    return np.array(res)


def permutation_null(table, cols, model, n_perm=20, seed=0):
    """R2 under shuffled outcomes; should be <= ~0. Sanity check against leakage."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        t = table.copy()
        t["y"] = rng.permutation(t["y"].values)
        out.append(group_cv(t, cols, model, n_repeats=1, seed=int(rng.integers(1_000_000)))[0, 0])
    return np.array(out)


def make_classifiers(seed=0):
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(imp(), StandardScaler(), LogisticRegression(
            max_iter=800, random_state=seed)),
        "random_forest": make_pipeline(imp(), RandomForestClassifier(
            300, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        "xgboost": make_pipeline(imp(), XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=1)),
    }


def group_cv_classify(table, cols, model, n_splits=5, n_repeats=3, seed=0):
    """Repeated subject-level CV for a classifier. Returns (accuracy, macro-F1) per repeat."""
    X, y, g = table[cols], table["y"].values, table["subject_id"].values
    labels = np.unique(y)
    ids = np.unique(g)
    res = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        fold_of = dict(zip(ids, rng.permutation(len(ids))))
        grp = np.array([fold_of[s] for s in g])
        pred = np.empty(len(y), dtype=float)
        for tr, te in GroupKFold(n_splits).split(X, y, grp):
            m = clone(model).fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
        res.append((accuracy_score(y, pred),
                    f1_score(y, pred, average="macro", labels=labels, zero_division=0)))
    return np.array(res)


def compare_classify(table, cols, n_repeats=3, seed=0) -> pd.DataFrame:
    rows = []
    for mname, model in make_classifiers(seed).items():
        r = group_cv_classify(table, cols, model, n_repeats=n_repeats, seed=seed)
        rows.append(dict(model=mname, accuracy_mean=r[:, 0].mean(), accuracy_sd=r[:, 0].std(),
                         macro_f1_mean=r[:, 1].mean(), macro_f1_sd=r[:, 1].std(),
                         n_windows=len(table), n_subjects=table["subject_id"].nunique()))
    return pd.DataFrame(rows)


def compare(table, sets, n_repeats=5, seed=0) -> pd.DataFrame:
    rows = []
    for mname, model in make_models(seed).items():
        use = {"none": sets["fusion"]} if mname == "mean_baseline" else sets
        for sname, cols in use.items():
            r = group_cv(table, cols, model, n_repeats=n_repeats, seed=seed)
            rows.append(dict(model=mname, features=sname, r2_mean=r[:, 0].mean(),
                             r2_sd=r[:, 0].std(), mae_mean=r[:, 1].mean(), n=len(table)))
    return pd.DataFrame(rows)
