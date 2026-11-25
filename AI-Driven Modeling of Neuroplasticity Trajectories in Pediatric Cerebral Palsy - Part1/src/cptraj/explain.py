"""Explainability. Outputs are ASSOCIATIONS with the prediction target, not
causal effects. Permutation importance is computed on held-out subjects."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold


def heldout_permutation_importance(table, cols, model, n_splits=5, n_repeats=10, seed=0):
    X, y, g = table[cols], table["y"].values, table["subject_id"].values
    imps = []
    for tr, te in GroupKFold(n_splits).split(X, y, g):
        m = clone(model).fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], n_repeats=n_repeats, random_state=seed)
        imps.append(r.importances_mean)
    return pd.DataFrame({"feature": cols, "importance": np.mean(imps, axis=0),
                         "fold_sd": np.std(imps, axis=0)}).sort_values("importance", ascending=False)
