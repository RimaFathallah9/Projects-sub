"""Train the project models on the local cohorts.

Jordanian compiled CP cohort: one assessment per child, so this is concurrent
prediction of GMFCS, not a longitudinal trajectory.

NTNU-HARChildren: 1-second accelerometer windows, activity classification with
subject-held-out folds. Not an age trajectory.

The SPHN JSON next to these files is a catalog record (restricted gait-lab
metadata). It has no measurement rows and is not used.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from cptraj import schema, evaluate, explain, jordanian, har_children

JORDAN = ROOT / "data" / "csv" / "Jordanian children with CP -Compiled dataset.xlsx"
HAR = ROOT / "data" / "csv"
OUT = ROOT / "results"
PROC = ROOT / "data" / "processed"


def train_jordanian():
    visits = jordanian.load(JORDAN)
    schema.validate(visits.drop(columns=["clin_age_months"]))
    print("JORDANIAN", schema.quality_report(visits.drop(columns=["clin_age_months"])))
    print("Single visit per child: mixed-effects trajectories are not identifiable. "
          "Training a concurrent GMFCS model instead.")
    table = jordanian.prediction_table(visits)
    sets = jordanian.feature_sets(table)
    print("feature counts", {k: len(v) for k, v in sets.items()})
    res = evaluate.compare(table, sets, n_repeats=5, seed=0)
    print(res.round(3).to_string(index=False))
    null = evaluate.permutation_null(
        table, sets["fusion"], evaluate.make_models()["ridge"], n_perm=20, seed=0)
    print(f"permutation-null R2 (ridge, fusion): mean={null.mean():.3f} sd={null.std():.3f}")
    imp = explain.heldout_permutation_importance(
        table, sets["fusion"], evaluate.make_models()["random_forest"], n_repeats=5)
    print(imp.head(12).round(3).to_string(index=False))

    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    visits.to_csv(PROC / "jordanian_visits.csv", index=False)
    res.to_csv(OUT / "jordanian_gmfcs_metrics.csv", index=False)
    imp.to_csv(OUT / "jordanian_gmfcs_importance.csv", index=False)

    # One subject-level split, for a predicted-vs-observed plot. Not the CV estimate.
    model = evaluate.make_models()["random_forest"]
    X, y, g = table[sets["fusion"]], table["y"].values, table["subject_id"].values
    pred = pd.Series(index=table.index, dtype=float)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred.iloc[te] = clone(model).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(y, pred, alpha=0.45, s=18, c="#1f4e79")
    ax.plot([1, 5], [1, 5], color="grey", lw=1)
    ax.set(xlabel="Observed GMFCS", ylabel="Predicted GMFCS (one 5-fold split)",
           title="Jordanian cohort: concurrent GMFCS, random forest")
    fig.tight_layout()
    fig.savefig(OUT / "jordanian_gmfcs_pred.png", dpi=120)
    plt.close(fig)
    return res


def train_har():
    print("Reading wearable files and building 1-second windows...")
    windows = har_children.load_windows(HAR, max_per_label=25, seed=0)
    cols = har_children.feature_columns(windows)
    counts = windows.groupby(["cohort_filename", "subject_id"]).size().groupby(level=0).size()
    print("children by filename tag", counts.to_dict(), "windows", len(windows))
    print(windows["y"].map(har_children.ACTIVITY).fillna(windows["y"].astype(str)).value_counts().to_string())
    codes = sorted(windows["y"].unique())
    windows = windows.copy()
    windows["y"] = windows["y"].map({code: i for i, code in enumerate(codes)}).astype(int)
    print("label codes", {i: har_children.ACTIVITY.get(code, str(code)) for i, code in enumerate(codes)})
    res = evaluate.compare_classify(windows, cols, n_repeats=3, seed=0)
    print(res.round(3).to_string(index=False))
    windows.head(0).to_csv(PROC / "har_window_columns.csv", index=False)
    windows.sample(n=min(5000, len(windows)), random_state=0).to_csv(
        PROC / "har_windows_sample.csv", index=False)
    res.to_csv(OUT / "har_activity_metrics.csv", index=False)
    return res


if __name__ == "__main__":
    print("SPHN JSON is a restricted-catalog record with no measurement rows; skipped.")
    print("CPRN file is a registry data dictionary, not patient records; skipped.")
    train_jordanian()
    train_har()
