"""End-to-end smoke test on SYNTHETIC data (not patients). Verifies plumbing only."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cptraj import schema, synthetic, features, longitudinal, evaluate, explain

out = ROOT / "results"
out.mkdir(exist_ok=True)
df = schema.validate(synthetic.simulate())
rep = schema.quality_report(df)
print("QUALITY:", {k: v for k, v in rep.items() if k != "missing_fraction"})

fit = longitudinal.fit_lme(df, "mot_score")
print(fit.summary().tables[1])
coefs = longitudinal.subject_growth_coefs(df, "mot_score")
cls, bic = longitudinal.two_stage_classes(coefs)
print("BIC by k:", {k: round(v) for k, v in bic.items()})

t = features.build_landmark_table(df, "mot_score", landmark_months=18, horizon_months=48)
sets = features.feature_sets(t, outcome="mot_score")
print(f"landmark table n={len(t)}; sets={ {k: len(v) for k, v in sets.items()} }")
res = evaluate.compare(t, sets, n_repeats=3)
print(res.round(3).to_string(index=False))
null = evaluate.permutation_null(t, sets["fusion"], evaluate.make_models()["ridge"], n_perm=10)
print(f"permutation-null R2 (ridge, fusion): mean={null.mean():.3f} (expect <= ~0)")
imp = explain.heldout_permutation_importance(t, sets["fusion"], evaluate.make_models()["random_forest"])
print(imp.head(6).round(3))

fig, ax = plt.subplots()
for sid, g in df.groupby("subject_id"):
    ax.plot(g.age_months, g.mot_score, color="grey", alpha=.15)
ax.set(xlabel="age (months)", ylabel="mot_score (SYNTHETIC)",
       title="SYNTHETIC data - pipeline test only")
fig.savefig(out / "synthetic_trajectories.png", dpi=120)
