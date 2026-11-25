# Experiment plan (proposed, not run on real data)

Terminology: derived MRI metrics (FA, cortical thickness, volumes) are *structural
correlates*, not measures of neuroplasticity. Say "structural/developmental
trajectories", not "plasticity", unless a task/intervention design supports it.

## E1 - Trajectory structure
- H: motor/cognitive trajectories are better described by >1 latent class than a single mean curve.
- Models: LME (baseline) vs GAMM vs LCGA/GMM (k=1..5). Compare BIC, entropy, smallest-class size (>5%), class stability across bootstrap.
- Caveat: class number is sensitive to missingness, age coverage, and model form; classes are descriptive, not "subtypes".
- Current code: LME + a two-stage approximation only. True LCGA/GMM and GAMM still to add.

## E2 - Early prediction (landmark design)
- H: observations <= landmark age predict outcome at horizon better than a baseline.
- Required baselines (beyond mean): (a) landmark-time outcome only (autoregressive), (b) GMFCS only. **A fusion model that does not beat (a) adds nothing.** The smoke test shows `mot_score` dominating importance, which is expected for a persistence-heavy outcome.
- Validation: repeated subject-level GroupKFold; nested CV for any tuning; imputation inside folds; permutation null.

## E3 - Modality ablation
- H: fusion > best single modality. Compare clinical / imaging / cognitive-motor / fusion with paired differences across repeats and bootstrap CI over subjects. Report negative results.
- Deep multimodal nets only if n is in the high hundreds+; otherwise justify not using them.

## E4 - Explainability
- Held-out permutation importance, SHAP as secondary; stability across folds. Associations only.

## E5 - External validation
- Train on one site/cohort, test on another (site-held-out). If not feasible, state it as the main limitation.

## Fallback if no CP cohort is accessible
Method-development paper on a public pediatric longitudinal cohort (e.g. normative development) framed honestly as such; or a registry/clinical-score-only study.
