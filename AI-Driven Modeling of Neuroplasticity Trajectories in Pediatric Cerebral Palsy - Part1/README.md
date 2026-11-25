# Multimodal modeling of neurodevelopmental trajectories in pediatric cerebral palsy

## Abstract
Cerebral palsy is a permanent disorder of movement and posture caused by a non-progressive injury to the developing brain. Motor and cognitive function still change across childhood, and those changes are the object of this project. The aim is to test whether longitudinal clinical, motor, and imaging measures describe more than one developmental trajectory, and whether observations up to a landmark age predict later function better than simple baselines. Derived MRI metrics are treated as structural correlates of development, not as direct measures of neuroplasticity. Model explanations are associations, not causal effects. Every learner is compared with a mean predictor and a label-permutation null. Deep learning is used only if it beats those baselines at a sample size that can support it.

The longitudinal experiments are specified in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). They have been exercised on simulated visits (`cptraj.synthetic`) so the leakage checks run. They have not been fit to repeated real assessments, because no such table is in hand yet.

## Study design
One row is one child at one visit: `subject_id`, `age_months`, and modality prefixes `clin_*`, `img_*`, `cog_*`, and `mot_*`. Prediction uses only observations at or before the landmark age. Cross-validation holds out whole children. Imputation and scaling are fit inside each training fold.

Planned comparisons, once repeated visits exist:

1. Trajectory structure: a linear mixed-effects mean curve against latent growth classes.
2. Early prediction: history up to the landmark age against an autoregressive baseline and against GMFCS alone.
3. Modality ablation: clinical, imaging, and cognitive-motor blocks against their combination.
4. Explanation: held-out permutation importance, with SHAP as a secondary view.
5. External validation: train on one site or cohort and test on another.

## Data sources
Files used for the interim models live in `data/csv/` and are gitignored. The Jordan spreadsheet contains dates of birth and must not be committed. Adapters are `cptraj.jordanian` and `cptraj.har_children`. Neither source has repeated visits, so neither can carry the trajectory aim above.

### CPUP-Jordan compiled cohort
Al-Sowi AM, AlMasri N, Hammo B, Al-Qwaqzeh FA. Cerebral palsy classification based on multi-feature analysis using machine learning. *Informatics in Medicine Unlocked.* 2023. [doi:10.1016/j.imu.2023.101197](https://doi.org/10.1016/j.imu.2023.101197).

The table is the CPUP-Jordan follow-up registry, exported in February 2020 and released as Mendeley Data, [doi:10.17632/snsk3xvjb2.2](https://doi.org/10.17632/snsk3xvjb2.2). It contains 305 children and one assessment each (about 6 months to 16 years). The interim model predicts Gross Motor Function Classification System (GMFCS) level at that same visit from birth history, comorbidities, CP topography, tone, and range of motion. Functional Mobility Scale items, the Manual Ability Classification System, and the Communication Function Classification System are withheld, because they describe function in the same way GMFCS does. Dates of birth are dropped before modeling.

Under subject-level cross-validation, ridge regression on history plus examination reaches R² ≈ 0.31 and a mean absolute error of about 0.94 GMFCS levels. Predicting the cohort mean errs by about 1.20 levels. Age alone does not predict GMFCS. A permutation of the outcome returns R² ≈ 0. The dominant associate is spastic quadriplegia. This is a same-day association between topography and gross-motor level.

### NTNU-HARChildren wearable recordings
Tørring MF, Logacjov A, Brændvik SM, Ustad A, Roeleveld K, Bardal EM. Validation of two novel human activity recognition models for typically developing children and children with cerebral palsy. *PLOS ONE.* 2024;19(9):e0308853. [doi:10.1371/journal.pone.0308853](https://doi.org/10.1371/journal.pone.0308853).

Accelerometer files are the public Dataverse release, [doi:10.18710/EPCXCC](https://doi.org/10.18710/EPCXCC): one session per child, sensors on the thigh and lower back, video labels of activity. The interim model classifies 1-second windows and keeps every window from a child inside a single fold. XGBoost accuracy is about 0.86, against a majority-class accuracy of 0.15. The task is activity recognition during that session. It is not a classifier of cerebral palsy versus typical development. The release note that numeric filenames are children with cerebral palsy, and `TD*` / `PM*` filenames are typically developing children, does not match the paper’s sample of 16 and 63. That filename tag is stored and is not used as a label.

### Records inspected and not analyzed
The SPHN JSON in `data/` is a catalog entry for restricted clinical gait-laboratory data (UKBB and HUG). It contains no measurement rows. The CPRN file is a registry data dictionary, not a cohort. Controlled-access candidates that could support the longitudinal aim, including CP-NET and Hemi-NET on Brain-CODE, are tracked in [docs/DATASETS.md](docs/DATASETS.md). The ABCD Study is listed there only as a normative reference: enrollment is at 9–10 years, which misses early cerebral palsy development, and the sample is not a cerebral palsy cohort.

## Upcoming steps
1. Obtain a cerebral palsy cohort with at least about 100 children and at least three visits. The first request is CP-NET / Hemi-NET through the Brain-CODE data-release portal, asking whether the release is longitudinal and whether an unaffiliated investigator can apply.
2. Map that release onto the visit schema and run the five experiments above. Report a fusion model only if it beats the landmark-time outcome and GMFCS alone.
3. Until those visits exist, keep the Jordanian and wearable results in their present frame: a cross-sectional GMFCS association, and session-level activity recognition. Do not describe them as neurodevelopmental trajectories.
4. On the Jordanian table, leave FMS, MACS, and CFCS out of the GMFCS predictors. On the wearable table, compare the 1-second model with the original Tørring protocol ([repository](https://github.com/ntnu-ai-lab/harth-ml-experiments)), still holding out children.
5. Add true latent-class growth and generalized additive mixed models before any claim about the number of trajectories. The code now has a linear mixed-effects baseline and a two-stage approximation only. External validation on a second cohort remains open.

## Reproducibility
```
pip install -e .[dev]
python -m pytest
python scripts/run_synthetic_smoke.py
python scripts/run_real_data.py
```

`src/cptraj/` holds the schema, features, longitudinal models, evaluation, explanation, synthetic data, and the two adapters. `tests/` checks leakage. `scripts/run_synthetic_smoke.py` is the plumbing check on simulated visits. `scripts/run_real_data.py` retrains the interim models and writes metrics under `results/`. Patient files, processed tables, and results stay out of git.
