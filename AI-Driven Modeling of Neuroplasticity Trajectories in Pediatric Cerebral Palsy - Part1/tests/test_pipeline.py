import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from cptraj import schema, synthetic, features, evaluate


def test_validate_rejects_duplicates():
    df = synthetic.simulate(5)
    with pytest.raises(schema.SchemaError):
        schema.validate(pd.concat([df, df.iloc[[0]]]))


def test_landmark_features_use_only_past():
    df = schema.validate(synthetic.simulate(30, missing_rate=0))
    t1 = features.build_landmark_table(df, "mot_score", 18, 48)
    d2 = df.copy()
    d2.loc[d2.age_months > 18, ["cog_score", "img_fa", "mot_score"]] = 999
    d2.loc[d2.age_months == 48, "mot_score"] = df.loc[df.age_months == 48, "mot_score"]  # keep target
    t2 = features.build_landmark_table(d2, "mot_score", 18, 48)
    cols = [c for c in t1.columns if c not in ("subject_id", "y")]
    assert np.allclose(t1[cols], t2[cols], equal_nan=True)


def test_group_cv_never_splits_subject():
    df = synthetic.simulate(40)
    g = df.subject_id.values
    for tr, te in GroupKFold(5).split(df, groups=g):
        assert not set(g[tr]) & set(g[te])


def test_jordanian_table_drops_identifiers_and_outcome():
    from cptraj.jordanian import YES_NO, MAS, _rom_columns, to_visit_table, prediction_table, feature_sets
    row = {
        "Main1_ID": 7,
        "Main3_Age_at_Assessment": 2.0,
        "A2_Child_Gender": 2,
        "A1_Child_DoB": "2019-01-01",
        "B2_Birth_Weight": 2800,
        "B3_Gestation_Age": 38,
        "B11_Age_at_Diagnosis": 12,
        "F51_GMFCS": 2,
        "F2_CP_Type_Assessor_Report": 1,
        "B15_CP_Cause": 14,
    }
    row.update({src: 1 for src in YES_NO})
    row.update({src: 2 for src in MAS})
    row.update({src: 40.0 for src, _ in _rom_columns()})
    visits = to_visit_table(pd.DataFrame([row]))
    assert "A1_Child_DoB" not in visits.columns
    assert set(visits["mot_gmfcs"]) == {2}
    visits["clin_age_months"] = visits["age_months"]
    table = prediction_table(visits)
    assert "mot_gmfcs" not in feature_sets(table)["fusion"]
    assert "y" not in feature_sets(table)["fusion"]
    assert table.loc[0, "clin_female"] == 1
    assert table.loc[0, "clin_nicu"] == 1
    assert table.loc[0, "clin_cp_diplegia"] == 1


def test_permutation_null_near_zero():
    df = schema.validate(synthetic.simulate(120))
    t = features.build_landmark_table(df, "mot_score", 18, 48)
    cols = features.feature_sets(t)["fusion"]
    null = evaluate.permutation_null(t, cols, evaluate.make_models()["ridge"], n_perm=8)
    assert null.mean() < 0.05
