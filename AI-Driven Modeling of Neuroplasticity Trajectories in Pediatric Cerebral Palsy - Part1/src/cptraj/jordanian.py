"""Adapter for the Jordanian pediatric CP compiled cohort.

One row per child (a single assessment). Dates of birth are never copied out.
GMFCS is the outcome. Functional mobility items (FMS, wheelchair, standing,
stairs) are the same construct as GMFCS and are not used as predictors.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

YES_NO = {
    "B4_Consanguinity": "clin_consanguinity",
    "B5_NICU": "clin_nicu",
    "C60_Impairment_1_Vision": "clin_vision_impairment",
    "C62_Impairment_2_Hearing": "clin_hearing_impairment",
    "C64_Impairment_3_Epilepsy_Seizures": "clin_epilepsy",
    "C66_Impairment_4_Speech": "clin_speech_impairment",
    "C68_Impairment_5_Cognitive": "clin_cognitive_impairment",
}
MAS = {
    "F7_MAS_Elbow_R1": "mot_mas_elbow_r",
    "F10_MAS_Elbow_L1": "mot_mas_elbow_l",
    "F13_MAS_Knee_R1": "mot_mas_knee_r",
    "F16_MAS_Knee_L1": "mot_mas_knee_l",
}


def _rom_columns() -> list[tuple[str, str]]:
    """F19 L / F20 R through F37 L / F38 R. Concurrent impairment, not GMFCS."""
    cols = []
    k = 1
    for n in range(19, 39, 2):
        cols.append((f"F{n}_Lower_Extrimities_RoM_{k}_L", f"mot_rom_{k:02d}_l"))
        cols.append((f"F{n+1}_Lower_Extrimities_RoM_{k}_R", f"mot_rom_{k:02d}_r"))
        k += 1
    return cols


def _yes(s: pd.Series) -> pd.Series:
    """Dictionary: 1 = yes, 2 = no. Any other code is missing."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out = out.mask(s.eq(1), 1.0).mask(s.eq(2), 0.0)
    return out


def _cp_type(code: pd.Series) -> pd.DataFrame:
    """Assessor CP topography. Codes are nominal, so they are indicators."""
    groups = {
        "clin_cp_diplegia": {1},
        "clin_cp_quadriplegia": {2},
        "clin_cp_hemiplegia": {3, 4},
        "clin_cp_other_spastic": {5, 6},
        "clin_cp_dyskinetic": {7, 8, 9},
        "clin_cp_ataxic": {10},
        "clin_cp_hypotonia": {12},
        "clin_cp_unspecified": {11, 13},
    }
    return pd.DataFrame({name: code.isin(vals).astype(float) for name, vals in groups.items()})


def _cp_cause(code: pd.Series) -> pd.DataFrame:
    """Parent-reported cause, collapsed. Other specified causes are the reference."""
    return pd.DataFrame({
        "clin_cause_hypoxia": code.eq(14).astype(float),
        "clin_cause_prematurity": code.eq(26).astype(float),
        "clin_cause_unknown": code.eq(27).astype(float),
    })


def to_visit_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One visit per child. Does not copy dates of birth or the GMFCS text label."""
    out = pd.DataFrame({
        "subject_id": raw["Main1_ID"].astype(str),
        "age_months": raw["Main3_Age_at_Assessment"].astype(float) * 12.0,
        "clin_female": raw["A2_Child_Gender"].eq(2).astype(float),
        "clin_birth_weight_g": pd.to_numeric(raw["B2_Birth_Weight"], errors="coerce"),
        "clin_gestation_weeks": pd.to_numeric(raw["B3_Gestation_Age"], errors="coerce"),
        # Dictionary says days. The recorded values run 0-60, which is also
        # compatible with months, so the number is kept unconverted.
        "clin_dx_age_recorded": pd.to_numeric(raw["B11_Age_at_Diagnosis"], errors="coerce"),
        "mot_gmfcs": pd.to_numeric(raw["F51_GMFCS"], errors="coerce"),
    })
    for src, name in YES_NO.items():
        out[name] = _yes(raw[src])
    out = pd.concat([out, _cp_type(raw["F2_CP_Type_Assessor_Report"]),
                     _cp_cause(raw["B15_CP_Cause"])], axis=1)
    for src, name in MAS.items():
        out[name] = pd.to_numeric(raw[src], errors="coerce")
    for src, name in _rom_columns():
        out[name] = pd.to_numeric(raw[src], errors="coerce")
    keep = [c for c in out.columns if c == "subject_id" or out[c].notna().any()]
    return out[keep]


def prediction_table(visits: pd.DataFrame) -> pd.DataFrame:
    """Outcome is GMFCS. The outcome column is not a feature."""
    t = visits.dropna(subset=["mot_gmfcs"]).copy()
    t = t.rename(columns={"mot_gmfcs": "y"})
    return t.reset_index(drop=True)


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in table.columns if c not in ("subject_id", "y", "age_months")]
    history = [c for c in cols if c.startswith("clin_")]
    exam = [c for c in cols if c.startswith("mot_")]
    return {
        "age_only": ["clin_age_months"] if "clin_age_months" in table.columns else ["age_months"],
        "history": history,
        "exam": exam,
        "fusion": history + exam,
    }


def load(path: Path | str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Compiled CP Dataset")
    visits = to_visit_table(raw)
    # age is also a clinical feature; keep a prefixed copy for the model matrix
    visits.insert(2, "clin_age_months", visits["age_months"])
    return visits
