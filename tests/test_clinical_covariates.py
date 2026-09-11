"""Tests de clinical_covariates.harmonize_stage.

Incluye la regresion del 2026-09-11: una columna de estadio numerica con
un solo faltante la lee pandas como float ("2.0"), y antes quedaba con
0 muestras mapeadas -- el Cox ajustado por estadio se caia en silencio
para GSE39582.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_covariates import harmonize_stage, prepare_covariates


def test_float_coded_stages_are_mapped():
    stages = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0, np.nan])
    out = harmonize_stage(stages, verbose=False)
    assert out.tolist()[:4] == [1, 2, 3, 4]
    assert np.isnan(out.iloc[4])  # estadio 0 (in situ) -> faltante a proposito
    assert np.isnan(out.iloc[5])


def test_integer_text_dukes_and_ajcc_still_map():
    stages = pd.Series(["1", "IIB", "Dukes C", "AJCC stage II CRC", "iv", "unknown"])
    out = harmonize_stage(stages, verbose=False)
    assert out.tolist()[:5] == [1, 2, 3, 2, 4]
    assert np.isnan(out.iloc[5])


def test_prepare_covariates_drops_stage_iv_with_float_input():
    df = pd.DataFrame({
        "cohort": ["A"] * 5,
        "stage": [1.0, 2.0, 4.0, 3.0, np.nan],
        "relapse_free_months": [10, 20, 30, 40, 50],
        "relapse_event": [1, 0, 1, 0, 1],
    })
    out = prepare_covariates(df, verbose=False)
    assert 4 not in out["stage_harmonized"].dropna().tolist()
    assert len(out) == 4
    assert out["stage_harmonized"].notna().sum() == 3
