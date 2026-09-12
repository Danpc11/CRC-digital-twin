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


def test_harmonize_stage_is_robust_to_duplicate_index_inf_and_nullable_na():
    # indice duplicado (concat sin ignore_index) no debe tronar
    out = harmonize_stage(pd.Series([1.0, 2.0, np.nan, "iii"], index=[0, 0, 1, 1]), verbose=False)
    assert out.tolist()[:2] == [1, 2] and np.isnan(out.iloc[2]) and out.iloc[3] == 3
    # inf y numeros enormes: faltantes, sin OverflowError/TypeError
    out = harmonize_stage(pd.Series([float("inf"), 1e300, 2.0]), verbose=False)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1]) and out.iloc[2] == 2
    # NA de dtypes anulables no se reporta como "<NA>" no reconocido
    out = harmonize_stage(pd.Series([1, 2, None], dtype="Int64"), verbose=False)
    assert out.tolist()[:2] == [1, 2] and np.isnan(out.iloc[2])
    out = harmonize_stage(pd.Series(["II", None], dtype="string"), verbose=False)
    assert out.iloc[0] == 2 and np.isnan(out.iloc[1])
