"""
Tests de regresion para cox_diagnostics.py -- verificados con dos
escenarios sinteticos de respuesta conocida: heterogeneidad real
inyectada (debe detectarse) y control negativo sin heterogeneidad
(NO debe detectarse falsamente).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cox_diagnostics import (
    check_heterogeneity_across_cohorts,
    check_influential_observations,
    check_proportional_hazards,
)
from lifelines import CoxPHFitter


def _make_two_cohorts(hr_a: float, hr_b: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, hr in [("A", hr_a), ("B", hr_b)]:
        for _ in range(150):
            es_x = rng.random() < 0.4
            h = hr if es_x else 1.0
            duration = rng.exponential(30 / h)
            event = int(rng.random() < 0.5)
            rows.append({"x": int(es_x), "relapse_free_months": round(duration, 1),
                         "relapse_event": event, "cohort": cohort})
    return pd.DataFrame(rows)


def test_heterogeneity_detected_when_real():
    """Con HR muy distinto entre cohortes (3.0 vs 1.0), la prueba de
    interaccion debe salir significativa."""
    df = _make_two_cohorts(hr_a=3.5, hr_b=1.0, seed=11)
    result = check_heterogeneity_across_cohorts(
        df, "relapse_free_months", "relapse_event", ["x"], cohort_col="cohort")
    p_het = result["test_interaccion"].loc["x", "p_heterogeneidad"]
    assert p_het < 0.05, f"deberia detectar heterogeneidad real, p={p_het}"


def test_heterogeneity_not_falsely_detected_when_absent():
    """Con el MISMO HR en ambas cohortes, la prueba de interaccion NO
    debe salir significativa (control negativo)."""
    df = _make_two_cohorts(hr_a=2.0, hr_b=2.0, seed=22)
    result = check_heterogeneity_across_cohorts(
        df, "relapse_free_months", "relapse_event", ["x"], cohort_col="cohort")
    p_het = result["test_interaccion"].loc["x", "p_heterogeneidad"]
    assert p_het > 0.05, f"NO deberia detectar heterogeneidad falsa, p={p_het}"


def test_proportional_hazards_check_runs_and_returns_expected_columns():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "duration": rng.exponential(30, n), "event": rng.integers(0, 2, n),
        "x": rng.integers(0, 2, n),
    })
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    result = check_proportional_hazards(cph, df)
    assert "p" in result.columns
    assert "viola_supuesto" in result.columns
    assert result["viola_supuesto"].dtype == bool


def test_influential_observations_returns_top_n_sorted_by_magnitude():
    rng = np.random.default_rng(4)
    n = 100
    df = pd.DataFrame({
        "duration": rng.exponential(20, n), "event": rng.integers(0, 2, n),
        "x": rng.integers(0, 2, n),
    })
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    result = check_influential_observations(cph, df, top_n=5)
    assert len(result) == 5
    assert result["magnitud_total"].is_monotonic_decreasing


def test_run_full_diagnostics_uses_stratified_model_by_default():
    """
    Regresion de un hallazgo real de revision externa: run_full_diagnostics
    descartaba la columna 'cohort' y ajustaba un Cox pooled SIN
    estratificar, distinto del modelo real (pooled_cox_validation.py
    SI estratifica). Verificar que el modelo interno de cph SI tiene
    'cohort' entre sus estratos cuando stratify=True (default).
    """
    df = _make_two_cohorts(hr_a=2.0, hr_b=2.0, seed=99)
    df = df.rename(columns={"x": "cms_X"})
    from cox_diagnostics import run_full_diagnostics
    resultado = run_full_diagnostics(
        df, "relapse_free_months", "relapse_event", ["cms_X"], cohort_col="cohort")
    assert resultado["cph"].strata == "cohort"


def test_run_full_diagnostics_no_stratify_flag_fits_pooled_model():
    """Con stratify=False (solo para comparacion/debug explicito), el
    modelo NO debe tener estratos."""
    df = _make_two_cohorts(hr_a=2.0, hr_b=2.0, seed=98)
    df = df.rename(columns={"x": "cms_X"})
    from cox_diagnostics import run_full_diagnostics
    resultado = run_full_diagnostics(
        df, "relapse_free_months", "relapse_event", ["cms_X"],
        cohort_col="cohort", stratify=False)
    assert resultado["cph"].strata is None


def test_heterogeneity_interaction_handles_constant_covariate_cohort_gracefully():
    """
    Regresion de un hallazgo real con datos de produccion: GSE33113 es
    una cohorte de estadio II homogeneo (stage_harmonized constante
    dentro de ella) -- el termino de interaccion stage_harmonized x
    dummy_GSE33113 queda perfectamente colineal con el dummy solo,
    causando 'matriz singular' en lifelines. Debe detectarse
    proactivamente con un mensaje claro, no dejar que lifelines truene
    con su error interno.
    """
    df = _make_two_cohorts(hr_a=2.0, hr_b=2.0, seed=7)
    df["stage_harmonized"] = 2  # CONSTANTE en ambas cohortes -- caso extremo
    df.loc[df["cohort"] == "B", "stage_harmonized"] = df.loc[df["cohort"] == "B"].apply(
        lambda r: 2, axis=1)  # A y B ambas constantes=2 (mismo valor, sigue siendo constante por cohorte)

    resultado = check_heterogeneity_across_cohorts(
        df, "relapse_free_months", "relapse_event", ["x", "stage_harmonized"], cohort_col="cohort")
    fila = resultado["test_interaccion"].loc["stage_harmonized"]
    assert "No estimable" in str(fila["error"])
    assert "constante" in str(fila["error"])
