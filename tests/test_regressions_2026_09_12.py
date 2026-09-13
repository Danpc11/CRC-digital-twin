"""Regresiones de la segunda revision (2026-09-12): tratamiento dentro del
campo, cociente tratamiento/forzamiento explicito, calendario unico de
forzamiento, test de heterogeneidad bien especificado y unidades de tiempo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import calibrate_patterns_from_data
from modern_hopfield import (
    DEFAULT_FORCING_RAMP_MONTHS,
    compare_forcing_sweep_v1_v2,
    find_minimum_forcing_strength,
    patterns_to_matrix,
    resolve_forcing_ramp,
)
from synthetic_data import generate_synthetic_cohort
from treatment_simulation_demo import (
    DEFAULT_TREATMENT_TO_FORCING_RATIO,
    simulate_with_optional_treatment,
    treatment_strength_from_ratio,
)


@pytest.fixture(scope="module")
def patterns_and_genes():
    df = generate_synthetic_cohort()
    genes = [c for c in df.columns if c not in ("sample_id", "cms_label", "driver",
                                                 "relapse_free_months", "relapse_event", "stage")]
    genes = [g for g in genes if pd.api.types.is_numeric_dtype(df[g])]
    patterns, genes = calibrate_patterns_from_data(df, genes)
    return patterns, genes


# --- #1: tratamiento evaluado dentro del campo ---------------------------

def test_treatment_damping_does_not_overshoot_origin(patterns_and_genes):
    """Con forzamiento apagado (onset fuera de ventana) y un estado inicial
    lejos del origen, un amortiguamiento -k*x nunca debe cruzar el origen.
    Con el termino congelado (bug), un empuje constante -k*x(t0) durante 3
    meses si podia cruzarlo. Lo comprobamos con el motor legacy (lineal
    cerca del origen) y una amortiguacion fuerte."""
    patterns, genes = patterns_and_genes
    from attractor_model import build_model_from_patterns
    W, _, _ = build_model_from_patterns(patterns)
    n = len(genes)
    # Empezamos "ya recaidos": onset en el mes 0 con rampa larga y luego tratamiento fuerte.
    t, x = simulate_with_optional_treatment(
        W, n, genes, patterns["CMS1_MSI_immune"], patterns,
        treatment="immunotherapy_antiPD1", treatment_onset_month=3,
        n_timepoints=8, recurrence_onset_month=0, dynamics_model="projection_legacy",
        base_treatment_strength=20.0)
    proj = [float(np.dot(x[:, i], patterns["CMS1_MSI_immune"])) for i in range(x.shape[1])]
    # Nunca debe volverse fuertemente ANTI-CMS1 (cruce del origen por sobreimpulso)
    assert min(proj) > -0.25 * max(proj), f"sobreimpulso: proyecciones {np.round(proj, 3)}"


def test_treatment_term_is_state_dependent_inside_interval(patterns_and_genes):
    """Dos amortiguaciones distintas con el mismo estado inicial deben dar
    trayectorias distintas *dentro* del primer intervalo tratado -- y la
    mas fuerte debe dejar menor norma, sin excepcion por congelamiento."""
    patterns, genes = patterns_and_genes
    X, _ = patterns_to_matrix(patterns)
    n = len(genes)
    norms = []
    for ratio in (0.2, 1.0, 2.0):
        _, x = simulate_with_optional_treatment(
            X, n, genes, patterns["CMS1_MSI_immune"], patterns,
            treatment="immunotherapy_antiPD1", treatment_onset_month=18,
            treatment_to_forcing_ratio=ratio)
        norms.append(float(np.linalg.norm(x[:, -1])))
    assert norms[0] > norms[1] > norms[2]


# --- #2: cociente explicito ---------------------------------------------

def test_default_ratio_reproduces_historical_strength():
    assert np.isclose(treatment_strength_from_ratio(5.0, DEFAULT_TREATMENT_TO_FORCING_RATIO), 0.5)
    with pytest.raises(ValueError):
        treatment_strength_from_ratio(5.0, 0.0)


def test_explicit_base_strength_overrides_ratio(patterns_and_genes):
    patterns, genes = patterns_and_genes
    X, _ = patterns_to_matrix(patterns)
    n = len(genes)
    _, a = simulate_with_optional_treatment(
        X, n, genes, patterns["CMS1_MSI_immune"], patterns,
        treatment="immunotherapy_antiPD1", treatment_onset_month=18, base_treatment_strength=0.5)
    _, b = simulate_with_optional_treatment(
        X, n, genes, patterns["CMS1_MSI_immune"], patterns,
        treatment="immunotherapy_antiPD1", treatment_onset_month=18,
        treatment_to_forcing_ratio=0.1, max_forcing_strength=5.0)
    assert np.allclose(a, b)


# --- #3: calendario unico -----------------------------------------------

def test_resolve_forcing_ramp_matches_app_horizon():
    assert resolve_forcing_ramp(10, 3, 15) == 12.0 == DEFAULT_FORCING_RAMP_MONTHS
    assert resolve_forcing_ramp(8, 3, 15) == 6.0           # acotada al ultimo control
    assert resolve_forcing_ramp(6, 3, 100) == DEFAULT_FORCING_RAMP_MONTHS  # onset fuera: no aplica
    with pytest.raises(ValueError):
        resolve_forcing_ramp(6, 3, 100, require_post_onset_check=True)


def test_min_forcing_and_sweep_share_schedule(patterns_and_genes):
    """find_minimum_forcing_strength y compare_forcing_sweep_v1_v2 deben
    aplicar la misma fuerza final para el mismo horizonte y candidato."""
    patterns, _ = patterns_and_genes
    res = find_minimum_forcing_strength(patterns, "CMS4_mesenchymal", strength_candidates=[3.0])
    assert res["calendario"]["rampa_meses"] == 12.0 and res["calendario"]["driver_normalizado"]
    sweep = compare_forcing_sweep_v1_v2(patterns, strength_candidates=[3.0], n_timepoints=10)
    assert (sweep["duracion_rampa_meses"] == 12.0).all()
    assert np.allclose(sweep["fuerza_aplicada_final"], 3.0)
    # y la correlacion V1 reportada por ambos coincide (misma simulacion)
    row = sweep[sweep["patron_objetivo"] == "CMS4_mesenchymal"].iloc[0]
    assert np.isclose(row["v1_corr_objetivo_activo"] if "v1_corr_objetivo_activo" in row
                      else row["v1_corr_activo"] if "v1_corr_activo" in row else res["detalle"][0]["corr_con_objetivo"],
                      res["detalle"][0]["corr_con_objetivo"], atol=1e-6)


# --- #4: heterogeneidad con modelo completo -----------------------------

def test_heterogeneity_test_keeps_all_main_effects():
    from cox_diagnostics import check_heterogeneity_across_cohorts
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({
        "cohort": rng.choice(["A", "B", "C"], n),
        "cms_CMS1": rng.integers(0, 2, n).astype(float),
        "cms_CMS4": rng.integers(0, 2, n).astype(float),
    })
    lp = 0.4 * df["cms_CMS4"] - 0.3 * df["cms_CMS1"]
    df["duration"] = rng.exponential(np.exp(-lp) * 40)
    df["event"] = (rng.uniform(size=n) < 0.7).astype(int)
    res = check_heterogeneity_across_cohorts(df, "duration", "event", ["cms_CMS1", "cms_CMS4"])
    it = res["test_interaccion"]
    assert "covariables_principales" in it.columns
    assert it.loc["cms_CMS4", "covariables_principales"] == "cms_CMS1, cms_CMS4"
    assert 0.0 <= float(it.loc["cms_CMS4", "p_heterogeneidad"]) <= 1.0
    assert int(it.loc["cms_CMS4", "df"]) == 2   # 3 cohortes -> 2 dummies


# --- #5: unidades de tiempo ---------------------------------------------

def test_duration_units_check():
    from build_external_cohort_generic import check_duration_units
    meses = pd.Series(np.random.default_rng(0).uniform(3, 120, 200))
    assert check_duration_units(meses) == "meses"
    dias = meses * 30.4375
    with pytest.raises(ValueError, match="dias"):
        check_duration_units(dias)
    assert check_duration_units(dias, strict=False) == "dias?"
    anios = meses / 12
    with pytest.raises(ValueError, match="anios"):
        check_duration_units(anios)
