"""Regresiones de la revision del 2026-09-15."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_external_cohort_generic import ensure_log2_scale
from clinical_covariates import expand_stage_categorical
from cox_clinical_adjustment import sparse_level_warnings
from cox_treatment_interaction import (cell_counts, chemo_hr_within_cms,
                                       interaction_test, prepare_interaction_frame)
from pooled_cox_validation import (DEFAULT_CMS_REFERENCE, build_cox_frame,
                                   nested_model_increment, stratified_c_index)
from prognosis import hazard_from_trajectory, state_norm_from_trajectory

LEVELS = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]


def _synthetic_survival(n=400, seed=0, two_cohorts=True):
    rng = np.random.default_rng(seed)
    cms = rng.choice(LEVELS, n)
    stage = rng.choice([1, 2, 3], n, p=[0.2, 0.45, 0.35])
    cohort = rng.choice(["A", "B"], n) if two_cohorts else np.array(["A"] * n)
    # riesgo: estadio III mucho peor que II, CMS4 peor; cohorte B con base mas alta
    lam = 0.01 * np.exp(0.9 * (stage == 3) - 0.3 * (stage == 1) + 0.7 * (cms == "CMS4_mesenchymal")
                        + 0.5 * (cohort == "B"))
    t = rng.exponential(1.0 / lam)
    c = rng.uniform(20, 120, n)
    dur = np.minimum(t, c)
    ev = (t <= c).astype(int)
    chemo = np.where(stage == 3, rng.choice(["Y", "N"], n, p=[0.8, 0.2]),
                     rng.choice(["Y", "N"], n, p=[0.3, 0.7]))
    return pd.DataFrame({"predicted_cms": cms, "stage": stage.astype(str), "cohort": cohort,
                         "relapse_free_months": dur, "relapse_event": ev, "adjuvant_chemo": chemo})


def test_expand_stage_categorical_reference_ii_and_nan_propagation():
    df = pd.DataFrame({"stage_harmonized": [1, 2, 3, np.nan, 3]})
    out, cols = expand_stage_categorical(df)
    assert cols == ["stage_I", "stage_III"]
    assert out["stage_I"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert out["stage_III"].tolist()[:3] == [0.0, 0.0, 1.0]
    assert np.isnan(out.loc[3, "stage_I"]) and np.isnan(out.loc[3, "stage_III"])


def test_build_cox_frame_uses_stage_dummies_not_linear():
    df = _synthetic_survival()
    df["stage_harmonized"] = df["stage"].astype(int)
    frame = build_cox_frame(df, "relapse_free_months", "relapse_event", DEFAULT_CMS_REFERENCE,
                            ["stage_harmonized"])
    assert "stage_harmonized" not in frame.columns
    assert {"stage_I", "stage_III"} <= set(frame.columns)
    linear = build_cox_frame(df, "relapse_free_months", "relapse_event", DEFAULT_CMS_REFERENCE,
                             ["stage_harmonized"], stage_categorical=False)
    assert "stage_harmonized" in linear.columns


def test_categorical_stage_recovers_nonlinear_effect():
    """Con un salto II->III mucho mayor que I->II, el modelo lineal fuerza un
    coeficiente unico; el categorico deja HR_III >> 1/HR_I."""
    df = _synthetic_survival(n=1500, seed=3)
    df["stage_harmonized"] = df["stage"].astype(int)
    frame = build_cox_frame(df, "relapse_free_months", "relapse_event", DEFAULT_CMS_REFERENCE,
                            ["stage_harmonized"])
    cph = CoxPHFitter().fit(frame, "duration", "event", strata=["cohort"])
    hr_iii = cph.summary.loc["stage_III", "exp(coef)"]
    hr_i = cph.summary.loc["stage_I", "exp(coef)"]
    assert hr_iii > 1.8 and hr_i < 1.0
    assert abs(np.log(hr_iii)) > 1.5 * abs(np.log(hr_i))


def test_stratified_c_index_equals_lifelines_with_single_stratum():
    df = _synthetic_survival(two_cohorts=False)
    df["stage_harmonized"] = df["stage"].astype(int)
    frame = build_cox_frame(df, "relapse_free_months", "relapse_event", DEFAULT_CMS_REFERENCE,
                            ["stage_harmonized"])
    cph = CoxPHFitter().fit(frame.drop(columns=["cohort"]), "duration", "event")
    assert abs(stratified_c_index(cph, frame) - cph.concordance_index_) < 1e-9


def test_stratified_c_index_ignores_between_cohort_baseline_difference():
    """Un predictor que solo codifica la cohorte (B tiene riesgo basal 4x)
    parece 'discriminar' en el C agrupado; en el estratificado vale ~0.5
    porque dentro de cada cohorte no ordena a nadie."""
    from lifelines.utils import concordance_index
    from pooled_cox_validation import stratified_c_index_from_lp
    rng = np.random.default_rng(5)
    n = 600
    cohort = np.repeat(["A", "B"], n // 2)
    lam = 0.01 * np.where(cohort == "B", 4.0, 1.0)
    t = rng.exponential(1.0 / lam)
    frame = pd.DataFrame({"duration": t, "event": 1.0, "cohort": cohort})
    lp = pd.Series((cohort == "B").astype(float) + 1e-3 * rng.normal(size=n), index=frame.index)
    pooled = concordance_index(frame["duration"], -lp, frame["event"])
    strat = stratified_c_index_from_lp(lp, frame)
    assert pooled > 0.65
    assert 0.45 < strat < 0.55


def test_nested_increment_reports_stratified_when_frames_given():
    df = _synthetic_survival()
    df["stage_harmonized"] = df["stage"].astype(int)
    f0 = build_cox_frame(df, "relapse_free_months", "relapse_event", DEFAULT_CMS_REFERENCE,
                         ["stage_harmonized"], include_cms=False)
    f1 = build_cox_frame(df, "relapse_free_months", "relapse_event", DEFAULT_CMS_REFERENCE,
                         ["stage_harmonized"], include_cms=True)
    c0 = CoxPHFitter().fit(f0, "duration", "event", strata=["cohort"])
    c1 = CoxPHFitter().fit(f1, "duration", "event", strata=["cohort"])
    inc = nested_model_increment(c0, c1, f0, f1)
    assert "delta_c_index_stratified" in inc
    assert np.isfinite(inc["c_index_stratified_stage_plus_cms"])


def test_ensure_log2_scale_transforms_linear_and_keeps_log2():
    log2m = pd.DataFrame(np.random.default_rng(0).uniform(3, 14, (50, 8)))
    out, tag = ensure_log2_scale(log2m)
    assert tag == "log2" and np.allclose(out.to_numpy(), log2m.to_numpy())
    linear = 2 ** log2m
    out2, tag2 = ensure_log2_scale(linear)
    assert tag2 == "lineal->log2"
    assert np.allclose(out2.to_numpy(), np.log2(linear.to_numpy() + 1.0))
    out3, tag3 = ensure_log2_scale(linear, strict=False)
    assert tag3 == "lineal_sin_transformar" and np.allclose(out3.to_numpy(), linear.to_numpy())


def test_sparse_level_warning_fires_for_small_cells():
    df = pd.DataFrame({"predicted_cms": ["CMS1_MSI_immune"] * 4 + ["CMS2_canonical_WNT"] * 40,
                       "relapse_event": [1, 0, 0, 0] + [1] * 15 + [0] * 25})
    msgs = sparse_level_warnings(df, "predicted_cms", "relapse_event", "E")
    assert len(msgs) == 1 and "CMS1_MSI_immune" in msgs[0]


def test_state_norm_alias_and_semantics():
    x = np.array([[3.0, 0.0], [4.0, 0.0]])
    assert np.allclose(state_norm_from_trajectory(x), [5.0, 0.0])
    assert np.allclose(hazard_from_trajectory(x), state_norm_from_trajectory(x))


def test_interaction_module_runs_and_detects_planted_interaction():
    rng = np.random.default_rng(11)
    df = _synthetic_survival(n=1200, seed=11)
    # plantar beneficio de quimio solo en CMS1: acortar/alargar tiempos
    mask = (df["predicted_cms"] == "CMS1_MSI_immune") & (df["adjuvant_chemo"] == "Y")
    df.loc[mask, "relapse_free_months"] *= 3.0
    prep = prepare_interaction_frame(df, "adjuvant_chemo", "Y", verbose=False)
    assert set(prep["stage_harmonized"].unique()) <= {2, 3}
    cells = cell_counts(prep, "predicted_cms", "relapse_event")
    assert len(cells) == 8
    res = interaction_test(prep)
    assert res["df"] == 3 and 0 <= res["p_interaction"] <= 1
    assert res["p_interaction"] < 0.05
    within = chemo_hr_within_cms(prep)
    hr_cms1 = within.set_index("cms").loc["CMS1_MSI_immune", "HR_quimio"]
    hr_cms2 = within.set_index("cms").loc["CMS2_canonical_WNT", "HR_quimio"]
    assert hr_cms1 < hr_cms2


def test_convert_duration_units_days_to_months():
    """GSE33113 anota el tiempo a recurrencia en DIAS; sin convertir entraba
    al Cox como 'meses' (mediana 1179)."""
    from build_external_cohort_generic import check_duration_units, convert_duration_units
    dias = pd.Series([1179.5, 3599.0, 365.0])
    meses = convert_duration_units(dias, "days")
    assert abs(meses.iloc[0] - 38.75) < 0.1
    assert abs(meses.iloc[2] - 12.0) < 0.1
    assert check_duration_units(meses, strict=True) == "meses"
    with pytest.raises(ValueError):
        check_duration_units(dias, strict=True)
    assert convert_duration_units(pd.Series([3.0]), "years").iloc[0] == 36.0
    with pytest.raises(ValueError):
        convert_duration_units(dias, "weeks")


def test_align_to_model_fills_absent_stage_levels():
    """GSE33113 es todo estadio II y GSE37892 no tiene estadio I: el frame de
    prueba no genera esas dummies y el LOCO fallaba con KeyError."""
    from pooled_cox_validation import align_to_model
    df = _synthetic_survival(n=300, seed=4)
    df["stage_harmonized"] = df["stage"].astype(int)
    train = build_cox_frame(df, "relapse_free_months", "relapse_event",
                            DEFAULT_CMS_REFERENCE, ["stage_harmonized"])
    model = CoxPHFitter().fit(train, "duration", "event", strata=["cohort"])
    homogenea = df[df["stage"] == "2"]           # sin estadio I ni III
    test = build_cox_frame(homogenea, "relapse_free_months", "relapse_event",
                           DEFAULT_CMS_REFERENCE, ["stage_harmonized"])
    assert "stage_III" not in test.columns
    alineado = align_to_model(test, model)
    lp = alineado[model.params_.index] @ model.params_   # no debe lanzar
    assert len(lp) == len(test)
    assert (alineado["stage_III"] == 0).all()


def test_check_event_coding_detects_inverted_column():
    """GSE14333: 'DFS_Cens' con 1=censurado leido como evento -> 79% de
    'eventos' y seguimiento mas largo en los supuestos eventos."""
    from build_external_cohort_generic import check_event_coding
    rng = np.random.default_rng(2)
    n = 200
    censurado = rng.random(n) < 0.75
    dur = np.where(censurado, rng.uniform(60, 120, n), rng.uniform(5, 40, n))
    invertido = pd.Series(censurado.astype(float))      # 1 = censurado (mal)
    correcto = pd.Series((~censurado).astype(float))
    with pytest.raises(ValueError, match="INVERTIDA"):
        check_event_coding(invertido, pd.Series(dur), strict=True)
    res = check_event_coding(invertido, pd.Series(dur), strict=False)
    assert res["sospechoso"] is True
    assert check_event_coding(correcto, pd.Series(dur), strict=True)["sospechoso"] is False


def test_event_map_applies_to_numeric_column_before_validation():
    """Regresion: check_event_coding corria ANTES de --event-map, asi que
    abortaba sobre la columna cruda y el mapa 0=1,1=0 nunca se aplicaba.
    Ademas las claves del mapa son texto y la columna de GSE14333 es
    numerica, asi que el .map() no casaba con nada."""
    from build_external_cohort_generic import _looks_numeric, check_event_coding
    assert _looks_numeric("0") and _looks_numeric("1") and not _looks_numeric("yes")

    # columna numerica con claves de texto: el mapeo debe funcionar igual
    cens = pd.Series([1.0, 1.0, 0.0, 1.0, 0.0])
    event_map = {float(k): v for k, v in {"0": 1, "1": 0}.items()}
    recodificado = cens.map(event_map)
    assert recodificado.tolist() == [0, 0, 1, 0, 1]

    # y tras recodificar, el validador debe aceptar la columna
    rng = np.random.default_rng(9)
    n = 200
    censurado = rng.random(n) < 0.75
    dur = np.where(censurado, rng.uniform(60, 120, n), rng.uniform(5, 40, n))
    invertida = pd.Series(censurado.astype(float))
    corregida = invertida.map({0.0: 1, 1.0: 0})
    assert check_event_coding(corregida, pd.Series(dur), strict=True)["sospechoso"] is False


def test_collapse_sparse_stage_levels_merges_level_without_events():
    """Con el evento bien codificado casi nadie recae en estadio I: stage_I
    determina la ausencia de evento (separacion completa) y lifelines avisa
    ConvergenceWarning con norm(delta) alto en cada ajuste."""
    from clinical_covariates import collapse_sparse_stage_levels, expand_stage_categorical
    df = pd.DataFrame({
        "stage_harmonized": [1] * 40 + [2] * 60 + [3] * 60,
        "relapse_event": [0] * 39 + [1] + [0] * 40 + [1] * 20 + [0] * 35 + [1] * 25,
    })
    work, cols = expand_stage_categorical(df)
    assert cols == ["stage_I", "stage_III"]
    keep = collapse_sparse_stage_levels(work, cols, "relapse_event")
    assert keep == ["stage_III"]          # stage_I tiene 1 solo evento
    # con eventos suficientes en ambos niveles, no se descarta nada
    df2 = df.copy()
    df2.loc[:9, "relapse_event"] = 1
    work2, cols2 = expand_stage_categorical(df2)
    assert collapse_sparse_stage_levels(work2, cols2, "relapse_event") == cols2


def test_build_cox_frame_drops_separated_stage_level():
    df = _synthetic_survival(n=400, seed=6)
    df["stage_harmonized"] = df["stage"].astype(int)
    df.loc[df["stage"] == "1", "relapse_event"] = 0     # estadio I sin eventos
    frame = build_cox_frame(df, "relapse_free_months", "relapse_event",
                            DEFAULT_CMS_REFERENCE, ["stage_harmonized"])
    assert "stage_I" not in frame.columns
    assert "stage_III" in frame.columns
    cph = CoxPHFitter().fit(frame, "duration", "event", strata=["cohort"])
    assert np.isfinite(cph.params_).all()
