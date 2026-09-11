"""
Tests de cox_clinical_adjustment.py -- cohorte sintetica con un
confusor clinico binario (tipo dMMR) deliberadamente solapado con CMS1
y un efecto real de CMS4 independiente de el. Se verifica que:
  - el indicador binario se construye bien (incluidos faltantes),
  - los modelos anidados corren sobre la misma muestra,
  - CMS4 conserva HR>1 tras ajustar por el confusor (efecto inyectado),
  - el modelo de subgrupo corre y se omite si el subgrupo es muy chico.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cox_clinical_adjustment import (
    binary_indicator,
    crosstab_group_by_covariate,
    fit_nested_clinical_models,
    fit_subgroup_model,
    indicator_name,
    parse_level_spec,
    prepare_single_frame,
)

LABELS = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]


def _synthetic_cohort(n=500, seed=7) -> pd.DataFrame:
    """CMS1 es dMMR en ~80%; dMMR protege (HR 0.5); CMS4 tiene HR 2.2 propio;
    el estadio sube el riesgo. Algunos faltantes en msi_status."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        label = rng.choice(LABELS, p=[0.2, 0.4, 0.15, 0.25])
        p_dmmr = 0.8 if label == "CMS1_MSI_immune" else 0.05
        dmmr = rng.random() < p_dmmr
        stage = int(rng.integers(1, 5))  # 1-4; el IV se excluye luego
        hazard = (2.2 if label == "CMS4_mesenchymal" else 1.0) * (0.5 if dmmr else 1.0) \
            * (1.6 ** (stage - 1)) / 60
        t_event = rng.exponential(1 / hazard)
        t_cens = rng.exponential(70)
        msi = "dMMR" if dmmr else "pMMR"
        if rng.random() < 0.05:
            msi = np.nan
        rows.append({
            "sample_id": f"S{_}", "predicted_cms": label, "stage": stage,
            "msi_status": msi,
            "relapse_free_months": round(min(t_event, t_cens), 2),
            "relapse_event": int(t_event <= t_cens),
        })
    return pd.DataFrame(rows)


def test_parse_level_spec_and_indicator():
    assert parse_level_spec("msi_status=dMMR") == ("msi_status", "dMMR")
    with pytest.raises(ValueError):
        parse_level_spec("msi_status")
    s = pd.Series(["dMMR", " dmmr ", "pMMR", None, "NA", "pmmr"])
    ind = binary_indicator(s, "dMMR")
    assert ind.tolist()[:3] == [1.0, 1.0, 0.0]
    assert np.isnan(ind.iloc[3]) and np.isnan(ind.iloc[4])
    assert ind.iloc[5] == 0.0
    assert indicator_name("msi_status", "dMMR") == "msi_status_dMMR"


def test_prepare_single_frame_drops_stage_iv_and_missing():
    df = _synthetic_cohort()
    data = prepare_single_frame(df, [("msi_status", "dMMR")], verbose=False)
    assert "stage_harmonized" in data.columns and "cohort" in data.columns
    assert (data["stage_harmonized"] != 4).all()
    assert data["msi_status_dMMR"].notna().all()
    assert len(data) < len(df)
    ct = crosstab_group_by_covariate(df, "predicted_cms", "msi_status")
    # CMS1 concentra los dMMR -- el solapamiento que motiva el analisis
    assert ct.loc["CMS1_MSI_immune", "dMMR"] > ct.loc["CMS4_mesenchymal", "dMMR"]


def test_nested_models_keep_cms4_effect_after_clinical_adjustment():
    df = _synthetic_cohort()
    data = prepare_single_frame(df, [("msi_status", "dMMR")], verbose=False)
    res = fit_nested_clinical_models(data, ["msi_status_dMMR"], "CMS2_canonical_WNT")
    summary = res["summary"]
    assert set(summary["modelo"]) == {"A_cms", "B_cms_estadio", "C_estadio_clinica",
                                      "D_cms_estadio_clinica"}
    # misma muestra en los 4 modelos
    assert summary["n"].nunique() == 1 and summary["eventos"].nunique() == 1
    hr_cms4_full = summary.loc[(summary["modelo"] == "D_cms_estadio_clinica"),
                               "HR"].loc["cms_CMS4_mesenchymal"]
    assert hr_cms4_full > 1.5, f"CMS4 deberia conservar su efecto inyectado, HR={hr_cms4_full:.2f}"
    # el confusor tiene el signo correcto (protector)
    hr_dmmr = summary.loc[(summary["modelo"] == "C_estadio_clinica"), "HR"].loc["msi_status_dMMR"]
    assert hr_dmmr < 1.0
    inc = res["increment"]
    assert 0.0 <= inc["p_incremental_cms_sobre_clinica"] <= 1.0
    assert inc["df"] == 3
    assert inc["p_incremental_cms_sobre_clinica"] < 0.05  # CMS4 aporta de verdad


def test_subgroup_model_runs_and_skips_when_too_small():
    df = _synthetic_cohort()
    data = prepare_single_frame(df, [("msi_status", "dMMR")], verbose=False)
    sub = fit_subgroup_model(data, "msi_status", "pMMR", "CMS2_canonical_WNT")
    assert sub is not None
    cph, table = sub
    assert table["modelo"].iloc[0].startswith("E_cms_estadio_solo_msi_status=pMMR")
    assert table.loc["cms_CMS4_mesenchymal", "HR"] > 1.0
    # subgrupo dMMR: pocos eventos -> debe omitirse sin tronar
    tiny = fit_subgroup_model(data.head(40), "msi_status", "dMMR", "CMS2_canonical_WNT")
    assert tiny is None
