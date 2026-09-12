"""Tests de cms_subgroup_split.py con una cohorte sintetica: CMS1 partido por
una covariable binaria donde un subgrupo tiene riesgo inyectado alto y el
otro bajo. Se verifica la construccion de grupos, que el Cox recupere la
direccion, la comparacion dentro del subtipo y el manejo de subgrupos chicos."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cms_subgroup_split import (
    fit_group_cox,
    prepare_frame,
    split_group_labels,
    within_split_comparison,
)

LABELS = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]


def _cohort(n=600, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = rng.choice(LABELS, p=[0.3, 0.4, 0.15, 0.15])
        msi = rng.choice(["dMMR", "pMMR"], p=[0.6, 0.4]) if label == "CMS1_MSI_immune" else "pMMR"
        if rng.random() < 0.03:
            msi = np.nan
        stage = int(rng.integers(1, 5))
        hr = 1.0
        if label == "CMS1_MSI_immune":
            hr = 0.5 if msi == "dMMR" else 2.5   # el subgrupo pMMR es el malo
        hazard = hr * (1.5 ** (stage - 1)) / 60
        t = rng.exponential(1 / hazard)
        c = rng.exponential(80)
        rows.append({"sample_id": f"S{i}", "predicted_cms": label, "msi_status": msi,
                     "braf_status": rng.choice(["WT", "M"], p=[0.85, 0.15]), "stage": stage,
                     "relapse_free_months": round(min(t, c), 2), "relapse_event": int(t <= c)})
    return pd.DataFrame(rows)


def test_split_group_labels_only_splits_target_and_handles_missing():
    df = pd.DataFrame({"predicted_cms": ["CMS1_MSI_immune"] * 3 + ["CMS2_canonical_WNT"],
                       "msi_status": ["dMMR", "pMMR", np.nan, "pMMR"]})
    g = split_group_labels(df, "predicted_cms", "CMS1_MSI_immune", "msi_status")
    assert g.tolist()[:2] == ["CMS1_MSI_immune|dMMR", "CMS1_MSI_immune|pMMR"]
    assert pd.isna(g.iloc[2])
    assert g.iloc[3] == "CMS2_canonical_WNT"  # el resto no se toca


def test_cox_recovers_opposite_risks_within_split_cms():
    data = prepare_frame(_cohort(), "predicted_cms", "CMS1_MSI_immune", "msi_status",
                         "relapse_free_months", "relapse_event", verbose=False)
    assert (data["stage_harmonized"] != 4).all()
    cph, summary = fit_group_cox(data, "CMS2_canonical_WNT")
    hr_d = summary.loc["grp_CMS1_MSI_immune|dMMR", "HR"]
    hr_p = summary.loc["grp_CMS1_MSI_immune|pMMR", "HR"]
    assert hr_d < 1.0 < hr_p, f"dMMR={hr_d:.2f}, pMMR={hr_p:.2f}"
    assert summary.loc["grp_CMS1_MSI_immune|dMMR", "n_grupo"] == (data["group"] == "CMS1_MSI_immune|dMMR").sum()
    within = within_split_comparison(data, "CMS1_MSI_immune", "msi_status")
    assert within["cox_HR_B_vs_A"] > 1.0  # B = pMMR (orden alfabetico) vs A = dMMR
    assert within["logrank_p"] < 0.05
    assert within["rfs60_CMS1_MSI_immune|pMMR"] < within["rfs60_CMS1_MSI_immune|dMMR"]


def test_within_comparison_reports_note_when_subgroup_too_small():
    data = prepare_frame(_cohort(), "predicted_cms", "CMS1_MSI_immune", "msi_status",
                         "relapse_free_months", "relapse_event", verbose=False)
    tiny = pd.concat([data[data["group"] == "CMS1_MSI_immune|pMMR"].head(4),
                      data[data["group"] != "CMS1_MSI_immune|pMMR"]])
    within = within_split_comparison(tiny, "CMS1_MSI_immune", "msi_status")
    assert "note" in within and "cox_p" not in within


def test_extra_indicator_is_added_to_model():
    data = prepare_frame(_cohort(), "predicted_cms", "CMS1_MSI_immune", "msi_status",
                         "relapse_free_months", "relapse_event", verbose=False)
    data["braf_status_M"] = (data["braf_status"] == "M").astype(float)
    _, summary = fit_group_cox(data, "CMS2_canonical_WNT", extra_indicator="braf_status_M")
    assert "braf_status_M" in summary.index


def test_prepare_frame_rejects_unknown_split_cms_and_drops_unclassified():
    import pytest
    df = _cohort()
    with pytest.raises(ValueError, match="no aparece"):
        prepare_frame(df, "predicted_cms", "CMS1", "msi_status",
                      "relapse_free_months", "relapse_event", verbose=False)
    df2 = df.copy()
    df2.loc[df2.index[:5], "predicted_cms"] = "indeterminado"
    data = prepare_frame(df2, "predicted_cms", "CMS1_MSI_immune", "msi_status",
                         "relapse_free_months", "relapse_event", verbose=False)
    assert "indeterminado" not in set(data["group"])


def test_group_counts_match_fitted_sample_with_extra_indicator():
    data = prepare_frame(_cohort(), "predicted_cms", "CMS1_MSI_immune", "msi_status",
                         "relapse_free_months", "relapse_event", verbose=False)
    data["braf_status_M"] = (data["braf_status"] == "M").astype(float)
    data.loc[data.index[:40], "braf_status_M"] = np.nan  # 40 pacientes sin BRAF
    _, summary = fit_group_cox(data, "CMS2_canonical_WNT", extra_indicator="braf_status_M")
    fitted = data.dropna(subset=["braf_status_M"])
    assert summary.loc["stage_harmonized", "n_grupo"] == len(fitted)
    for lv in fitted["group"].unique():
        if lv == "CMS2_canonical_WNT":
            continue
        assert summary.loc[f"grp_{lv}", "n_grupo"] == (fitted["group"] == lv).sum()
