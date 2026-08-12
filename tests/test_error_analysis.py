"""Regresiones para evitar resúmenes de cohorte codificados manualmente."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from error_analysis import cohen_kappa_from_labels, load, summary


def test_summary_calculates_cohort_kappa_and_most_frequent_error(capsys):
    df = pd.DataFrame({
        "cms_label": ["CMS1_MSI_immune", "CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS4_mesenchymal"],
        "predicted_cms": ["CMS1_MSI_immune", "CMS4_mesenchymal", "CMS3_metabolic", "CMS4_mesenchymal"],
        "correct": [True, False, False, True],
        "error_type": ["correct", "CMS1→CMS4", "CMS2→CMS3", "correct"],
    })
    best = pd.Series({"threshold": 0.4, "accuracy": 0.75, "coverage": 0.5})
    result = summary(df, best, cohort_name="COHORTE_X")
    output = capsys.readouterr().out

    assert result["cohort"] == "COHORTE_X"
    assert result["n_errors"] == 2
    assert 0 <= result["cohen_kappa"] <= 1
    assert "COHORTE_X" in output
    assert "GSE17536" not in output
    assert "Empate entre tipos de error" in output


def test_cohen_kappa_is_one_for_perfect_agreement():
    labels = pd.Series(["CMS1", "CMS2", "CMS3", "CMS4"])
    assert cohen_kappa_from_labels(labels, labels) == 1.0


def test_load_can_analyze_modern_hopfield_and_excludes_abstentions(tmp_path):
    path = tmp_path / "scored.tsv"
    pd.DataFrame({
        "cms_label": ["CMS1_MSI_immune", "CMS4_mesenchymal"],
        "predicted_cms": ["CMS1_MSI_immune", "CMS4_mesenchymal"],
        "classification_confidence": [0.7, 0.8],
        "modern_hopfield_cms": ["CMS1_MSI_immune", "indeterminado"],
        "modern_hopfield_correlation": [0.95, 0.4],
        "modern_hopfield_input_margin": [0.25, 0.05],
    }).to_csv(path, sep="\t", index=False)
    result = load(str(path), prediction_col="modern_hopfield_cms")
    assert len(result) == 1
    assert result.iloc[0]["predicted_cms"] == "CMS1_MSI_immune"
    assert result.iloc[0]["classification_confidence"] == 0.25
