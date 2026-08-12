"""
Tests de regresion para prognosis_demo.py -- verifica que la
integracion attractor_model + calibration + prognosis funcione
end-to-end con patrones calibrados (no solo con los placeholders de
attractor_model.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prognosis_demo import (
    simulate_longitudinal_patient,
    classify_current_state,
    applicable_treatments,
    EVIDENCE_STRENGTH,
)
from calibration import calibrate_patterns_from_data
from attractor_model import build_model_from_patterns
from modern_hopfield import patterns_to_matrix
from synthetic_data import generate_synthetic_cohort
from prognosis import detect_recurrence_signal, hazard_from_trajectory


@pytest.fixture(scope="module")
def real_calibrated_patterns():
    df = generate_synthetic_cohort(n_per_class=50, seed=42)
    patterns, gene_order = calibrate_patterns_from_data(df)
    return patterns, gene_order


def test_baseline_period_stays_at_origin(real_calibrated_patterns):
    """Antes del inicio de recaida, sin forzamiento, el estado debe permanecer en cero."""
    patterns, gene_order = real_calibrated_patterns
    X, labels = patterns_to_matrix(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns["CMS4_mesenchymal"]

    t_checks, x_series = simulate_longitudinal_patient(
        X, gene_order, recurrence_pattern, n_genes,
        n_timepoints=6, months_between_checks=3, recurrence_onset_month=100,  # nunca llega
    )
    assert np.allclose(x_series, 0.0, atol=1e-6)


def test_recurrence_signal_detected_after_onset(real_calibrated_patterns):
    """Tras el inicio de recaida, se debe detectar una alerta en algun punto posterior."""
    patterns, gene_order = real_calibrated_patterns
    X, labels = patterns_to_matrix(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns["CMS4_mesenchymal"]

    t_checks, x_series = simulate_longitudinal_patient(
        X, gene_order, recurrence_pattern, n_genes,
        n_timepoints=8, months_between_checks=3, recurrence_onset_month=15,
    )
    hazard = hazard_from_trajectory(x_series)
    alert, alert_idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)

    assert alert is True
    assert t_checks[alert_idx] >= 15, "la alerta no deberia dispararse antes del inicio de la recaida"


def test_trajectory_converges_toward_recurrence_pattern_direction(real_calibrated_patterns):
    """El estado final debe correlacionar positivamente con el patron de recaida simulado."""
    patterns, gene_order = real_calibrated_patterns
    X, labels = patterns_to_matrix(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns["CMS4_mesenchymal"]

    t_checks, x_series = simulate_longitudinal_patient(
        X, gene_order, recurrence_pattern, n_genes,
        n_timepoints=8, months_between_checks=3, recurrence_onset_month=15,
    )
    x_final = x_series[:, -1]
    corr = np.corrcoef(x_final, recurrence_pattern)[0, 1]
    assert corr > 0.5, f"el estado final deberia acercarse al patron de recaida simulado (corr={corr:.3f})"


def test_different_recurrence_targets_produce_different_trajectories(real_calibrated_patterns):
    """Simular recaida hacia CMS1 vs CMS4 debe producir trayectorias claramente distintas."""
    patterns, gene_order = real_calibrated_patterns
    X, labels = patterns_to_matrix(patterns)
    n_genes = len(gene_order)

    _, x_cms4 = simulate_longitudinal_patient(
        X, gene_order, patterns["CMS4_mesenchymal"], n_genes,
        n_timepoints=8, months_between_checks=3, recurrence_onset_month=15,
    )
    _, x_cms1 = simulate_longitudinal_patient(
        X, gene_order, patterns["CMS1_MSI_immune"], n_genes,
        n_timepoints=8, months_between_checks=3, recurrence_onset_month=15,
    )
    assert not np.allclose(x_cms4[:, -1], x_cms1[:, -1], atol=0.1)


def test_evidence_strength_covers_all_four_cms():
    expected = {"CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"}
    assert set(EVIDENCE_STRENGTH.keys()) == expected


def test_projection_legacy_remains_explicitly_available(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    W, _, _ = build_model_from_patterns(patterns)
    t, x = simulate_longitudinal_patient(
        W, gene_order, patterns["CMS4_mesenchymal"], len(gene_order),
        n_timepoints=6, recurrence_onset_month=15,
        dynamics_model="projection_legacy")
    assert x.shape == (len(gene_order), len(t))


def test_cms4_evidence_is_temporally_qualified():
    evidence = EVIDENCE_STRENGTH["CMS4_mesenchymal"]
    assert evidence["level"] == "moderada/provisional"
    assert "riesgos proporcionales" in evidence["detail"]
    assert EVIDENCE_STRENGTH["CMS1_MSI_immune"]["level"] != "fuerte"
    assert EVIDENCE_STRENGTH["CMS3_metabolic"]["level"] != "fuerte"


def test_classify_current_state_zero_vector_returns_none(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x_zero = np.zeros(len(gene_order))
    label, corr = classify_current_state(x_zero, patterns)
    assert label == "none"
    assert corr == 0.0


def test_classify_current_state_matches_own_pattern(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    for expected_label, pattern in patterns.items():
        label, corr = classify_current_state(pattern, patterns)
        assert label == expected_label
        assert corr > 0.99


def test_applicable_treatments_includes_immunotherapy_near_cms1(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x_cms1 = patterns["CMS1_MSI_immune"] * 0.8
    treatments = applicable_treatments(x_cms1, gene_order, patterns)
    names = [name for name, _ in treatments]
    assert "immunotherapy_antiPD1" in names


def test_applicable_treatments_excludes_immunotherapy_far_from_cms1(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x_cms4 = patterns["CMS4_mesenchymal"] * 0.8
    treatments = applicable_treatments(x_cms4, gene_order, patterns)
    names = [name for name, _ in treatments]
    assert "immunotherapy_antiPD1" not in names


def test_applicable_treatments_sorted_by_efficacy_descending(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x_cms1 = patterns["CMS1_MSI_immune"] * 0.8
    treatments = applicable_treatments(x_cms1, gene_order, patterns)
    efficacies = [eff for _, eff in treatments]
    assert efficacies == sorted(efficacies, reverse=True)
