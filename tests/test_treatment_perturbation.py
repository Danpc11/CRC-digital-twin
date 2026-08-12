"""
Tests de regresion para treatment_perturbation.py y
treatment_simulation_demo.py -- usando patrones calibrados con datos
sinteticos (no placeholders), mismo patron que test_prognosis_demo.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import calibrate_patterns_from_data
from synthetic_data import generate_synthetic_cohort
from treatment_perturbation import (
    TREATMENT_MECHANISMS,
    apply_treatment_perturbation,
    describe_treatment,
)
from treatment_simulation_demo import simulate_with_optional_treatment
from attractor_model import build_model_from_patterns


@pytest.fixture(scope="module")
def real_calibrated_patterns():
    df = generate_synthetic_cohort(n_per_class=50, seed=123)
    patterns, gene_order = calibrate_patterns_from_data(df)
    return patterns, gene_order


def test_all_mechanisms_have_required_fields():
    for name, spec in TREATMENT_MECHANISMS.items():
        assert "target_genes" in spec
        assert "criterio" in spec
        assert "evidence" in spec
        assert len(spec["target_genes"]) > 0


def test_describe_treatment_raises_on_unknown():
    with pytest.raises(ValueError):
        describe_treatment("tratamiento_inventado")


def test_apply_treatment_raises_on_unknown_treatment(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x = patterns["CMS1_MSI_immune"] * 0.5
    with pytest.raises(ValueError):
        apply_treatment_perturbation(x, gene_order, "no_existe", patterns)


def test_immunotherapy_zero_effect_far_from_cms1(real_calibrated_patterns):
    """Sin beneficio (jalon ~0) en un paciente que va hacia CMS4, no CMS1."""
    patterns, gene_order = real_calibrated_patterns
    x_cms4 = patterns["CMS4_mesenchymal"] * 0.8
    I = apply_treatment_perturbation(x_cms4, gene_order, "immunotherapy_antiPD1", patterns)
    assert np.linalg.norm(I) < 0.05


def test_immunotherapy_nonzero_effect_near_cms1(real_calibrated_patterns):
    """Efecto real (jalon hacia origen) en un paciente cerca de CMS1."""
    patterns, gene_order = real_calibrated_patterns
    x_cms1 = patterns["CMS1_MSI_immune"] * 0.8
    I = apply_treatment_perturbation(x_cms1, gene_order, "immunotherapy_antiPD1", patterns)
    assert np.linalg.norm(I) > 0.1
    # el jalon debe apuntar hacia el origen (direccion opuesta a x)
    assert np.dot(I, x_cms1) < 0


def test_anti_egfr_zero_when_mutant(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x = patterns["CMS3_metabolic"] * 0.8
    I = apply_treatment_perturbation(x, gene_order, "anti_egfr", patterns, ras_braf_wildtype=False)
    assert np.allclose(I, 0.0)


def test_anti_egfr_wildtype_stronger_than_unknown_when_near_cms3(real_calibrated_patterns):
    """Con estatus real wild-type, la eficacia debe ser >= que con estatus desconocido
    (el proxy penaliza por incertidumbre cuando el paciente esta cerca de CMS3)."""
    patterns, gene_order = real_calibrated_patterns
    x_cms3 = patterns["CMS3_metabolic"] * 0.8
    I_known = apply_treatment_perturbation(x_cms3, gene_order, "anti_egfr", patterns, ras_braf_wildtype=True)
    I_unknown = apply_treatment_perturbation(x_cms3, gene_order, "anti_egfr", patterns, ras_braf_wildtype=None)
    assert np.linalg.norm(I_known) >= np.linalg.norm(I_unknown)


def test_chemo_less_effective_in_cms4_than_cms2(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    x_cms4 = patterns["CMS4_mesenchymal"] * 0.8
    x_cms2 = patterns["CMS2_canonical_WNT"] * 0.8
    I_cms4 = apply_treatment_perturbation(x_cms4, gene_order, "cytotoxic_chemo", patterns)
    I_cms2 = apply_treatment_perturbation(x_cms2, gene_order, "cytotoxic_chemo", patterns)
    assert np.linalg.norm(I_cms4) < np.linalg.norm(I_cms2)


def test_treatment_raises_if_panel_missing_mechanism_genes(real_calibrated_patterns):
    patterns, gene_order = real_calibrated_patterns
    reduced_gene_order = [g for g in gene_order if g not in ("GNLY", "USP18")]
    x = patterns["CMS1_MSI_immune"] * 0.5
    with pytest.raises(ValueError):
        apply_treatment_perturbation(x, reduced_gene_order, "immunotherapy_antiPD1", patterns)


def test_counterfactual_treated_trajectory_has_lower_final_hazard(real_calibrated_patterns):
    """La demostracion central: un paciente recayendo hacia CMS1, tratado con
    inmunoterapia, debe terminar con hazard mas bajo que sin tratamiento."""
    from prognosis import hazard_from_trajectory

    patterns, gene_order = real_calibrated_patterns
    W, labels, _ = build_model_from_patterns(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns["CMS1_MSI_immune"]

    _, x_baseline = simulate_with_optional_treatment(
        W, n_genes, gene_order, recurrence_pattern, patterns,
        treatment=None, n_timepoints=8, recurrence_onset_month=15,
    )
    _, x_treated = simulate_with_optional_treatment(
        W, n_genes, gene_order, recurrence_pattern, patterns,
        treatment="immunotherapy_antiPD1", treatment_onset_month=18,
        n_timepoints=8, recurrence_onset_month=15,
    )

    hazard_baseline_final = hazard_from_trajectory(x_baseline)[-1]
    hazard_treated_final = hazard_from_trajectory(x_treated)[-1]
    assert hazard_treated_final < hazard_baseline_final
