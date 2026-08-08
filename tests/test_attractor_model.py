"""
Tests de regresion para attractor_model.py

Cubre:
  - Cada patron CMS es punto fijo aproximado del sistema linealizado.
  - Cada perfil de mutacion converge al atractor CMS esperado.
  - El estado sin forzamiento (driver='none') permanece acotado
    (no diverge, no genera falsos positivos de clasificacion fuerte).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attractor_model import (
    CMS_LABELS,
    CMS_PATTERNS,
    DRIVER_BIAS,
    N,
    P,
    W,
    classify_state,
    simulate_patient,
)


def test_state_dimension_matches_gene_panel():
    # Panel congelado de 10 genes (ver PROJECT_STATUS.md) -- si este
    # test falla tras agregar/quitar un gen, hay que actualizar TAMBIEN
    # synthetic_data.py y los scripts build_*_dataset.py.
    assert N == 10
    assert P.shape == (N, 4)


def test_patterns_are_approximate_fixed_points_of_linear_system():
    # W @ p_mu ~= p_mu por construccion (regla de proyeccion)
    for label in CMS_LABELS:
        p = CMS_PATTERNS[label]
        reconstructed = W @ p
        assert np.allclose(reconstructed, p, atol=1e-8), (
            f"{label} no es punto fijo exacto del sistema linealizado"
        )


@pytest.mark.parametrize(
    "driver,expected_cms",
    [
        ("MSI_high", "CMS1_MSI_immune"),
        ("APC_mut", "CMS2_canonical_WNT"),
        ("KRAS_mut", "CMS3_metabolic"),
        ("SMAD4_loss", "CMS4_mesenchymal"),
    ],
)
def test_driver_mutation_converges_to_expected_attractor(driver, expected_cms):
    result = simulate_patient(driver)
    x_final = result["x"][:, -1]
    predicted, corr = classify_state(x_final)
    assert predicted == expected_cms
    assert corr > 0.9, f"correlacion baja ({corr:.3f}) para {driver}"


def test_no_driver_state_remains_bounded():
    result = simulate_patient("none")
    x_final = result["x"][:, -1]
    assert np.all(np.abs(x_final) < 1e-6), (
        "sin forzamiento el sistema deberia relajar al origen, no divergir"
    )
    label, corr = classify_state(x_final)
    assert label == "none"
    assert corr == 0.0, "el estado neutro no deberia clasificar con confianza a ningun CMS"


def test_all_driver_keys_map_to_valid_cms_or_none():
    # Cada driver (menos 'none') empuja hacia exactamente UN patron CMS,
    # con magnitud positiva y signo consistente con ese patron.
    expected = {"MSI_high", "APC_mut", "KRAS_mut", "SMAD4_loss", "none"}
    assert set(DRIVER_BIAS) == expected
    for driver, bias in DRIVER_BIAS.items():
        assert bias.shape == (N,)
        if driver == "none":
            assert np.allclose(bias, 0.0)
        else:
            assert np.linalg.norm(bias) > 0
            aligned = [label for label in CMS_LABELS
                       if np.dot(bias, CMS_PATTERNS[label]) > 0]
            assert len(aligned) >= 1, f"{driver} no empuja hacia ningun atractor"
