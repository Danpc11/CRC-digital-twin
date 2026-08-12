"""
Tests de regresion para dynamics_diagnostics.py -- capturan los tres
hallazgos de la investigacion real:

  1. El jacobiano analitico coincide con diferenciacion numerica
     (verificado con error < 1e-6 antes de confiar en el).
  2. El metodo de calibracion (z-score global + centroides por clase)
     SIEMPRE colapsa el rango de los patrones a n_clases-1=3, sin
     importar los datos -- consecuencia matematica inevitable, no un
     accidente de una corrida especifica.
  3. Existe una transicion de estabilidad en beta=1.0 (justificada
     analiticamente: J(0) = -I + beta*W, W proyector con eigenvalores
     en {0,1}, estabilidad en el origen requiere beta<1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamics_diagnostics import (
    jacobian_at_state,
    stability_at_equilibrium,
    sweep_beta_stability,
    vector_field,
)
from calibration import calibrate_patterns_from_data
from attractor_model import build_model_from_patterns
from synthetic_data import generate_synthetic_cohort


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_cohort(n_per_class=60, noise_sigma=1.0, seed=1)


def test_jacobian_matches_finite_differences():
    """El jacobiano analitico debe coincidir con diferenciacion numerica --
    esto es lo que se verifico con sympy antes de codificar la formula."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.3, size=(10, 10))
    x = rng.normal(0, 1, size=10)
    beta = 2.0

    J_analitico = jacobian_at_state(x, W, beta)

    eps = 1e-6
    J_numerico = np.zeros((10, 10))
    for k in range(10):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        J_numerico[:, k] = (vector_field(xp, W, beta) - vector_field(xm, W, beta)) / (2 * eps)

    assert np.max(np.abs(J_analitico - J_numerico)) < 1e-6


def test_calibration_always_collapses_rank_by_one(synthetic_df):
    """
    Hallazgo estructural: con z-score global + centroides por clase,
    el rango de los patrones SIEMPRE es <= n_clases-1, sin importar
    los datos -- consecuencia matematica de que las desviaciones
    respecto a la media global suman cero.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    P = np.array(list(patterns.values())).T
    rank = np.linalg.matrix_rank(P)
    n_clases = len(patterns)
    assert rank <= n_clases - 1, (
        f"rango={rank} deberia ser <= {n_clases-1} -- si esto falla, "
        "el fenomeno de colapso de rango pudo haberse corregido en calibration.py"
    )
    suma = sum(patterns.values())
    assert np.linalg.norm(suma) < 1e-6, "la suma de los centroides deberia ser ~0"


def test_beta_stability_transition_near_one(synthetic_df):
    """
    Con la calibracion REAL (4 clases CMS, via calibrate_patterns_from_data
    -- que colapsa el rango a 3, ver test_calibration_always_collapses_rank_by_one),
    debe existir una transicion clara de estable a inestable cerca de
    beta=1.0. Verificado empiricamente con datos reales del proyecto
    antes de escribir este test -- la transicion de la ESTABILIDAD DEL
    ORIGEN en beta=1 esta probada analiticamente (J(0)=-I+beta*W, W
    proyector con eigenvalor maximo 1), pero la estabilidad de los
    PATRONES especificos puede variar segun la configuracion -- para
    la calibracion real de 4 clases, coincide de cerca con beta=1.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    barrido = sweep_beta_stability(patterns, W, beta_values=[0.3, 0.7, 1.5, 2.0])
    assert barrido[barrido["beta"] == 0.3]["todos_estables"].iloc[0] == True
    assert barrido[barrido["beta"] == 0.7]["todos_estables"].iloc[0] == True
    assert barrido[barrido["beta"] == 1.5]["todos_estables"].iloc[0] == False
    assert barrido[barrido["beta"] == 2.0]["todos_estables"].iloc[0] == False
    assert barrido["peor_parte_real"].is_monotonic_increasing


def test_origin_stability_transition_is_exactly_at_beta_equals_one_over_max_eigenvalue():
    """
    Esto SI esta probado analiticamente para cualquier configuracion:
    en el origen, J(0)=-I+beta*W. Con W proyector ortogonal (nuestro
    caso real), sus eigenvalores estan en {0,1} -- estabilidad en el
    origen requiere beta < 1/max_eigenvalor(W) = 1 (ya que projector
    ortogonal tiene eigenvalor maximo exactamente 1). Verificado en un
    caso simple de 2 patrones ortogonales, donde se puede calcular a mano.
    """
    patterns = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    W, _, _ = build_model_from_patterns(patterns)
    eig_W = np.linalg.eigvals(W)
    beta_critico = 1.0 / np.max(eig_W.real)
    assert abs(beta_critico - 1.0) < 1e-8, "para un proyector ortogonal, beta critico debe ser exactamente 1"

    x0 = np.zeros(2)
    _, estable_bajo, _ = stability_at_equilibrium(x0, W, beta_critico * 0.5)
    _, estable_alto, _ = stability_at_equilibrium(x0, W, beta_critico * 1.5)
    assert estable_bajo == True
    assert estable_alto == False
