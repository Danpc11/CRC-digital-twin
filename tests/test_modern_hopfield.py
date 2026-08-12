"""
Tests para modern_hopfield.py -- verifican las propiedades que
justifican este rediseno frente a attractor_model.py: jacobiano
correcto y SIMETRICO (flujo de gradiente genuino), energia decreciente,
y los 4 patrones como atractores genuinos con cuencas robustas.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modern_hopfield import (
    empirical_basin_membership_hopfield,
    find_true_equilibrium_hopfield,
    hopfield_energy,
    modern_hopfield_field,
    modern_hopfield_jacobian,
    patterns_to_matrix,
    stability_at_equilibrium_hopfield,
    sweep_beta_hopfield,
    verify_all_patterns_hopfield,
    verify_energy_decreases,
)
from calibration import calibrate_patterns_from_data
from synthetic_data import generate_synthetic_cohort


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_cohort(n_per_class=60, noise_sigma=1.0, seed=1)


@pytest.fixture(scope="module")
def real_patterns(synthetic_df):
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    return patterns


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    n_genes, n_patterns = 10, 4
    X = rng.normal(0, 1.5, size=(n_genes, n_patterns))
    x = rng.normal(0, 1, size=n_genes)
    beta = 2.0

    J_analitico = modern_hopfield_jacobian(x, X, beta)
    eps = 1e-6
    J_numerico = np.zeros((n_genes, n_genes))
    for k in range(n_genes):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        J_numerico[:, k] = (modern_hopfield_field(xp, X, beta) - modern_hopfield_field(xm, X, beta)) / (2 * eps)

    assert np.max(np.abs(J_analitico - J_numerico)) < 1e-6


def test_jacobian_is_symmetric():
    """Propiedad central que distingue esta construccion de la anterior
    (attractor_model.py, cuyo jacobiano se probo NO simetrico con
    sympy) -- simetria implica que es un flujo de gradiente genuino."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1.5, size=(10, 4))
    x = rng.normal(0, 1, size=10)
    J = modern_hopfield_jacobian(x, X, beta=2.0)
    assert np.allclose(J, J.T)


def test_energy_decreases_monotonically_along_trajectories():
    """Propiedad garantizada matematicamente para cualquier flujo de
    gradiente -- si esto fallara, seria un bug de implementacion, no
    una sorpresa cientifica."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1.5, size=(10, 4))
    resultado = verify_energy_decreases(X, beta=2.0, n_trajectories=15)
    assert resultado["todas_decrecientes"] is True


def test_stored_patterns_are_near_exact_fixed_points_at_beta_2(real_patterns):
    """Con beta=2.0 (el default del proyecto), los patrones calibrados
    deben ser casi exactamente los equilibrios (desplazamiento minimo),
    a diferencia de attractor_model.py donde el desplazamiento es
    sustancial (~0.5-1.5) con el mismo beta."""
    tabla = verify_all_patterns_hopfield(real_patterns, beta=2.0)
    assert tabla["localmente_estable"].all()
    assert (tabla["desplazamiento_vs_patron_calibrado"] < 0.1).all()
    assert (tabla["correlacion_equilibrio_vs_patron"] > 0.99).all()


def test_all_four_patterns_qualify_across_wide_beta_range(real_patterns):
    """A diferencia de la dinamica anterior (donde ningun beta hasta 50
    califico los 4 con correlacion>=0.8 en la calibracion real), aqui
    debe existir un rango AMPLIO (no una ventana angosta de una
    centesima) donde los 4 califican simultaneamente."""
    barrido = sweep_beta_hopfield(real_patterns, beta_values=[0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    # todos los beta desde 0.7 en adelante en este barrido deben calificar
    assert barrido["todos_4_ok"].sum() >= 5


def test_basin_coverage_is_near_complete_at_beta_2(real_patterns):
    """A diferencia de la dinamica anterior (72-82% 'ninguno_claro' con
    datos reales), aqui la cobertura de cuencas debe ser casi
    completa."""
    cuencas = empirical_basin_membership_hopfield(real_patterns, beta=2.0, n_samples=150, seed=5)
    proporcion_sin_clasificar = cuencas["proporcion"].get("ninguno_claro", 0.0)
    proporcion_origen = cuencas["proporcion"].get("origen", 0.0)
    assert (proporcion_sin_clasificar + proporcion_origen) < 0.1


def test_find_true_equilibrium_hopfield_converges_from_pattern_itself():
    X = np.array([[2.0, -1.0], [1.0, 2.0], [-1.0, 0.5]])  # 3 genes, 2 patrones
    p = X[:, 0]
    x_eq, converged, desp = find_true_equilibrium_hopfield(p, X, beta=2.0)
    assert converged
    assert desp < 0.1


def test_patterns_to_matrix_preserves_order():
    patterns = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0]), "C": np.array([5.0, 6.0])}
    X, labels = patterns_to_matrix(patterns)
    assert labels == ["A", "B", "C"]
    assert np.array_equal(X[:, 0], patterns["A"])
    assert np.array_equal(X[:, 2], patterns["C"])
