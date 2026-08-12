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


def test_forcing_strength_0_7_converges_to_dominant_pattern_not_target(real_patterns):
    """
    Regresion de un hallazgo real: con fuerza_max=0.7 (el valor viejo,
    calibrado para attractor_model.py), forzar hacia un patron NO
    dominante puede terminar convergiendo al patron DOMINANTE en su
    lugar -- la atraccion nativa de esta dinamica (eigenvalor ~-1.0
    para el patron de mayor norma) es mucho mas fuerte que la del
    sistema anterior (~-0.08 cerca de su region critica), y 0.7 no
    basta para vencerla.
    """
    from modern_hopfield import patterns_to_matrix, simulate_longitudinal_patient_hopfield
    X, labels = patterns_to_matrix(real_patterns)
    n_genes = X.shape[0]

    # identificar el patron de MAYOR norma (el "dominante" esperado)
    normas = {l: np.linalg.norm(X[:, i]) for i, l in enumerate(labels)}
    dominante = max(normas, key=normas.get)
    no_dominante = min(normas, key=normas.get)
    if dominante == no_dominante:
        pytest.skip("normas identicas en este dataset sintetico, no aplica el escenario")

    idx_target = labels.index(no_dominante)
    p_target = X[:, idx_target]

    t, x = simulate_longitudinal_patient_hopfield(
        X, p_target, n_genes, beta=3.0, max_forcing_strength=0.7,
        n_timepoints=10, recurrence_onset_month=15)
    x_final = x[:, -1]

    corr_target = np.corrcoef(x_final, p_target)[0, 1] if np.std(x_final) > 1e-12 else np.nan
    # con fuerza insuficiente, no deberia lograr una correlacion alta
    # y limpia con el objetivo -- puede ser bajo, negativo, o ambiguo,
    # pero no un 1.0 perfecto como con fuerza suficiente
    assert corr_target < 0.9 or np.isnan(corr_target)


def test_forcing_strength_1_5_converges_correctly_to_any_target(real_patterns):
    """Con fuerza_max=1.5 (el nuevo default), forzar hacia CUALQUIER
    patron -- incluido el de menor norma -- debe converger limpiamente
    a ese objetivo, no al patron dominante."""
    from modern_hopfield import patterns_to_matrix, simulate_longitudinal_patient_hopfield
    X, labels = patterns_to_matrix(real_patterns)
    n_genes = X.shape[0]

    normas = {l: np.linalg.norm(X[:, i]) for i, l in enumerate(labels)}
    no_dominante = min(normas, key=normas.get)
    idx_target = labels.index(no_dominante)
    p_target = X[:, idx_target]

    t, x = simulate_longitudinal_patient_hopfield(
        X, p_target, n_genes, beta=3.0, max_forcing_strength=1.5,
        n_timepoints=10, recurrence_onset_month=15)
    x_final = x[:, -1]
    corr_target = np.corrcoef(x_final, p_target)[0, 1] if np.std(x_final) > 1e-12 else np.nan
    assert corr_target > 0.95


def test_find_minimum_forcing_strength_finds_correct_threshold(real_patterns):
    """
    Regresion de un hallazgo real: el umbral de fuerza suficiente NO es
    universal (1.5 basto con datos sinteticos de prueba pero fallo con
    la calibracion real de GSE39582, donde CMS2 termino con correlacion
    NEGATIVA respecto al objetivo). find_minimum_forcing_strength debe
    encontrar el umbral real para CUALQUIER calibracion, buscando en
    vez de asumir un numero fijo.
    """
    from modern_hopfield import find_minimum_forcing_strength, patterns_to_matrix
    X, labels = patterns_to_matrix(real_patterns)
    normas = {l: np.linalg.norm(X[:, i]) for i, l in enumerate(labels)}
    no_dominante = min(normas, key=normas.get)

    resultado = find_minimum_forcing_strength(
        real_patterns, no_dominante, beta=3.0,
        strength_candidates=[0.7, 1.5, 3.0, 5.0, 8.0, 12.0])

    assert resultado["umbral_minimo_encontrado"] is not None, (
        "deberia encontrar ALGUN umbral suficiente dentro de los candidatos probados"
    )
    # verificar que el umbral encontrado realmente funciona (corr alta)
    detalle_en_umbral = next(
        d for d in resultado["detalle"] if d["fuerza"] == resultado["umbral_minimo_encontrado"])
    assert detalle_en_umbral["corr_con_objetivo"] >= 0.9

    # y que el candidato ANTERIOR (mas chico) en la lista, si existe, NO
    # bastaba -- confirma que es el minimo, no cualquier valor que funcione
    candidatos_ordenados = sorted(d["fuerza"] for d in resultado["detalle"])
    idx_umbral = candidatos_ordenados.index(resultado["umbral_minimo_encontrado"])
    if idx_umbral > 0:
        fuerza_anterior = candidatos_ordenados[idx_umbral - 1]
        detalle_anterior = next(d for d in resultado["detalle"] if d["fuerza"] == fuerza_anterior)
        assert detalle_anterior["corr_con_objetivo"] < 0.9


def test_origin_is_fixed_point_but_unstable_saddle(real_patterns):
    """
    HALLAZGO CRITICO de produccion (con datos reales de GSE39582): el
    origen SI es un punto fijo exacto (consecuencia de que los
    patrones, z-scoreados contra la media global, suman cero), pero
    es una SILLA inestable (eigenvalores positivos grandes), no un
    reposo estable como en attractor_model.py. Esto rompe la fase
    "sin recaida" de las simulaciones clinicas -- el ruido numerico se
    amplifica exponencialmente y el estado colapsa a la cuenca
    dominante ANTES de que cualquier forzamiento comience.
    """
    from modern_hopfield import patterns_to_matrix, modern_hopfield_field, stability_at_equilibrium_hopfield
    X, labels = patterns_to_matrix(real_patterns)
    n_genes = X.shape[0]
    origen = np.zeros(n_genes)

    campo_en_origen = modern_hopfield_field(origen, X, beta=3.0)
    assert np.linalg.norm(campo_en_origen) < 1e-6, "el origen deberia ser un punto fijo (o casi)"

    eigvals, stable, max_eig = stability_at_equilibrium_hopfield(origen, X, beta=3.0)
    assert stable is False, "el origen deberia ser INESTABLE bajo esta dinamica (silla, no reposo)"
    assert max_eig > 0, "deberia haber al menos un eigenvalor positivo (direccion inestable)"


def test_quiescent_phase_drifts_away_from_origin_before_forcing_begins(real_patterns):
    """
    Regresion directa del hallazgo: simulando SOLO la fase "sin
    recaida" (sin ningun forzamiento), el estado NO debe quedarse en
    el origen -- debe alejarse sustancialmente por la inestabilidad de
    silla, confirmando que la fase pre-recaida esta rota bajo esta
    dinamica (a diferencia de attractor_model.py, donde SI se queda en
    el origen establemente).
    """
    from modern_hopfield import patterns_to_matrix, modern_hopfield_field
    from scipy.integrate import solve_ivp

    X, labels = patterns_to_matrix(real_patterns)
    n_genes = X.shape[0]
    x_current = np.zeros(n_genes)

    for t in np.arange(0, 15, 3):
        sol = solve_ivp(lambda tt, xx: modern_hopfield_field(xx, X, beta=3.0), (0, 3), x_current,
                         method="RK45", rtol=1e-8, atol=1e-10)
        x_current = sol.y[:, -1]

    # para mes 15 (cuando "empezaria" la recaida forzada), el estado
    # NO deberia seguir cerca del origen -- deberia haber colapsado
    # hacia alguna cuenca por la inestabilidad numerica amplificada
    assert np.linalg.norm(x_current) > 1.0, (
        "se esperaba que el estado se alejara sustancialmente del origen "
        "durante la fase 'sin recaida' -- si esto falla, la inestabilidad "
        "de silla pudo haberse corregido en una version posterior"
    )


def test_compute_stabilizing_k_makes_origin_genuinely_stable(real_patterns):
    """
    Regresion del hallazgo critico: con k=0 (sin estabilizador), el
    origen es una silla inestable (eigenvalores positivos). Con
    k=compute_stabilizing_k(), debe pasar a ser un minimo local
    genuino (TODOS los eigenvalores negativos).
    """
    from modern_hopfield import (patterns_to_matrix, compute_stabilizing_k,
                                   modern_hopfield_jacobian_stabilized)
    X, labels = patterns_to_matrix(real_patterns)
    beta = 3.0
    origen = np.zeros(X.shape[0])

    k = compute_stabilizing_k(X, beta)
    assert k > 0, "deberia ser positivo si el origen sin estabilizar es inestable"

    J_estabilizado = modern_hopfield_jacobian_stabilized(origen, X, beta, k)
    eigvals = np.linalg.eigvalsh(J_estabilizado)
    assert np.all(eigvals < 0), "el origen debe ser estable CON el estabilizador aplicado"


def test_stabilized_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.5, size=(10, 4))
    x = rng.normal(0, 1, size=10)
    beta, k = 2.0, 3.0

    from modern_hopfield import modern_hopfield_field_stabilized, modern_hopfield_jacobian_stabilized
    J_analitico = modern_hopfield_jacobian_stabilized(x, X, beta, k)
    eps = 1e-6
    J_numerico = np.zeros((10, 10))
    for i in range(10):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        J_numerico[:, i] = (modern_hopfield_field_stabilized(xp, X, beta, k) -
                              modern_hopfield_field_stabilized(xm, X, beta, k)) / (2 * eps)
    assert np.max(np.abs(J_analitico - J_numerico)) < 1e-6


def test_quiescent_phase_stays_at_origin_with_stabilizer(real_patterns):
    """
    Regresion directa del hallazgo critico: CON el estabilizador
    activo, la fase 'sin recaida' debe mantener el estado en el
    origen (a diferencia de sin estabilizador, donde colapsaba a la
    cuenca dominante por inestabilidad numerica).
    """
    from modern_hopfield import (patterns_to_matrix, compute_stabilizing_k,
                                   modern_hopfield_field_stabilized)
    from scipy.integrate import solve_ivp

    X, labels = patterns_to_matrix(real_patterns)
    beta = 3.0
    k = compute_stabilizing_k(X, beta)
    x_current = np.zeros(X.shape[0])

    for t in np.arange(0, 15, 3):
        sol = solve_ivp(lambda tt, xx: modern_hopfield_field_stabilized(xx, X, beta, k),
                         (0, 3), x_current, method="RK45", rtol=1e-8, atol=1e-10)
        x_current = sol.y[:, -1]

    assert np.linalg.norm(x_current) < 1e-3, (
        "con el estabilizador activo, el estado deberia quedarse en el origen "
        "durante toda la fase pre-recaida"
    )


def test_v2_simulation_reaches_all_targets_with_synthetic_data(real_patterns):
    """
    Con datos sinteticos, la version v2 (estabilizador en fase
    pre-recaida + forzamiento normal en fase de recaida) debe converger
    correctamente a CUALQUIER patron objetivo -- CUIDADO: esto NO
    garantiza que resuelva el caso real de GSE39582 (CMS1-CMS2 con
    r=-0.825), donde las reproducciones sinteticas han dado resultados
    distintos al comportamiento real varias veces en esta investigacion.
    Verificar con datos reales antes de confiar en esto como solucion
    definitiva.
    """
    from modern_hopfield import patterns_to_matrix, simulate_longitudinal_patient_hopfield_v2
    X, labels = patterns_to_matrix(real_patterns)
    n_genes = X.shape[0]

    for i, label in enumerate(labels):
        p = X[:, i]
        t, x = simulate_longitudinal_patient_hopfield_v2(
            X, p, n_genes, beta=3.0, max_forcing_strength=1.5,
            n_timepoints=10, recurrence_onset_month=15)
        x_final = x[:, -1]
        corr = np.corrcoef(x_final, p)[0, 1] if np.std(x_final) > 1e-12 else np.nan
        assert corr > 0.9, f"{label} no convergio correctamente (corr={corr:.3f})"
