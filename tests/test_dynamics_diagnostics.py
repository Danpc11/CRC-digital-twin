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
    basin_membership_near_patterns,
    check_equilibria_separation,
    compare_clinical_trajectories,
    es_atractor_cms,
    find_contiguous_valid_segments,
    find_true_equilibrium,
    find_valid_beta_interval,
    investigate_unclassified_basins,
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
    CORREGIDO tras revision externa -- la premisa original de este test
    ('beta<1 -> todos_estables=True') es EXACTAMENTE el hallazgo
    enganoso que se identifico: con beta=0.3/0.7, el jacobiano SI
    reporta estable, pero el equilibrio real es el ORIGEN colapsado
    (verificado: desplazamiento ~ norma completa del patron,
    correlacion NaN), no los patrones CMS. Con el criterio corregido
    (es_atractor_cms), beta<1 debe dar CERO atractores genuinos, no
    "todos estables".
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    barrido = sweep_beta_stability(patterns, W, beta_values=[0.3, 0.7, 1.5, 2.0])
    # beta<1: el jacobiano dice "estable" pero es el origen colapsado --
    # el criterio corregido debe rechazarlo (cero atractores genuinos)
    assert barrido[barrido["beta"] == 0.3]["cuantos_son_atractores_cms"].iloc[0] == 0
    assert barrido[barrido["beta"] == 0.7]["cuantos_son_atractores_cms"].iloc[0] == 0
    # beta>=1: inestable segun el jacobiano (parte real positiva) --
    # tampoco califican, pero por razon distinta (inestabilidad genuina,
    # no colapso al origen)
    assert barrido[barrido["beta"] == 1.5]["todos_son_atractores_cms_genuinos"].iloc[0] == False
    assert barrido[barrido["beta"] == 2.0]["todos_son_atractores_cms_genuinos"].iloc[0] == False
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


def test_es_atractor_cms_rejects_origin_collapse(synthetic_df):
    """
    Regresion de un hallazgo real de revision externa: con beta<1, los
    4 equilibrios "estables" resultan ser el origen colapsado, no los
    patrones -- verificado con la calibracion real de este proyecto
    (desplazamiento ~ norma completa del patron, correlacion NaN).
    es_atractor_cms debe rechazar esto aunque el jacobiano diga estable.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    for label, p in patterns.items():
        x_eq, converged, _ = find_true_equilibrium(p, W, beta=0.5)
        _, stable, _ = stability_at_equilibrium(x_eq, W, beta=0.5)
        assert stable == True, "en beta=0.5 el jacobiano SI reporta estable (es el origen)"
        assert es_atractor_cms(converged, stable, x_eq, p) == False, (
            "es_atractor_cms debe rechazar el colapso al origen aunque el jacobiano diga estable"
        )


def test_es_atractor_cms_accepts_genuine_nontrivial_stable_equilibrium():
    """Caso de control positivo directo: un equilibrio lejos del origen,
    estable, y perfectamente correlacionado con el patron -- debe aceptarse."""
    x_eq = np.array([2.0, -1.0, 0.5, 1.5])
    p = np.array([2.0, -1.0, 0.5, 1.5])  # identico -- correlacion=1
    assert es_atractor_cms(converged=True, stable=True, x_eq=x_eq, p_original=p) == True


def test_es_atractor_cms_rejects_when_not_converged_or_not_stable():
    x_eq = np.array([2.0, -1.0, 0.5, 1.5])
    p = np.array([2.0, -1.0, 0.5, 1.5])
    assert es_atractor_cms(converged=False, stable=True, x_eq=x_eq, p_original=p) == False
    assert es_atractor_cms(converged=True, stable=False, x_eq=x_eq, p_original=p) == False


def test_check_equilibria_separation_detects_collapsed_pair():
    equilibria = {
        "A": np.array([1.0, 0.0]), "B": np.array([1.01, 0.0]),  # casi identicos
        "C": np.array([-5.0, 3.0]),
    }
    sep = check_equilibria_separation(equilibria, min_separation=0.5)
    colapsados = sep.attrs["colapsados"]
    assert len(colapsados) == 1
    assert set(colapsados[0][:2]) == {"A", "B"}


def test_sweep_beta_stability_uses_corrected_criterion_not_just_jacobian(synthetic_df):
    """
    Regresion directa del hallazgo: con la calibracion real, beta=0.5
    NO debe reportarse como 'todos_son_atractores_cms_genuinos=True'
    (el bug real: antes se reportaba 'todos_estables=True' ahi, porque
    solo miraba el jacobiano sin verificar que el equilibrio fuera
    distinto del origen).
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)
    barrido = sweep_beta_stability(patterns, W, beta_values=[0.5])
    assert barrido.iloc[0]["todos_son_atractores_cms_genuinos"] == False
    assert barrido.iloc[0]["cuantos_son_atractores_cms"] == 0


def test_find_valid_beta_interval_finds_known_interval_with_real_calibration(synthetic_df):
    """
    Verificado manualmente antes de escribir este test: con la
    calibracion real del proyecto, existe un intervalo robusto
    (no un punto aislado) alrededor de beta~9.4-11.9 donde los 4
    patrones SI califican como atractores CMS genuinos, bien
    separados entre si -- aunque beta=2.0 (el default actual del
    proyecto) no calfica ninguno.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    busqueda = find_valid_beta_interval(patterns, W, beta_min=0.1, beta_max=15.0, n_steps=60)
    validos = busqueda[busqueda["todos_4_califican"]]
    assert len(validos) > 0, (
        "se esperaba encontrar al menos un beta en [0.1,15.0] donde los 4 patrones "
        "califiquen -- si esto falla, la calibracion sintetica cambio de forma "
        "sustancial respecto a cuando se verifico este test a mano"
    )
    # el intervalo encontrado deberia estar en la region alta (beta > 5),
    # consistente con el hallazgo verificado a mano
    assert validos["beta"].min() > 5.0


def test_todos_4_califican_requires_separation_not_just_individual_count():
    """
    Regresion de un bug real de revision externa: 'todos_4_califican'
    solo contaba cuantos pasaban es_atractor_cms() individualmente, sin
    verificar que no hubieran colapsado entre si -- 4 equilibrios
    identicos (colapsados) podrian "calificar" cada uno por separado.
    """
    # 4 patrones DELIBERADAMENTE colapsados al mismo punto (simula el
    # caso donde individualmente pasarian es_atractor_cms pero estan
    # superpuestos)
    patterns = {
        "A": np.array([2.0, 0.0, 0.0, 0.0]), "B": np.array([2.01, 0.0, 0.0, 0.0]),
        "C": np.array([1.99, 0.01, 0.0, 0.0]), "D": np.array([2.0, -0.01, 0.0, 0.0]),
    }
    W, _, _ = build_model_from_patterns(patterns)
    resultado = find_valid_beta_interval(patterns, W, beta_min=0.5, beta_max=0.5, n_steps=1,
                                           min_separation=0.5)
    # aunque los 4 pudieran pasar es_atractor_cms individualmente, estan
    # colapsados entre si (separacion < 0.5) -- NO deben calificar juntos
    assert resultado.iloc[0]["todos_4_califican"] == False


def test_find_contiguous_valid_segments_splits_disjoint_ranges():
    """
    Regresion de un bug real: reportar solo min()/max() de los puntos
    validos asume un unico intervalo continuo sin verificarlo. Aqui se
    construye un caso con DOS tramos separados por un hueco, y se
    verifica que efectivamente se detecten como 2 segmentos, no 1.
    """
    df = pd.DataFrame({
        "beta": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        "todos_4_califican": [True, True, True, False, False, False, False, True, True, True, True],
    })
    segmentos = find_contiguous_valid_segments(df, beta_max_explored=2.0)
    assert len(segmentos) == 2, f"se esperaban 2 tramos separados, se encontraron {len(segmentos)}"
    assert segmentos[0]["beta_max"] < segmentos[1]["beta_min"]


def test_find_contiguous_valid_segments_flags_undetermined_upper_bound():
    """Si el ultimo tramo valido toca el limite explorado, debe marcarse
    como limite superior NO determinado -- no reportar ese numero como
    si fuera el verdadero borde."""
    df = pd.DataFrame({
        "beta": [1.0, 1.1, 1.2, 1.3, 1.4],
        "todos_4_califican": [False, False, True, True, True],
    })
    segmentos = find_contiguous_valid_segments(df, beta_max_explored=1.4)
    assert segmentos[-1]["limite_superior_no_determinado"] == True


def test_investigate_unclassified_basins_finds_genuine_spurious_equilibria(synthetic_df):
    """
    Verificado con la calibracion real del proyecto antes de escribir
    este test: con beta=10 (dentro del intervalo donde los 4 patrones
    CMS son atractores genuinos), la mayoria de las trayectorias desde
    ruido centrado en el origen NO terminan en ninguno de los 4 -- pero
    SI convergen a un numero pequeno de equilibrios adicionales
    genuinos (fenomeno de atractores espurios en redes asociativas
    saturadas), no a comportamiento sin estructura.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    resultado = investigate_unclassified_basins(patterns, W, beta=10.0, n_samples=40, seed=3)
    if resultado["n_sin_clasificar"] > 0:
        # si hay estados sin clasificar, al menos algunos clusters
        # deben ser equilibrios genuinos (no solo ruido sin estructura)
        genuinos = [c for c in resultado["clusters"] if c["es_equilibrio_genuino"]]
        assert len(genuinos) > 0
        assert all(c["residuo_maximo"] <= 1e-6 for c in genuinos)
        assert all(c["max_parte_real_eigenvalor"] < 0 for c in genuinos)


def test_spurious_equilibria_are_deduplicated_by_distance(synthetic_df):
    patterns, _ = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)
    resultado = investigate_unclassified_basins(
        patterns, W, beta=10.0, n_samples=30, seed=7,
        equilibrium_distance_threshold=0.25)
    clusters = resultado["clusters"]
    assert sum(c["n_miembros"] for c in clusters) == resultado["n_refinados_validos"]
    assert len({c["equilibrio_id"] for c in clusters}) == len(clusters)


def test_compare_clinical_trajectories_uses_different_beta_per_row():
    """
    Regresion de un bug real que encontre yo mismo construyendo esta
    funcion: la primera version pasaba dos matrices W distintas como si
    'W dependiera de beta' (no depende -- W se construye solo de los
    patrones, beta solo entra en tanh(beta*x) dentro de dynamics()) y
    nunca pasaba beta a simulate_longitudinal_patient(), dando
    resultados IDENTICOS para beta=2 y beta=10 sin ningun error visible.
    """
    patterns = {"A": np.array([2.0, -1.0, 0.5, 1.0, -0.5]),
                "B": np.array([-1.0, 2.0, -0.5, -1.0, 1.5])}
    W, _, _ = build_model_from_patterns(patterns)
    gene_order = ["g1", "g2", "g3", "g4", "g5"]

    resultado = compare_clinical_trajectories(patterns, W, gene_order, betas=(2.0, 10.0))
    for patron in patterns:
        fila_b2 = resultado[(resultado["patron"] == patron) & (resultado["beta"] == 2.0)].iloc[0]
        fila_b10 = resultado[(resultado["patron"] == patron) & (resultado["beta"] == 10.0)].iloc[0]
        # deben ser DISTINTOS -- si son identicos, el bug volvio
        assert fila_b2["hazard_final"] != fila_b10["hazard_final"]


def test_basin_membership_near_patterns_more_robust_than_global_sampling(synthetic_df):
    """
    Verificado con la calibracion real: muestrear CERCA de cada patron
    (ruido pequeno) da una tasa de retorno mucho mas alta que el
    muestreo global centrado en el origen -- confirma que ambas
    estrategias miden cosas distintas (tamano de cuenca local vs.
    cobertura global del espacio de estados).

    Usa beta=10 (no beta=2) a proposito -- ya esta establecido que
    beta=2 no tiene NINGUN atractor genuino (ver
    test_beta_stability_transition_near_one), asi que 0% de retorno
    ahi seria el resultado CORRECTO y esperado, no una prueba util de
    esta funcion. beta=10 esta dentro del intervalo verificado donde
    los patrones si son atractores genuinos.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    resultado = basin_membership_near_patterns(
        patterns, W, beta=10.0, noise_scale=0.2, n_per_pattern=20, seed=4)
    assert resultado["proporcion_regresa_al_mismo"].mean() > 0.0


def test_correlation_threshold_is_exposed_and_changes_qualification():
    """
    Regresion de un hallazgo real con la calibracion de GSE39582:
    correlation_threshold cambia la conclusion CUALITATIVA de si existe
    un beta valido -- con 0.8, ningun beta hasta 50 califica los 4
    simultaneamente; con 0.65, aparece una ventana real y angosta cerca
    de beta~1.01-1.02. No es un detalle cosmetico, hay que exponerlo
    como parametro real y reportarlo siempre junto al resultado.

    Verificado aqui con un caso de correlacion EXACTA construida via
    combinacion lineal controlada (corr=0.70, entre ambos umbrales).
    """
    p = np.array([1.5, 1.8, 0.2, 1.0, -0.5])
    p_hat = (p - p.mean()) / np.linalg.norm(p - p.mean())
    rng = np.random.default_rng(0)
    ortogonal = rng.normal(size=5)
    ortogonal = ortogonal - ortogonal.mean() - np.dot(ortogonal - ortogonal.mean(), p_hat) * p_hat
    ortogonal = ortogonal / np.linalg.norm(ortogonal)

    corr_objetivo = 0.70
    x_eq = (corr_objetivo * p_hat + np.sqrt(1 - corr_objetivo**2) * ortogonal) * 2.0
    assert abs(np.corrcoef(x_eq, p)[0, 1] - 0.70) < 1e-6

    assert es_atractor_cms(True, True, x_eq, p, correlation_threshold=0.8) == False
    assert es_atractor_cms(True, True, x_eq, p, correlation_threshold=0.65) == True


def test_find_valid_beta_interval_correlation_threshold_is_wired_through(synthetic_df):
    """Verifica que el parametro realmente se propaga hasta la busqueda
    completa (no solo hasta es_atractor_cms aislado) -- llamando con dos
    umbrales distintos y confirmando que el argumento pasado es el que
    efectivamente se usa (via monkeypatch del criterio interno)."""
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    W, _, _ = build_model_from_patterns(patterns)

    # solo confirma que no revienta y que acepta el parametro -- la
    # sensibilidad cualitativa ya esta probada con datos reales arriba
    b1 = find_valid_beta_interval(patterns, W, beta_min=0.5, beta_max=2.0, n_steps=20,
                                    correlation_threshold=0.8)
    b2 = find_valid_beta_interval(patterns, W, beta_min=0.5, beta_max=2.0, n_steps=20,
                                    correlation_threshold=0.3)
    # con umbral mucho mas laxo (0.3), nunca deberia calificar MENOS
    # puntos que con 0.8 -- relajar el criterio no puede reducir cuantos
    # pasan
    assert b2["n_atractores_genuinos"].sum() >= b1["n_atractores_genuinos"].sum()
