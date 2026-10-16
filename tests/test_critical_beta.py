"""Regresión del umbral crítico de beta (2026-10-12).

A beta=2.0 —el valor que usaban por defecto los subcomandos de dinámica—
el centroide de CMS2 no es un atractor del campo: su equilibrio no
converge, el autovalor máximo es ~0 y su cuenca empírica sale vacía. No
es un defecto de CMS2 sino de la elección de beta: el umbral de
existencia depende de la norma del centroide y se mueve cada vez que
cambia el panel.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_hopfield import (DEFAULT_BETA, empirical_basin_membership_hopfield,
                             find_critical_beta, patterns_to_matrix,
                             report_beta_validity, validate_modern_hopfield_beta)

PATTERNS = ROOT / "results_gse39582" / "calibrated_patterns.tsv"
pytestmark = pytest.mark.skipif(
    not PATTERNS.exists(), reason="requiere calibración vigente")


@pytest.fixture(scope="module")
def patterns():
    from calibration import load_calibrated_patterns
    p, _ = load_calibrated_patterns(PATTERNS)
    return p


def test_default_beta_sostiene_los_cuatro_atractores(patterns):
    v = validate_modern_hopfield_beta(patterns, DEFAULT_BETA)
    assert v["valid"], f"a beta={DEFAULT_BETA} fallan: {v['invalid_patterns']}"


def test_existe_umbral_critico_y_el_default_queda_por_encima(patterns):
    crit = find_critical_beta(patterns)
    bc = crit["beta_critico"]
    assert np.isfinite(bc)
    assert DEFAULT_BETA > bc
    assert (DEFAULT_BETA - bc) / bc > 0.15     # margen mínimo del 15 %


def test_por_debajo_del_umbral_falla_y_por_encima_no(patterns):
    bc = find_critical_beta(patterns)["beta_critico"]
    assert not validate_modern_hopfield_beta(patterns, bc * 0.95)["valid"]
    assert validate_modern_hopfield_beta(patterns, bc * 1.10)["valid"]


def test_el_subtipo_de_menor_norma_es_el_primero_en_perderse(patterns):
    """El softmax satura según beta por la brecha de logit, que escala con
    la norma; el centroide de menor norma es el que deja de ser atractor."""
    bc = find_critical_beta(patterns)["beta_critico"]
    invalidos = validate_modern_hopfield_beta(patterns, bc * 0.95)["invalid_patterns"]
    X, labels = patterns_to_matrix(patterns)
    normas = {l: float(np.linalg.norm(X[:, i])) for i, l in enumerate(labels)}
    menor_norma = min(normas, key=normas.get)
    assert menor_norma in invalidos


def test_sin_atractor_la_cuenca_empirica_queda_vacia(patterns):
    bc = find_critical_beta(patterns)["beta_critico"]
    invalidos = validate_modern_hopfield_beta(patterns, bc * 0.95)["invalid_patterns"]
    cuencas = empirical_basin_membership_hopfield(
        patterns, beta=bc * 0.95, n_samples=120, seed=0)
    for etiqueta in invalidos:
        assert etiqueta not in cuencas.index
    # al beta por defecto, en cambio, los cuatro reciben trayectorias
    cuencas_ok = empirical_basin_membership_hopfield(
        patterns, beta=DEFAULT_BETA, n_samples=200, seed=0)
    assert len(cuencas_ok) == 4


def test_report_avisa_cuando_el_beta_es_insuficiente(patterns, capsys):
    bc = find_critical_beta(patterns)["beta_critico"]
    report_beta_validity(patterns, bc * 0.95)
    salida = capsys.readouterr().out
    assert "AVISO" in salida and "NO son atractores genuinos" in salida
    report_beta_validity(patterns, DEFAULT_BETA)
    assert "atractores genuinos" in capsys.readouterr().out
