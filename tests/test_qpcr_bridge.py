"""
Tests para qpcr_bridge.py -- puente entre Delta-Ct crudo de RT-qPCR y
la escala de referencia congelada (expresion log2 de microarreglos).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qpcr_bridge import (
    apply_qpcr_bridge,
    compute_delta_ct,
    fit_qpcr_bridge,
    fit_qpcr_bridge_from_known_cms,
)


def test_compute_delta_ct_increases_with_expression():
    """Delta-Ct con signo invertido: mas expresion (menos ciclos Ct)
    debe dar un Delta-Ct MAS ALTO, igual direccion que expresion log2."""
    ct_alto = compute_delta_ct({"MYC": 20.0}, ct_reference=25.0)  # Ct bajo = mucha expresion
    ct_bajo = compute_delta_ct({"MYC": 30.0}, ct_reference=25.0)  # Ct alto = poca expresion
    assert ct_alto["MYC"] > ct_bajo["MYC"]


def test_fit_recovers_known_linear_transform_with_paired_anchors():
    """Caso gold standard: con anclas PAREADAS (valor real conocido en
    la escala de referencia), el ajuste debe recuperar la
    transformacion verdadera con precision alta."""
    rng = np.random.default_rng(0)
    a_real, b_real = 1.3, -2.1
    valores_referencia = rng.uniform(-2, 2, 15)
    delta_ct = (valores_referencia - b_real) / a_real + rng.normal(0, 0.05, 15)

    bridge = fit_qpcr_bridge({"MYC": delta_ct}, {"MYC": valores_referencia})
    assert abs(bridge["MYC"]["a"] - a_real) < 0.1
    assert abs(bridge["MYC"]["b"] - b_real) < 0.3
    assert bridge["MYC"]["r2"] > 0.95


def test_few_anchors_warns_despite_perfect_r2():
    """Regresion de una trampa real: con 2 anclas, una recta siempre da
    R2=1.0 (pasa exacto por los 2 puntos) -- eso NO es evidencia de que
    el ajuste generalice. Debe avisarse explicitamente, no dar falsa
    confianza con un R2 perfecto sin contexto."""
    bridge = fit_qpcr_bridge({"MYC": [10.0, 15.0]}, {"MYC": [1.0, 2.0]})
    assert bridge["MYC"]["r2"] == 1.0
    assert "aviso" in bridge["MYC"]
    assert "fragil" in bridge["MYC"]["aviso"] or "2 anclas" in bridge["MYC"]["aviso"]


def test_poor_linear_fit_flagged_with_low_r2():
    rng = np.random.default_rng(2)
    dct_ruidoso = rng.uniform(5, 20, 10)
    objetivo_sin_relacion = rng.uniform(-2, 2, 10)
    bridge = fit_qpcr_bridge({"MYC": dct_ruidoso}, {"MYC": objetivo_sin_relacion})
    assert bridge["MYC"]["r2"] < 0.5
    assert "aviso" in bridge["MYC"]


def test_single_anchor_cannot_fit_a_line():
    bridge = fit_qpcr_bridge({"MYC": [10.0]}, {"MYC": [1.0]})
    assert bridge["MYC"]["a"] is None
    assert "aviso" in bridge["MYC"]


def test_fit_from_known_cms_uses_pattern_centroid_as_proxy_target():
    """Modo practico: sin anclas pareadas, usa el centroide calibrado
    de la clase CMS conocida de cada ancla como objetivo aproximado."""
    patterns = {
        "CMS1_MSI_immune": np.array([2.0, -1.0]),
        "CMS2_canonical_WNT": np.array([-1.5, 1.8]),
    }
    gene_order = ["MYC", "AXIN2"]
    rng = np.random.default_rng(3)
    a_real, b_real = 0.8, 1.0

    delta_ct_anclas = {"MYC": [], "AXIN2": []}
    etiquetas = []
    for label, centroide in patterns.items():
        for _ in range(3):
            for i, g in enumerate(gene_order):
                dct = (centroide[i] - b_real) / a_real + rng.normal(0, 0.05)
                delta_ct_anclas[g].append(dct)
            etiquetas.append(label)

    bridge = fit_qpcr_bridge_from_known_cms(delta_ct_anclas, etiquetas, patterns, gene_order)
    assert abs(bridge["MYC"]["a"] - a_real) < 0.15
    assert bridge["MYC"]["r2"] > 0.9


def test_fit_from_known_cms_rejects_unknown_label():
    patterns = {"CMS1_MSI_immune": np.array([2.0])}
    with pytest.raises(ValueError, match="no esta en los patrones"):
        fit_qpcr_bridge_from_known_cms(
            {"MYC": [1.0, 2.0]}, ["CMS1_MSI_immune", "CMS_INVENTADO"], patterns, ["MYC"])


def test_apply_bridge_end_to_end_classifies_correctly():
    """Prueba completa: simular un paciente con una transformacion de
    escala conocida, aplicar el puente ajustado con anclas por
    centroide, y confirmar que la clasificacion final (via correlacion
    con los patrones) da el CMS correcto."""
    patterns = {
        "CMS1_MSI_immune": np.array([2.0, -1.0, 0.5]),
        "CMS4_mesenchymal": np.array([-1.5, 1.8, -0.9]),
    }
    gene_order = ["MYC", "AXIN2", "VIM"]
    rng = np.random.default_rng(4)
    a_real = {g: rng.uniform(0.6, 1.4) for g in gene_order}
    b_real = {g: rng.uniform(-2, 2) for g in gene_order}

    delta_ct_anclas = {g: [] for g in gene_order}
    etiquetas = []
    for label, centroide in patterns.items():
        for _ in range(4):
            for i, g in enumerate(gene_order):
                dct = (centroide[i] - b_real[g]) / a_real[g] + rng.normal(0, 0.05)
                delta_ct_anclas[g].append(dct)
            etiquetas.append(label)

    bridge = fit_qpcr_bridge_from_known_cms(delta_ct_anclas, etiquetas, patterns, gene_order)

    # paciente nuevo, CMS4 real, con la MISMA transformacion de escala
    p_real = patterns["CMS4_mesenchymal"] + rng.normal(0, 0.1, 3)
    dct_paciente = {g: (p_real[i] - b_real[g]) / a_real[g] + rng.normal(0, 0.05)
                     for i, g in enumerate(gene_order)}

    valor_calibrado = apply_qpcr_bridge(dct_paciente, bridge, gene_order)
    corrs = {label: np.corrcoef(valor_calibrado, c)[0, 1] for label, c in patterns.items()}
    predicho = max(corrs, key=corrs.get)
    assert predicho == "CMS4_mesenchymal"


def test_apply_bridge_raises_clear_error_for_missing_gene_fit():
    bridge = {"MYC": {"a": 1.0, "b": 0.0, "r2": 0.9, "n_anclas": 5}}
    with pytest.raises(ValueError, match="AXIN2"):
        apply_qpcr_bridge({"MYC": 10.0, "AXIN2": 5.0}, bridge, ["MYC", "AXIN2"])
