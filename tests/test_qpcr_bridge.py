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


# ----------------------------------------------------------------------
# Regresion del bug de doble normalizacion (2026-09-11)
# ----------------------------------------------------------------------

def _cohorte_sintetica_calibrada():
    """Cohorte sintetica en escala 'cruda' (medias por gen lejos de 0,
    como microarreglo), patrones calibrados y estadisticas congeladas."""
    from calibration import calibrate_patterns_from_data, compute_gene_stats
    import pandas as pd

    rng = np.random.default_rng(7)
    genes = ["G1", "G2", "G3", "G4", "G5", "G6"]
    ref_mean = np.array([8.0, 6.5, 10.2, 5.1, 9.4, 7.7])  # escala log2 tipica
    ref_sd = np.array([0.8, 1.1, 0.6, 1.3, 0.9, 0.7])
    centroids_z = {
        "CMS1_MSI_immune":    np.array([ 1.2,  1.0, -0.4, -0.5, -0.6, -0.5]),
        "CMS2_canonical_WNT": np.array([-0.5, -0.4,  1.1,  1.0, -0.5, -0.4]),
        "CMS3_metabolic":     np.array([-0.4, -0.5, -0.5,  1.0,  1.1, -0.6]),
        "CMS4_mesenchymal":   np.array([-0.5, -0.5, -0.4, -0.5,  0.9,  1.2]),
    }
    rows = []
    for label, cz in centroids_z.items():
        for i in range(60):
            z = cz + rng.normal(0, 0.6, len(genes))
            raw = ref_mean + ref_sd * z
            rows.append({"sample_id": f"{label}_{i}", "cms_label": label,
                         **dict(zip(genes, raw))})
    df = pd.DataFrame(rows)
    patterns, genes = calibrate_patterns_from_data(df, genes)
    stats = compute_gene_stats(df, genes)
    return df, genes, patterns, stats


def test_bridge_with_gene_stats_then_frozen_zscore_classifies_correctly():
    """Flujo REAL de la app: puente ajustado con gene_stats -> apply ->
    zscore congelado -> correlacion. Antes del fix, el puente devolvia
    escala z y el zscore congelado se aplicaba encima (doble
    normalizacion): un CMS1 claro salia CMS2."""
    from calibration import zscore_genes
    import pandas as pd

    df, genes, patterns, stats = _cohorte_sintetica_calibrada()
    # Delta-Ct simulado = transformacion lineal desconocida de la escala cruda
    a_real, b_real = 0.9, -4.0
    anclas = df.groupby("cms_label").head(6)
    dct_anclas = {g: (a_real * anclas[g] + b_real).tolist() for g in genes}
    bridge = fit_qpcr_bridge_from_known_cms(
        dct_anclas, anclas["cms_label"].tolist(), patterns, genes, gene_stats=stats)
    assert bridge["_meta"]["output_scale"] == "raw"

    aciertos = 0
    pacientes = df.groupby("cms_label").tail(10)
    for _, pt in pacientes.iterrows():
        dct = {g: a_real * pt[g] + b_real for g in genes}
        crudo = apply_qpcr_bridge(dct, bridge, genes, expected_scale="raw")
        z = zscore_genes(pd.DataFrame([dict(zip(genes, crudo))]), genes, stats=stats).iloc[0].to_numpy()
        corrs = {k: np.corrcoef(z, v)[0, 1] for k, v in patterns.items()}
        aciertos += max(corrs, key=corrs.get) == pt["cms_label"]
    assert aciertos / len(pacientes) >= 0.8


def test_classify_delta_ct_matches_manual_path():
    from calibration import zscore_genes
    from qpcr_bridge import classify_delta_ct
    import pandas as pd

    df, genes, patterns, stats = _cohorte_sintetica_calibrada()
    anclas = df.groupby("cms_label").head(6)
    dct_anclas = {g: anclas[g].tolist() for g in genes}
    bridge = fit_qpcr_bridge_from_known_cms(
        dct_anclas, anclas["cms_label"].tolist(), patterns, genes, gene_stats=stats)
    pt = df.iloc[0]
    dct = {g: float(pt[g]) for g in genes}
    label, corrs, z = classify_delta_ct(dct, bridge, genes, patterns, stats)
    crudo = apply_qpcr_bridge(dct, bridge, genes)
    z_manual = zscore_genes(pd.DataFrame([dict(zip(genes, crudo))]), genes, stats=stats).iloc[0].to_numpy()
    assert np.allclose(z, z_manual)
    assert label == max(corrs, key=corrs.get)


def test_apply_bridge_refuses_scale_mismatch():
    """La proteccion explicita contra volver a normalizar un puente en escala z."""
    df, genes, patterns, stats = _cohorte_sintetica_calibrada()
    anclas = df.groupby("cms_label").head(4)
    dct_anclas = {g: anclas[g].tolist() for g in genes}
    bridge_z = fit_qpcr_bridge_from_known_cms(
        dct_anclas, anclas["cms_label"].tolist(), patterns, genes)  # sin gene_stats
    assert bridge_z["_meta"]["output_scale"] == "z"
    with pytest.raises(ValueError, match="escala"):
        apply_qpcr_bridge({g: 1.0 for g in genes}, bridge_z, genes, expected_scale="raw")


def test_bridge_without_gene_stats_must_not_be_rezscored():
    """Documenta el bug: si se z-scorea un puente en escala z con stats
    congelados, la clasificacion se degrada de forma medible."""
    from calibration import zscore_genes
    import pandas as pd

    df, genes, patterns, stats = _cohorte_sintetica_calibrada()
    anclas = df.groupby("cms_label").head(6)
    dct_anclas = {g: anclas[g].tolist() for g in genes}
    bridge_z = fit_qpcr_bridge_from_known_cms(
        dct_anclas, anclas["cms_label"].tolist(), patterns, genes)
    pacientes = df.groupby("cms_label").tail(10)
    ok_correcto = ok_doble = 0
    for _, pt in pacientes.iterrows():
        dct = {g: float(pt[g]) for g in genes}
        z = apply_qpcr_bridge(dct, bridge_z, genes)  # ya es z
        c1 = {k: np.corrcoef(z, v)[0, 1] for k, v in patterns.items()}
        ok_correcto += max(c1, key=c1.get) == pt["cms_label"]
        z2 = zscore_genes(pd.DataFrame([dict(zip(genes, z))]), genes, stats=stats).iloc[0].to_numpy()
        c2 = {k: np.corrcoef(z2, v)[0, 1] for k, v in patterns.items()}
        ok_doble += max(c2, key=c2.get) == pt["cms_label"]
    assert ok_correcto > ok_doble
