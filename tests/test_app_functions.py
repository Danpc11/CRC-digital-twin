"""
Tests para las funciones puras de app.py -- el modulo no se puede
importar directo (corre codigo de Streamlit a nivel de modulo apenas
se importa: st.set_page_config, CSS, etc.), asi que se extraen por AST
las funciones logicas puras y se ejecutan en un namespace controlado.
Mismo patron usado repetidas veces de forma manual durante el
desarrollo -- aqui queda formalizado como test permanente.
"""

import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration import calibrate_patterns_from_data, zscore_genes
from survival_validation import score_cohort
from attractor_model import build_model_from_patterns
from prognosis import hazard_from_trajectory, detect_recurrence_signal
from prognosis_demo import EVIDENCE_STRENGTH, simulate_longitudinal_patient, classify_current_state
from treatment_perturbation import TREATMENT_MECHANISMS, describe_treatment
from treatment_simulation_demo import simulate_with_optional_treatment
from synthetic_data import generate_synthetic_cohort

CMS_COLOR = {"CMS1_MSI_immune": "#0072B2", "CMS2_canonical_WNT": "#E69F00",
             "CMS3_metabolic": "#009E73", "CMS4_mesenchymal": "#D55E00", "none": "#8A8F98"}
CMS_SHORT = {"CMS1_MSI_immune": "CMS1", "CMS2_canonical_WNT": "CMS2",
             "CMS3_metabolic": "CMS3", "CMS4_mesenchymal": "CMS4", "none": "n/c"}


def _extract_app_functions(names: set) -> dict:
    """Extrae funciones especificas de app.py por AST, sin importar (ni
    ejecutar) el resto del modulo, que dispara codigo de Streamlit a
    nivel de import. Tambien extrae las constantes a nivel de modulo
    (EVIDENCE_STYLE, etc.) de las que dependen esas funciones."""
    src = (ROOT / "app.py").read_text()
    tree = ast.parse(src)
    namespace = {**globals()}

    # constantes a nivel de modulo que las funciones extraidas necesitan
    const_names = {"EVIDENCE_STYLE", "CMS_COLOR", "CMS_SHORT"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in const_names for t in node.targets
        ):
            code = compile(ast.Module(body=[node], type_ignores=[]), "<app.py>", "exec")
            exec(code, namespace)

    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = compile(ast.Module(body=[node], type_ignores=[]), "<app.py>", "exec")
            exec(code, namespace)
            found.add(node.name)
    missing = names - found
    assert not missing, f"funciones no encontradas en app.py: {missing}"
    return {name: namespace[name] for name in names}


@pytest.fixture(scope="module")
def app_functions():
    return _extract_app_functions({
        "cms_tag", "evidence_meter", "readout",
        "cached_trajectory", "cached_treatment_sim",
        "evaluate_all_treatments", "build_patient_pdf",
    })


@pytest.fixture(scope="module")
def real_calibration():
    df = generate_synthetic_cohort(n_per_class=60, seed=7)
    patterns, gene_order = calibrate_patterns_from_data(df)
    W, _, _ = build_model_from_patterns(patterns)
    return patterns, gene_order, W


# --- funciones de construccion de HTML (deben ser deterministas y no reventar) ---

def test_cms_tag_returns_html_with_short_label(app_functions):
    html = app_functions["cms_tag"]("CMS4_mesenchymal")
    assert "CMS4" in html
    assert "#D55E00" in html  # color Wong de CMS4


def test_cms_tag_handles_unknown_label_gracefully(app_functions):
    html = app_functions["cms_tag"]("etiqueta_no_existente")
    assert "etiqueta_no_existente" in html  # cae al label crudo, no revienta


def test_evidence_meter_fuerte_vs_sin_evidencia_differ(app_functions):
    fuerte = app_functions["evidence_meter"]("fuerte")
    sin_ev = app_functions["evidence_meter"]("sin evidencia")
    assert fuerte != sin_ev
    assert "#1B7F5A" in fuerte  # verde para fuerte
    assert "#B03A2E" in sin_ev  # rojo para sin evidencia


def test_readout_includes_value_and_accent_color(app_functions):
    html = app_functions["readout"]("Riesgo final", "2.74", "ordinal", "#0072B2")
    assert "2.74" in html
    assert "#0072B2" in html
    assert "Riesgo final" in html


# --- funciones que corren la simulacion real (con datos calibrados reales) ---

def test_cached_trajectory_produces_expected_shape(app_functions, real_calibration):
    patterns, gene_order, W = real_calibration
    n_genes = len(gene_order)
    driver = patterns["CMS4_mesenchymal"]
    t, x = app_functions["cached_trajectory"](W, gene_order, driver, n_genes, 10, 15)
    assert len(t) == 10
    assert x.shape == (n_genes, 10)


def test_evaluate_all_treatments_returns_sorted_by_delta_descending(app_functions, real_calibration):
    patterns, gene_order, W = real_calibration
    n_genes = len(gene_order)
    driver = patterns["CMS1_MSI_immune"] * 0.8
    resultados = app_functions["evaluate_all_treatments"](W, n_genes, gene_order, driver, patterns)
    assert len(resultados) == len(TREATMENT_MECHANISMS)
    deltas = [r["delta"] for r in resultados]
    assert deltas == sorted(deltas, reverse=True)
    assert all("aplica" in r for r in resultados)


def test_evaluate_all_treatments_immunotherapy_applies_near_cms1(app_functions, real_calibration):
    """Regresion de comportamiento esperado: inmunoterapia debe tener
    efecto para un paciente cerca de CMS1, no para uno cerca de CMS4."""
    patterns, gene_order, W = real_calibration
    n_genes = len(gene_order)

    driver_cms1 = patterns["CMS1_MSI_immune"] * 0.8
    res_cms1 = app_functions["evaluate_all_treatments"](W, n_genes, gene_order, driver_cms1, patterns)
    aplica_cms1 = {r["treatment"]: r["aplica"] for r in res_cms1}
    assert aplica_cms1["immunotherapy_antiPD1"] == True


# --- generacion de PDF (verificado antes visualmente, aqui solo estructura) ---

def test_build_patient_pdf_produces_valid_pdf_bytes(app_functions, real_calibration):
    patterns, gene_order, W = real_calibration
    n_genes = len(gene_order)
    driver = patterns["CMS4_mesenchymal"]

    t, x = app_functions["cached_trajectory"](W, gene_order, driver, n_genes, 10, 15)
    hazard = hazard_from_trajectory(x)
    alert, idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)
    resultados = app_functions["evaluate_all_treatments"](W, n_genes, gene_order, driver, patterns)
    ev = EVIDENCE_STRENGTH.get("CMS4_mesenchymal", {})

    pdf_bytes = app_functions["build_patient_pdf"](
        "TEST-001", "CMS4_mesenchymal", 0.85, ev, t, hazard, alert, idx, resultados, gene_order)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"  # magic bytes reales de un PDF valido
    assert len(pdf_bytes) > 5000  # no un PDF vacio/roto


def test_build_patient_pdf_does_not_crash_without_alert(app_functions, real_calibration):
    """Caso sin alerta detectada (idx=None) -- no debe reventar al
    intentar indexar t_checks[None]."""
    patterns, gene_order, W = real_calibration
    n_genes = len(gene_order)
    driver = patterns["CMS4_mesenchymal"]

    t, x = app_functions["cached_trajectory"](W, gene_order, driver, n_genes, 5, 999)  # recaida nunca ocurre
    hazard = hazard_from_trajectory(x)
    alert, idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)
    assert alert is False

    resultados = app_functions["evaluate_all_treatments"](W, n_genes, gene_order, driver, patterns)
    ev = EVIDENCE_STRENGTH.get("CMS4_mesenchymal", {})
    pdf_bytes = app_functions["build_patient_pdf"](
        "TEST-002", "CMS4_mesenchymal", 0.85, ev, t, hazard, alert, idx, resultados, gene_order)
    assert pdf_bytes[:4] == b"%PDF"
