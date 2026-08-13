"""
Tests para build_tcga_rnaseq_dataset.py -- reconstruye la cohorte TCGA
con los 10 genes completos del panel (el archivo anterior solo tenia 5,
de una version pre-congelamiento del panel).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_tcga_rnaseq_dataset as mod


def _setup_fixture(tmp_path):
    raw = tmp_path / "raw_synapse" / "tcga_rnaseq"
    labels_dir = tmp_path / "raw_synapse" / "tcga_cms_labels"
    raw.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    expr = pd.DataFrame({
        "TCGA-A6-6653": [5.2, 1.1, 4.1, 0.5, 2.1, 1.5, 3.1, 0.9, 2.5, 1.8, 0.4],
        "TCGA-F4-6461": [4.6, 2.2, 5.2, 0.6, 2.2, 1.6, 3.2, 1.0, 2.6, 1.9, 0.5],
        "TCGA-XX-0000": [3.3, 3.3, 1.3, 0.7, 2.3, 1.7, 3.3, 1.1, 2.7, 2.0, 0.6],
    }, index=["A1BG"] + mod.GENES)
    expr.index.name = "feature"
    expr.to_csv(raw / "expr.tsv", sep="\t")

    labels = pd.DataFrame({
        "sample": ["TCGA-A6-6653", "TCGA-F4-6461", "TCGA-OTRO"],
        "dataset": ["tcga", "tcga", "otro_dataset"],
        "CMS_network": ["CMS1", "UNK", "CMS2"],
        "CMS_RFclassifier": ["CMS1", "CMS4", "CMS2"],
        "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples": [
            "CMS1", "CMS4", "CMS2"],  # forma CORTA -- asi viene en el archivo real
    })
    labels.to_csv(labels_dir / "labels.tsv", sep="\t", index=False)

    clinical = pd.DataFrame({
        "id": ["TCGA-A6-6653", "TCGA-F4-6461", "TCGA-XX-0000"],
        "age": [31, 45, 50], "gender": ["male", "female", "male"],
        "stage": ["Stage IIA", "Stage IIIA", "Stage I"],
        "tStage": ["T3", "T2", "T1"], "nStage": ["N0", "N1b", "N0"], "mStage": ["M0", "M0", "M0"],
        "tumorLocation": ["Cecum", "Rectum", "Colon"],
        "dfsMo": ["NA", "NA", "NA"], "dfsStat": ["NA", "NA", "NA"],
        "osMo": [4.24, 7.36, 10.0], "osStat": [0, 1, 0],
        "batch": ["NA"]*3, "microsatelite": ["MSS"]*3, "cimp": ["NA"]*3, "adjChemo": ["NA"]*3,
    })
    clinical.to_csv(raw / "clinical.tsv", sep="\t", index=False)

    return raw / "expr.tsv", raw / "clinical.tsv", labels_dir / "labels.tsv"


def test_extracts_all_10_panel_genes(tmp_path, monkeypatch):
    expr_path, clinical_path, labels_path = _setup_fixture(tmp_path)
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--expression", str(expr_path), "--clinical", str(clinical_path),
        "--labels", str(labels_path), "--output", str(out_path),
    ])
    mod.main()
    result = pd.read_csv(out_path, sep="\t")
    for gene in mod.GENES:
        assert gene in result.columns
    assert "A1BG" not in result.columns  # no debe colarse un gen fuera del panel


def test_labels_from_correct_dataset_only(tmp_path, monkeypatch):
    """Debe usar solo las etiquetas con dataset='tcga', no las de otros
    datasets aunque coincidan por casualidad en otras columnas."""
    expr_path, clinical_path, labels_path = _setup_fixture(tmp_path)
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--expression", str(expr_path), "--clinical", str(clinical_path),
        "--labels", str(labels_path), "--output", str(out_path),
    ])
    mod.main()
    result = pd.read_csv(out_path, sep="\t").set_index("sample_id")
    assert result.loc["TCGA-A6-6653", "cms_label"] == "CMS1_MSI_immune"
    assert result.loc["TCGA-F4-6461", "cms_label"] == "CMS4_mesenchymal"


def test_sample_without_official_label_marked_none_not_dropped(tmp_path, monkeypatch):
    """Una muestra con expresion pero sin etiqueta CMS oficial debe
    quedarse en el dataset marcada 'none', no excluirse -- igual que
    el resto de las cohortes del proyecto sin verdad de referencia
    completa para todas sus muestras."""
    expr_path, clinical_path, labels_path = _setup_fixture(tmp_path)
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--expression", str(expr_path), "--clinical", str(clinical_path),
        "--labels", str(labels_path), "--output", str(out_path),
    ])
    mod.main()
    result = pd.read_csv(out_path, sep="\t").set_index("sample_id")
    assert len(result) == 3  # las 3 muestras de expresion, ninguna se perdio
    assert result.loc["TCGA-XX-0000", "cms_label"] == "none"


def test_short_form_cms_labels_mapped_to_long_form(tmp_path, monkeypatch):
    """
    Regresion de un bug real de produccion: el archivo central de
    etiquetas da la forma CORTA (CMS1, CMS4, NOLBL), pero
    load_labeled_dataset() del resto del pipeline exige la forma LARGA
    (CMS1_MSI_immune, ...) -- sin este mapeo, external_validation.py
    truena con 'Etiquetas CMS no reconocidas'. Debe usar el MISMO
    mapeo canonico (CMS_RENAME) que build_external_cohort_generic.py,
    no uno inventado -- incluye el caso NOLBL, visto en produccion con
    61 muestras reales de TCGA.
    """
    expr_path, clinical_path, labels_path = _setup_fixture(tmp_path)
    labels = pd.read_csv(labels_path, sep="\t")
    labels.loc[len(labels)] = ["TCGA-XX-0000", "tcga", "UNK", "NOLBL", "NOLBL"]
    labels.to_csv(labels_path, sep="\t", index=False)

    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--expression", str(expr_path), "--clinical", str(clinical_path),
        "--labels", str(labels_path), "--output", str(out_path),
    ])
    mod.main()
    result = pd.read_csv(out_path, sep="\t").set_index("sample_id")

    esperadas = {"CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic",
                 "CMS4_mesenchymal", "none"}
    assert set(result["cms_label"].unique()).issubset(esperadas), (
        "todas las etiquetas deben quedar en forma larga reconocida por "
        "load_labeled_dataset(), no en forma corta cruda"
    )
    assert result.loc["TCGA-A6-6653", "cms_label"] == "CMS1_MSI_immune"
    assert result.loc["TCGA-XX-0000", "cms_label"] == "none"  # NOLBL -> none


def test_output_is_accepted_by_load_labeled_dataset(tmp_path, monkeypatch):
    """Prueba end-to-end minima: el archivo generado debe pasar la
    misma validacion que usa external_validation.py en produccion, no
    solo verse bien por inspeccion manual."""
    expr_path, clinical_path, labels_path = _setup_fixture(tmp_path)
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--expression", str(expr_path), "--clinical", str(clinical_path),
        "--labels", str(labels_path), "--output", str(out_path),
    ])
    mod.main()

    from calibration import load_labeled_dataset
    load_labeled_dataset(str(out_path))  # no debe lanzar ValueError


def test_missing_panel_gene_is_reported_not_silently_dropped(tmp_path, monkeypatch, capsys):
    expr_path, clinical_path, labels_path = _setup_fixture(tmp_path)
    # quitar un gen del panel del archivo de expresion
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr = expr.drop(index="MLH1")
    expr.to_csv(expr_path, sep="\t")

    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--expression", str(expr_path), "--clinical", str(clinical_path),
        "--labels", str(labels_path), "--output", str(out_path),
    ])
    mod.main()
    salida = capsys.readouterr().out
    assert "MLH1" in salida
    assert "9/10" in salida
    result = pd.read_csv(out_path, sep="\t")
    assert "MLH1" not in result.columns
