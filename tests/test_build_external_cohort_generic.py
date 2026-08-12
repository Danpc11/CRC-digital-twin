"""
Tests para build_external_cohort_generic.py -- el puente de ID (GSM vs.
'col001' del consorcio) y el manejo de tokens NA fueron bugs reales
encontrados en produccion con GSE33113/GSE14333. Se monkeypatchean
RAW_GEO/RAW_SYNAPSE (constantes de modulo con rutas fijas relativas al
repo) para aislar cada test en un directorio temporal, sin tocar datos
reales ni requerir descargas de red.
"""

import gzip
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_external_cohort_generic as mod


def _setup_fixture(tmp_path, sample_titles, geo_accessions, expr_values,
                     dataset_labels, duration_vals, event_vals, gene_symbol="MYC", probe_id="201649_at"):
    """Arma un series_matrix.txt.gz + anotacion de plataforma + etiquetas
    CMS minimas, en un directorio temporal aislado."""
    raw_geo = tmp_path / "data" / "raw_geo"
    raw_synapse = tmp_path / "data" / "raw_synapse" / "tcga_cms_labels"
    raw_geo.mkdir(parents=True)
    raw_synapse.mkdir(parents=True)

    titles_str = "\t".join(f'"{t}"' for t in sample_titles)
    gsm_str = "\t".join(f'"{g}"' for g in geo_accessions)
    dur_str = "\t".join(f'"duration: {d}"' for d in duration_vals)
    ev_str = "\t".join(f'"event: {e}"' for e in event_vals)
    expr_str = "\t".join(str(v) for v in expr_values)
    gsm_header = "\t".join(f'"{g}"' for g in geo_accessions)

    content = (
        f'!Series_platform_id\t"GPL_TEST"\n'
        f'!Sample_title\t{titles_str}\n'
        f'!Sample_geo_accession\t{gsm_str}\n'
        f'!Sample_characteristics_ch1\t{dur_str}\n'
        f'!Sample_characteristics_ch1\t{ev_str}\n'
        f'!series_matrix_table_begin\n'
        f'"ID_REF"\t{gsm_header}\n'
        f'"{probe_id}"\t{expr_str}\n'
        f'!series_matrix_table_end\n'
    )
    with gzip.open(raw_geo / "GSETEST_series_matrix.txt.gz", "wt") as f:
        f.write(content)

    annot = f"!platform_table_begin\nID\tGene Symbol\n{probe_id}\t{gene_symbol}\n!platform_table_end\n"
    (raw_geo / "GPL_TEST.txt").write_text(annot)

    labels = pd.DataFrame({
        "sample": list(dataset_labels.keys()),
        "dataset": ["testset"] * len(dataset_labels),
        "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples": list(dataset_labels.values()),
    })
    labels.to_csv(raw_synapse / "cms_labels_public_all.txt", sep="\t", index=False)

    return raw_geo, raw_synapse.parent


@pytest.fixture(autouse=True)
def _patch_paths(tmp_path, monkeypatch):
    """Redirige RAW_GEO/RAW_SYNAPSE del modulo a un directorio temporal
    para cada test, y vuelve al directorio de trabajo original al salir."""
    monkeypatch.setattr(mod, "RAW_GEO", tmp_path / "data" / "raw_geo")
    monkeypatch.setattr(mod, "RAW_SYNAPSE", tmp_path / "data" / "raw_synapse")
    monkeypatch.chdir(tmp_path)


def test_direct_gsm_match_no_bridge_needed(tmp_path, monkeypatch, capsys):
    """Caso normal: las etiquetas CMS usan el mismo ID (GSM) que la
    expresion -- no hace falta ningun puente."""
    _setup_fixture(
        tmp_path,
        sample_titles=["t1", "t2"], geo_accessions=["GSM1", "GSM2"],
        expr_values=[5.2, 4.8], dataset_labels={"GSM1": "CMS1", "GSM2": "CMS2"},
        duration_vals=["10.0", "20.0"], event_vals=["1", "0"],
    )
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--gse", "GSETEST", "--dataset", "testset",
        "--duration-col", "characteristics__duration", "--event-col", "characteristics__event",
        "--output", str(out_path),
    ])
    mod.main()
    result = pd.read_csv(out_path, sep="\t")
    assert len(result) == 2
    assert set(result["sample_id"]) == {"GSM1", "GSM2"}


def test_regression_id_bridge_via_sample_title(tmp_path, monkeypatch, capsys):
    """
    Bug real (GSE33113): las etiquetas CMS usan 'col001' etc., pero la
    expresion viene indexada por GSM -- el cruce directo da 0 y debe
    usar Sample_title como puente automaticamente.
    """
    _setup_fixture(
        tmp_path,
        sample_titles=["col001", "col002"], geo_accessions=["GSM999", "GSM998"],
        expr_values=[5.2, 4.8], dataset_labels={"col001": "CMS1", "col002": "CMS4"},
        duration_vals=["15.0", "25.0"], event_vals=["0", "1"],
    )
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--gse", "GSETEST", "--dataset", "testset",
        "--duration-col", "characteristics__duration", "--event-col", "characteristics__event",
        "--output", str(out_path),
    ])
    mod.main()
    salida = capsys.readouterr().out
    assert "puente" in salida.lower()

    result = pd.read_csv(out_path, sep="\t")
    assert len(result) == 2, "el puente de ID deberia haber encontrado las 2 muestras"
    assert set(result["sample_id"]) == {"col001", "col002"}


def test_regression_no_bridge_found_raises_clear_error(tmp_path, monkeypatch):
    """Si ni GSM ni Sample_title/Sample_description coinciden con las
    etiquetas, debe fallar con un mensaje claro, no un resultado vacio
    silencioso (el bug original antes del fix: 0 muestras sin error)."""
    _setup_fixture(
        tmp_path,
        sample_titles=["titulo_random_1", "titulo_random_2"],
        geo_accessions=["GSM1", "GSM2"], expr_values=[5.2, 4.8],
        dataset_labels={"ID_QUE_NO_EXISTE_A": "CMS1", "ID_QUE_NO_EXISTE_B": "CMS2"},
        duration_vals=["10.0", "20.0"], event_vals=["1", "0"],
    )
    monkeypatch.setattr(sys, "argv", [
        "prog", "--gse", "GSETEST", "--dataset", "testset",
        "--duration-col", "characteristics__duration", "--event-col", "characteristics__event",
    ])
    with pytest.raises(ValueError, match="puente"):
        mod.main()


def test_event_map_translates_text_values(tmp_path, monkeypatch):
    """El mapeo texto->numero (ej. 'yes'/'no') debe aplicarse correctamente."""
    _setup_fixture(
        tmp_path,
        sample_titles=["t1", "t2"], geo_accessions=["GSM1", "GSM2"],
        expr_values=[5.2, 4.8], dataset_labels={"GSM1": "CMS1", "GSM2": "CMS2"},
        duration_vals=["10.0", "20.0"], event_vals=["yes", "no"],
    )
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--gse", "GSETEST", "--dataset", "testset",
        "--duration-col", "characteristics__duration", "--event-col", "characteristics__event",
        "--event-map", "yes=1,no=0", "--output", str(out_path),
    ])
    mod.main()
    result = pd.read_csv(out_path, sep="\t").set_index("sample_id")
    assert result.loc["GSM1", "relapse_event"] == 1
    assert result.loc["GSM2", "relapse_event"] == 0


def test_na_token_becomes_missing_not_error(tmp_path, monkeypatch):
    """
    Bug real relacionado (GSE33113): valores 'NA' en texto literal
    deben convertirse a faltante, no reventar la validacion de
    --event-map (que antes los marcaba como 'no cubiertos').
    """
    _setup_fixture(
        tmp_path,
        sample_titles=["t1", "t2"], geo_accessions=["GSM1", "GSM2"],
        expr_values=[5.2, 4.8], dataset_labels={"GSM1": "CMS1", "GSM2": "CMS2"},
        duration_vals=["10.0", "20.0"], event_vals=["yes", "NA"],
    )
    out_path = tmp_path / "out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--gse", "GSETEST", "--dataset", "testset",
        "--duration-col", "characteristics__duration", "--event-col", "characteristics__event",
        "--event-map", "yes=1,no=0", "--output", str(out_path),
    ])
    mod.main()  # no debe reventar por el 'NA'
    result = pd.read_csv(out_path, sep="\t").set_index("sample_id")
    assert result.loc["GSM1", "relapse_event"] == 1
    assert pd.isna(result.loc["GSM2", "relapse_event"])


def test_diagnose_mode_does_not_write_output_file(tmp_path, monkeypatch, capsys):
    """--diagnose solo debe inspeccionar, nunca escribir un archivo de salida."""
    _setup_fixture(
        tmp_path,
        sample_titles=["t1", "t2"], geo_accessions=["GSM1", "GSM2"],
        expr_values=[5.2, 4.8], dataset_labels={"GSM1": "CMS1", "GSM2": "CMS2"},
        duration_vals=["10.0", "20.0"], event_vals=["1", "0"],
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--gse", "GSETEST", "--diagnose"])
    mod.main()
    salida = capsys.readouterr().out
    assert "Columnas de fenotipo disponibles" in salida
    assert not (tmp_path / "testset_cms_labeled.tsv").exists()
