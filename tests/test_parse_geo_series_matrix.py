"""
Tests para parse_geo_series_matrix.py -- el archivo con MAS bugs reales
encontrados en produccion durante el desarrollo de este proyecto (dos
distintos, en dos cohortes distintas) y CERO cobertura de tests hasta
ahora. Cada bug real se reproduce aqui como caso de regresion.

  1. Desalineacion posicional (GSE17537): una version anterior asumia
     que todas las muestras tienen sus '!Sample_characteristics_ch1'
     en el mismo orden, y usaba el nombre de atributo de la PRIMERA
     muestra para etiquetar toda la fila -- si dos muestras tienen sus
     caracteristicas en distinto orden, los valores se desalinean.
  2. Orden de lineas inesperado (GSE33113): '!Sample_title' aparecia
     ANTES de '!Sample_geo_accession' en el archivo -- una version de
     una sola pasada asumia el orden contrario y descartaba el titulo
     en silencio (justo el campo que servia de puente de ID).
"""

import gzip
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parse_geo_series_matrix import parse_series_matrix


def _write_gz(tmp_path, name, content):
    path = tmp_path / name
    with gzip.open(path, "wt") as f:
        f.write(content)
    return path


# --- caso base: formato simple, una linea por atributo -----------------

def test_simple_format_one_attribute_per_line(tmp_path):
    content = (
        '!Series_platform_id\t"GPL570"\n'
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t"age: 65"\t"age: 72"\n'
        '!Sample_characteristics_ch1\t"dfs_event: no death"\t"dfs_event: death"\n'
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"1007_s_at"\t5.2\t4.8\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "simple.txt.gz", content)
    pheno, expr = parse_series_matrix(path)

    assert list(pheno.index) == ["GSM1", "GSM2"]
    assert pheno.loc["GSM1", "characteristics__age"] == "65"
    assert pheno.loc["GSM2", "characteristics__dfs_event"] == "death"
    assert expr.loc["1007_s_at", "GSM1"] == 5.2


# --- bug real #1: desalineacion posicional entre muestras --------------

def test_regression_misaligned_characteristics_order_between_samples(tmp_path):
    """
    Bug real de GSE17537: si dos muestras tienen sus caracteristicas en
    ORDEN DISTINTO (GSM2 trae 'grade' antes de 'dfs_event', GSM1 al
    reves), la version por-posicion mezclaba los valores entre
    atributos distintos. El parser correcto extrae cada celda por su
    propia clave 'atributo: valor', sin asumir alineacion posicional.
    """
    content = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t"dfs_event: no death"\t"grade: 2 - Moderately differentiated"\n'
        '!Sample_characteristics_ch1\t"grade: 1 - Well differentiated"\t"dfs_event: death"\n'
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"1007_s_at"\t5.2\t4.8\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "misaligned.txt.gz", content)
    pheno, expr = parse_series_matrix(path)

    # cada muestra debe tener SU propio valor correcto, no mezclado
    assert pheno.loc["GSM1", "characteristics__dfs_event"] == "no death"
    assert pheno.loc["GSM1", "characteristics__grade"] == "1 - Well differentiated"
    assert pheno.loc["GSM2", "characteristics__dfs_event"] == "death"
    assert pheno.loc["GSM2", "characteristics__grade"] == "2 - Moderately differentiated"


# --- bug real #2: Sample_title antes de Sample_geo_accession -----------

def test_regression_sample_title_before_geo_accession(tmp_path):
    """
    Bug real de GSE33113: '!Sample_title' (que trae el ID interno del
    consorcio, 'col001') aparecia ANTES de '!Sample_geo_accession' en
    el archivo real. Una version de una sola pasada asumia el orden
    contrario y descartaba el titulo en silencio -- justo el campo que
    servia de puente de ID entre la expresion (indexada por GSM) y las
    etiquetas CMS del consorcio (indexadas por 'col001').
    """
    content = (
        '!Sample_title\t"col001"\t"col002"\n'
        '!Sample_geo_accession\t"GSM820048"\t"GSM820049"\n'
        '!Sample_characteristics_ch1\t"disease status: AJCC stage II CRC"\t"disease status: AJCC stage II CRC"\n'
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM820048"\t"GSM820049"\n'
        '"1007_s_at"\t5.2\t4.8\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "title_first.txt.gz", content)
    pheno, expr = parse_series_matrix(path)

    assert "Sample_title" in pheno.columns, "Sample_title se descarto -- bug real reproducido"
    assert pheno.loc["GSM820048", "Sample_title"] == "col001"
    assert pheno.loc["GSM820049", "Sample_title"] == "col002"
    # Sample_title NO debe llevar el prefijo characteristics__ (era otro
    # bug real: el prefijo generico rompia el puente de ID en downstream)
    assert "characteristics__Sample_title" not in pheno.columns


# --- formato empacado (GSE14333): varios atributos en una sola celda ---

def test_packed_format_multiple_attributes_in_one_cell(tmp_path):
    content = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t'
        '"Location: Right; DukesStage: A; DFS_Time: 3.64; DFS_Cens: 1"\t'
        '"Location: Left; DukesStage: B; DFS_Time: 9.1; DFS_Cens: 0"\n'
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"1007_s_at"\t5.2\t4.8\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "packed.txt.gz", content)
    pheno, expr = parse_series_matrix(path)

    assert pheno.loc["GSM1", "characteristics__DukesStage"] == "A"
    assert pheno.loc["GSM1", "characteristics__DFS_Time"] == "3.64"
    assert pheno.loc["GSM2", "characteristics__DukesStage"] == "B"
    assert pheno.loc["GSM2", "characteristics__DFS_Cens"] == "0"


# --- archivos malformados -----------------------------------------------

def test_malformed_missing_geo_accession_raises_clear_error(tmp_path):
    """Sin '!Sample_geo_accession' no hay forma de saber cuantas muestras
    hay ni sus IDs -- debe fallar con un mensaje claro, no silenciosamente
    ni con un traceback críptico de pandas/numpy."""
    content = (
        '!Sample_characteristics_ch1\t"age: 65"\t"age: 72"\n'
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"1007_s_at"\t5.2\t4.8\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "no_accession.txt.gz", content)
    with pytest.raises(ValueError, match="Sample_geo_accession"):
        parse_series_matrix(path)


def test_malformed_empty_file_raises_clear_error(tmp_path):
    path = _write_gz(tmp_path, "empty.txt.gz", "")
    with pytest.raises(ValueError):
        parse_series_matrix(path)


def test_malformed_characteristics_cell_without_colon_is_skipped_not_crashed(tmp_path):
    """Una celda sin ':' no se puede asociar a un atributo de forma segura
    -- debe omitirse, no adivinar ni reventar."""
    content = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t"age: 65"\t"texto sin dos puntos"\n'
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"1007_s_at"\t5.2\t4.8\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "no_colon.txt.gz", content)
    pheno, expr = parse_series_matrix(path)  # no debe reventar
    assert pheno.loc["GSM1", "characteristics__age"] == "65"
    # GSM2 no tiene ningun atributo valido en esa linea -- no debe
    # aparecer 'characteristics__age' para GSM2 con un valor inventado
    assert pd.isna(pheno.loc["GSM2"].get("characteristics__age", pd.NA))


def test_malformed_mismatched_sample_count_in_characteristics_line(tmp_path):
    """Si una linea de characteristics tiene MENOS valores que muestras
    declaradas (archivo truncado/corrupto), no debe reventar con
    IndexError -- las muestras faltantes simplemente no reciben ese
    atributo (zip corta en el mas corto, comportamiento seguro)."""
    content = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\n'
        '!Sample_characteristics_ch1\t"age: 65"\t"age: 72"\n'  # falta GSM3
        '!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\n'
        '"1007_s_at"\t5.2\t4.8\t3.1\n'
        '!series_matrix_table_end\n'
    )
    path = _write_gz(tmp_path, "truncated.txt.gz", content)
    pheno, expr = parse_series_matrix(path)  # no debe reventar
    assert pheno.loc["GSM1", "characteristics__age"] == "65"
    assert "GSM3" in pheno.index  # la muestra existe (por geo_accession)
