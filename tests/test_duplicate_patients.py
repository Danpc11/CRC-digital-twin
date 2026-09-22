"""Pruebas de la detección de pacientes depositados dos veces en GEO.

El caso que motiva el módulo: GSE14333 y GSE17536 incluyen ambas la serie
del H. Lee Moffitt Cancer Center sin declararlo, y en supervivencia un
paciente duplicado aporta su evento dos veces.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detect_duplicate_patients import (
    CLINICAL_KEY_FIELDS,
    DEFAULT_COHORT_PRIORITY,
    chance_match_rate,
    clinical_key_table,
    detect_across_cohorts,
    drop_duplicate_patients,
    find_duplicate_pairs,
)
from parse_geo_series_matrix import parse_series_matrix

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_geo"


def _keys(cohort, rows):
    """Tabla de claves ya normalizada, como la devuelve clinical_key_table."""
    return pd.DataFrame(rows).assign(cohort=cohort)


def test_clinical_key_normaliza_sexo_edad_y_estadio():
    pheno = pd.DataFrame(
        {
            "characteristics__Gender": ["M", "F"],
            "characteristics__Age_Diag": ["78", "53"],
            "characteristics__DFS_Time": ["3.64", "14.53"],
            "characteristics__DukesStage": ["A", "C"],
        },
        index=["GSM1", "GSM2"],
    )
    keys = clinical_key_table("GSE14333", pheno)
    assert list(keys.sex) == ["M", "F"]
    assert list(keys.age) == [78.0, 53.0]
    assert keys.time_months.tolist() == [3.64, 14.53]
    # Dukes A/C se traduce a la escala AJCC para poder compararlo con series
    # que reportan el estadio como número
    assert keys.stage.tolist() == [1.0, 3.0]


def test_clinical_key_convierte_dias_a_meses_y_coma_decimal():
    """GSE33113 da el tiempo en días y la edad con coma decimal ('41,6')."""
    pheno = pd.DataFrame(
        {
            "characteristics__Sex": ["m"],
            "characteristics__age at diagnosis": ["41,6"],
            "characteristics__time to meta or recurrence": ["2000"],
        },
        index=["GSM1"],
    )
    keys = clinical_key_table("GSE33113", pheno)
    assert keys.sex.iloc[0] == "M"
    assert keys.age.iloc[0] == pytest.approx(41.6)
    assert keys.time_months.iloc[0] == pytest.approx(2000 / 30.4375)
    # la serie entera es estadio II: no discrimina, se declara ausente
    assert keys.stage.iloc[0] is None or pd.isna(keys.stage.iloc[0])


def test_clinical_key_deriva_la_duracion_de_dos_fechas():
    """GSE37892 no deposita duración, solo fecha de cirugía y último contacto."""
    pheno = pd.DataFrame(
        {
            "characteristics__gender": ["male"],
            "characteristics__age at diagnosis": ["70"],
            "characteristics__date at surgery": ["2000-05-02"],
            "characteristics__date at last contact": ["2005-01-01"],
            "characteristics__Stage": ["2"],
        },
        index=["GSM1"],
    )
    keys = clinical_key_table("GSE37892", pheno)
    assert keys.time_months.iloc[0] == pytest.approx(1705 / 30.4375, abs=0.01)


def test_encuentra_la_pareja_y_descarta_la_coincidencia_de_edad_distinta():
    a = _keys("A", [
        {"sample_id": "a1", "sex": "M", "age": 78.0, "time_months": 3.64, "stage": 1.0},
        {"sample_id": "a2", "sex": "M", "age": 60.0, "time_months": 20.00, "stage": 3.0},
    ])
    b = _keys("B", [
        {"sample_id": "b1", "sex": "M", "age": 78.0, "time_months": 3.64, "stage": 1.0},
        # mismo sexo y mismo tiempo que a2, pero otra persona
        {"sample_id": "b2", "sex": "M", "age": 45.0, "time_months": 20.00, "stage": 3.0},
    ])
    pairs, n_descartadas = find_duplicate_pairs(a, b)

    assert len(pairs) == 1
    assert pairs.sample_id_a.iloc[0] == "a1" and pairs.sample_id_b.iloc[0] == "b1"
    assert pairs.clasificacion.iloc[0] == "confirmada"
    assert n_descartadas == 1


def test_el_estadio_discordante_baja_la_pareja_a_revisar():
    a = _keys("A", [{"sample_id": "a1", "sex": "F", "age": 61.0, "time_months": 9.11, "stage": 1.0}])
    b = _keys("B", [{"sample_id": "b1", "sex": "F", "age": 61.0, "time_months": 9.11, "stage": 4.0}])
    pairs, _ = find_duplicate_pairs(a, b)
    assert pairs.clasificacion.iloc[0] == "revisar"


def test_clave_que_empareja_con_varias_muestras_queda_marcada_ambigua():
    """Un tiempo redondo (0.0 meses) colisiona sin identificar a nadie."""
    a = _keys("A", [
        {"sample_id": "a1", "sex": "F", "age": 65.0, "time_months": 0.0, "stage": 2.0},
        {"sample_id": "a2", "sex": "F", "age": 65.0, "time_months": 0.0, "stage": 2.0},
    ])
    b = _keys("B", [
        {"sample_id": "b1", "sex": "F", "age": 65.0, "time_months": 0.0, "stage": 2.0},
    ])
    pairs, _ = find_duplicate_pairs(a, b)
    assert pairs.ambigua.all()
    assert (pairs.clasificacion == "revisar").all()


def test_drop_conserva_la_cohorte_prioritaria():
    pairs = pd.DataFrame([{
        "cohort_a": "GSE14333", "sample_id_a": "a1",
        "cohort_b": "GSE17536", "sample_id_b": "b1",
        "clasificacion": "confirmada",
    }])
    data = pd.DataFrame([
        {"cohort": "GSE14333", "sample_id": "a1"},
        {"cohort": "GSE17536", "sample_id": "b1"},
        {"cohort": "GSE17536", "sample_id": "b2"},
    ])
    out = drop_duplicate_patients(data, pairs, verbose=False)

    assert len(out) == 2
    # GSE17536 va antes que GSE14333 en la prioridad: trae OS/DSS/DFS y su
    # indicador de recaída es explícito, no el DFS_Cens invertido
    assert DEFAULT_COHORT_PRIORITY.index("GSE17536") < DEFAULT_COHORT_PRIORITY.index("GSE14333")
    assert set(out.sample_id) == {"b1", "b2"}


def test_drop_no_depende_de_como_se_llamen_las_cohortes_en_los_datos():
    """cox_diagnostics.py nombra las cohortes por el directorio de entrada.

    Con 'results_external_gse14333' en vez de 'GSE14333', emparejar por
    (cohorte, GSM) hacía que la deduplicación se saltara en silencio. El
    GSM basta: es identificador único en todo GEO.
    """
    pairs = pd.DataFrame([{
        "cohort_a": "GSE14333", "sample_id_a": "GSM358341",
        "cohort_b": "GSE17536", "sample_id_b": "GSM437093",
        "clasificacion": "confirmada",
    }])
    data = pd.DataFrame([
        {"cohort": "results_external_gse14333", "sample_id": "GSM358341"},
        {"cohort": "results_external_gse17536", "sample_id": "GSM437093"},
    ])
    out = drop_duplicate_patients(data, pairs, verbose=False)
    assert list(out.sample_id) == ["GSM437093"]


def test_drop_no_toca_al_paciente_cuya_otra_copia_ya_cayo():
    """Si una copia no sobrevivió a los filtros, no hay duplicación que corregir."""
    pairs = pd.DataFrame([{
        "cohort_a": "GSE14333", "sample_id_a": "a1",
        "cohort_b": "GSE17536", "sample_id_b": "b1",
        "clasificacion": "confirmada",
    }])
    data = pd.DataFrame([{"cohort": "GSE14333", "sample_id": "a1"}])
    out = drop_duplicate_patients(data, pairs, verbose=False)
    assert len(out) == 1


def test_drop_ignora_las_parejas_no_confirmadas():
    pairs = pd.DataFrame([{
        "cohort_a": "GSE14333", "sample_id_a": "a1",
        "cohort_b": "GSE17536", "sample_id_b": "b1",
        "clasificacion": "revisar",
    }])
    data = pd.DataFrame([
        {"cohort": "GSE14333", "sample_id": "a1"},
        {"cohort": "GSE17536", "sample_id": "b1"},
    ])
    assert len(drop_duplicate_patients(data, pairs, verbose=False)) == 2


def test_la_tasa_de_azar_detecta_una_coincidencia_sin_estructura():
    """Dos cohortes independientes con tiempos de baja precisión.

    Es el escenario que producía falsos positivos: cuando una serie reporta
    el seguimiento en meses enteros, dos series grandes coinciden en tiempo
    cientos de veces por azar y unas cuantas coinciden además en edad.
    """
    a = _keys("A", [{"sample_id": f"a{i}", "sex": "M", "age": float(40 + i % 30),
                     "time_months": float(i % 24), "stage": 2.0} for i in range(80)])
    b = _keys("B", [{"sample_id": f"b{i}", "sex": "M", "age": float(40 + (i * 7) % 30),
                     "time_months": float(i % 24), "stage": 2.0} for i in range(80)])
    esperadas = chance_match_rate(a, b, n_permutations=20)
    assert esperadas > 1.0


def test_parse_series_matrix_phenotype_only_no_devuelve_expresion():
    path = RAW_DIR / "GSE17537_series_matrix.txt.gz"
    if not path.exists():
        pytest.skip("hacen falta los series_matrix en data/raw_geo/")
    pheno_solo, expr_none = parse_series_matrix(path, phenotype_only=True)
    pheno_full, expr = parse_series_matrix(path)
    assert expr_none is None
    assert expr is not None and not expr.empty
    pd.testing.assert_frame_equal(pheno_solo, pheno_full)


def test_regresion_solapamiento_real_gse14333_gse17536():
    """Sobre los datos reales: 129 parejas confirmadas, y solo entre esas dos series.

    Cifra fijada el 2026-09-22 reproduciendo el reporte externo del proyecto
    crc_mra (133 candidatas por clave clínica, 131 inequívocas en su conteo).
    Las coincidencias sueltas contra GSE39582 no deben confirmarse: esa serie
    reporta el seguimiento en meses enteros y colisiona por azar.
    """
    faltan = [c for c in CLINICAL_KEY_FIELDS
              if not (RAW_DIR / f"{c}_series_matrix.txt.gz").exists()]
    if faltan:
        pytest.skip(f"hacen falta los series_matrix de {faltan}")

    pairs = detect_across_cohorts(sorted(CLINICAL_KEY_FIELDS), RAW_DIR, verbose=False)
    confirmadas = pairs[pairs.clasificacion == "confirmada"]

    assert len(confirmadas) == 129
    assert set(zip(confirmadas.cohort_a, confirmadas.cohort_b)) == {("GSE14333", "GSE17536")}
    # el estadio, que no forma parte de la clave, concuerda en todas
    comparables = confirmadas[confirmadas.stage_comparable]
    assert comparables.stage_match.all()
