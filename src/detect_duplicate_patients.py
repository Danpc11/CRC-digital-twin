"""
detect_duplicate_patients.py

Detecta pacientes depositados DOS VECES en series de GEO distintas.

El caso conocido: GSE14333 y GSE17536 incluyen ambas la serie del H. Lee
Moffitt Cancer Center, y ninguna de las dos lo declara en sus metadatos.
Los identificadores no coinciden ('T5266A1' vs 'MCC Patient 1') y la
expresion sola tampoco delata nada, porque cada serie trae su propio RMA:
sobre las matrices crudas dos arrays del MISMO paciente correlacionan
menos (r max 0.796) que dos pacientes distintos dentro de una misma serie
(hasta r 0.96). Solo tras corregir el efecto de lote emergen, y aun asi a
r ~0.88 -- son hibridaciones independientes del mismo tumor, no el mismo
CEL reprocesado.

Lo que si delata el solapamiento es la CLAVE CLINICA: sexo + tiempo de
seguimiento, que GEO reporta al centesimo de mes (3.64, 16.47, 110.79...).
Coincidir al centesimo, repetido decenas de veces, no es casualidad. La
edad y el estadio entran como confirmacion independiente, no como parte
de la clave, porque su precision varia entre series (GSE33113 da la edad
con decimales, GSE17536 en anios enteros) y exigirlos en la clave perderia
parejas reales.

Por que importa: en supervivencia un paciente duplicado aporta su evento
dos veces, el error estandar del Cox sale demasiado pequeno y el p-valor
optimista. Estratificar por cohorte NO lo arregla -- la estratificacion
permite riesgos base distintos por serie, pero el mismo paciente sigue
contribuyendo dos observaciones tratadas como independientes.

USO:
    python3 src/detect_duplicate_patients.py \\
        --cohort GSE14333 --cohort GSE17536 \\
        --output data/duplicate_patients_geo.tsv

Sin --cohort barre las seis series con clave clinica declarada, todas
contra todas. Necesita los series_matrix en data/raw_geo/.
"""

import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_geo_series_matrix import parse_series_matrix

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_geo"

# Tolerancia al comparar tiempos de seguimiento. GEO los da al centesimo
# de mes, pero GSE33113 los da en DIAS y GSE37892 hay que derivarlos de
# dos fechas: convertir introduce un residuo de menos de medio dia.
TIME_TOLERANCE_MONTHS = 0.02
AGE_TOLERANCE_YEARS = 0.5
DAYS_PER_MONTH = 30.4375

# Cuantas veces por encima del ruido esperado tiene que estar el conteo
# observado para tratar un par de cohortes como solapamiento real. Con 5x
# el caso conocido (129 observadas, ~0.1 esperadas) pasa por tres ordenes
# de magnitud, y las coincidencias sueltas contra GSE39582 no pasan.
CHANCE_EXCESS_FACTOR = 5.0

# Nombres de campo por serie. Se declaran a mano porque GEO no impone
# vocabulario: la misma variable se llama 'Gender', 'gender' o 'Sex'
# segun quien deposito la serie.
CLINICAL_KEY_FIELDS = {
    "GSE14333": {
        "sex": "characteristics__Gender",
        "age": "characteristics__Age_Diag",
        "time": "characteristics__DFS_Time",
        "time_unit": "months",
        "stage": "characteristics__DukesStage",
        "stage_scale": "dukes",
    },
    "GSE17536": {
        "sex": "characteristics__gender",
        "age": "characteristics__age",
        "time": "characteristics__dfs_time",
        "time_unit": "months",
        "stage": "characteristics__ajcc_stage",
        "stage_scale": "ajcc",
    },
    "GSE17537": {
        "sex": "characteristics__gender",
        "age": "characteristics__age",
        "time": "characteristics__dfs_time",
        "time_unit": "months",
        "stage": "characteristics__ajcc_stage",
        "stage_scale": "ajcc",
    },
    "GSE33113": {
        "sex": "characteristics__Sex",
        "age": "characteristics__age at diagnosis",
        "time": "characteristics__time to meta or recurrence",
        "time_unit": "days",
        "stage": None,  # la serie entera es estadio II: no discrimina
        "stage_scale": None,
    },
    "GSE37892": {
        "sex": "characteristics__gender",
        "age": "characteristics__age at diagnosis",
        # no deposita una duracion, solo las dos fechas que la producen
        "time_start": "characteristics__date at surgery",
        "time_end": "characteristics__date at last contact",
        "stage": "characteristics__Stage",
        "stage_scale": "ajcc",
    },
    "GSE39582": {
        "sex": "characteristics__Sex",
        "age": "characteristics__age.at.diagnosis (year)",
        "time": "characteristics__rfs.delay",
        "time_unit": "months",
        "stage": "characteristics__tnm.stage",
        "stage_scale": "ajcc",
    },
}

# Conservar SIEMPRE la copia de la cohorte que aparece antes en esta lista.
# GSE17536 gana a GSE14333 por dos razones concretas: trae OS, DSS y DFS
# (GSE14333 solo DFS), y su indicador de recaida es un campo explicito
# ("recurrence"/"no recurrence") en vez del DFS_Cens de GSE14333, que
# codifica 1 = CENSURADO y ya causo una lectura invertida en este proyecto.
DEFAULT_COHORT_PRIORITY = ["GSE17536", "GSE39582", "GSE17537",
                           "GSE37892", "GSE33113", "GSE14333"]

DUKES_TO_AJCC = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}


def _normalize_sex(value):
    """'M', 'm', 'male', 'Male' -> 'M'. Todo lo demas -> None."""
    if value is None or pd.isna(value):
        return None
    letter = str(value).strip().upper()[:1]
    return letter if letter in {"M", "F"} else None


def _to_float(value):
    """Numero tolerante a la coma decimal ('41,6' en GSE33113)."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_stage(value, scale):
    if scale is None or value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if scale == "dukes":
        return DUKES_TO_AJCC.get(text[:1])
    return _to_float(text)


def clinical_key_table(cohort: str, phenotype: pd.DataFrame) -> pd.DataFrame:
    """Extrae sexo / edad / tiempo / estadio normalizados de un fenotipo GEO."""
    spec = CLINICAL_KEY_FIELDS[cohort]
    missing = [spec[k] for k in ("sex", "age") if spec[k] not in phenotype.columns]
    if missing:
        raise ValueError(f"{cohort}: el fenotipo no trae {missing}. "
                         f"Revisa CLINICAL_KEY_FIELDS contra las columnas reales.")

    out = pd.DataFrame(index=phenotype.index)
    out["cohort"] = cohort
    out["sample_id"] = phenotype.index
    out["sex"] = phenotype[spec["sex"]].map(_normalize_sex)
    out["age"] = phenotype[spec["age"]].map(_to_float)

    if spec.get("time"):
        factor = {"months": 1.0, "days": 1.0 / DAYS_PER_MONTH, "years": 12.0}[spec["time_unit"]]
        out["time_months"] = phenotype[spec["time"]].map(_to_float) * factor
    else:
        start = pd.to_datetime(phenotype[spec["time_start"]], errors="coerce")
        end = pd.to_datetime(phenotype[spec["time_end"]], errors="coerce")
        out["time_months"] = (end - start).dt.days / DAYS_PER_MONTH

    # float y no None aunque la serie no traiga estadio: una columna de
    # objetos toda vacia hace que pd.concat cambie de dtype mas adelante
    out["stage"] = (phenotype[spec["stage"]].map(lambda v: _normalize_stage(v, spec["stage_scale"]))
                    if spec.get("stage") else np.nan)
    out["stage"] = out["stage"].astype(float)
    return out.reset_index(drop=True)


def find_duplicate_pairs(keys_a: pd.DataFrame, keys_b: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Parejas entre dos cohortes: mismo sexo, mismo tiempo Y misma edad.

    Devuelve (parejas, n_descartadas_por_edad).

    Los TRES campos son obligatorios, no solo el tiempo. Exigir solo sexo
    y tiempo produce cientos de falsos positivos en cuanto una de las dos
    series reporta el seguimiento con poca precision: GSE39582 lo da en
    meses enteros, asi que choca por azar con cualquier otra serie en los
    valores redondos (640 coincidencias espurias contra GSE17536 con ese
    criterio, ninguna real). El tiempo al centesimo de mes es lo que
    discrimina; la edad es lo que descarta la coincidencia.

    El estadio entra como tercera confirmacion cuando las dos series lo
    traen en una escala convertible, pero no se exige: GSE33113 es de
    estadio unico y no discrimina nada.

    Producto cruzado explicito en vez de un merge sobre tiempo redondeado:
    las cohortes son de cientos de muestras, el costo es irrelevante y un
    umbral de tolerancia no se puede expresar como clave de union sin
    perder las parejas que caen justo en el borde de un bin.
    """
    a = keys_a.dropna(subset=["sex", "time_months", "age"])
    b = keys_b.dropna(subset=["sex", "time_months", "age"])
    if a.empty or b.empty:
        return pd.DataFrame(), 0

    cross = a.merge(b, on="sex", suffixes=("_a", "_b"))
    cross = cross[(cross.time_months_a - cross.time_months_b).abs() <= TIME_TOLERANCE_MONTHS]
    if cross.empty:
        return pd.DataFrame(), 0

    edad_ok = (cross.age_a - cross.age_b).abs() <= AGE_TOLERANCE_YEARS
    n_descartadas = int((~edad_ok).sum())
    cross = cross[edad_ok].copy()
    if cross.empty:
        return pd.DataFrame(), n_descartadas

    cross["stage_comparable"] = cross.stage_a.notna() & cross.stage_b.notna()
    cross["stage_match"] = cross.stage_comparable & (cross.stage_a == cross.stage_b)

    # Una clave que empareja con varias muestras del otro lado no
    # identifica a un paciente: son colisiones de tiempos redondos.
    # Se marcan y se dejan fuera del descarte automatico.
    cross["ambigua"] = (cross.sample_id_a.duplicated(keep=False)
                        | cross.sample_id_b.duplicated(keep=False))

    cross["clasificacion"] = "revisar"
    confirmada = ~cross.ambigua & (cross.stage_match | ~cross.stage_comparable)
    cross.loc[confirmada, "clasificacion"] = "confirmada"

    cols = ["cohort_a", "sample_id_a", "cohort_b", "sample_id_b", "sex",
            "age_a", "age_b", "time_months_a", "time_months_b",
            "stage_a", "stage_b", "stage_match", "stage_comparable",
            "ambigua", "clasificacion"]
    pairs = cross[cols].sort_values(["cohort_a", "cohort_b", "sample_id_a"]).reset_index(drop=True)
    return pairs, n_descartadas


def chance_match_rate(keys_a: pd.DataFrame, keys_b: pd.DataFrame,
                      n_permutations: int = 50, seed: int = 2026) -> float:
    """Cuantas parejas saldrian por azar entre estas dos cohortes.

    Hace falta porque el poder del metodo depende de la PRECISION con que
    cada serie reporta el seguimiento, y no todas lo hacen igual: GSE14333
    y GSE17536 lo dan al centesimo de mes, pero GSE39582 lo da en meses
    enteros. Con meses enteros, dos series grandes coinciden en tiempo
    cientos de veces por puro azar (624 entre GSE17536 y GSE39582), y de
    esas unas cuantas coinciden ademas en edad. Sin una referencia de
    cuanto ruido esperar, esas coincidencias se leerian como duplicados.

    Null: se permuta la edad DENTRO de la cohorte b, rompiendo el vinculo
    edad-tiempo pero conservando ambas distribuciones marginales y, por
    tanto, la tasa de colisiones de tiempo. Permutar las filas completas
    no serviria de nada -- el emparejamiento es por valor, no por posicion,
    asi que reordenar no cambia una sola pareja.
    """
    generator = np.random.default_rng(seed)
    shuffled = keys_b.copy()
    total = 0
    for _ in range(n_permutations):
        shuffled["age"] = generator.permutation(keys_b["age"].to_numpy())
        pairs, _ = find_duplicate_pairs(keys_a, shuffled)
        total += 0 if not len(pairs) else int((pairs.clasificacion == "confirmada").sum())
    return total / n_permutations


def detect_across_cohorts(cohorts: list[str], raw_dir: Path = RAW_DIR,
                          verbose: bool = True) -> pd.DataFrame:
    """Barre todas las parejas de cohortes, cada una contra cada otra."""
    keys = {}
    for cohort in cohorts:
        path = raw_dir / f"{cohort}_series_matrix.txt.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}. Descargalo con el curl documentado en el README.")
        phenotype, _ = parse_series_matrix(path, phenotype_only=True)
        keys[cohort] = clinical_key_table(cohort, phenotype)
        if verbose:
            usable = keys[cohort].dropna(subset=["sex", "time_months", "age"])
            print(f"{cohort}: {len(keys[cohort])} muestras depositadas, "
                  f"{len(usable)} con clave clinica completa")

    found = []
    for first, second in itertools.combinations(cohorts, 2):
        pairs, n_edad = find_duplicate_pairs(keys[first], keys[second])
        n_conf = int((pairs.clasificacion == "confirmada").sum()) if len(pairs) else 0
        esperadas = chance_match_rate(keys[first], keys[second]) if n_conf else 0.0

        # El solapamiento real entre dos series es un fenomeno de BLOQUE:
        # cuando comparten una subcohorte aparecen decenas de parejas de
        # golpe. Un punado de coincidencias sueltas entre dos series
        # grandes es ruido, y la permutacion dice cuanto ruido esperar.
        # Solo si lo observado supera el umbral de azar se autoriza el
        # descarte automatico; si no, las parejas quedan como 'revisar'.
        es_bloque = n_conf >= CHANCE_EXCESS_FACTOR * max(esperadas, 1.0)
        if len(pairs):
            pairs = pairs.copy()
            pairs["parejas_esperadas_por_azar"] = round(esperadas, 2)
            if not es_bloque:
                pairs.loc[pairs.clasificacion == "confirmada", "clasificacion"] = "revisar"
            found.append(pairs)
        if verbose:
            veredicto = "solapamiento real" if es_bloque else "compatible con azar"
            print(f"  {first} vs {second}: {n_conf} coincidencias completas "
                  f"(esperadas por azar {esperadas:.1f}) -> {veredicto}"
                  f"; {n_edad} coincidencias de tiempo descartadas por edad distinta")
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame()


def drop_duplicate_patients(data: pd.DataFrame, pairs: pd.DataFrame,
                            priority: list[str] | None = None,
                            id_col: str = "sample_id",
                            cohort_col: str = "cohort",
                            verbose: bool = True) -> pd.DataFrame:
    """Deja una sola fila por paciente cuando ambas copias estan presentes.

    Identifica las copias SOLO por su GSM, que es identificador unico en
    todo GEO. No por (cohorte, GSM): cada script nombra las cohortes a su
    manera -- pooled_cox_validation.py toma el nombre del argumento
    ('GSE14333') y cox_diagnostics.py lo deriva del directorio
    ('results_external_gse14333'), asi que exigir que el nombre coincida
    con el de la tabla de duplicados hacia que la deduplicacion se
    saltara en silencio en la mitad de los analisis.

    La cohorte sigue haciendo falta, pero solo para decidir cual de las
    dos copias se conserva, y esa se lee de la tabla de parejas.

    Solo actua sobre parejas 'confirmada' y solo cuando LAS DOS copias
    sobrevivieron a los filtros del analisis: si una ya cayo antes (sin
    etiqueta CMS, sin estadio, sin seguimiento) no hay duplicacion que
    corregir y quitar la otra seria perder un paciente real.
    """
    if pairs is None or not len(pairs):
        return data
    priority = priority or DEFAULT_COHORT_PRIORITY
    rank = {name: i for i, name in enumerate(priority)}
    present = set(data[id_col])

    to_drop = set()
    for row in pairs[pairs.clasificacion == "confirmada"].itertuples():
        if row.sample_id_a not in present or row.sample_id_b not in present:
            continue
        # sin prioridad declarada, la cohorte va al final: se descarta ella
        rank_a = rank.get(row.cohort_a, len(rank))
        rank_b = rank.get(row.cohort_b, len(rank))
        to_drop.add(row.sample_id_b if rank_a <= rank_b else row.sample_id_a)

    if not to_drop:
        if verbose:
            print("Deduplicacion: ninguna pareja tiene sus dos copias en esta muestra.")
        return data

    mask = data[id_col].isin(to_drop)
    if verbose:
        detalle = (data.loc[mask, cohort_col].value_counts().to_dict()
                   if cohort_col in data.columns else {})
        print(f"Deduplicacion: {int(mask.sum())} copias retiradas {detalle}; "
              f"se conserva una fila por paciente. La n reportada es de PACIENTES.")
    return data[~mask].copy()


def load_duplicate_pairs(path) -> pd.DataFrame:
    """Lee la tabla que produce este script, validando que trae lo necesario."""
    pairs = pd.read_csv(path, sep="\t")
    required = {"cohort_a", "sample_id_a", "cohort_b", "sample_id_b", "clasificacion"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"'{path}' no parece una tabla de duplicados: faltan {missing}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", action="append", choices=sorted(CLINICAL_KEY_FIELDS),
                        help="Serie a incluir en el barrido; repetir. "
                             "Sin este argumento se barren todas.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR),
                        help="Directorio con los *_series_matrix.txt.gz")
    parser.add_argument("--output", default="data/duplicate_patients_geo.tsv")
    args = parser.parse_args()

    cohorts = args.cohort or sorted(CLINICAL_KEY_FIELDS)
    if len(cohorts) < 2:
        parser.error("hacen falta al menos dos cohortes para buscar duplicados")

    pairs = detect_across_cohorts(cohorts, Path(args.raw_dir))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, sep="\t", index=False)

    print(f"\n{len(pairs)} parejas candidatas en total.")
    if len(pairs):
        print(pairs.clasificacion.value_counts().to_string())
        confirmadas = pairs[pairs.clasificacion == "confirmada"]
        if len(confirmadas):
            comparables = confirmadas[confirmadas.stage_comparable]
            print(f"Estadio concordante en las confirmadas: "
                  f"{int(comparables.stage_match.sum())}/{len(comparables)}")
    print(f"Tabla guardada en: {out_path}")
    print("\nPasala a pooled-cox y cox-diagnostics con --duplicates para "
          "que el Cox cuente pacientes y no muestras.")


if __name__ == "__main__":
    main()
