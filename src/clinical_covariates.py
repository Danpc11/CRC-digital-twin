"""
clinical_covariates.py

Armoniza covariables clinicas entre cohortes que las codifican distinto,
para poder ajustar el modelo de Cox por estadio -- el factor pronostico
dominante en cancer colorrectal.

POR QUE IMPORTA
---------------
El analisis actual del proyecto (pooled_cox_validation.py sin
covariables) reporta el efecto CRUDO del subtipo CMS. Un revisor
preguntara si ese efecto es independiente del estadio o si solo refleja
que los tumores CMS4 tienden a diagnosticarse mas avanzados. Solo el
modelo AJUSTADO responde eso.

EL PROBLEMA DE ARMONIZACION
---------------------------
Las cuatro cohortes usan sistemas distintos:
    GSE39582   tnm.stage      -> 0, 1, 2, 3, 4  (numerico)
    GSE14333   DukesStage     -> A, B, C, D
    GSE17536   ajcc_stage     -> 1, 2, 3, 4 (o texto)
    GSE17537   ajcc_stage     -> idem

La correspondencia Dukes <-> TNM/AJCC es la aceptada clinicamente:
    Dukes A  ~ estadio I     (tumor limitado a la pared)
    Dukes B  ~ estadio II    (invade a traves de la pared, sin ganglios)
    Dukes C  ~ estadio III   (ganglios positivos)
    Dukes D  ~ estadio IV    (metastasis a distancia)

No es una equivalencia exacta (Dukes B agrupa T3 y T4; los sistemas
difieren en subdivisiones), pero es la aproximacion estandar para
analisis agrupados. SE DEBE DECLARAR EXPLICITAMENTE en cualquier
manuscrito -- por eso esta funcion imprime lo que hizo, y por eso el
modelo de Cox se estratifica por cohorte (cada cohorte conserva su
propia funcion de riesgo basal, lo que absorbe parte de la diferencia
residual entre sistemas de estadificacion).

TRATAMIENTO DEL ESTADIO IV
--------------------------
Los pacientes en estadio IV (Dukes D) ya tienen metastasis al
diagnostico -- no tienen un estado "libre de enfermedad" que perder,
asi que su supervivencia libre de recidiva no es comparable. Por
default se EXCLUYEN del analisis de RFS ajustado (drop_stage_iv=True),
que es la practica habitual. En GSE14333 esto es automatico: esos
pacientes ya vienen con DFS_Cens = NA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mapeo a una escala ordinal comun (1-4). Se aceptan las variantes de
# texto y numericas que aparecen en las cuatro cohortes.
STAGE_MAP = {
    # Dukes (GSE14333)
    "a": 1, "b": 2, "c": 3, "d": 4,
    "dukes a": 1, "dukes b": 2, "dukes c": 3, "dukes d": 4,
    # TNM / AJCC numerico
    "1": 1, "2": 2, "3": 3, "4": 4,
    "0": np.nan,   # estadio 0 (in situ): muy pocos casos, no comparable
    # Romanos
    "i": 1, "ii": 2, "iii": 3, "iv": 4,
    "stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4,
    # Formato "AJCC stage X CRC" -- visto en GSE33113 (disease status:
    # "AJCC stage II CRC"), cohorte diseñada como estadio II homogeneo.
    "ajcc stage i crc": 1, "ajcc stage ii crc": 2,
    "ajcc stage iii crc": 3, "ajcc stage iv crc": 4,
    # Subdivisiones AJCC -- se colapsan al estadio principal
    "iia": 2, "iib": 2, "iic": 2,
    "iiia": 3, "iiib": 3, "iiic": 3,
    "iva": 4, "ivb": 4,
    "stage iia": 2, "stage iib": 2, "stage iiia": 3, "stage iiib": 3, "stage iiic": 3,
    # Faltantes explicitos
    "na": np.nan, "n/a": np.nan, "": np.nan, "nan": np.nan, "unknown": np.nan,
}


def harmonize_stage(values: pd.Series, cohort_name: str = "", verbose: bool = True) -> pd.Series:
    """
    Convierte una columna de estadio (Dukes / TNM / AJCC, texto o
    numerica) a una escala ordinal comun 1-4.

    Imprime un reporte de lo que mapeo y lo que no pudo mapear -- no
    falla en silencio, porque un valor no reconocido convertido a NaN
    sin aviso es exactamente el tipo de error que corrompe un analisis
    sin que nadie lo note.
    """
    # Faltantes primero, ANTES de convertir a texto: en dtypes anulables
    # (Int64, string) astype(str) produce el literal "<NA>", que no es un
    # valor "no reconocido" sino un faltante.
    missing = values.isna().to_numpy()
    raw_arr = values.astype(str).str.strip().str.lower().to_numpy(dtype=object)
    # Estadios numericos leidos como float ("2.0") -- pasa en cuanto la
    # columna trae un solo faltante, porque pandas la sube a float. Sin
    # esto, GSE39582 (2 NaN en tnm.stage) quedaba con 0/566 mapeados y el
    # modelo ajustado se caia en silencio (bug encontrado 2026-09-11).
    # Se trabaja posicionalmente (numpy) para no depender del indice de la
    # Series (indices duplicados rompian Series.where), y solo se
    # normalizan enteros finitos representables (evita inf / 1e300).
    numeric = pd.to_numeric(pd.Series(raw_arr), errors="coerce").to_numpy(dtype=float)
    is_int_like = np.isfinite(numeric) & (np.abs(numeric) < 2 ** 53) & (numeric == np.round(numeric))
    raw_arr = np.where(is_int_like, np.where(is_int_like, numeric, 0).astype(np.int64).astype(str), raw_arr)
    raw_arr = np.where(missing, "nan", raw_arr)
    raw = pd.Series(raw_arr, index=values.index, dtype=object)
    mapped = raw.map(STAGE_MAP)

    # "No reconocido" = no esta en STAGE_MAP. Los que si estan pero mapean
    # a NaN a proposito ("0", "unknown"...) no son un error de mapeo.
    unmapped = sorted(set(raw[~raw.isin(STAGE_MAP.keys())].unique()) - {"nan"})
    if verbose:
        prefix = f"[{cohort_name}] " if cohort_name else ""
        n_ok = int(mapped.notna().sum())
        print(f"{prefix}estadio armonizado: {n_ok}/{len(raw)} muestras mapeadas a escala 1-4")
        if unmapped:
            print(f"{prefix}AVISO: valores NO reconocidos (quedan como faltantes): {unmapped}")
            print(f"{prefix}       si alguno es un estadio valido, agregalo a STAGE_MAP "
                  "en clinical_covariates.py")
        dist = mapped.value_counts().sort_index()
        if len(dist):
            print(f"{prefix}distribucion: " +
                  ", ".join(f"E{int(k)}={v}" for k, v in dist.items()))
    return mapped


STAGE_REFERENCE = 2           # estadio II como referencia (el mas frecuente en RFS I-III)
STAGE_DUMMY_COLS = {1: "stage_I", 3: "stage_III", 4: "stage_IV"}


def expand_stage_categorical(
    df: pd.DataFrame,
    stage_col: str = "stage_harmonized",
    reference: int = STAGE_REFERENCE,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convierte el estadio armonizado (1-4) en indicadores 0/1 con estadio
    `reference` como categoria de referencia. Devuelve (df_con_dummies,
    nombres_de_columnas_dummy) -- solo las de niveles PRESENTES en el df
    (si estadio IV se excluyo, no aparece stage_IV).

    POR QUE: meter stage_harmonized como numero (1,2,3) en el Cox asume
    que el salto I->II multiplica el riesgo igual que II->III, y en CRC
    el salto a III es mucho mayor. Un solo coeficiente "por estadio" es
    un modelo mal especificado; el ajuste por estadio deja de ser un
    ajuste real y la afirmacion "CMS conserva su HR tras ajustar por
    estadio" queda sin sustento. Con indicadores, cada estadio tiene su
    propio HR frente a la referencia. Los NaN se conservan como NaN en
    todas las dummies para que dropna() los excluya.
    """
    out = df.copy()
    st = pd.to_numeric(out[stage_col], errors="coerce")
    cols = []
    for level, name in STAGE_DUMMY_COLS.items():
        if level == reference or not (st == level).any():
            continue
        out[name] = np.where(st.isna(), np.nan, (st == level).astype(float))
        cols.append(name)
    return out, cols


MIN_EVENTS_PER_STAGE_LEVEL = 5


def collapse_sparse_stage_levels(
    df: pd.DataFrame, stage_cols: list[str], event_col: str,
    min_events: int = MIN_EVENTS_PER_STAGE_LEVEL, verbose: bool = True,
) -> list[str]:
    """
    Descarta las dummies de estadio con muy pocos eventos, devolviendo la
    lista de columnas que SI deben entrar al modelo.

    POR QUE: con el evento bien codificado, casi ningun paciente en
    estadio I recae. Entonces stage_I determina practicamente la ausencia
    de evento ("complete separation"): su coeficiente tiende a -infinito,
    la verosimilitud no tiene maximo unico y lifelines avisa
    ConvergenceWarning con norm(delta) alto en cada ajuste. El estimador
    de ese nivel no es interpretable y desestabiliza el resto del modelo.

    Descartar la dummy equivale a fusionar ese estadio con la categoria
    de referencia (estadio II). Clinicamente es razonable: I y II son
    ambos enfermedad sin afectacion ganglionar, y la mayoria de los
    estudios de RFS los agrupan. Se imprime siempre que ocurre, para que
    quede en el reporte que el modelo compara "I+II vs III".
    """
    keep, dropped = [], []
    for col in stage_cols:
        mask = pd.to_numeric(df[col], errors="coerce") == 1
        n_events = int(pd.to_numeric(df.loc[mask, event_col], errors="coerce").fillna(0).sum())
        if n_events < min_events:
            dropped.append((col, int(mask.sum()), n_events))
        else:
            keep.append(col)
    if dropped and verbose:
        for col, n, ev in dropped:
            print(f"AVISO: '{col}' tiene n={n} con solo {ev} eventos (<{min_events}); "
                  "se fusiona con la categoria de referencia (estadio II) para evitar "
                  "separacion completa. El modelo compara estadio I+II vs III.")
    return keep


def prepare_covariates(
    df: pd.DataFrame,
    stage_col: str = "stage",
    drop_stage_iv: bool = True,
    cohort_col: str = "cohort",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepara un dataframe combinado para el modelo de Cox ajustado:
    armoniza el estadio por cohorte y (por default) excluye estadio IV.
    """
    out = df.copy()
    if stage_col not in out.columns:
        raise ValueError(
            f"No se encontro la columna '{stage_col}'. Los scripts build_*_dataset.py "
            "deben incluirla -- ver la opcion de estadio en cada uno."
        )

    harmonized = []
    if cohort_col in out.columns:
        for cohort, sub in out.groupby(cohort_col, sort=False):
            harmonized.append(harmonize_stage(sub[stage_col], str(cohort), verbose))
        out["stage_harmonized"] = pd.concat(harmonized).reindex(out.index)
    else:
        out["stage_harmonized"] = harmonize_stage(out[stage_col], verbose=verbose)

    n_before = len(out)
    if drop_stage_iv:
        n_iv = int((out["stage_harmonized"] == 4).sum())
        out = out[out["stage_harmonized"] != 4]
        if verbose and n_iv:
            print(f"\nExcluidos {n_iv} pacientes en estadio IV: ya tienen metastasis al "
                  "diagnostico, su supervivencia libre de recidiva no es comparable.")

    n_missing = int(out["stage_harmonized"].isna().sum())
    if verbose and n_missing:
        print(f"AVISO: {n_missing} pacientes sin estadio utilizable -- se excluiran del "
              "modelo ajustado (pero SI aparecen en el modelo crudo).")

    if verbose:
        print(f"\nn tras preparacion de covariables: {len(out)} (de {n_before})")
    return out
