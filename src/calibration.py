"""
calibration.py

Calibra los patrones de atractor (CMS_PATTERNS) contra datos reales
etiquetados, en vez de los placeholders hechos a mano en
attractor_model.py.

ESQUEMA DE DATOS ESPERADO (TSV, no CSV -- ver preferencia estandar del
proyecto):

    sample_id   cms_label            GENE1   GENE2   ...  relapse_free_months  relapse_event
    TCGA-01     CMS1_MSI_immune      4.21    -1.02   ...  38.2                 0
    TCGA-02     CMS2_canonical_WNT   -0.88   3.55    ...  12.1                 1
    ...

Requisitos minimos de columnas:
    - sample_id: identificador unico
    - cms_label: uno de CMS1_MSI_immune / CMS2_canonical_WNT /
      CMS3_metabolic / CMS4_mesenchymal (o "none"/NA para no clasificado
      -- el consorcio original deja ~13-15% de muestras sin clasificar
      limpiamente, no se deben forzar a una etiqueta)
    - columnas de genes: cualquier subconjunto, no tiene que ser
      exactamente el panel placeholder de 8 genes de attractor_model.py
    - relapse_free_months / relapse_event: OPCIONALES, solo necesarias
      para el modulo de validacion de supervivencia (survival_validation.py)

FUENTES RECOMENDADAS PARA CONSTRUIR ESTE TSV:
    - Etiquetas CMS: consorcio CMS, Synapse syn2623706 (Guinney et al. 2015)
    - Expresion + supervivencia: GSE39582 (GEO, Marisa et al. 2013)
    - Alternativa/validacion externa: GSE17536, GSE17537
    - Mutaciones/MSI: TCGA-COAD/READ via GDC o cBioPortal

Este modulo NO descarga datos -- opera sobre el TSV ya construido por
el usuario a partir de esas fuentes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VALID_CMS_LABELS = [
    "CMS1_MSI_immune",
    "CMS2_canonical_WNT",
    "CMS3_metabolic",
    "CMS4_mesenchymal",
]

MIN_SAMPLES_PER_CLASS = 30  # aviso, no bloqueo -- ver warn_low_n


def load_labeled_dataset(path: str | Path) -> pd.DataFrame:
    """Carga el TSV de calibracion y valida columnas minimas."""
    df = pd.read_csv(path, sep="\t")
    if "sample_id" not in df.columns or "cms_label" not in df.columns:
        raise ValueError(
            "El TSV debe tener al menos las columnas 'sample_id' y 'cms_label'. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    unknown = set(df["cms_label"].dropna().unique()) - set(VALID_CMS_LABELS) - {"none"}
    if unknown:
        raise ValueError(
            f"Etiquetas CMS no reconocidas: {unknown}. "
            f"Esperadas: {VALID_CMS_LABELS + ['none']}"
        )
    return df


def infer_gene_columns(df: pd.DataFrame, non_gene_cols: set[str] | None = None) -> list[str]:
    """
    Infiere las columnas de genes como todas las columnas numericas que
    no son metadata conocida (sample_id, cms_label, columnas de
    supervivencia, covariables clinicas, etc).

    CUIDADO -- ESTA LISTA ES CRITICA: cualquier columna numerica que NO
    este aqui se tratara como un gen, se z-scoreara y entrara a la
    calibracion como un rasgo mas del panel. Eso ya paso una vez con
    'stage' (estadio clinico numerico en GSE17536/17537): contamino los
    patrones calibrados sin ningun error visible, solo p-valores que
    cambiaban sin explicacion. Al agregar cualquier covariable clinica
    nueva a los datasets, agregarla TAMBIEN aqui.
    """
    non_gene_cols = non_gene_cols or {
        "sample_id", "cms_label", "cohort",
        # supervivencia
        "relapse_free_months", "relapse_event",
        "overall_survival_months", "death_event",
        # covariables clinicas
        "stage", "stage_harmonized", "age", "sex", "gender",
        "tumor_location", "adjuvant_chemo",
        # estatus molecular (no expresion)
        "kras_status", "braf_status", "msi_status",
        # salidas del propio pipeline
        "predicted_cms", "classification_confidence",
    }
    candidate = [c for c in df.columns if c not in non_gene_cols]
    numeric = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def zscore_genes(
    df: pd.DataFrame, gene_cols: list[str], stats: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """
    Normaliza cada gen a z-score.

    stats: opcional, {gen: (media, sigma)} -- si se da, usa estos valores
    CONGELADOS (de la cohorte de calibracion) en vez de recalcular
    media/desviacion del propio df. Necesario para clasificar un
    PACIENTE INDIVIDUAL: con n=1, la varianza de una sola muestra es
    exactamente cero, z-score sin referencia externa es matematicamente
    imposible -- ver load_gene_reference_stats().

    Para cohortes (n grande, ej. validacion externa), el comportamiento
    por defecto (auto-normalizar cada cohorte contra si misma, stats=None)
    sigue siendo el correcto: ayuda a corregir efectos de lote/plataforma
    entre estudios. NO forzar stats congelados ahi tambien.
    """
    z = df.copy()
    n = len(df)
    for gene in gene_cols:
        if stats is not None and gene in stats:
            mu, sigma = stats[gene]
            if sigma == 0 or np.isnan(sigma):
                raise ValueError(
                    f"El gen '{gene}' tiene sigma de referencia congelado invalido "
                    "(cero o NaN) -- revisa el archivo de calibracion."
                )
        else:
            if n < 10:
                print(
                    f"AVISO: normalizando '{gene}' con solo {n} muestra(s) SIN "
                    "referencia externa congelada -- estadisticamente inestable "
                    "o directamente imposible si n=1. Si estas clasificando un "
                    "paciente individual, pasa 'stats' de "
                    "load_gene_reference_stats() en vez de dejar que se "
                    "autocalcule."
                )
            mu = df[gene].mean()
            sigma = df[gene].std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                raise ValueError(
                    f"El gen '{gene}' tiene desviacion estandar cero o NaN en "
                    f"esta muestra/cohorte (n={n}). Si estas clasificando UN "
                    "paciente individual, esto es ESPERADO (la varianza de un "
                    "solo punto es cero por definicion) -- necesitas pasar "
                    "'stats' congelados de la cohorte de calibracion "
                    "(load_gene_reference_stats), no recalcular desde este "
                    "archivo de una sola muestra."
                )
        z[gene] = (df[gene] - mu) / sigma
    return z


def warn_low_n(df: pd.DataFrame) -> dict[str, int]:
    """Reporta el tamano de muestra por clase CMS y marca los que estan bajos."""
    counts = df["cms_label"].value_counts().to_dict()
    for label in VALID_CMS_LABELS:
        n = counts.get(label, 0)
        if n < MIN_SAMPLES_PER_CLASS:
            print(
                f"AVISO: solo {n} muestras para {label} "
                f"(minimo recomendado: {MIN_SAMPLES_PER_CLASS}). "
                "El centroide calibrado para esta clase sera inestable."
            )
    return counts


def compute_gene_stats(df: pd.DataFrame, gene_cols: list[str]) -> dict[str, tuple[float, float]]:
    """
    Media y desviacion estandar por gen, para CONGELAR como referencia
    de normalizacion -- se calculan una vez sobre la cohorte de
    calibracion (con n grande, estadisticamente estables) y se guardan
    junto con los patrones. Necesarias para clasificar un paciente
    individual despues (n=1, donde no se puede calcular varianza
    propia).
    """
    return {g: (float(df[g].mean()), float(df[g].std(ddof=0))) for g in gene_cols}


def calibrate_patterns_from_data(
    df: pd.DataFrame,
    gene_cols: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Calcula los patrones de atractor (centroides por CMS) a partir de
    datos reales etiquetados.

    Devuelve:
        patterns: {cms_label: np.ndarray} -- centroide z-score por clase
        gene_cols: orden de genes usado (para reconstruir vectores luego)
    """
    gene_cols = gene_cols or infer_gene_columns(df)
    if len(gene_cols) == 0:
        raise ValueError("No se encontraron columnas de genes numericas en el dataframe.")

    warn_low_n(df)
    z = zscore_genes(df, gene_cols)

    patterns = {}
    for label in VALID_CMS_LABELS:
        subset = z[z["cms_label"] == label]
        if len(subset) == 0:
            raise ValueError(f"No hay muestras para la clase {label} en el dataset.")
        centroid = subset[gene_cols].mean(axis=0).to_numpy()
        patterns[label] = centroid

    return patterns, gene_cols


def save_calibrated_patterns(
    patterns: dict[str, np.ndarray], gene_cols: list[str], path: str | Path,
    gene_stats: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Guarda los patrones calibrados en TSV (gen x subtipo).

    gene_stats: opcional -- si se da (ver compute_gene_stats), se
    guardan como dos columnas extra (_ref_mean, _ref_std) en el mismo
    archivo. Permite clasificar un paciente individual despues sin
    tener que recalcular estadisticas sobre n=1 (ver
    load_gene_reference_stats / zscore_genes).
    """
    df = pd.DataFrame(patterns, index=gene_cols)
    if gene_stats is not None:
        df["_ref_mean"] = [gene_stats[g][0] for g in gene_cols]
        df["_ref_std"] = [gene_stats[g][1] for g in gene_cols]
    df.index.name = "gene"
    df.to_csv(path, sep="\t")
    print(f"Patrones calibrados guardados en: {path}")
    if gene_stats is not None:
        print("  (incluye estadisticas de referencia -- permite clasificar pacientes individuales)")


def load_calibrated_patterns(path: str | Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Carga patrones calibrados previamente guardados. Firma sin cambios
    (2-tuple) para no romper a los llamadores existentes -- las
    columnas _ref_mean/_ref_std (si existen) se excluyen aqui de los
    patrones CMS y se leen aparte con load_gene_reference_stats().
    """
    df = pd.read_csv(path, sep="\t", index_col="gene")
    gene_cols = list(df.index)
    pattern_cols = [c for c in df.columns if not c.startswith("_ref_")]
    patterns = {col: df[col].to_numpy() for col in pattern_cols}
    return patterns, gene_cols


def load_gene_reference_stats(path: str | Path) -> dict[str, tuple[float, float]] | None:
    """
    Carga las estadisticas de referencia congeladas (media, sigma por
    gen) de un archivo de patrones calibrados, si las tiene.

    Devuelve None si el archivo es de una version anterior sin estas
    columnas (compatibilidad hacia atras).
    """
    df = pd.read_csv(path, sep="\t", index_col="gene")
    if "_ref_mean" not in df.columns or "_ref_std" not in df.columns:
        return None
    return {gene: (float(row["_ref_mean"]), float(row["_ref_std"])) for gene, row in df.iterrows()}
