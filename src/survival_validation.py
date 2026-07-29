"""
survival_validation.py

Valida el modelo calibrado contra desenlaces REALES de supervivencia
(relapse_free_months, relapse_event), no solo contra concordancia de
subtipo. Esta es la prueba de fuego: un modelo puede clasificar CMS
perfectamente y aun asi no aportar nada pronostico si no separa curvas
de supervivencia.

Requiere columnas relapse_free_months y relapse_event en el TSV de
entrada (ver calibration.py para el esquema completo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def risk_score_from_expression(
    row: np.ndarray, patterns: dict[str, np.ndarray]
) -> tuple[str, float]:
    """
    Clasifica una fila de expresion (ya z-scoreada) por correlacion
    maxima con los patrones calibrados. Reutiliza la misma logica que
    classify_state en attractor_model.py pero desacoplada del modelo
    dinamico, para poder correr sobre un dataframe completo sin
    integrar ODEs (mas rapido para validacion en cohortes grandes).
    """
    if np.linalg.norm(row) < 1e-8:
        return "none", 0.0
    correlations = {
        label: float(np.corrcoef(row, p)[0, 1]) for label, p in patterns.items()
    }
    best = max(correlations, key=correlations.get)
    return best, correlations[best]


def score_cohort(
    df: pd.DataFrame, gene_cols: list[str], patterns: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Asigna subtipo predicho y score de correlacion a cada fila del dataframe."""
    out = df.copy()
    predicted, scores = [], []
    for _, row in df[gene_cols].iterrows():
        label, corr = risk_score_from_expression(row.to_numpy(), patterns)
        predicted.append(label)
        scores.append(corr)
    out["predicted_cms"] = predicted
    out["classification_confidence"] = scores
    return out


def validate_survival_by_subtype(
    scored_df: pd.DataFrame,
    duration_col: str = "relapse_free_months",
    event_col: str = "relapse_event",
) -> dict:
    """
    Corre Kaplan-Meier estratificado por subtipo predicho y el test
    log-rank multivariado (equivalente a comparar >2 grupos).

    Devuelve un diccionario con el resultado del test estadistico y los
    fitters de KM por grupo (para graficar despues).
    """
    required = {duration_col, event_col, "predicted_cms"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para validacion: {missing}")

    clean = scored_df.dropna(subset=[duration_col, event_col])
    groups = clean["predicted_cms"].unique()

    if len(groups) < 2:
        raise ValueError(
            "Se necesitan al menos 2 grupos de subtipo predicho con datos "
            "de supervivencia para correr el log-rank test."
        )

    result = multivariate_logrank_test(
        clean[duration_col], clean["predicted_cms"], clean[event_col]
    )

    fitters = {}
    for label in groups:
        subset = clean[clean["predicted_cms"] == label]
        if len(subset) < 5:
            continue  # muy pocos para un KM fiable
        kmf = KaplanMeierFitter()
        kmf.fit(subset[duration_col], subset[event_col], label=label)
        fitters[label] = kmf

    return {
        "logrank_p_value": result.p_value,
        "logrank_statistic": result.test_statistic,
        "n_groups": len(groups),
        "n_patients": len(clean),
        "km_fitters": fitters,
    }


def interpret_validation_result(result: dict) -> str:
    """
    Traduce el resultado estadistico a una conclusion en lenguaje llano,
    sin sobreclamar. p < 0.05 no es "el modelo funciona"; es evidencia
    de que el subtipo predicho separa curvas de supervivencia en ESTA
    cohorte, lo cual es necesario pero no suficiente para uso clinico.
    """
    p = result["logrank_p_value"]
    n = result["n_patients"]
    lines = [
        f"n = {n} pacientes con datos de supervivencia completos, "
        f"{result['n_groups']} grupos de subtipo predicho.",
        f"log-rank p = {p:.4g}",
    ]
    if p < 0.05:
        lines.append(
            "El subtipo predicho por el modelo separa significativamente las "
            "curvas de supervivencia en esta cohorte. Esto es evidencia de "
            "utilidad pronostica -- NO es lo mismo que validacion clinica, "
            "que requeriria replicacion en cohorte externa independiente "
            "(ej. GSE17536/GSE17537) antes de cualquier uso mas alla de "
            "investigacion."
        )
    else:
        lines.append(
            "No hay separacion significativa de supervivencia por subtipo "
            "predicho en esta cohorte. Esto no invalida el modelo "
            "necesariamente -- puede reflejar tamano de muestra insuficiente, "
            "calibracion pobre, o que el panel de genes elegido no captura "
            "senal pronostica -- pero significa que el modelo NO tiene, "
            "todavia, evidencia de utilidad pronostica."
        )
    return "\n".join(lines)
