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

from clinical_selection import cms_continuous_profile


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
    predicted, scores, profiles = [], [], []
    for _, row in df[gene_cols].iterrows():
        values = row.to_numpy()
        label, corr = risk_score_from_expression(values, patterns)
        predicted.append(label)
        scores.append(corr)
        profiles.append(cms_continuous_profile(values, patterns))
    out["predicted_cms"] = predicted
    out["classification_confidence"] = scores
    out["cms_interpretation"] = [p["interpretation"] for p in profiles]
    out["cms_primary_tendency"] = [p["primary"] for p in profiles]
    out["cms_secondary_tendency"] = [p["secondary"] for p in profiles]
    out["cms_margin"] = [p["margin"] for p in profiles]
    out["cms_entropy"] = [p["entropy"] for p in profiles]
    for label in patterns:
        short = label.split("_")[0].lower()
        out[f"{short}_tendency"] = [p["scores"].get(label, 0.0) for p in profiles]
    return out


def validate_survival_by_subtype(
    scored_df: pd.DataFrame,
    duration_col: str = "relapse_free_months",
    event_col: str = "relapse_event",
    endpoint_label: str = "supervivencia libre de recidiva (RFS)",
    group_col: str = "predicted_cms",
) -> dict:
    """
    Corre Kaplan-Meier estratificado por grupo y el test log-rank
    multivariado (equivalente a comparar >2 grupos).

    group_col: columna a usar para estratificar -- por default el
    subtipo RECLASIFICADO por el modelo ("predicted_cms"), pero puede
    pasarse "cms_label" para validar contra la etiqueta CMS OFICIAL del
    consorcio como linea base. Esto separa dos preguntas distintas: (1)
    existe asociacion CMS-supervivencia en esta cohorte/endpoint, y (2)
    el panel reducido del modelo la recupera. Un resultado negativo con
    predicted_cms y positivo con cms_label apunta a perdida de senal
    por reduccion de panel, no a ausencia real de asociacion.

    endpoint_label: descripcion humana del desenlace que se esta
    validando -- IMPORTANTE especificarlo correctamente (supervivencia
    global / OS no es lo mismo que supervivencia libre de recidiva /
    RFS; son desenlaces clinicamente distintos, no intercambiables).

    Devuelve un diccionario con el resultado del test estadistico y los
    fitters de KM por grupo (para graficar despues).
    """
    required = {duration_col, event_col, group_col}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para validacion: {missing}")

    clean = scored_df.dropna(subset=[duration_col, event_col, group_col])
    groups = clean[group_col].unique()

    if len(groups) < 2:
        raise ValueError(
            "Se necesitan al menos 2 grupos con datos de supervivencia "
            "para correr el log-rank test."
        )

    result = multivariate_logrank_test(
        clean[duration_col], clean[group_col], clean[event_col]
    )

    fitters = {}
    for label in groups:
        subset = clean[clean[group_col] == label]
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
        "endpoint_label": endpoint_label,
        "group_col": group_col,
    }


def interpret_validation_result(result: dict) -> str:
    """
    Traduce el resultado estadistico a una conclusion en lenguaje llano,
    sin sobreclamar. p < 0.05 no es "el modelo funciona"; es evidencia
    de que el subtipo predicho separa curvas de supervivencia en ESTA
    cohorte para ESTE endpoint especifico, lo cual es necesario pero no
    suficiente para uso clinico.
    """
    p = result["logrank_p_value"]
    n = result["n_patients"]
    endpoint = result.get("endpoint_label", "supervivencia (endpoint no especificado)")
    group_col = result.get("group_col", "predicted_cms")
    group_descriptions = {
        "cms_label": "etiqueta CMS OFICIAL del consorcio (linea base)",
        "predicted_cms": "subtipo RECLASIFICADO por correlacion con el panel reducido",
        "modern_hopfield_cms": "subtipo recuperado dinamicamente por Modern Hopfield",
    }
    group_desc = group_descriptions.get(group_col, f"grupos de la columna '{group_col}'")
    lines = [
        f"Endpoint: {endpoint}",
        f"Agrupacion: {group_desc}",
        f"n = {n} pacientes con datos completos, "
        f"{result['n_groups']} grupos.",
        f"log-rank p = {p:.4g}",
    ]
    if p < 0.05:
        subject = "La agrupacion CMS oficial" if group_col == "cms_label" else "El subtipo predicho por el modelo"
        lines.append(
            f"{subject} separa significativamente las "
            f"curvas de {endpoint} en esta cohorte. Esto es evidencia de "
            "utilidad pronostica para ESTE endpoint especificamente -- NO "
            "generaliza automaticamente a otro desenlace (ej. supervivencia "
            "global no implica lo mismo que supervivencia libre de recidiva), "
            "y NO es lo mismo que validacion clinica, que requeriria "
            "replicacion en una cohorte confirmatoria que no haya intervenido "
            "en decisiones del modelo antes de cualquier uso mas alla de investigacion."
        )
    else:
        lines.append(
            f"No hay separacion significativa de {endpoint} por subtipo "
            "predicho en esta cohorte. Esto no invalida el modelo "
            "necesariamente -- puede reflejar tamano de muestra insuficiente, "
            "calibracion pobre, o que el panel de genes elegido no captura "
            "senal pronostica para este endpoint -- pero significa que el "
            "modelo NO tiene, todavia, evidencia de utilidad pronostica para "
            "este desenlace especifico."
        )
    return "\n".join(lines)
