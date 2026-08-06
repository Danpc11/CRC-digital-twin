"""
prognosis.py

Modulo de pronostico longitudinal: convierte una serie temporal de
mediciones qPCR/RT-qPCR post-quirurgicas en una senal de riesgo
continua, en vez de una clasificacion puntual.

Logica clinica que formaliza (ver conversacion sobre DYNAMIC trial):
    - Vector de estado cerca de cero a lo largo del tiempo -> sin
      enfermedad residual detectable -> buen pronostico
      (analogo a pacientes ctDNA-negativos en DYNAMIC, ~92-97% RFS a 3 anios)
    - Vector que se aleja del origen hacia un atractor -> reaparicion de
      senal molecular -> alerta de recurrencia, con el atractor
      especifico dando pronostico diferencial

IMPORTANTE: esto es un esqueleto de la LOGICA, no un modelo calibrado.
La funcion hazard_from_trajectory() usa la norma del vector de estado
como proxy de riesgo -- una eleccion razonable pero arbitraria hasta
que se calibre contra datos reales de seguimiento longitudinal + tiempo
a recurrencia (que no existen en fuentes publicas facilmente accesibles;
DYNAMIC/GALAXY no son datos abiertos). Sin esa calibracion, el output
de este modulo es ORDINAL (mas alto = mas riesgo relativo dentro del
mismo paciente a lo largo del tiempo), NO una probabilidad calibrada
de recurrencia.
"""

from __future__ import annotations

import numpy as np


def hazard_from_trajectory(x_series: np.ndarray) -> np.ndarray:
    """
    x_series: array (n_genes, n_timepoints) -- una medicion por
    timepoint de seguimiento post-quirurgico.

    Devuelve un score de riesgo ORDINAL por timepoint (no calibrado a
    probabilidad), proporcional a la distancia del vector de estado al
    origen (estado "sin enfermedad residual").
    """
    if x_series.ndim != 2:
        raise ValueError("x_series debe ser un array 2D (n_genes, n_timepoints)")
    return np.linalg.norm(x_series, axis=0)


def detect_recurrence_signal(
    hazard_series: np.ndarray,
    baseline_window: int = 2,
    threshold_sigma: float = 3.0,
) -> tuple[bool, int | None]:
    """
    Deteccion simple de senal de alerta: compara cada timepoint contra
    la media + threshold_sigma * desviacion estandar de una ventana
    basal (los primeros `baseline_window` puntos, tipicamente las
    mediciones inmediatamente post-quirurgicas donde se espera
    enfermedad residual minima).

    Devuelve (alerta_detectada, indice_del_primer_timepoint_de_alerta).

    NOTA: threshold_sigma=3.0 es un valor de partida conservador
    tipico en control estadistico de procesos, NO esta calibrado
    contra datos clinicos de este contexto especifico.
    """
    if len(hazard_series) <= baseline_window:
        raise ValueError(
            f"Se necesitan mas de {baseline_window} timepoints para "
            "establecer una ventana basal."
        )

    baseline = hazard_series[:baseline_window]
    mu, sigma = baseline.mean(), baseline.std(ddof=0)

    if sigma < 1e-8:
        # Ventana basal perfectamente plana (ej. todos ceros) -- usar
        # umbral absoluto pequenio en vez de dividir por sigma=0
        threshold = mu + 0.1
    else:
        threshold = mu + threshold_sigma * sigma

    for i in range(baseline_window, len(hazard_series)):
        if hazard_series[i] > threshold:
            return True, i

    return False, None


def summarize_patient_trajectory(
    t_points: np.ndarray, x_series: np.ndarray, baseline_window: int = 2
) -> dict:
    """Resumen de alto nivel de una trayectoria de seguimiento de un paciente."""
    hazard = hazard_from_trajectory(x_series)
    alert, idx = detect_recurrence_signal(hazard, baseline_window=baseline_window)
    return {
        "t": t_points,
        "hazard_series": hazard,
        "alert_detected": alert,
        "alert_timepoint": t_points[idx] if idx is not None else None,
        "final_hazard": hazard[-1],
    }
