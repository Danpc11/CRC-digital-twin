"""
prognosis.py

Modulo de pronostico longitudinal: convierte una serie temporal de
mediciones qPCR/RT-qPCR post-quirurgicas en una senal de riesgo
continua, en vez de una clasificacion puntual.

ADVERTENCIA CONCEPTUAL (2026-09-11): el "origen" del espacio de estado
es x=0 en z-score, es decir, el TUMOR PROMEDIO de la cohorte de
calibracion -- no tejido sano ni ausencia de tumor. La identificacion
"cerca del origen = sin enfermedad residual" es una convencion de la
simulacion, no una propiedad biologica del espacio. El panel mide
expresion tisular tumoral; no existe hoy ninguna fuente de datos donde
estos 10 genes se midan longitudinalmente post-cirugia, asi que la
analogia con ctDNA/MRD (DYNAMIC) es conceptual, no operativa.

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

import warnings

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


MIN_BASELINE_FOR_SIGMA = 3  # con menos puntos, sigma basal no es estimable


def detect_recurrence_signal(
    hazard_series: np.ndarray,
    baseline_window: int = 2,
    threshold_sigma: float = 3.0,
    absolute_floor: float = 0.1,
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

    absolute_floor: umbral minimo sobre la media basal. Con
    baseline_window < MIN_BASELINE_FOR_SIGMA la desviacion estandar
    basal no es estimable (con 2 puntos, sigma es la mitad de su
    diferencia -- puro ruido), asi que en ese caso se usa
    mu + absolute_floor y se emite un warning. Con sigma ~ 0 (ventana
    basal plana, ej. todo ceros) se usa el mismo piso absoluto.
    """
    if baseline_window < MIN_BASELINE_FOR_SIGMA:
        warnings.warn(
            f"baseline_window={baseline_window} < {MIN_BASELINE_FOR_SIGMA}: la sigma basal "
            "no es estimable; se usa el umbral absoluto mu + absolute_floor. "
            "Considera mas puntos basales.", stacklevel=2)
    if len(hazard_series) <= baseline_window:
        raise ValueError(
            f"Se necesitan mas de {baseline_window} timepoints para "
            "establecer una ventana basal."
        )

    baseline = hazard_series[:baseline_window]
    mu, sigma = baseline.mean(), baseline.std(ddof=0)

    if baseline_window < MIN_BASELINE_FOR_SIGMA or sigma < 1e-8:
        threshold = mu + absolute_floor
    else:
        threshold = max(mu + threshold_sigma * sigma, mu + absolute_floor)

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
