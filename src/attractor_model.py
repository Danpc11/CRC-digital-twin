"""
attractor_model.py

Modelo dinamico de atractores para los subtipos moleculares consensuados
(Consensus Molecular Subtypes, CMS1-4) de cancer colorrectal.

Formalismo:
    Red tipo Hopfield continua sobre un vector de estado x en R^N, donde
    cada componente representa el nivel de expresion normalizado
    (z-score) de un gen marcador. Cada subtipo CMS se codifica como un
    patron objetivo p_mu en R^N, y la matriz de acoplamiento W se
    construye con la regla de proyeccion (pseudo-inversa de
    Personnaz-Guyon-Dreyfus) para que cada patron sea un punto fijo
    EXACTO del sistema linealizado, evitando el limite de capacidad de
    la regla de Hebb clasica (~0.14 N patrones).

    Dinamica:
        dx/dt = -x + W @ tanh(beta * x) + I_driver + I_noise

    donde I_driver es un termino de forzamiento constante que representa
    el sesgo introducido por mutaciones conductoras (drivers), e
    I_noise es ruido gaussiano opcional (dinamica de Langevin) para
    estudiar transiciones estocasticas entre cuencas de atraccion.

NOTA DE ALCANCE: los patrones p_mu de CMS_PATTERNS abajo son
PLACEHOLDERS cualitativos sobre el panel actual de 10 genes (ver
docstring de CMS_PATTERNS), pensados para demos y tests de la
dinamica. Para cualquier analisis real se usan los patrones
CALIBRADOS contra GSE39582 (calibration.py), cargados con
load_calibrated_patterns() -- nunca estos placeholders.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------
# 1. Definicion del espacio de estado: genes marcadores por subtipo
# ---------------------------------------------------------------------

# Panel actual de 10 genes, todos medibles por RT-qPCR (congelado, ver
# PROJECT_STATUS.md). MLH1 va PRIMERO y con signo negativo en CMS1: la
# senal real es BAJA expresion (silenciamiento epigenetico de MLH1 como
# causa de MSI esporadica, ver CHANGELOG) -- no alta.
GENES = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]
N = len(GENES)

CMS_LABELS = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]

# Patrones objetivo (placeholders cualitativos -- para analisis real se
# usan los patrones calibrados de calibration.py, nunca estos).
# Orden de columnas = GENES. Cada fila = patron atractor de un subtipo.
# Signos por eje, consistentes con los centroides de synthetic_data.py:
#   CMS1: MLH1 BAJO (silenciamiento -> MSI), GNLY/USP18 altos (inmune)
#   CMS2: MYC/AXIN2 altos (WNT)
#   CMS3: FABP1/CPS1/SI altos (metabolico)
#   CMS4: VIM/TGFB1 altos (mesenquimal)
CMS_PATTERNS = {
    "CMS1_MSI_immune":    np.array([-0.9,  0.9,  0.9, -0.6, -0.5, -0.4, -0.4, -0.4, -0.6, -0.4]),
    "CMS2_canonical_WNT": np.array([-0.5, -0.4, -0.4,  0.9,  0.9, -0.4, -0.3, -0.3, -0.5, -0.4]),
    "CMS3_metabolic":     np.array([-0.4, -0.3, -0.3, -0.4, -0.3,  0.9,  0.9,  0.9, -0.4, -0.3]),
    "CMS4_mesenchymal":   np.array([-0.5, -0.4, -0.4, -0.5, -0.4, -0.3, -0.3, -0.3,  0.9,  0.9]),
}

P = np.stack([CMS_PATTERNS[label] for label in CMS_LABELS], axis=1)  # (N, 4)


# ---------------------------------------------------------------------
# 2. Matriz de acoplamiento: regla de proyeccion (pseudo-inversa)
# ---------------------------------------------------------------------

def projection_weight_matrix(patterns: np.ndarray) -> np.ndarray:
    """
    Construye W tal que W @ p_mu = p_mu para cada patron (punto fijo
    exacto del sistema linealizado dx/dt = -x + W x), usando la regla
    de proyeccion de Personnaz-Guyon-Dreyfus (Kohonen, 1972).

    patterns: array (N, M) con M patrones como columnas.

    Usa pseudo-inversa (Moore-Penrose), no solve exacto -- si los M
    patrones no son linealmente independientes (puede pasar con
    cohortes de calibracion chicas/desbalanceadas, donde dos
    centroides empiricos terminan casi colineales), gram = P^T P sale
    singular o numericamente mal condicionada, y np.linalg.solve
    revienta con LinAlgError. La formula P(P^T P)^+ P^T sigue siendo
    el proyector ortogonal correcto sobre el subespacio generado por
    los patrones incluso cuando son dependientes -- preserva W @ p_mu
    = p_mu exactamente para cada patron, sin importar el rango.
    """
    gram = patterns.T @ patterns
    W = patterns @ np.linalg.pinv(gram) @ patterns.T
    return W


W = projection_weight_matrix(P)


# ---------------------------------------------------------------------
# 3. Terminos de forzamiento por mutacion conductora (driver)
# ---------------------------------------------------------------------

# Mapeo mutacion -> vector de sesgo en el espacio de genes marcadores.
# Estos tambien son placeholders biologicamente motivados:
#   - Perdida de MLH1/MSH2 (MSI) empuja hacia CMS1
#   - APC mutante (activacion WNT constitutiva) empuja hacia CMS2
#   - KRAS mutante empuja hacia CMS3
#   - Perdida de SMAD4 / activacion TGF-beta empuja hacia CMS4
DRIVER_BIAS = {
    "MSI_high":      0.6 * CMS_PATTERNS["CMS1_MSI_immune"],
    "APC_mut":       0.6 * CMS_PATTERNS["CMS2_canonical_WNT"],
    "KRAS_mut":      0.6 * CMS_PATTERNS["CMS3_metabolic"],
    "SMAD4_loss":    0.6 * CMS_PATTERNS["CMS4_mesenchymal"],
    "none":          np.zeros(N),
}


# ---------------------------------------------------------------------
# 4. Dinamica
# ---------------------------------------------------------------------

def dynamics(t, x, W, I_driver, beta=2.0, noise_sigma=0.0, rng=None):
    dxdt = -x + W @ np.tanh(beta * x) + I_driver
    if noise_sigma > 0.0:
        rng = rng or np.random.default_rng()
        dxdt = dxdt + noise_sigma * rng.standard_normal(x.shape)
    return dxdt


def simulate_patient(
    driver: str,
    x0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 20.0),
    n_points: int = 400,
    beta: float = 2.0,
) -> dict:
    """Integra la trayectoria de un paciente con un perfil de mutacion dado."""
    if driver not in DRIVER_BIAS:
        raise ValueError(f"driver desconocido: {driver}. Opciones: {list(DRIVER_BIAS)}")

    x0 = np.zeros(N) if x0 is None else x0
    I_driver = DRIVER_BIAS[driver]
    t_eval = np.linspace(*t_span, n_points)

    sol = solve_ivp(
        dynamics, t_span, x0, t_eval=t_eval,
        args=(W, I_driver, beta, 0.0, None),
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    return {"t": sol.t, "x": sol.y, "driver": driver}


def classify_state(x: np.ndarray, norm_floor: float = 1e-8) -> tuple[str, float]:
    """
    Clasifica un vector de estado por correlacion maxima con los patrones CMS.

    Si la norma de x esta por debajo de norm_floor (estado neutro/sin
    forzamiento), corrcoef es indefinido (division por desviacion
    estandar cero); en ese caso se devuelve correlacion 0.0 en vez de
    NaN, para reflejar que no hay evidencia de ningun subtipo.
    """
    if np.linalg.norm(x) < norm_floor:
        return "none", 0.0

    correlations = {
        label: float(np.corrcoef(x, CMS_PATTERNS[label])[0, 1])
        for label in CMS_LABELS
    }
    best = max(correlations, key=correlations.get)
    return best, correlations[best]


def build_model_from_patterns(patterns: dict) -> tuple:
    """
    Construye (W, gene_order, cms_labels) a partir de un diccionario
    arbitrario {cms_label: np.ndarray} — usado tanto por los patrones
    default de demo como por patrones calibrados contra datos reales
    (ver calibration.py).
    """
    labels = list(patterns.keys())
    stacked = np.stack([patterns[l] for l in labels], axis=1)
    W = projection_weight_matrix(stacked)
    return W, labels, stacked


def simulate_patient_with_model(
    driver_bias: np.ndarray,
    W: np.ndarray,
    n_genes: int,
    x0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 20.0),
    n_points: int = 400,
    beta: float = 2.0,
) -> dict:
    """Version generica de simulate_patient que acepta W y dimension arbitrarios."""
    x0 = np.zeros(n_genes) if x0 is None else x0
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(
        dynamics, t_span, x0, t_eval=t_eval,
        args=(W, driver_bias, beta, 0.0, None),
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    return {"t": sol.t, "x": sol.y}


if __name__ == "__main__":
    print(f"Dimension del espacio de estado (genes marcadores): N = {N}")
    print(f"Genes: {GENES}\n")

    for driver, expected_cms in [
        ("MSI_high", "CMS1_MSI_immune"),
        ("APC_mut", "CMS2_canonical_WNT"),
        ("KRAS_mut", "CMS3_metabolic"),
        ("SMAD4_loss", "CMS4_mesenchymal"),
    ]:
        result = simulate_patient(driver)
        x_final = result["x"][:, -1]
        predicted, corr = classify_state(x_final)
        status = "OK" if predicted == expected_cms else "MISMATCH"
        print(f"[{status}] driver={driver:12s} -> predicho={predicted:20s} "
              f"(r={corr:.3f}), esperado={expected_cms}")
