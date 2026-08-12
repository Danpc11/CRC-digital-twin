"""
modern_hopfield.py

Rediseno de la dinamica de atractores usando la construccion de
"Modern Hopfield Networks" (Ramsauer et al. 2020, "Hopfield Networks
is All You Need" -- la misma matematica detras de la atencion de los
Transformers), en vez de la regla de proyeccion + tanh usada en
attractor_model.py.

POR QUE ESTE REDISENIO, NO SOLO OTRO INTENTO
-----------------------------------------------
Se verifico (dynamics_diagnostics.py, con datos reales de GSE39582)
que la dinamica actual (dx/dt = -x + W tanh(beta x)) NO tiene los 4
patrones CMS como atractores genuinos del sistema no lineal en ningun
beta razonable -- y se probo simbolicamente por que: aunque W es
simetrica, el jacobiano de esa dinamica NO lo es (verificado con
sympy), lo que demuestra que esa dinamica NO es un flujo de gradiente.
Sin una funcion de energia que garantice decrecimiento monotono, nada
impide comportamiento errante, ciclos, o la proliferacion de
atractores espurios que efectivamente se encontraron.

La construccion de Hopfield moderno SI es, por diseno, un flujo de
gradiente genuino (verificado tambien simbolicamente aqui: jacobiano
simetrico exacto) sobre la energia continua de Ramsauer et al.:

    E(x) = -(1/beta) * logsumexp(beta * X^T x) + (1/2) ||x||^2

cuya dinamica de descenso de gradiente es:

    dx/dt = -grad E(x) = -x + X softmax(beta * X^T x)

donde X = [p^1 | p^2 | p^3 | p^4] es la matriz de patrones (una
columna por subtipo CMS). El nuevo estado es una combinacion convexa
de los patrones almacenados, ponderada por similitud (producto punto)
con el estado actual -- entre mas alto beta, mas "todo o nada" (mas
parecido a recuperar exactamente el patron mas cercano).

GARANTIA FORMAL (Ramsauer et al. 2020, Teorema 3): si los patrones
estan suficientemente "separados" (una condicion especifica sobre las
similitudes cruzadas vs. beta), cada patron almacenado es un punto
fijo con una cuenca de atraccion demostrable, y la convergencia desde
cerca del patron es una contraccion (Banach) -- tipicamente en un solo
paso de la iteracion discreta. Aqui se usa la version de TIEMPO
CONTINUO (compatible con el resto del proyecto, que integra ODEs), no
la iteracion discreta de un solo paso del paper original.

Este modulo es el motor dinamico predeterminado de las simulaciones
clinicas exploratorias. attractor_model.py permanece disponible bajo
la opcion explicita projection_legacy para reproducibilidad y
comparacion, pero no es el valor predeterminado.
HALLAZGO CRITICO -- EL ORIGEN ES UNA SILLA INESTABLE, NO UN REPOSO
--------------------------------------------------------------------
Verificado con datos reales de GSE39582: el origen SI es un punto fijo
exacto de esta dinamica (consecuencia matematica de que los patrones,
calibrados via z-score contra la media global, suman cero -- ver
hallazgo de "colapso de rango" en dynamics_diagnostics.py), pero es
una SILLA con eigenvalores hasta +4.25 (7 negativos, 3 fuertemente
positivos), NO un reposo estable como en attractor_model.py (donde el
termino -x domina y el origen SI es localmente estable, representando
"sin enfermedad residual").

CONSECUENCIA PRACTICA GRAVE: en simulate_longitudinal_patient_hopfield
(y su version con sesgo), la fase "sin recaida" (antes de
recurrence_onset_month, sin forzamiento) NO mantiene al paciente en el
origen -- el ruido numerico de la integracion se amplifica
exponencialmente (tasa ~e^{4.25 t}) y el estado colapsa a la cuenca
dominante (CMS1 en la calibracion real) ANTES de que cualquier
forzamiento/sesgo hacia otro patron siquiera comience. Verificado:
para mes 9 el estado sigue esencialmente en el origen (norma 0.02),
para mes 12 ya esta en CMS1 (norma 1.7, corr=0.997), para mes 15
(cuando "empieza" la recaida forzada) ya esta CMS1 exacto (corr=1.000).
Ningun mecanismo de forzamiento posterior (aditivo o sesgo en softmax)
puede revertir esto de forma confiable, porque no esta partiendo de
un punto neutral -- ya esta profundamente en la cuenca mas fuerte
(eigenvalor -1.0) antes de empezar.

VEREDICTO HONESTO: este rediseno es una mejora demostrable para la
pregunta de EQUILIBRIOS/ATRACTORES (flujo de gradiente probado,
intervalo amplio de beta, cobertura completa de cuencas sin espurios)
-- pero NO es un reemplazo directo para el modulo de PRONOSTICO/
TRAYECTORIA CLINICA (prognosis_demo.py), que depende de que el origen
sea un reposo estable durante la fase pre-recaida. Usar esta dinamica
para verificar clasificacion/atractores es solido; usarla para
simular trayectorias clinicas requeriria rediseno arquitectonico
adicional (ej. un termino estabilizador especifico para la fase
pre-recaida, o redefinir que punto representa "sin enfermedad" bajo
esta energia), no solo ajustar la magnitud del forzamiento.
"""

import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax numericamente estable (resta el maximo antes de exp)."""
    z_shift = z - np.max(z)
    e = np.exp(z_shift)
    return e / np.sum(e)


def _logsumexp(z: np.ndarray) -> float:
    """log-sum-exp numericamente estable."""
    m = np.max(z)
    return float(m + np.log(np.sum(np.exp(z - m))))


def patterns_to_matrix(patterns: dict) -> tuple[np.ndarray, list]:
    """Convierte el diccionario {label: vector} al formato matricial
    X (N genes x M patrones) que usa este modulo, preservando el orden
    de las etiquetas."""
    labels = list(patterns.keys())
    X = np.column_stack([patterns[l] for l in labels])
    return X, labels


def validate_modern_pattern_matrix(
    X: np.ndarray, n_genes: int, n_patterns: int = 4,
) -> np.ndarray:
    """Valida que se recibio X=[p1|...|pM], no la matriz legacy W.

    W y X comparten el numero de filas y pueden multiplicarse sin lanzar
    errores, por lo que confundirlas produce resultados numericos plausibles
    pero semanticamente falsos. En este proyecto M=4 (CMS1--CMS4).
    """
    matrix = np.asarray(X, dtype=float)
    expected = (int(n_genes), int(n_patterns))
    if matrix.ndim != 2 or matrix.shape != expected:
        raise ValueError(
            f"Modern Hopfield requiere X con forma {expected} "
            f"(genes x patrones CMS); se recibio {matrix.shape}. "
            "No pases la matriz W de projection_legacy.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("La matriz X contiene valores no finitos")
    return matrix


def validate_modern_hopfield_beta(
    patterns: dict, beta: float, correlation_threshold: float = 0.9,
) -> dict:
    """Comprueba que los cuatro centroides produzcan atractores estables."""
    if beta <= 0:
        raise ValueError("beta debe ser > 0")
    table = verify_all_patterns_hopfield(patterns, beta)
    valid_rows = (
        table["convergio_a_equilibrio"].astype(bool)
        & table["localmente_estable"].astype(bool)
        & (table["correlacion_equilibrio_vs_patron"] >= correlation_threshold)
    )
    invalid = table.loc[~valid_rows, "patron"].tolist()
    return {
        "valid": len(invalid) == 0,
        "invalid_patterns": invalid,
        "correlation_threshold": float(correlation_threshold),
        "table": table,
    }


def hopfield_energy(x: np.ndarray, X: np.ndarray, beta: float) -> float:
    """
    E(x) = -(1/beta) logsumexp(beta X^T x) + (1/2) ||x||^2

    (Ramsauer et al. 2020, ecuacion continua -- se omiten constantes
    aditivas que no afectan la dinamica: beta^-1 log M + max||p_mu||^2/2)
    """
    z = X.T @ x
    lse = _logsumexp(beta * z) / beta
    return -lse + 0.5 * float(np.dot(x, x))


def modern_hopfield_field(x: np.ndarray, X: np.ndarray, beta: float) -> np.ndarray:
    """
    dx/dt = -x + X softmax(beta X^T x) -- descenso de gradiente sobre
    hopfield_energy(). El nuevo estado es una combinacion convexa de
    los patrones almacenados, ponderada por similitud con el estado
    actual.
    """
    z = X.T @ x
    weights = _softmax(beta * z)
    return -x + X @ weights


def modern_hopfield_jacobian(x: np.ndarray, X: np.ndarray, beta: float) -> np.ndarray:
    """
    Jacobiano analitico de modern_hopfield_field, derivado y verificado
    simbolicamente (sympy, ver docstring del modulo) antes de
    codificarlo -- SIMETRICO por construccion (es el hessiano de
    hopfield_energy, con signo cambiado):

        J = -I + beta * X (diag(s) - s s^T) X^T,   s = softmax(beta X^T x)

    (diag(s) - s s^T) es el jacobiano estandar de softmax, simetrico
    por construccion -- sandwichearlo entre X y X^T preserva la
    simetria.
    """
    n = len(x)
    z = X.T @ x
    s = _softmax(beta * z)
    softmax_jac = np.diag(s) - np.outer(s, s)
    return -np.eye(n) + beta * X @ softmax_jac @ X.T


def find_true_equilibrium_hopfield(x0: np.ndarray, X: np.ndarray, beta: float):
    """Resuelve modern_hopfield_field(x)=0 numericamente partiendo de x0."""
    sol, info, ier, msg = fsolve(
        lambda x: modern_hopfield_field(x, X, beta), x0, full_output=True, xtol=1e-12)
    converged = ier == 1
    desplazamiento = float(np.linalg.norm(sol - x0))
    return sol, converged, desplazamiento


def stability_at_equilibrium_hopfield(x_eq: np.ndarray, X: np.ndarray, beta: float):
    """Estabilidad local via eigenvalores del jacobiano -- todos reales
    (jacobiano simetrico garantizado), estable si todos son negativos."""
    J = modern_hopfield_jacobian(x_eq, X, beta)
    eigvals = np.linalg.eigvalsh(J)  # eigvalsh: mas preciso y solo reales, dado que J es simetrica
    stable = bool(np.all(eigvals < 0))
    max_eigval = float(np.max(eigvals))
    return eigvals, stable, max_eigval


def _correlation_profile(x: np.ndarray, X: np.ndarray, labels: list) -> dict:
    """Correlaciones seguras y etiqueta mas cercana para un estado."""
    correlations = {}
    for i, label in enumerate(labels):
        p = X[:, i]
        if np.std(x) < 1e-12 or np.std(p) < 1e-12:
            correlations[label] = float("nan")
        else:
            correlations[label] = float(np.corrcoef(x, p)[0, 1])
    finite = {k: v for k, v in correlations.items() if np.isfinite(v)}
    if not finite:
        return {"correlations": correlations, "best_label": "indeterminado",
                "best_correlation": float("nan"), "margin": float("nan")}
    ordered = sorted(finite.items(), key=lambda item: item[1], reverse=True)
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else float("nan")
    return {"correlations": correlations, "best_label": ordered[0][0],
            "best_correlation": ordered[0][1], "margin": float(margin)}


def classify_expression_modern_hopfield(
    expression: np.ndarray, patterns: dict, beta: float = 3.0,
    integration_time: float = 30.0, corr_threshold: float = 0.8,
    residual_threshold: float = 1e-6, input_margin_threshold: float = 0.15,
) -> dict:
    """Recuperacion dinamica experimental de una muestra ya normalizada.

    La etiqueta solo se acepta si la integracion y el refinamiento convergen,
    el equilibrio es estable, el residuo es pequeno y la correlacion supera
    el umbral. No reemplaza al clasificador CMS por correlacion validado.
    """
    if beta <= 0:
        raise ValueError("beta debe ser > 0")
    if input_margin_threshold < 0:
        raise ValueError("input_margin_threshold debe ser >= 0")
    x0 = np.asarray(expression, dtype=float)
    X, labels = patterns_to_matrix(patterns)
    if x0.shape != (X.shape[0],):
        raise ValueError(f"expression debe tener forma ({X.shape[0]},)")
    if not np.all(np.isfinite(x0)):
        raise ValueError("expression contiene valores no finitos")
    if np.linalg.norm(x0) < 1e-8 or np.std(x0) < 1e-12:
        return {
            "label": "indeterminado", "correlation": float("nan"),
            "margin": float("nan"), "converged": False, "stable": False,
            "residual": float("nan"), "displacement": 0.0,
            "energy_drop": 0.0, "correlations": {label: float("nan") for label in labels},
            "input_label": "indeterminado", "input_correlation": float("nan"),
            "input_margin": float("nan"), "abstention_reason": "entrada_degenerada",
        }

    input_profile = _correlation_profile(x0, X, labels)

    sol = solve_ivp(
        lambda t, x: modern_hopfield_field(x, X, beta), (0, integration_time), x0,
        method="RK45", rtol=1e-8, atol=1e-10)
    x_terminal = sol.y[:, -1]
    x_eq, refined, _ = find_true_equilibrium_hopfield(x_terminal, X, beta)
    residual = float(np.linalg.norm(modern_hopfield_field(x_eq, X, beta)))
    _, stable, max_eig = stability_at_equilibrium_hopfield(x_eq, X, beta)
    profile = _correlation_profile(x_eq, X, labels)
    converged = bool(sol.success and refined and residual <= residual_threshold)
    accepted = bool(
        converged and stable
        and profile["best_correlation"] >= corr_threshold
        and input_profile["margin"] >= input_margin_threshold)
    if not converged:
        abstention_reason = "no_convergio"
    elif not stable:
        abstention_reason = "equilibrio_no_estable"
    elif profile["best_correlation"] < corr_threshold:
        abstention_reason = "correlacion_final_baja"
    elif input_profile["margin"] < input_margin_threshold:
        abstention_reason = "entrada_hibrida_o_ambigua"
    else:
        abstention_reason = ""
    return {
        "label": profile["best_label"] if accepted else "indeterminado",
        "correlation": profile["best_correlation"], "margin": profile["margin"],
        "input_label": input_profile["best_label"],
        "input_correlation": input_profile["best_correlation"],
        "input_margin": input_profile["margin"],
        "input_correlations": input_profile["correlations"],
        "abstention_reason": abstention_reason,
        "converged": converged, "stable": stable, "max_eigenvalue": max_eig,
        "residual": residual, "displacement": float(np.linalg.norm(x_eq - x0)),
        "energy_drop": float(hopfield_energy(x0, X, beta) - hopfield_energy(x_eq, X, beta)),
        "correlations": profile["correlations"], "equilibrium": x_eq,
    }


def score_cohort_modern_hopfield(
    df, gene_cols: list[str], patterns: dict, beta: float = 3.0,
    corr_threshold: float = 0.8, integration_time: float = 30.0,
    input_margin_threshold: float = 0.15,
):
    """Agrega columnas de recuperacion Modern Hopfield a una cohorte."""
    out = df.copy()
    results = [classify_expression_modern_hopfield(
        row.to_numpy(dtype=float), patterns, beta=beta,
        corr_threshold=corr_threshold,
        integration_time=integration_time,
        input_margin_threshold=input_margin_threshold,
    ) for _, row in df[gene_cols].iterrows()]
    out["modern_hopfield_cms"] = [r["label"] for r in results]
    out["modern_hopfield_correlation"] = [r["correlation"] for r in results]
    out["modern_hopfield_margin"] = [r["margin"] for r in results]
    out["modern_hopfield_input_label"] = [r["input_label"] for r in results]
    out["modern_hopfield_input_correlation"] = [r["input_correlation"] for r in results]
    out["modern_hopfield_input_margin"] = [r["input_margin"] for r in results]
    out["modern_hopfield_abstention_reason"] = [r["abstention_reason"] for r in results]
    out["modern_hopfield_residual"] = [r["residual"] for r in results]
    out["modern_hopfield_converged"] = [r["converged"] for r in results]
    out["modern_hopfield_stable"] = [r["stable"] for r in results]
    out["modern_hopfield_displacement"] = [r["displacement"] for r in results]
    out["modern_hopfield_energy_drop"] = [r["energy_drop"] for r in results]
    if "predicted_cms" in out.columns:
        out["modern_hopfield_concordant"] = (
            out["modern_hopfield_cms"] == out["predicted_cms"])
    return out


def verify_energy_decreases(
    X: np.ndarray, beta: float, n_trajectories: int = 20,
    integration_time: float = 30.0, seed: int = 0,
) -> dict:
    """
    Verificacion EXTRA, unica de esta construccion (no aplicable a la
    dinamica anterior, que no es un flujo de gradiente): confirma
    numericamente que la energia decrece monotonamente a lo largo de
    trayectorias reales -- una propiedad garantizada matematicamente
    para cualquier flujo de gradiente genuino, asi que si esto fallara
    seria evidencia de un bug de implementacion, no una sorpresa
    cientifica legitima.
    """
    rng = np.random.default_rng(seed)
    n_genes = X.shape[0]
    scale = float(np.mean([np.linalg.norm(X[:, i]) for i in range(X.shape[1])]))

    violaciones = 0
    for _ in range(n_trajectories):
        x0 = rng.normal(0, scale / np.sqrt(n_genes), size=n_genes)
        sol = solve_ivp(
            lambda t, x: modern_hopfield_field(x, X, beta), (0, integration_time), x0,
            method="RK45", rtol=1e-9, atol=1e-11, dense_output=True)
        ts = np.linspace(0, integration_time, 50)
        energias = [hopfield_energy(sol.sol(t), X, beta) for t in ts]
        diffs = np.diff(energias)
        if np.any(diffs > 1e-8):
            violaciones += 1

    return {
        "n_trayectorias": n_trajectories,
        "violaciones_monotonicidad": violaciones,
        "todas_decrecientes": violaciones == 0,
    }


def verify_all_patterns_hopfield(patterns: dict, beta: float = 2.0) -> "pd.DataFrame":
    """Version de verify_all_patterns() (dynamics_diagnostics.py) para
    esta dinamica -- misma metodologia, para comparacion directa."""
    import pandas as pd
    X, labels = patterns_to_matrix(patterns)
    rows = []
    for i, label in enumerate(labels):
        p = X[:, i]
        x_eq, converged, desp = find_true_equilibrium_hopfield(p, X, beta)
        eigvals, stable, max_eig = stability_at_equilibrium_hopfield(x_eq, X, beta)
        corr = float(np.corrcoef(x_eq, p)[0, 1]) if np.std(x_eq) > 1e-12 else float("nan")
        rows.append({
            "patron": label, "convergio_a_equilibrio": converged,
            "desplazamiento_vs_patron_calibrado": desp, "localmente_estable": stable,
            "max_eigenvalor": max_eig, "correlacion_equilibrio_vs_patron": corr,
        })
    return pd.DataFrame(rows)


def empirical_basin_membership_hopfield(
    patterns: dict, beta: float = 2.0, n_samples: int = 300,
    integration_time: float = 30.0, corr_threshold: float = 0.8, seed: int = 0,
    residual_threshold: float = 1e-6,
) -> "pd.DataFrame":
    """Version de empirical_basin_membership() (dynamics_diagnostics.py)
    para esta dinamica -- misma metodologia exacta, para comparacion
    directa lado a lado con la dinamica anterior."""
    import pandas as pd
    X, labels = patterns_to_matrix(patterns)
    n_genes = X.shape[0]
    rng = np.random.default_rng(seed)
    scale = float(np.mean([np.linalg.norm(X[:, i]) for i in range(X.shape[1])]))

    resultados = []
    for _ in range(n_samples):
        x0 = rng.normal(0, scale / np.sqrt(n_genes), size=n_genes)
        sol = solve_ivp(
            lambda t, x: modern_hopfield_field(x, X, beta), (0, integration_time), x0,
            method="RK45", rtol=1e-8, atol=1e-10)
        x_terminal = sol.y[:, -1]
        x_final, refined, _ = find_true_equilibrium_hopfield(x_terminal, X, beta)
        residual = float(np.linalg.norm(modern_hopfield_field(x_final, X, beta)))
        _, stable, _ = stability_at_equilibrium_hopfield(x_final, X, beta)
        if not (sol.success and refined and stable and residual <= residual_threshold):
            resultados.append({"convergio_a": "no_convergente", "correlacion": np.nan,
                                "norma_final": float(np.linalg.norm(x_final))})
            continue
        norm_final = float(np.linalg.norm(x_final))
        if norm_final < 1e-6:
            resultados.append({"convergio_a": "origen", "correlacion": np.nan, "norma_final": norm_final})
            continue
        mejor_label, mejor_corr = None, -2.0
        for i, label in enumerate(labels):
            p = X[:, i]
            if np.std(x_final) < 1e-12 or np.std(p) < 1e-12:
                continue
            c = float(np.corrcoef(x_final, p)[0, 1])
            if c > mejor_corr:
                mejor_corr, mejor_label = c, label
        etiqueta = mejor_label if mejor_corr >= corr_threshold else "ninguno_claro"
        resultados.append({"convergio_a": etiqueta, "correlacion": mejor_corr, "norma_final": norm_final})

    df = pd.DataFrame(resultados)
    resumen = df["convergio_a"].value_counts(normalize=True).rename("proporcion").to_frame()
    resumen["n"] = df["convergio_a"].value_counts()
    return resumen


def sweep_beta_hopfield(
    patterns: dict, beta_values=None, min_separation: float = 0.5,
) -> "pd.DataFrame":
    """Barrido de beta para esta dinamica -- reporta si los 4 patrones
    son simultaneamente equilibrios exactos (o casi), estables, y
    separados entre si, para cada beta."""
    import pandas as pd
    if beta_values is None:
        beta_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    X, labels = patterns_to_matrix(patterns)
    rows = []
    for beta in beta_values:
        equilibria = {}
        n_ok = 0
        for i, label in enumerate(labels):
            p = X[:, i]
            x_eq, converged, desp = find_true_equilibrium_hopfield(p, X, beta)
            _, stable, _ = stability_at_equilibrium_hopfield(x_eq, X, beta)
            equilibria[label] = x_eq
            corr = float(np.corrcoef(x_eq, p)[0, 1]) if np.std(x_eq) > 1e-12 else float("nan")
            if converged and stable and not np.isnan(corr) and corr >= 0.8 and desp < 1.0:
                n_ok += 1
        distancias = [np.linalg.norm(equilibria[labels[i]] - equilibria[labels[j]])
                      for i in range(len(labels)) for j in range(i + 1, len(labels))]
        min_sep = float(min(distancias)) if distancias else float("nan")
        rows.append({
            "beta": beta, "n_patrones_ok": n_ok,
            "todos_4_ok": n_ok == len(labels) and min_sep >= min_separation,
            "min_separacion": min_sep,
        })
    return pd.DataFrame(rows)



def _scheduled_forcing_strength(
    months_since_onset: float, max_strength: float,
    ramp_duration_months: float | None = None,
) -> tuple[float, float]:
    """Devuelve fuerza y progreso [0,1] de una rampa reproducible.

    Sin ramp_duration conserva el calendario historico (0.15/mes). Con
    ramp_duration, el candidato se alcanza exactamente al final de esa
    ventana, necesario para comparar caps distintos en el mismo horizonte.
    """
    if months_since_onset <= 0:
        return 0.0, 0.0
    if ramp_duration_months is None:
        strength = min(0.15 * months_since_onset, max_strength)
        progress = min(strength / max_strength, 1.0) if max_strength > 0 else 0.0
        return float(strength), float(progress)
    if ramp_duration_months <= 0:
        raise ValueError("ramp_duration_months debe ser > 0")
    progress = min(months_since_onset / ramp_duration_months, 1.0)
    return float(max_strength * progress), float(progress)


def normalized_driver_direction(pattern: np.ndarray) -> np.ndarray:
    """Direccion unitaria: la fuerza pasa a ser una magnitud comparable."""
    p = np.asarray(pattern, dtype=float)
    norm = float(np.linalg.norm(p))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("El patron del driver debe tener norma positiva y finita")
    return p / norm


def simulate_longitudinal_patient_hopfield(
    X: np.ndarray, recurrence_pattern: np.ndarray, n_genes: int,
    n_timepoints: int = 8, months_between_checks: int = 3,
    recurrence_onset_month: int = 15, beta: float = 2.0,
    max_forcing_strength: float = 1.5,
    forcing_ramp_duration_months: float | None = None,
    normalize_driver: bool = False,
) -> tuple:
    """
    Version de simulate_longitudinal_patient (prognosis_demo.py) con
    la dinamica de Hopfield moderno en vez de attractor_model.dynamics
    -- MISMA estructura de forzamiento (I_driver creciente hacia
    recurrence_pattern despues de recurrence_onset_month), para
    comparacion directa y justa entre ambas dinamicas bajo el USO
    PRACTICO real de la app (con forzamiento activo, no cuencas
    libres).

    max_forcing_strength: EXPUESTO explicitamente, default 1.5, no 0.7
    (el valor usado en prognosis_demo.py para la dinamica anterior) --
    hallazgo real: la dinamica de Hopfield moderno tiene una atraccion
    NATIVA mucho mas fuerte hacia su patron dominante (eigenvalor ~-1.0
    para CMS1, el de mayor norma en la calibracion real) que la
    dinamica anterior (eigenvalores ~-0.08 cerca de su region critica).
    Con fuerza_max=0.7 (el valor viejo), forzar hacia un patron NO
    dominante (ej. CMS2) termina en el patron dominante en vez del
    objetivo (verificado: corr con CMS2=0.32, corr con CMS1=0.75).
    Con fuerza_max>=1.5, converge correctamente (corr con objetivo=1.0)
    de forma robusta -- verificado 1.5/3.0/5.0 dan resultados identicos.
    """
    t_checks = np.arange(0, n_timepoints * months_between_checks, months_between_checks)
    x_series = np.zeros((n_genes, n_timepoints))
    x_current = np.zeros(n_genes)
    driver_direction = (normalized_driver_direction(recurrence_pattern)
                        if normalize_driver else recurrence_pattern)

    for i, t in enumerate(t_checks):
        if t >= recurrence_onset_month:
            months_since_onset = t - recurrence_onset_month
            strength, _ = _scheduled_forcing_strength(
                months_since_onset, max_forcing_strength,
                forcing_ramp_duration_months)
            I_driver = strength * driver_direction
        else:
            I_driver = np.zeros(n_genes)

        sol = solve_ivp(
            lambda tt, xx: modern_hopfield_field(xx, X, beta) + I_driver,
            (0, months_between_checks), x_current, method="RK45", rtol=1e-8, atol=1e-10)
        x_current = sol.y[:, -1]
        x_series[:, i] = x_current

    return t_checks, x_series


def find_minimum_forcing_strength(
    patterns: dict, target_label: str, beta: float = 3.0,
    strength_candidates=None, corr_threshold: float = 0.9,
    recurrence_onset_month: int = 15, n_timepoints: int = 10,
) -> dict:
    """
    Busca la fuerza de forzamiento MINIMA necesaria para que forzar
    hacia target_label efectivamente converja ahi (corr >= corr_threshold),
    en vez de terminar en el patron dominante u otro lugar.

    NECESARIO tras un hallazgo real: el umbral de fuerza suficiente NO
    es universal, depende de que tan dominante es el patron de mayor
    norma en CADA calibracion especifica -- con datos sinteticos de
    prueba (cuencas ~parejas, 22-27% cada patron), fuerza=1.5 bastaba
    para los 4; con la calibracion real de GSE39582 (cuencas desiguales,
    CMS1=45% vs CMS2=12%), fuerza=1.5 NO basta para CMS2 (termina con
    correlacion NEGATIVA respecto al objetivo). Buscar el umbral real
    en vez de asumir un numero fijo.

    Devuelve, para cada candidato de fuerza probado: la correlacion
    lograda con el objetivo, Y con que patron distinto termina
    correlacionando mas si el forzamiento fallo (diagnostico completo,
    no solo pasa/no-pasa).
    """
    if strength_candidates is None:
        strength_candidates = [0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0]

    X, labels = patterns_to_matrix(patterns)
    n_genes = X.shape[0]
    idx_target = labels.index(target_label)
    p_target = X[:, idx_target]

    resultados = []
    umbral_minimo_encontrado = None
    for fuerza in strength_candidates:
        t, x = simulate_longitudinal_patient_hopfield(
            X, p_target, n_genes, beta=beta, max_forcing_strength=fuerza,
            n_timepoints=n_timepoints, recurrence_onset_month=recurrence_onset_month)
        x_final = x[:, -1]

        if np.std(x_final) < 1e-12:
            resultados.append({"fuerza": fuerza, "corr_con_objetivo": float("nan"),
                                "corr_con_mas_parecido": float("nan"), "mas_parecido_a": "origen"})
            continue

        corr_target = float(np.corrcoef(x_final, p_target)[0, 1])
        mejor_label, mejor_corr = None, -2.0
        for i, label in enumerate(labels):
            c = float(np.corrcoef(x_final, X[:, i])[0, 1])
            if c > mejor_corr:
                mejor_corr, mejor_label = c, label

        resultados.append({
            "fuerza": fuerza, "corr_con_objetivo": corr_target,
            "corr_con_mas_parecido": mejor_corr, "mas_parecido_a": mejor_label,
        })
        if corr_target >= corr_threshold and umbral_minimo_encontrado is None:
            umbral_minimo_encontrado = fuerza

    return {
        "patron_objetivo": target_label,
        "umbral_minimo_encontrado": umbral_minimo_encontrado,
        "detalle": resultados,
    }


def compute_stabilizing_k(X: np.ndarray, beta: float, safety_margin: float = 1.0) -> float:
    """
    Calcula la constante k MINIMA necesaria para estabilizar el origen
    bajo esta calibracion especifica -- NO un valor universal (misma
    leccion que con la fuerza de forzamiento: lo que basta para un
    dataset no basta para otro). Se calcula a partir del eigenvalor
    positivo mas grande del jacobiano en el origen (verificado con
    datos reales de GSE39582: hasta +4.25) mas un margen de seguridad.

    Con k = eigenvalor_max_positivo + safety_margin, el jacobiano
    modificado en el origen (J(0) - k*I) tiene TODOS los eigenvalores
    negativos -- el origen pasa de silla inestable a minimo local
    genuino.
    """
    n = X.shape[0]
    origin = np.zeros(n)
    _, _, max_eig = stability_at_equilibrium_hopfield(origin, X, beta)
    return max(0.0, max_eig) + safety_margin


def modern_hopfield_baseline(X: np.ndarray) -> np.ndarray:
    """Deriva el termino constante que hace al origen un punto fijo.

    En x=0 el softmax es uniforme, por lo que el campo sin corregir es
    X @ (1/M). Restarlo es equivalente a agregar un termino lineal a la
    energia y no cambia el Jacobiano.
    """
    return X @ np.full(X.shape[1], 1.0 / X.shape[1])


def modern_hopfield_field_stabilized(
    x: np.ndarray, X: np.ndarray, beta: float, k: float = 0.0,
    baseline_correction: np.ndarray | None = None,
) -> np.ndarray:
    """
    Campo con termino estabilizador extra -k*x, para usar SOLO durante
    la fase pre-recaida (k=0 durante la fase de forzamiento/sesgo --
    ahi se quiere la dinamica genuina, sin distorsionar el paisaje de
    energia que ya se verifico que tiene los 4 atractores correctos).

    La correccion constante b=X@(1/M) garantiza F(0)=0 incluso con
    centroides de clases desbalanceadas. El campo es descenso de la
    energia modificada E_k(x)=E(x)+b^T x+(k/2)||x||^2.
    """
    b = modern_hopfield_baseline(X) if baseline_correction is None else baseline_correction
    return modern_hopfield_field(x, X, beta) - b - k * x


def modern_hopfield_jacobian_stabilized(x: np.ndarray, X: np.ndarray, beta: float, k: float = 0.0) -> np.ndarray:
    """Jacobiano del campo estabilizado -- simplemente J - k*I."""
    n = len(x)
    return modern_hopfield_jacobian(x, X, beta) - k * np.eye(n)


def simulate_longitudinal_patient_hopfield_v2(
    X: np.ndarray, recurrence_pattern: np.ndarray, n_genes: int,
    n_timepoints: int = 8, months_between_checks: int = 3,
    recurrence_onset_month: int = 15, beta: float = 2.0,
    max_forcing_strength: float = 5.0, stabilizing_k: float | None = None,
    mechanism: str = "additive", smooth_transition: bool = True,
    forcing_ramp_duration_months: float | None = None,
    normalize_driver: bool = False,
) -> tuple:
    """
    Version CORREGIDA de simulate_longitudinal_patient_hopfield -- usa
    el termino estabilizador -k*x SOLO durante la fase pre-recaida
    (manteniendo el origen como reposo genuino, no una silla que
    colapsa por ruido numerico antes de que empiece el forzamiento),
    y lo QUITA por completo durante la fase de recaida (donde se quiere
    la dinamica genuina + forzamiento/sesgo, sin distorsion).

    stabilizing_k: si es None, se calcula automaticamente con
    compute_stabilizing_k() para esta calibracion especifica.
    mechanism: "additive" (fuerza sumada al campo) o "bias" (sesgo en
    softmax, ver modern_hopfield_field_biased) para la fase de recaida.
    """
    X = validate_modern_pattern_matrix(X, n_genes, n_patterns=4)
    if max_forcing_strength <= 0:
        raise ValueError("max_forcing_strength debe ser > 0")
    if mechanism not in {"additive", "bias"}:
        raise ValueError("mechanism debe ser 'additive' o 'bias'")
    if stabilizing_k is None:
        stabilizing_k = compute_stabilizing_k(X, beta)
    baseline = modern_hopfield_baseline(X)

    n_patterns = X.shape[1]
    t_checks = np.arange(0, n_timepoints * months_between_checks, months_between_checks)
    x_series = np.zeros((n_genes, n_timepoints))
    x_current = np.zeros(n_genes)
    driver_direction = (normalized_driver_direction(recurrence_pattern)
                        if normalize_driver else recurrence_pattern)

    target_idx = None
    if mechanism == "bias":
        labels_idx = np.where((X.T == recurrence_pattern).all(axis=1))[0]
        target_idx = int(labels_idx[0]) if len(labels_idx) > 0 else None

    for i, t in enumerate(t_checks):
        if t >= recurrence_onset_month:
            months_since_onset = t - recurrence_onset_month
            strength, forcing_progress = _scheduled_forcing_strength(
                months_since_onset, max_forcing_strength,
                forcing_ramp_duration_months)
            # Evita retirar el reposo en el mismo instante en que la fuerza
            # todavia vale cero. Al crecer la senal, se apagan suavemente
            # tanto k como la correccion basal.
            if smooth_transition:
                quiescent_weight = max(0.0, 1.0 - forcing_progress)
            else:
                quiescent_weight = 0.0
            if mechanism == "bias" and target_idx is not None:
                bias = np.zeros(n_patterns)
                bias[target_idx] = strength
                field = lambda tt, xx: (
                    modern_hopfield_field_biased(xx, X, beta, bias)
                    - quiescent_weight * baseline
                    - quiescent_weight * stabilizing_k * xx)
            else:
                I_driver = strength * driver_direction
                field = lambda tt, xx: (
                    modern_hopfield_field(xx, X, beta) + I_driver
                    - quiescent_weight * baseline
                    - quiescent_weight * stabilizing_k * xx)
        else:
            field = lambda tt, xx: modern_hopfield_field_stabilized(
                xx, X, beta, stabilizing_k, baseline)

        sol = solve_ivp(field, (0, months_between_checks), x_current,
                         method="RK45", rtol=1e-8, atol=1e-10)
        x_current = sol.y[:, -1]
        x_series[:, i] = x_current

    return t_checks, x_series


def diagnose_pre_recurrence_residual(
    patterns: dict, beta: float = 3.0, duration_months: int = 15,
    months_between_checks: int = 3, stabilizing_k: float | None = None,
) -> dict:
    """Cuantifica y orienta cualquier desplazamiento durante el reposo."""
    X, labels = patterns_to_matrix(patterns)
    k = compute_stabilizing_k(X, beta) if stabilizing_k is None else stabilizing_k
    origin = np.zeros(X.shape[0])
    baseline = modern_hopfield_baseline(X)
    initial_residual = float(np.linalg.norm(
        modern_hopfield_field_stabilized(origin, X, beta, k, baseline)))
    x = origin.copy()
    n_steps = int(np.ceil(duration_months / months_between_checks))
    for _ in range(n_steps):
        sol = solve_ivp(
            lambda t, xx: modern_hopfield_field_stabilized(xx, X, beta, k, baseline),
            (0, months_between_checks), x, method="RK45", rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
    profile = _correlation_profile(x, X, labels)
    return {
        "k": float(k), "baseline_correction_norm": float(np.linalg.norm(baseline)),
        "origin_field_residual": initial_residual, "final_norm": float(np.linalg.norm(x)),
        "final_field_residual": float(np.linalg.norm(
            modern_hopfield_field_stabilized(x, X, beta, k, baseline))),
        "closest_cms": profile["best_label"],
        "closest_correlation": profile["best_correlation"],
        "correlations": profile["correlations"], "final_state": x,
        "origin_is_fixed": initial_residual <= 1e-8,
    }


def relax_after_forcing_withdrawal(
    state_at_withdrawal: np.ndarray, X: np.ndarray, labels: list,
    beta: float = 3.0, withdrawal_time: float = 30.0,
    residual_threshold: float = 1e-6,
) -> dict:
    """Retira por completo el driver y determina el atractor libre final."""
    if withdrawal_time <= 0:
        raise ValueError("withdrawal_time debe ser > 0")
    x0 = np.asarray(state_at_withdrawal, dtype=float)
    sol = solve_ivp(
        lambda t, x: modern_hopfield_field(x, X, beta),
        (0, withdrawal_time), x0, method="RK45", rtol=1e-8, atol=1e-10)
    x_terminal = sol.y[:, -1]
    x_eq, refined, _ = find_true_equilibrium_hopfield(x_terminal, X, beta)
    residual = float(np.linalg.norm(modern_hopfield_field(x_eq, X, beta)))
    _, stable, max_eig = stability_at_equilibrium_hopfield(x_eq, X, beta)
    profile = _correlation_profile(x_eq, X, labels)
    converged = bool(sol.success and refined and residual <= residual_threshold)
    return {
        "state": x_eq, "converged": converged, "stable": stable,
        "residual": residual, "max_eigenvalue": max_eig,
        "label": profile["best_label"],
        "correlation": profile["best_correlation"],
        "correlations": profile["correlations"],
        "displacement_during_withdrawal": float(np.linalg.norm(x_eq - x0)),
    }


def compare_forcing_sweep_v1_v2(
    patterns: dict, beta: float = 3.0, strength_candidates=None,
    corr_threshold: float = 0.9, recurrence_onset_month: int = 15,
    n_timepoints: int = 10, months_between_checks: int = 3,
    smooth_transition: bool = True, withdrawal_time: float = 30.0,
    normalize_driver: bool = True, residual_threshold: float = 1e-6,
) -> "pd.DataFrame":
    """Barrido apples-to-apples de fuerza sin y con estabilizacion.

    V1 es la dinamica Modern Hopfield sin reposo estabilizado. V2 usa
    correccion basal, k calibrado y transicion opcionalmente suave.
    Ambas comparten beta, tiempos, objetivo y fuerza. Por defecto el
    driver es unitario; despues se retira y el exito se decide sobre el
    equilibrio libre alcanzado, no sobre la alineacion bajo fuerza activa.
    """
    import pandas as pd
    if strength_candidates is None:
        strength_candidates = [0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0]
    strengths = sorted({float(v) for v in strength_candidates})
    if not strengths or strengths[0] < 0:
        raise ValueError("strength_candidates debe contener valores >= 0")

    X, labels = patterns_to_matrix(patterns)
    k = compute_stabilizing_k(X, beta)
    last_check_month = (n_timepoints - 1) * months_between_checks
    ramp_duration = last_check_month - recurrence_onset_month
    if ramp_duration <= 0:
        raise ValueError("La simulacion debe incluir al menos un control posterior a la recaida")
    baseline_diag = diagnose_pre_recurrence_residual(
        patterns, beta=beta, duration_months=recurrence_onset_month,
        months_between_checks=months_between_checks, stabilizing_k=k)
    rows = []
    target_equilibria = {}
    for i, target in enumerate(labels):
        target_equilibria[target], _, _ = find_true_equilibrium_hopfield(
            X[:, i], X, beta)
    for i, target in enumerate(labels):
        p = X[:, i]
        driver_direction = normalized_driver_direction(p) if normalize_driver else p
        driver_direction_norm = float(np.linalg.norm(driver_direction))
        target_eq = target_equilibria[target]
        target_eq_norm = float(np.linalg.norm(target_eq))
        for strength in strengths:
            _, x_v1 = simulate_longitudinal_patient_hopfield(
                X, p, X.shape[0], n_timepoints=n_timepoints,
                months_between_checks=months_between_checks,
                recurrence_onset_month=recurrence_onset_month, beta=beta,
                max_forcing_strength=strength,
                forcing_ramp_duration_months=ramp_duration,
                normalize_driver=normalize_driver)
            _, x_v2 = simulate_longitudinal_patient_hopfield_v2(
                X, p, X.shape[0], n_timepoints=n_timepoints,
                months_between_checks=months_between_checks,
                recurrence_onset_month=recurrence_onset_month, beta=beta,
                max_forcing_strength=strength, stabilizing_k=k,
                smooth_transition=smooth_transition,
                forcing_ramp_duration_months=ramp_duration,
                normalize_driver=normalize_driver)
            profile_v1 = _correlation_profile(x_v1[:, -1], X, labels)
            profile_v2 = _correlation_profile(x_v2[:, -1], X, labels)
            active_corr_v1 = profile_v1["correlations"][target]
            active_corr_v2 = profile_v2["correlations"][target]
            withdrawn_v1 = relax_after_forcing_withdrawal(
                x_v1[:, -1], X, labels, beta=beta, withdrawal_time=withdrawal_time,
                residual_threshold=residual_threshold)
            withdrawn_v2 = relax_after_forcing_withdrawal(
                x_v2[:, -1], X, labels, beta=beta, withdrawal_time=withdrawal_time,
                residual_threshold=residual_threshold)
            corr_v1 = withdrawn_v1["correlations"][target]
            corr_v2 = withdrawn_v2["correlations"][target]
            distance_v1 = float(np.linalg.norm(withdrawn_v1["state"] - target_eq))
            distance_v2 = float(np.linalg.norm(withdrawn_v2["state"] - target_eq))
            relative_distance_v1 = distance_v1 / target_eq_norm if target_eq_norm > 0 else float("nan")
            relative_distance_v2 = distance_v2 / target_eq_norm if target_eq_norm > 0 else float("nan")
            success_v1 = bool(
                withdrawn_v1["converged"] and withdrawn_v1["stable"] and
                withdrawn_v1["label"] == target and np.isfinite(corr_v1) and
                corr_v1 >= corr_threshold)
            success_v2 = bool(
                withdrawn_v2["converged"] and withdrawn_v2["stable"] and
                withdrawn_v2["label"] == target and np.isfinite(corr_v2) and
                corr_v2 >= corr_threshold)
            rows.append({
                "patron_objetivo": target, "fuerza_maxima": strength, "beta": beta,
                "fuerza_aplicada_final": strength,
                "norma_driver_aplicado": strength * driver_direction_norm,
                "driver_normalizado": normalize_driver,
                "duracion_rampa_meses": ramp_duration,
                "tiempo_retirada": withdrawal_time,
                "v1_corr_objetivo_activo": active_corr_v1,
                "v1_cms_activo": profile_v1["best_label"],
                "v1_corr_objetivo": corr_v1, "v1_cms_final": withdrawn_v1["label"],
                "v1_residuo_post_retirada": withdrawn_v1["residual"],
                "v1_estable_post_retirada": withdrawn_v1["stable"],
                "v1_distancia_relativa_objetivo": relative_distance_v1,
                "v1_exito": success_v1,
                "v2_corr_objetivo_activo": active_corr_v2,
                "v2_cms_activo": profile_v2["best_label"],
                "v2_corr_objetivo": corr_v2, "v2_cms_final": withdrawn_v2["label"],
                "v2_residuo_post_retirada": withdrawn_v2["residual"],
                "v2_estable_post_retirada": withdrawn_v2["stable"],
                "v2_distancia_relativa_objetivo": relative_distance_v2,
                "v2_exito": success_v2,
                "v2_norma_al_retirar": float(np.linalg.norm(x_v2[:, -1])),
                "v2_norma_post_retirada": float(np.linalg.norm(withdrawn_v2["state"])),
                "norma_equilibrio_objetivo": target_eq_norm,
                "v2_norma_basal": baseline_diag["final_norm"],
                "v2_residuo_campo_en_origen": baseline_diag["origin_field_residual"],
                "stabilizing_k": k, "transicion_suave": smooth_transition,
            })
    return pd.DataFrame(rows)


def summarize_forcing_thresholds(sweep) -> "pd.DataFrame":
    """Extrae el primer candidato exitoso por CMS para V1 y V2."""
    import pandas as pd
    rows = []
    for target, group in sweep.groupby("patron_objetivo", sort=False):
        row = {"patron_objetivo": target}
        for version in ("v1", "v2"):
            passing = group[group[f"{version}_exito"]].sort_values("fuerza_maxima")
            row[f"umbral_{version}"] = (
                float(passing.iloc[0]["fuerza_maxima"]) if len(passing) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)



def modern_hopfield_field_biased(x: np.ndarray, X: np.ndarray, beta: float, bias: np.ndarray) -> np.ndarray:
    """
    Mecanismo ALTERNATIVO al forzamiento aditivo (I_driver sumado
    directamente al campo) -- en vez de pelear contra el paisaje de
    energia con una fuerza externa, sesga las similitudes ANTES del
    softmax (mas "nativo" a esta construccion: cambia CUAL combinacion
    de patrones se favorece, en vez de imponer una fuerza que compite
    con el propio flujo de gradiente).

    NO VERIFICADO como solucion real todavia -- un intento de
    reproducir el problema (CMS1 dominante casi opuesto a CMS2, r~-0.8
    en los datos reales) con un caso sintetico dio el resultado
    CONTRARIO (aditivo funciono, sesgo fallo) -- evidencia de que la
    reproduccion sintetica no capturo la estructura real del problema
    (probablemente depende de como se relacionan los 4 patrones entre
    si, no solo el par dominante-objetivo). Hay que probar esto
    directamente contra los datos reales antes de confiar en el.

    bias: vector de longitud M (numero de patrones), sumado a las
    similitudes X^T x antes del softmax -- un valor grande en el
    indice del patron objetivo aumenta su peso en la combinacion.
    """
    z = X.T @ x + bias
    weights = _softmax(beta * z)
    return -x + X @ weights


def simulate_longitudinal_patient_hopfield_biased(
    X: np.ndarray, target_idx: int, n_genes: int,
    n_timepoints: int = 8, months_between_checks: int = 3,
    recurrence_onset_month: int = 15, beta: float = 2.0,
    max_bias_strength: float = 5.0,
) -> tuple:
    """Version de simulate_longitudinal_patient_hopfield() usando el
    mecanismo de sesgo en softmax en vez de forzamiento aditivo."""
    n_patterns = X.shape[1]
    t_checks = np.arange(0, n_timepoints * months_between_checks, months_between_checks)
    x_series = np.zeros((n_genes, n_timepoints))
    x_current = np.zeros(n_genes)

    for i, t in enumerate(t_checks):
        bias = np.zeros(n_patterns)
        if t >= recurrence_onset_month:
            months_since_onset = t - recurrence_onset_month
            strength = min(0.15 * months_since_onset, max_bias_strength)
            bias[target_idx] = strength

        sol = solve_ivp(
            lambda tt, xx: modern_hopfield_field_biased(xx, X, beta, bias),
            (0, months_between_checks), x_current, method="RK45", rtol=1e-8, atol=1e-10)
        x_current = sol.y[:, -1]
        x_series[:, i] = x_current

    return t_checks, x_series


def find_minimum_bias_strength(
    patterns: dict, target_label: str, beta: float = 3.0,
    strength_candidates=None, corr_threshold: float = 0.9,
    recurrence_onset_month: int = 15, n_timepoints: int = 10,
) -> dict:
    """Version de find_minimum_forcing_strength() para el mecanismo de
    sesgo en vez de forzamiento aditivo -- misma logica de busqueda y
    diagnostico, mecanismo distinto."""
    if strength_candidates is None:
        strength_candidates = [1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]

    X, labels = patterns_to_matrix(patterns)
    n_genes = X.shape[0]
    idx_target = labels.index(target_label)
    p_target = X[:, idx_target]

    resultados = []
    umbral_minimo_encontrado = None
    for fuerza in strength_candidates:
        t, x = simulate_longitudinal_patient_hopfield_biased(
            X, idx_target, n_genes, beta=beta, max_bias_strength=fuerza,
            n_timepoints=n_timepoints, recurrence_onset_month=recurrence_onset_month)
        x_final = x[:, -1]

        if np.std(x_final) < 1e-12:
            resultados.append({"fuerza": fuerza, "corr_con_objetivo": float("nan"),
                                "corr_con_mas_parecido": float("nan"), "mas_parecido_a": "origen"})
            continue

        corr_target = float(np.corrcoef(x_final, p_target)[0, 1])
        mejor_label, mejor_corr = None, -2.0
        for i, label in enumerate(labels):
            c = float(np.corrcoef(x_final, X[:, i])[0, 1])
            if c > mejor_corr:
                mejor_corr, mejor_label = c, label

        resultados.append({
            "fuerza": fuerza, "corr_con_objetivo": corr_target,
            "corr_con_mas_parecido": mejor_corr, "mas_parecido_a": mejor_label,
        })
        if corr_target >= corr_threshold and umbral_minimo_encontrado is None:
            umbral_minimo_encontrado = fuerza

    return {
        "patron_objetivo": target_label,
        "umbral_minimo_encontrado": umbral_minimo_encontrado,
        "detalle": resultados,
    }


def compare_forced_trajectories_old_vs_new(
    patterns: dict, W_old: np.ndarray, gene_order: list,
    beta_old: float = 2.0, beta_new: float = 3.0, max_forcing_strength_new: float = 1.5,
    recurrence_onset_month: int = 15, n_timepoints: int = 10,
) -> "pd.DataFrame":
    """
    La pregunta central para decidir si este rediseno mejora el USO
    CLINICO real (no solo las propiedades matematicas abstractas):
    con el MISMO forzamiento activo que usa prognosis_demo.py, la
    dinamica NUEVA converge al patron objetivo tan bien o mejor que la
    ANTIGUA, incluso para los patrones con cuenca libre mas chica
    (CMS2/CMS4)?
    """
    import pandas as pd
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from prognosis_demo import simulate_longitudinal_patient as simulate_old
    from prognosis import hazard_from_trajectory

    X, labels = patterns_to_matrix(patterns)
    n_genes = len(gene_order)
    filas = []
    for i, label in enumerate(labels):
        p = X[:, i]

        t_old, x_old = simulate_old(W_old, gene_order, p, n_genes, beta=beta_old,
                                      n_timepoints=n_timepoints, recurrence_onset_month=recurrence_onset_month)
        h_old = hazard_from_trajectory(x_old)
        corr_old = float(np.corrcoef(x_old[:, -1], p)[0, 1]) if np.std(x_old[:, -1]) > 1e-12 else float("nan")

        t_new, x_new = simulate_longitudinal_patient_hopfield(
            X, p, n_genes, beta=beta_new, max_forcing_strength=max_forcing_strength_new,
            n_timepoints=n_timepoints, recurrence_onset_month=recurrence_onset_month)
        h_new = hazard_from_trajectory(x_new)
        corr_new = float(np.corrcoef(x_new[:, -1], p)[0, 1]) if np.std(x_new[:, -1]) > 1e-12 else float("nan")

        filas.append({
            "patron": label,
            "hazard_final_antigua": float(h_old[-1]), "correlacion_final_antigua": corr_old,
            "hazard_final_nueva": float(h_new[-1]), "correlacion_final_nueva": corr_new,
        })
    return pd.DataFrame(filas)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Verifica si los patrones calibrados son atractores genuinos "
                    "bajo la dinamica de Hopfield moderno (rediseno de attractor_model.py)")
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--sweep", action="store_true",
                         help="Barrer varios valores de beta en vez de solo el especificado")
    parser.add_argument("--compare-forced", action="store_true",
                         help="Comparar trayectorias clinicas FORZADAS (como las usa prognosis_demo.py) "
                              "entre la dinamica antigua y esta nueva")
    parser.add_argument("--max-forcing-strength", type=float, default=1.5,
                         help="Fuerza maxima de forzamiento para --compare-forced -- CUIDADO: el "
                              "umbral suficiente es ESPECIFICO de cada calibracion, no un numero "
                              "universal (con datos sinteticos de prueba, 1.5 basto; con la "
                              "calibracion real de GSE39582, no basto para CMS2 -- terminaba "
                              "correlacionando NEGATIVO con el objetivo). Usar "
                              "--find-forcing-threshold para buscar el valor correcto en tus datos "
                              "antes de confiar en un numero fijo.")
    parser.add_argument("--find-forcing-threshold",
                         help="Buscar la fuerza minima necesaria para converger correctamente hacia "
                              "el patron dado (nombre exacto, ej. CMS2_canonical_WNT) -- no asumir "
                              "que un valor fijo funciona para todos los patrones/calibraciones.")
    parser.add_argument("--find-bias-threshold",
                         help="Igual que --find-forcing-threshold pero con el mecanismo ALTERNATIVO "
                              "(sesgo en softmax en vez de fuerza aditiva) -- NO VERIFICADO como "
                              "solucion, un intento de reproduccion sintetica del problema real dio "
                              "resultados inconsistentes. Probar directamente, sin asumir que funciona.")
    parser.add_argument("--verify-stabilizer", action="store_true",
                         help="Verifica el termino estabilizador para la fase pre-recaida: calcula k "
                              "automaticamente, confirma que el origen pasa a ser estable, y prueba "
                              "la simulacion v2 completa (estabilizador + forzamiento) contra los 4 "
                              "patrones -- la verificacion definitiva pendiente en datos reales.")
    parser.add_argument("--compare-stabilized-sweep", action="store_true",
                         help="Barre la misma lista de fuerzas en V1 y V2 para los cuatro CMS, "
                              "incluyendo diagnostico del residuo pre-recaida")
    parser.add_argument("--forcing-candidates", nargs="+", type=float,
                         default=[0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0],
                         help="Fuerzas maximas para --compare-stabilized-sweep")
    parser.add_argument("--withdrawal-time", type=float, default=30.0,
                         help="Tiempo de relajacion libre despues de retirar el driver")
    parser.add_argument("--unnormalized-driver", action="store_true",
                         help="Diagnostico legado: usar fuerza*patron en vez de direccion unitaria")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibration import load_calibrated_patterns

    patterns, gene_order = load_calibrated_patterns(args.patterns)
    X, labels = patterns_to_matrix(patterns)

    print("=" * 78)
    print(f"1. VERIFICACION DE PROPIEDADES ESTRUCTURALES (flujo de gradiente)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    x_test = rng.normal(0, 1, size=X.shape[0])
    J = modern_hopfield_jacobian(x_test, X, args.beta)
    print(f"Jacobiano simetrico (flujo de gradiente genuino): {np.allclose(J, J.T)}")
    energia = verify_energy_decreases(X, args.beta, n_trajectories=20)
    print(f"Energia decrece monotonamente en {energia['n_trayectorias']} trayectorias: "
          f"{energia['todas_decrecientes']} ({energia['violaciones_monotonicidad']} violaciones)")

    print("\n" + "=" * 78)
    print(f"2. EQUILIBRIOS Y ESTABILIDAD (beta={args.beta})")
    print("=" * 78)
    tabla = verify_all_patterns_hopfield(patterns, args.beta)
    print(tabla.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"3. CUENCAS DE ATRACCION (empirico, n={args.n_samples})")
    print("=" * 78)
    cuencas = empirical_basin_membership_hopfield(patterns, args.beta, n_samples=args.n_samples)
    print(cuencas.to_string())

    if args.sweep:
        print("\n" + "=" * 78)
        print("4. BARRIDO DE BETA")
        print("=" * 78)
        barrido = sweep_beta_hopfield(patterns)
        print(barrido.to_string(index=False))

    if args.find_forcing_threshold:
        print("\n" + "=" * 78)
        print(f"BUSQUEDA DE FUERZA MINIMA DE FORZAMIENTO -- objetivo: {args.find_forcing_threshold}")
        print("=" * 78)
        resultado_umbral = find_minimum_forcing_strength(
            patterns, args.find_forcing_threshold, beta=args.beta)
        for r in resultado_umbral["detalle"]:
            print(f"  fuerza={r['fuerza']:6.1f}  corr_objetivo={r['corr_con_objetivo']:+.3f}  "
                  f"mas_parecido_a={r['mas_parecido_a']} (corr={r['corr_con_mas_parecido']:.3f})")
        if resultado_umbral["umbral_minimo_encontrado"] is not None:
            print(f"\nUmbral minimo encontrado: {resultado_umbral['umbral_minimo_encontrado']}")
        else:
            print("\nAVISO: ningun candidato probado (hasta 20.0) logro correlacion >= 0.9 con "
                  "el objetivo -- ampliar strength_candidates o revisar si este patron tiene un "
                  "problema mas de fondo (ej. muy cerca de otro patron dominante).")

    if args.find_bias_threshold:
        print("\n" + "=" * 78)
        print(f"BUSQUEDA DE SESGO MINIMO (mecanismo alternativo) -- objetivo: {args.find_bias_threshold}")
        print("=" * 78)
        resultado_bias = find_minimum_bias_strength(patterns, args.find_bias_threshold, beta=args.beta)
        for r in resultado_bias["detalle"]:
            print(f"  fuerza={r['fuerza']:6.1f}  corr_objetivo={r['corr_con_objetivo']:+.3f}  "
                  f"mas_parecido_a={r['mas_parecido_a']} (corr={r['corr_con_mas_parecido']:.3f})")
        if resultado_bias["umbral_minimo_encontrado"] is not None:
            print(f"\nUmbral minimo encontrado (sesgo): {resultado_bias['umbral_minimo_encontrado']}")
        else:
            print("\nAVISO: el mecanismo de sesgo TAMPOCO logro correlacion >= 0.9 con "
                  "los candidatos probados.")

    if args.verify_stabilizer:
        print("\n" + "=" * 78)
        print("VERIFICACION DEL TERMINO ESTABILIZADOR (fase pre-recaida)")
        print("=" * 78)
        k = compute_stabilizing_k(X, args.beta)
        print(f"k calculado automaticamente para esta calibracion: {k:.3f}")

        origen = np.zeros(X.shape[0])
        J0 = modern_hopfield_jacobian_stabilized(origen, X, args.beta, k)
        eigvals0 = np.linalg.eigvalsh(J0)
        residuo_origen = float(np.linalg.norm(
            modern_hopfield_field_stabilized(origen, X, args.beta, k)))
        print(f"Eigenvalores en el origen CON estabilizador: {np.round(eigvals0, 3)}")
        print(f"Residuo del campo en el origen: {residuo_origen:.3e}")
        print(f"Origen genuinamente estable ahora: "
              f"{bool(residuo_origen <= 1e-8 and np.all(eigvals0 < 0))}")

        x_current = np.zeros(X.shape[0])
        for t in np.arange(0, 15, 3):
            sol = solve_ivp(lambda tt, xx: modern_hopfield_field_stabilized(xx, X, args.beta, k),
                             (0, 3), x_current, method="RK45", rtol=1e-8, atol=1e-10)
            x_current = sol.y[:, -1]
        print(f"\nNorma del estado al final de la fase pre-recaida (deberia ser ~0): "
              f"{np.linalg.norm(x_current):.6f}")

        print("\nSimulacion v2 completa (estabilizador + forzamiento) contra los 4 patrones:")
        for i, label in enumerate(labels):
            p = X[:, i]
            _, x_sim = simulate_longitudinal_patient_hopfield_v2(
                X, p, X.shape[0], beta=args.beta, max_forcing_strength=args.max_forcing_strength,
                stabilizing_k=k, n_timepoints=10, recurrence_onset_month=15,
                normalize_driver=True)
            x_final = x_sim[:, -1]
            corr = float(np.corrcoef(x_final, p)[0, 1]) if np.std(x_final) > 1e-12 else float("nan")
            print(f"  {label:20s} corr_con_objetivo={corr:+.3f}")

    barrido_estabilizado = None
    resumen_estabilizado = None
    if args.compare_stabilized_sweep:
        print("\n" + "=" * 78)
        print("BARRIDO DE FUERZA V1 vs V2 ESTABILIZADA")
        print("=" * 78)
        diagnostico_basal = diagnose_pre_recurrence_residual(patterns, beta=args.beta)
        print(f"Residuo del campo en origen: {diagnostico_basal['origin_field_residual']:.3e}")
        print(f"Norma al final de fase pre-recaida: {diagnostico_basal['final_norm']:.6f}")
        print(f"Direccion residual: {diagnostico_basal['closest_cms']} "
              f"(corr={diagnostico_basal['closest_correlation']:+.3f})")
        barrido_estabilizado = compare_forcing_sweep_v1_v2(
            patterns, beta=args.beta, strength_candidates=args.forcing_candidates,
            withdrawal_time=args.withdrawal_time,
            normalize_driver=not args.unnormalized_driver)
        resumen_estabilizado = summarize_forcing_thresholds(barrido_estabilizado)
        print("\nDetalle:")
        print(barrido_estabilizado.to_string(index=False))
        print("\nPrimer candidato exitoso por CMS:")
        print(resumen_estabilizado.to_string(index=False))

    comparacion_forzada = None
    if args.compare_forced:
        print("\n" + "=" * 78)
        print("5. TRAYECTORIAS CLINICAS FORZADAS: ANTIGUA vs NUEVA")
        print("=" * 78)
        from attractor_model import build_model_from_patterns
        W_old, _, _ = build_model_from_patterns(patterns)
        comparacion_forzada = compare_forced_trajectories_old_vs_new(
            patterns, W_old, gene_order, beta_old=2.0, beta_new=args.beta,
            max_forcing_strength_new=args.max_forcing_strength)
        print(comparacion_forzada.to_string(index=False))

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(out / "modern_hopfield_equilibria.tsv", sep="\t", index=False)
        cuencas.to_csv(out / "modern_hopfield_basin_membership.tsv", sep="\t")
        if args.sweep:
            barrido.to_csv(out / "modern_hopfield_beta_sweep.tsv", sep="\t", index=False)
        if comparacion_forzada is not None:
            comparacion_forzada.to_csv(out / "modern_hopfield_forced_comparison.tsv", sep="\t", index=False)
        if barrido_estabilizado is not None:
            barrido_estabilizado.to_csv(
                out / "modern_hopfield_stabilized_forcing_sweep.tsv", sep="\t", index=False)
            resumen_estabilizado.to_csv(
                out / "modern_hopfield_stabilized_thresholds.tsv", sep="\t", index=False)
        print(f"\nTablas guardadas en: {out}")


if __name__ == "__main__":
    main()
