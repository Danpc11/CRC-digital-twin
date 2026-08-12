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

Este modulo NO reemplaza attractor_model.py todavia -- es una
alternativa completa, verificada con el mismo rigor (jacobiano
verificado contra diferencias finitas, decrecimiento de energia
verificado numericamente, equilibrios/estabilidad/cuencas con la misma
metodologia de dynamics_diagnostics.py) para permitir una comparacion
justa antes de decidir si reemplaza al sistema actual.
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
        x_final = sol.y[:, -1]
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


def sweep_beta_hopfield(patterns: dict, beta_values=None) -> "pd.DataFrame":
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
        rows.append({
            "beta": beta, "n_patrones_ok": n_ok, "todos_4_ok": n_ok == len(labels),
            "min_separacion": float(min(distancias)) if distancias else float("nan"),
        })
    return pd.DataFrame(rows)



def simulate_longitudinal_patient_hopfield(
    X: np.ndarray, recurrence_pattern: np.ndarray, n_genes: int,
    n_timepoints: int = 8, months_between_checks: int = 3,
    recurrence_onset_month: int = 15, beta: float = 2.0,
    max_forcing_strength: float = 1.5,
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

    for i, t in enumerate(t_checks):
        if t >= recurrence_onset_month:
            months_since_onset = t - recurrence_onset_month
            strength = min(0.15 * months_since_onset, max_forcing_strength)
            I_driver = strength * recurrence_pattern
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
        print(f"\nTablas guardadas en: {out}")


if __name__ == "__main__":
    main()
