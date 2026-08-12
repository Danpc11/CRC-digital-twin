"""
dynamics_diagnostics.py

Verifica lo que build_model_from_patterns() NUNCA verifico: que los
patrones calibrados sean equilibrios REALES del sistema no lineal
completo, que esos equilibrios sean localmente ESTABLES (atractores
en sentido estricto, no solo puntos fijos), y caracteriza
empiricamente sus cuencas de atraccion.

POR QUE ESTO IMPORTA
---------------------
projection_weight_matrix() garantiza W @ p = p para el sistema
LINEALIZADO (dx/dt = -x + Wx). Pero la dinamica real del proyecto es
NO lineal:

    dx/dt = -x + W tanh(beta x)

y tanh(beta x) != x salvo en el origen -- para x != 0, W @ tanh(beta x)
puede no ser igual a x incluso si W @ x = x. Es decir: los patrones
calibrados p^mu podrian NO ser equilibrios del sistema real, y aunque
lo fueran, nunca se verifico si son ESTABLES (que una perturbacion
pequena vuelva al equilibrio) -- llamarlos "atractores" sin esto es,
en sentido estricto de sistemas dinamicos, una afirmacion no probada.

TRES DIAGNOSTICOS
------------------
1. find_true_equilibrium() -- resuelve f(x)=0 numericamente partiendo
   de cada patron calibrado como semilla (scipy.optimize.fsolve).
   Compara la posicion del equilibrio real contra el patron original.
2. stability_at_equilibrium() -- jacobiano de f en el equilibrio,
   J = -I + W @ diag(beta*(1-tanh(beta*x)^2)) (derivado y verificado
   simbolicamente con sympy antes de codificar). Estable si y solo si
   todos los eigenvalores tienen parte real negativa.
3. empirical_basin_membership() -- muestrea puntos iniciales al azar,
   integra la dinamica, y clasifica a que patron converge cada uno.
   No es un mapeo geometrico completo de las cuencas (inviable de
   visualizar en 10 dimensiones), pero da una caracterizacion empirica
   real: que fraccion del espacio de estados alrededor de los patrones
   efectivamente termina en cada atractor, y que fraccion no converge
   claramente a ninguno (señal de comportamiento no capturado por los
   4 patrones -- oscilacion, otro equilibrio, o divergencia).

USO
    python3 src/dynamics_diagnostics.py \\
        --patterns results_gse39582_final/calibrated_patterns.tsv \\
        --output results_dynamics_diagnostics/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def vector_field(x: np.ndarray, W: np.ndarray, beta: float) -> np.ndarray:
    """f(x) = -x + W tanh(beta x) -- lado derecho de la dinamica, sin el termino I_driver."""
    return -x + W @ np.tanh(beta * x)


def jacobian_at_state(x: np.ndarray, W: np.ndarray, beta: float) -> np.ndarray:
    """
    Jacobiano de f(x) = -x + W tanh(beta x), derivado y verificado
    simbolicamente (sympy) antes de codificar:

        J[i,k] = -delta_ik + W[i,k] * beta * (1 - tanh(beta*x[k])^2)
    """
    n = len(x)
    dtanh = beta * (1.0 - np.tanh(beta * x) ** 2)  # derivada elemento a elemento, shape (n,)
    return -np.eye(n) + W * dtanh[np.newaxis, :]  # W[i,k] * dtanh[k], broadcast por columnas


def find_true_equilibrium(x0: np.ndarray, W: np.ndarray, beta: float):
    """
    Resuelve f(x)=0 numericamente partiendo de x0 (tipicamente el
    patron calibrado, que es exacto solo para el sistema linealizado).
    Devuelve (equilibrio, convergio: bool, desplazamiento respecto a x0).
    """
    sol, info, ier, msg = fsolve(
        lambda x: vector_field(x, W, beta), x0, full_output=True, xtol=1e-12)
    converged = ier == 1
    desplazamiento = float(np.linalg.norm(sol - x0))
    return sol, converged, desplazamiento


def stability_at_equilibrium(x_eq: np.ndarray, W: np.ndarray, beta: float):
    """
    Estabilidad local via eigenvalores del jacobiano. Estable
    (asintoticamente, localmente) si y solo si todas las partes reales
    son negativas -- una perturbacion pequena decae de vuelta al
    equilibrio en vez de alejarse.
    """
    J = jacobian_at_state(x_eq, W, beta)
    eigvals = np.linalg.eigvals(J)
    stable = bool(np.all(eigvals.real < 0))
    max_real_part = float(np.max(eigvals.real))
    return eigvals, stable, max_real_part


def verify_all_patterns(patterns: dict, W: np.ndarray, beta: float = 2.0) -> pd.DataFrame:
    """Corre find_true_equilibrium + stability_at_equilibrium para cada patron calibrado."""
    rows = []
    for label, p in patterns.items():
        x_eq, converged, desplazamiento = find_true_equilibrium(p, W, beta)
        eigvals, stable, max_real = stability_at_equilibrium(x_eq, W, beta)
        if np.linalg.norm(x_eq) > 1e-8 and np.std(x_eq) > 1e-12:
            corr = float(np.corrcoef(x_eq, p)[0, 1])
        else:
            corr = float("nan")
        rows.append({
            "patron": label,
            "convergio_a_equilibrio": converged,
            "desplazamiento_vs_patron_calibrado": desplazamiento,
            "localmente_estable": stable,
            "max_parte_real_eigenvalor": max_real,
            "correlacion_equilibrio_vs_patron": corr,
        })
    return pd.DataFrame(rows)


def empirical_basin_membership(
    patterns: dict, W: np.ndarray, beta: float = 2.0,
    n_samples: int = 300, integration_time: float = 60.0,
    corr_threshold: float = 0.8, seed: int = 0,
) -> pd.DataFrame:
    """
    Muestrea n_samples puntos iniciales aleatorios (escala similar a
    los patrones calibrados), integra la dinamica hasta
    integration_time, y clasifica a que patron converge cada uno por
    correlacion. No es un mapeo geometrico completo de las cuencas
    (inviable en 10D) -- es una caracterizacion empirica de que
    proporcion del espacio alrededor de los patrones efectivamente
    termina en cada atractor.
    """
    rng = np.random.default_rng(seed)
    gene_labels = list(patterns.keys())
    n_genes = len(next(iter(patterns.values())))
    scale = float(np.mean([np.linalg.norm(p) for p in patterns.values()]))

    resultados = []
    for _ in range(n_samples):
        x0 = rng.normal(0, scale / np.sqrt(n_genes), size=n_genes)
        sol = solve_ivp(
            lambda t, x: vector_field(x, W, beta), (0, integration_time), x0,
            method="RK45", rtol=1e-8, atol=1e-10)
        x_final = sol.y[:, -1]
        norm_final = float(np.linalg.norm(x_final))

        if norm_final < 1e-6:
            resultados.append({"convergio_a": "origen", "correlacion": np.nan, "norma_final": norm_final})
            continue

        mejor_label, mejor_corr = None, -2.0
        for label, p in patterns.items():
            if np.linalg.norm(x_final) < 1e-8 or np.linalg.norm(p) < 1e-8:
                continue
            c = float(np.corrcoef(x_final, p)[0, 1])
            if not np.isnan(c) and c > mejor_corr:
                mejor_corr, mejor_label = c, label

        etiqueta = mejor_label if mejor_corr >= corr_threshold else "ninguno_claro"
        resultados.append({"convergio_a": etiqueta, "correlacion": mejor_corr, "norma_final": norm_final})

    df = pd.DataFrame(resultados)
    resumen = df["convergio_a"].value_counts(normalize=True).rename("proporcion").to_frame()
    resumen["n"] = df["convergio_a"].value_counts()
    return resumen


def sweep_beta_stability(patterns: dict, W: np.ndarray, beta_values=None) -> pd.DataFrame:
    """
    Barre beta buscando la transicion de estabilidad. Justificacion
    analitica del punto critico: en el origen, tanh(beta*x)~=beta*x,
    asi que el jacobiano ahi es J(0) = -I + beta*W. Como W es un
    proyector ortogonal (eigenvalores en {0,1}), estabilidad en el
    origen requiere beta < 1 -- el mismo fenomeno de "ganancia critica"
    conocido en redes tipo Hopfield con no linealidad saturante.
    """
    if beta_values is None:
        beta_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]
    rows = []
    for beta in beta_values:
        tabla = verify_all_patterns(patterns, W, beta)
        rows.append({
            "beta": beta,
            "todos_estables": bool(tabla["localmente_estable"].all()),
            "peor_parte_real": float(tabla["max_parte_real_eigenvalor"].max()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attractor_model import build_model_from_patterns
    from calibration import load_calibrated_patterns

    patterns, gene_order = load_calibrated_patterns(args.patterns)
    W, _, _ = build_model_from_patterns(patterns)

    print("=" * 78)
    print(f"1+2. EQUILIBRIOS REALES Y ESTABILIDAD (beta={args.beta})")
    print("=" * 78)
    tabla = verify_all_patterns(patterns, W, args.beta)
    print(tabla.to_string(index=False))

    if not tabla["localmente_estable"].all():
        inestables = tabla[~tabla["localmente_estable"]]["patron"].tolist()
        print(f"\nAVISO: {inestables} NO son localmente estables en el sistema no lineal "
              f"con beta={args.beta} -- no califican como atractores en sentido estricto. "
              "Corriendo un barrido de beta para ubicar el punto de transicion...")
        barrido = sweep_beta_stability(patterns, W)
        print(barrido.to_string(index=False))
    else:
        print("\nTodos los patrones son equilibrios localmente estables del sistema no "
              "lineal completo -- confirmado, no solo asumido.")

    print("\n" + "=" * 78)
    print(f"3. CUENCAS DE ATRACCION (empirico, n={args.n_samples} muestras aleatorias)")
    print("=" * 78)
    cuencas = empirical_basin_membership(patterns, W, args.beta, n_samples=args.n_samples)
    print(cuencas.to_string())

    if "ninguno_claro" in cuencas.index and cuencas.loc["ninguno_claro", "proporcion"] > 0.05:
        print(f"\nAVISO: {cuencas.loc['ninguno_claro', 'proporcion']*100:.1f}% de las "
              "trayectorias no convergieron claramente a ninguno de los 4 patrones -- "
              "podria haber equilibrios adicionales no capturados por la calibracion.")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(out / "dynamics_equilibria_stability.tsv", sep="\t", index=False)
        cuencas.to_csv(out / "dynamics_basin_membership.tsv", sep="\t")
        print(f"\nTablas guardadas en: {out}")


if __name__ == "__main__":
    main()
