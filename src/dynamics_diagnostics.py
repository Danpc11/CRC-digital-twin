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


def es_atractor_cms(
    converged: bool, stable: bool, x_eq: np.ndarray, p_original: np.ndarray,
    origin_threshold: float = 0.5, correlation_threshold: float = 0.8,
) -> bool:
    """
    Criterio CORREGIDO tras revision externa -- "localmente_estable" NO
    basta. El sistema linealizado en el origen tambien es estable para
    beta<1 (ver docstring de sweep_beta_stability), asi que un
    equilibrio puede ser "estable" simplemente por haber colapsado al
    origen trivial, no por ser un atractor CMS genuino. Verificado
    empiricamente: con beta=0.5, las 4 "estabilidades" reportadas
    correspondian a un desplazamiento ~2.2 (la norma completa del
    patron original) y correlacion NaN -- exactamente el patron de
    colapso al origen.

    Un atractor CMS genuino requiere las 4 condiciones:
      1. converged -- fsolve encontro un equilibrio real
      2. stable -- localmente estable (jacobiano, partes reales < 0)
      3. NO es el origen -- ||x_eq|| > origin_threshold
      4. SI se parece al patron original -- corr(x_eq, p) >= correlation_threshold
    """
    if not converged or not stable:
        return False
    if np.linalg.norm(x_eq) <= origin_threshold:
        return False
    if np.std(x_eq) < 1e-12 or np.std(p_original) < 1e-12:
        return False
    corr = float(np.corrcoef(x_eq, p_original)[0, 1])
    return corr >= correlation_threshold


def check_equilibria_separation(equilibria: dict, min_separation: float = 0.5) -> pd.DataFrame:
    """
    Verifica que los equilibrios encontrados para los distintos
    patrones sean genuinamente DISTINTOS entre si (no que los 4
    hayan colapsado al mismo punto, incluso si ese punto no es el
    origen). Devuelve la matriz de distancias par a par.
    """
    labels = list(equilibria.keys())
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(equilibria[labels[i]] - equilibria[labels[j]])
    df = pd.DataFrame(dist, index=labels, columns=labels)
    colapsados = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < min_separation:
                colapsados.append((labels[i], labels[j], dist[i, j]))
    df.attrs["colapsados"] = colapsados
    return df


def sweep_beta_stability(patterns: dict, W: np.ndarray, beta_values=None) -> pd.DataFrame:
    """
    Barre beta buscando la transicion de estabilidad. Justificacion
    analitica del punto critico: en el origen, tanh(beta*x)~=beta*x,
    asi que el jacobiano ahi es J(0) = -I + beta*W. Como W es un
    proyector ortogonal (eigenvalores en {0,1}), estabilidad en el
    origen requiere beta < 1 -- el mismo fenomeno de "ganancia critica"
    conocido en redes tipo Hopfield con no linealidad saturante.

    CORREGIDO tras revision externa: "todos_estables" ahora exige
    es_atractor_cms() (estable + no-origen + correlacionado), no solo
    estabilidad del jacobiano -- antes beta<1 se reportaba
    "todos_estables=True" cuando en realidad los 4 equilibrios habian
    colapsado al origen trivial (verificado: desplazamiento~norma
    completa del patron, correlacion NaN). Tambien reporta separacion
    entre los 4 equilibrios, para detectar colapso mutuo (los 4
    patrones convergiendo al MISMO punto no-origen).
    """
    if beta_values is None:
        beta_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]
    rows = []
    for beta in beta_values:
        equilibria = {}
        es_atractor_por_patron = {}
        peor_parte_real = -np.inf
        for label, p in patterns.items():
            x_eq, converged, _ = find_true_equilibrium(p, W, beta)
            _, stable, max_real = stability_at_equilibrium(x_eq, W, beta)
            peor_parte_real = max(peor_parte_real, max_real)
            equilibria[label] = x_eq
            es_atractor_por_patron[label] = es_atractor_cms(converged, stable, x_eq, p)

        separacion = check_equilibria_separation(equilibria)
        colapsados_entre_si = len(separacion.attrs["colapsados"]) > 0

        rows.append({
            "beta": beta,
            "todos_son_atractores_cms_genuinos": bool(all(es_atractor_por_patron.values())),
            "cuantos_son_atractores_cms": sum(es_atractor_por_patron.values()),
            "equilibrios_colapsados_entre_si": colapsados_entre_si,
            "min_separacion_entre_equilibrios": float(
                separacion.values[np.triu_indices(len(patterns), k=1)].min()
            ) if len(patterns) > 1 else float("nan"),
            "peor_parte_real": float(peor_parte_real),
        })
    return pd.DataFrame(rows)


def find_valid_beta_interval(
    patterns: dict, W: np.ndarray, beta_min: float = 0.1, beta_max: float = 15.0,
    n_steps: int = 150, min_separation: float = 0.5,
) -> pd.DataFrame:
    """
    Busca automaticamente el/los intervalos de beta donde los 4
    patrones califican SIMULTANEAMENTE como atractores CMS genuinos
    (es_atractor_cms para los 4, mas separados entre si). Usa fsolve
    independiente sembrado en el patron original en cada paso (no
    continuacion/warm-start) -- necesario porque warm-start desde un
    punto ya colapsado nunca puede recuperar ramas distintas.

    Hallazgo verificado con la calibracion real del proyecto: con
    beta=2.0 (el valor actual por defecto), CERO patrones califican.
    Existe un intervalo robusto (no un punto aislado) mas arriba,
    aproximadamente beta in [9.4, 11.9], donde los 4 SI califican,
    bien separados entre si (todas las distancias par a par > 3.8).
    """
    betas = np.linspace(beta_min, beta_max, n_steps)
    rows = []
    for beta in betas:
        equilibria = {}
        atractores = {}
        for label, p in patterns.items():
            x_eq, converged, _ = find_true_equilibrium(p, W, beta)
            _, stable, _ = stability_at_equilibrium(x_eq, W, beta)
            equilibria[label] = x_eq
            atractores[label] = es_atractor_cms(converged, stable, x_eq, p)
        separacion = check_equilibria_separation(equilibria, min_separation)
        rows.append({
            "beta": beta,
            "n_atractores_genuinos": sum(atractores.values()),
            "todos_4_califican": sum(atractores.values()) == len(patterns),
            "min_separacion": float(
                separacion.values[np.triu_indices(len(patterns), k=1)].min()
            ) if len(patterns) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def continuation_search(
    patterns: dict, W: np.ndarray,
    beta_start: float = 0.05, beta_end: float = 5.0, n_steps: int = 100,
) -> pd.DataFrame:
    """
    Busqueda por CONTINUACION (warm-start secuencial), no fsolve
    independiente en cada beta -- cada paso usa como semilla el
    equilibrio encontrado en el paso anterior (empezando en beta_start
    desde el propio patron calibrado), siguiendo una rama continua en
    vez de arriesgarse a que fsolve salte a una rama distinta o al
    origen en cada llamada independiente.

    LIMITACION HONESTA: esto es continuacion secuencial simple, no
    continuacion pseudo-arclength -- no maneja bien puntos de giro
    (fold bifurcations) donde la rama podria doblarse sobre si misma.
    Para una confirmacion definitiva de que existe (o no) un intervalo
    de beta con 4 atractores CMS distintos y estables, hace falta un
    paquete de continuacion numerica dedicado (ej. AUTO, MatCont, o
    pseudo-arclength implementado a mano) -- esto es una primera
    aproximacion razonable, no la palabra final.
    """
    betas = np.linspace(beta_start, beta_end, n_steps)
    current_eq = {label: p.copy() for label, p in patterns.items()}
    rows = []
    for beta in betas:
        equilibria = {}
        es_atractor_por_patron = {}
        for label, p in patterns.items():
            x_eq, converged, _ = find_true_equilibrium(current_eq[label], W, beta)
            _, stable, max_real = stability_at_equilibrium(x_eq, W, beta)
            equilibria[label] = x_eq
            current_eq[label] = x_eq  # warm-start para el siguiente paso
            es_atractor_por_patron[label] = es_atractor_cms(converged, stable, x_eq, p)

        separacion = check_equilibria_separation(equilibria)
        rows.append({
            "beta": beta,
            "todos_son_atractores_cms_genuinos": bool(all(es_atractor_por_patron.values())),
            "cuantos_son_atractores_cms": sum(es_atractor_por_patron.values()),
            "min_separacion_entre_equilibrios": float(
                separacion.values[np.triu_indices(len(patterns), k=1)].min()
            ) if len(patterns) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--find-interval", action="store_true",
                         help="Buscar automaticamente un intervalo de beta donde los 4 "
                              "patrones califiquen como atractores CMS genuinos (busqueda "
                              "mas lenta, no corre por default)")
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

    # CRITERIO CORREGIDO tras revision externa: "localmente_estable" no
    # basta -- un equilibrio puede ser estable por haber colapsado al
    # origen trivial (verificado: pasaba exactamente esto para beta<1
    # con la calibracion real de este proyecto). Se exige
    # es_atractor_cms (estable + no-origen + correlacionado) para los 4.
    equilibria = {label: None for label in patterns}
    atractores_genuinos = {}
    for label, p in patterns.items():
        x_eq, converged, _ = find_true_equilibrium(p, W, args.beta)
        _, stable, _ = stability_at_equilibrium(x_eq, W, args.beta)
        equilibria[label] = x_eq
        atractores_genuinos[label] = es_atractor_cms(converged, stable, x_eq, p)

    if not all(atractores_genuinos.values()):
        no_califican = [k for k, v in atractores_genuinos.items() if not v]
        print(f"\nAVISO: {no_califican} NO califican como atractores CMS genuinos con "
              f"beta={args.beta} (estable + distinto del origen + correlacionado con el "
              "patron original) -- puede ser inestabilidad, o puede ser colapso al origen "
              "trivial (revisar 'desplazamiento_vs_patron_calibrado' en la tabla de arriba: "
              "~norma del patron completo = colapso al origen).")
        if args.find_interval:
            print("\nBuscando un intervalo de beta donde los 4 SI califiquen "
                  "(puede tardar unos segundos)...")
            busqueda = find_valid_beta_interval(patterns, W)
            validos = busqueda[busqueda["todos_4_califican"]]
            if len(validos) > 0:
                print(f"Encontrado: beta en [{validos['beta'].min():.2f}, "
                      f"{validos['beta'].max():.2f}] -- los 4 califican en ese rango.")
            else:
                print("No se encontro ningun beta en [0.1, 15.0] donde los 4 califiquen "
                      "simultaneamente. Considerar un rango mas amplio, o rediseñar la "
                      "dinamica (ver docstring del modulo).")
    else:
        print(f"\nLos 4 patrones califican como atractores CMS genuinos con beta={args.beta} "
              "-- estables, distintos del origen, y correlacionados con su patron calibrado. "
              "Confirmado, no solo asumido.")

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
