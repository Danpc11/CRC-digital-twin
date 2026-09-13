"""
treatment_simulation_demo.py

Demo contrafactual: simula la MISMA trayectoria de recaida de un
paciente dos veces -- una vez sin tratamiento (linea base, igual que
prognosis_demo.py) y otra vez con un tratamiento aplicado en el
momento en que se detecta la alerta de recurrencia. Compara ambas.

Esta es la demostracion real de "gemelo digital" en el sentido fuerte:
no solo clasifica o alerta, simula el efecto contrafactual de una
intervencion -- "que hubiera pasado si...". Ver treatment_perturbation.py
para las limitaciones explicitas de este modulo (direccion fundamentada
en literatura, magnitud NO calibrada contra datos reales).

USO:
    python3 src/treatment_simulation_demo.py \\
        --patterns results_gse39582_final/calibrated_patterns.tsv \\
        --treatment immunotherapy_antiPD1
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attractor_model import build_model_from_patterns, dynamics
from calibration import load_calibrated_patterns
from modern_hopfield import (
    DEFAULT_MONTHS_BETWEEN_CHECKS,
    DEFAULT_N_TIMEPOINTS,
    DEFAULT_RECURRENCE_ONSET_MONTH,
    _scheduled_forcing_strength,
    resolve_forcing_ramp,
    compute_stabilizing_k,
    modern_hopfield_baseline,
    modern_hopfield_field,
    modern_hopfield_field_stabilized,
    normalized_driver_direction,
    patterns_to_matrix,
    validate_modern_pattern_matrix,
)
from prognosis import hazard_from_trajectory
from prognosis_demo import classify_current_state
from treatment_perturbation import TREATMENT_MECHANISMS, apply_treatment_perturbation, describe_treatment

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# Cociente tratamiento/forzamiento por defecto. Historicamente la app usaba
# base_treatment_strength=0.5 con max_forcing_strength=5.0 (cociente 0.1):
# el tratamiento era 10x mas debil que la recaida por construccion, y el
# "beneficio simulado" reflejaba ese cociente arbitrario, no al paciente.
# Ahora el cociente es un parametro explicito y visible (ver
# treatment_strength_from_ratio y la barra lateral de la app).
DEFAULT_TREATMENT_TO_FORCING_RATIO = 0.1


def treatment_strength_from_ratio(max_forcing_strength: float, ratio: float) -> float:
    """base_treatment_strength = ratio * max_forcing_strength.

    Expresar la intensidad del tratamiento RELATIVA al forzamiento de
    recaida hace explicito el unico numero que gobierna el signo y la
    magnitud del beneficio simulado. ratio=1 significa que, en el
    estado donde la eficacia direccional es 1, el tratamiento empuja
    tanto como la recaida.
    """
    if max_forcing_strength <= 0 or ratio <= 0:
        raise ValueError("max_forcing_strength y ratio deben ser > 0")
    return float(ratio * max_forcing_strength)


def simulate_with_optional_treatment(
    model_matrix, n_genes, gene_order, recurrence_pattern, patterns,
    treatment=None, treatment_onset_month=None, ras_braf_wildtype=None,
    n_timepoints=DEFAULT_N_TIMEPOINTS,
    months_between_checks=DEFAULT_MONTHS_BETWEEN_CHECKS,
    recurrence_onset_month=DEFAULT_RECURRENCE_ONSET_MONTH,
    beta=None, base_treatment_strength=None,
    dynamics_model="modern_hopfield", max_forcing_strength=5.0,
    treatment_to_forcing_ratio=DEFAULT_TREATMENT_TO_FORCING_RATIO,
    forcing_ramp_duration_months=None,
):
    """Simula la trayectoria con o sin tratamiento.

    CORRECCION 2026-09-12 (bug real): antes el termino de tratamiento se
    evaluaba UNA vez con el estado al inicio de cada intervalo de 3
    meses y se pasaba constante a solve_ivp. Como el tratamiento esta
    definido como amortiguamiento -efficacy*x, congelarlo lo convertia
    en un empuje constante en la direccion -x(t0): no frena al llegar al
    origen y puede cruzarlo. Ahora la perturbacion se evalua dentro del
    campo, con el estado instantaneo xx. La eficacia direccional
    (correlacion con el patron relevante) se congela al inicio del
    intervalo -- es una propiedad del "estado clinico" en ese control,
    no algo que cambie de un dia a otro -- y solo el factor -x es vivo.

    base_treatment_strength: si es None se deriva de
    treatment_to_forcing_ratio * max_forcing_strength (ver
    treatment_strength_from_ratio). Pasarlo explicitamente sigue
    funcionando para reproducir resultados previos.
    """
    if dynamics_model not in {"modern_hopfield", "projection_legacy"}:
        raise ValueError("dynamics_model debe ser 'modern_hopfield' o 'projection_legacy'")
    resolved_beta = (3.0 if dynamics_model == "modern_hopfield" else 2.0) if beta is None else float(beta)
    if base_treatment_strength is None:
        base_treatment_strength = treatment_strength_from_ratio(
            max_forcing_strength, treatment_to_forcing_ratio)
    if dynamics_model == "modern_hopfield":
        X = validate_modern_pattern_matrix(
            model_matrix, n_genes, n_patterns=len(patterns))
        if max_forcing_strength <= 0:
            raise ValueError("max_forcing_strength debe ser > 0")
        stabilizing_k = compute_stabilizing_k(X, resolved_beta)
        baseline = modern_hopfield_baseline(X)
        driver_direction = normalized_driver_direction(recurrence_pattern)
        ramp = resolve_forcing_ramp(n_timepoints, months_between_checks,
                                    recurrence_onset_month, forcing_ramp_duration_months)
    else:
        W = model_matrix
    t_checks = np.arange(0, n_timepoints * months_between_checks, months_between_checks)
    x_series = np.zeros((n_genes, n_timepoints))
    x_current = np.zeros(n_genes)

    for i, t in enumerate(t_checks):
        I_relapse = np.zeros(n_genes)
        forcing_progress = 0.0
        if t >= recurrence_onset_month:
            months_since_onset = t - recurrence_onset_month
            if dynamics_model == "modern_hopfield":
                strength, forcing_progress = _scheduled_forcing_strength(
                    months_since_onset, max_forcing_strength, ramp)
                I_relapse = strength * driver_direction
            else:
                strength = min(0.15 * months_since_onset, 0.7)
                forcing_progress = 1.0
                I_relapse = strength * recurrence_pattern

        treatment_active = (treatment is not None and treatment_onset_month is not None
                            and t >= treatment_onset_month)
        if treatment_active:
            # Eficacia direccional congelada en este control; el factor -x
            # se evalua vivo dentro del campo (ver docstring).
            I_probe = apply_treatment_perturbation(
                x_current, gene_order, treatment, patterns,
                base_strength=base_treatment_strength, ras_braf_wildtype=ras_braf_wildtype)
            norm_x = float(np.linalg.norm(x_current))
            damping = float(np.linalg.norm(I_probe)) / norm_x if norm_x > 1e-12 else 0.0
            # I_probe = -damping * x_current por construccion (apply_treatment_perturbation
            # devuelve -efficacy * x); recuperamos el escalar para aplicarlo a xx.
            treatment_term = lambda xx: -damping * xx
        else:
            treatment_term = lambda xx: 0.0

        if dynamics_model == "modern_hopfield":
            if t < recurrence_onset_month:
                field = lambda tt, xx: (
                    modern_hopfield_field_stabilized(xx, X, resolved_beta, stabilizing_k, baseline)
                    + treatment_term(xx))
            else:
                quiescent_weight = max(0.0, 1.0 - forcing_progress)
                field = lambda tt, xx: (
                    modern_hopfield_field(xx, X, resolved_beta) + I_relapse
                    - quiescent_weight * baseline
                    - quiescent_weight * stabilizing_k * xx
                    + treatment_term(xx))
        else:
            field = lambda tt, xx: dynamics(tt, xx, W, I_relapse, resolved_beta) + treatment_term(xx)
        sol = solve_ivp(field, (0, months_between_checks), x_current,
                        method="RK45", rtol=1e-8, atol=1e-10)
        x_current = sol.y[:, -1]
        x_series[:, i] = x_current

    return t_checks, x_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--treatment", required=True, choices=list(TREATMENT_MECHANISMS.keys()))
    parser.add_argument("--recurrence-target", default="CMS4_mesenchymal")
    parser.add_argument("--dynamics-model",
                        choices=["modern_hopfield", "projection_legacy"],
                        default="modern_hopfield")
    parser.add_argument("--beta", type=float, default=None,
                        help="Default dependiente del motor: 3.0 moderno, 2.0 legacy")
    parser.add_argument("--max-forcing-strength", type=float, default=5.0,
                        help="Fuerza maxima del driver normalizado Modern Hopfield; "
                             "es especifica de la calibracion, no una dosis clinica")
    parser.add_argument("--n-timepoints", type=int, default=10)
    parser.add_argument("--treatment-to-forcing-ratio", type=float,
                        default=DEFAULT_TREATMENT_TO_FORCING_RATIO,
                        help="Intensidad del tratamiento RELATIVA a la fuerza maxima del "
                             "driver. Es el numero que gobierna el beneficio simulado; "
                             "0.1 reproduce el comportamiento historico (0.5 vs 5.0).")
    parser.add_argument("--treatment-onset-month", type=int, default=18,
                         help="Mes en que se inicia el tratamiento (ej. al detectarse la alerta)")
    parser.add_argument("--ras-braf-wildtype", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--output", default="figures/treatment_simulation.png")
    args = parser.parse_args()
    if args.n_timepoints < 2:
        raise ValueError("n_timepoints debe ser >= 2")

    print(f"Cargando patrones reales calibrados: {args.patterns}")
    patterns, gene_order = load_calibrated_patterns(args.patterns)
    print(f"Genes ({len(gene_order)}): {gene_order}")

    print(f"\n{describe_treatment(args.treatment)}")

    ras_braf_map = {"true": True, "false": False, "unknown": None}
    ras_braf_wildtype = ras_braf_map[args.ras_braf_wildtype]
    if args.treatment == "anti_egfr":
        print(f"Estatus RAS/BRAF asumido: {args.ras_braf_wildtype}")

    if args.dynamics_model == "modern_hopfield":
        model_matrix, _ = patterns_to_matrix(patterns)
    else:
        model_matrix, _, _ = build_model_from_patterns(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns[args.recurrence_target]

    print("\nSimulando SIN tratamiento (linea base)...")
    t_checks, x_baseline = simulate_with_optional_treatment(
        model_matrix, n_genes, gene_order, recurrence_pattern, patterns, treatment=None,
        dynamics_model=args.dynamics_model, beta=args.beta,
        max_forcing_strength=args.max_forcing_strength,
        n_timepoints=args.n_timepoints,
    )
    hazard_baseline = hazard_from_trajectory(x_baseline)

    print(f"Simulando CON tratamiento ({args.treatment}, inicio mes {args.treatment_onset_month})...")
    t_checks2, x_treated = simulate_with_optional_treatment(
        model_matrix, n_genes, gene_order, recurrence_pattern, patterns, treatment=args.treatment,
        treatment_onset_month=args.treatment_onset_month, ras_braf_wildtype=ras_braf_wildtype,
        dynamics_model=args.dynamics_model, beta=args.beta,
        max_forcing_strength=args.max_forcing_strength,
        n_timepoints=args.n_timepoints,
        treatment_to_forcing_ratio=args.treatment_to_forcing_ratio,
    )
    hazard_treated = hazard_from_trajectory(x_treated)
    print(f"  Cociente tratamiento/forzamiento = {args.treatment_to_forcing_ratio:.2f} "
          f"(base_treatment_strength = {args.treatment_to_forcing_ratio * args.max_forcing_strength:.2f}). "
          "El beneficio simulado es funcion directa de este cociente.")

    print("\nComparacion de hazard ordinal (sin tratamiento vs. con tratamiento):")
    for t, h_b, h_t in zip(t_checks, hazard_baseline, hazard_treated):
        marker = " <- inicio tratamiento" if t == args.treatment_onset_month else ""
        print(f"  mes {t:3d}: sin_tx={h_b:.3f}  con_tx={h_t:.3f}{marker}")

    label_base, corr_base = classify_current_state(x_baseline[:, -1], patterns)
    label_treated, corr_treated = classify_current_state(x_treated[:, -1], patterns)
    delta_final = float(hazard_baseline[-1] - hazard_treated[-1])
    auc_base = float(np.trapezoid(hazard_baseline, t_checks))
    auc_treated = float(np.trapezoid(hazard_treated, t_checks2))
    print("\nResumen al final de la ventana:")
    print(f"  Sin tratamiento: {label_base} (corr={corr_base:.3f})")
    print(f"  Con tratamiento: {label_treated} (corr={corr_treated:.3f})")
    print(f"  Diferencia final de riesgo ordinal (sin_tx - con_tx): {delta_final:+.3f}")
    print(f"  Diferencia de area ordinal acumulada: {auc_base - auc_treated:+.3f}")
    if np.any(hazard_treated > hazard_baseline):
        print("  AVISO: el efecto simulado no es monotono; hay puntos intermedios donde "
              "con_tx > sin_tx. Interpretar la trayectoria completa, no un punto aislado.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_checks, hazard_baseline, color="#D55E00", marker="o", linewidth=1.8, label="Sin tratamiento")
    ax.plot(t_checks2, hazard_treated, color="#0072B2", marker="o", linewidth=1.8, label=f"Con {args.treatment}")
    ax.axvline(15, color="grey", linestyle="--", linewidth=1, label="Inicio recaida (simulado)")
    ax.axvline(args.treatment_onset_month, color="#0072B2", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Meses desde cirugia")
    ax.set_ylabel("Hazard ordinal (distancia al origen)")
    ax.set_title(
        "Simulacion contrafactual -- NO calibrado a magnitud real, solo direccion fundamentada en literatura\n"
        f"(ver treatment_perturbation.py para evidencia y limitaciones)",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en: {out_path}")
    print(
        "\nRECORDATORIO: esta simulacion muestra DIRECCION del efecto esperado segun "
        "mecanismo de accion clinico establecido, NO una prediccion cuantitativa validada. "
        "No usar para decisiones de tratamiento reales."
    )


if __name__ == "__main__":
    main()
