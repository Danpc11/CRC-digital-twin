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
from prognosis import hazard_from_trajectory
from treatment_perturbation import TREATMENT_MECHANISMS, apply_treatment_perturbation, describe_treatment

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def simulate_with_optional_treatment(
    W, n_genes, gene_order, recurrence_pattern, patterns,
    treatment=None, treatment_onset_month=None, ras_braf_wildtype=None,
    n_timepoints=10, months_between_checks=3, recurrence_onset_month=15,
    beta=2.0, base_treatment_strength=0.5,
):
    t_checks = np.arange(0, n_timepoints * months_between_checks, months_between_checks)
    x_series = np.zeros((n_genes, n_timepoints))
    x_current = np.zeros(n_genes)

    for i, t in enumerate(t_checks):
        I_relapse = np.zeros(n_genes)
        if t >= recurrence_onset_month:
            months_since_onset = t - recurrence_onset_month
            strength = min(0.15 * months_since_onset, 0.7)
            I_relapse = strength * recurrence_pattern

        I_total = I_relapse
        if treatment is not None and treatment_onset_month is not None and t >= treatment_onset_month:
            I_treatment = apply_treatment_perturbation(
                x_current, gene_order, treatment, patterns,
                base_strength=base_treatment_strength, ras_braf_wildtype=ras_braf_wildtype,
            )
            I_total = I_relapse + I_treatment

        sol = solve_ivp(
            dynamics, (0, months_between_checks), x_current,
            args=(W, I_total, beta, 0.0, None), method="RK45",
            rtol=1e-8, atol=1e-10,
        )
        x_current = sol.y[:, -1]
        x_series[:, i] = x_current

    return t_checks, x_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--treatment", required=True, choices=list(TREATMENT_MECHANISMS.keys()))
    parser.add_argument("--recurrence-target", default="CMS4_mesenchymal")
    parser.add_argument("--treatment-onset-month", type=int, default=18,
                         help="Mes en que se inicia el tratamiento (ej. al detectarse la alerta)")
    parser.add_argument("--ras-braf-wildtype", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--output", default="figures/treatment_simulation.png")
    args = parser.parse_args()

    print(f"Cargando patrones reales calibrados: {args.patterns}")
    patterns, gene_order = load_calibrated_patterns(args.patterns)
    print(f"Genes ({len(gene_order)}): {gene_order}")

    print(f"\n{describe_treatment(args.treatment)}")

    ras_braf_map = {"true": True, "false": False, "unknown": None}
    ras_braf_wildtype = ras_braf_map[args.ras_braf_wildtype]
    if args.treatment == "anti_egfr":
        print(f"Estatus RAS/BRAF asumido: {args.ras_braf_wildtype}")

    W, labels, _ = build_model_from_patterns(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns[args.recurrence_target]

    print("\nSimulando SIN tratamiento (linea base)...")
    t_checks, x_baseline = simulate_with_optional_treatment(
        W, n_genes, gene_order, recurrence_pattern, patterns, treatment=None,
    )
    hazard_baseline = hazard_from_trajectory(x_baseline)

    print(f"Simulando CON tratamiento ({args.treatment}, inicio mes {args.treatment_onset_month})...")
    t_checks2, x_treated = simulate_with_optional_treatment(
        W, n_genes, gene_order, recurrence_pattern, patterns, treatment=args.treatment,
        treatment_onset_month=args.treatment_onset_month, ras_braf_wildtype=ras_braf_wildtype,
    )
    hazard_treated = hazard_from_trajectory(x_treated)

    print("\nComparacion de hazard ordinal (sin tratamiento vs. con tratamiento):")
    for t, h_b, h_t in zip(t_checks, hazard_baseline, hazard_treated):
        marker = " <- inicio tratamiento" if t == args.treatment_onset_month else ""
        print(f"  mes {t:3d}: sin_tx={h_b:.3f}  con_tx={h_t:.3f}{marker}")

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
