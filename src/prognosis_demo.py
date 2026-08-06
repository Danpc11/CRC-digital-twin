"""
prognosis_demo.py

Conecta tres piezas que hasta ahora vivian por separado:
    1. attractor_model.py -- la dinamica (ODE tipo Hopfield)
    2. calibration.py -- los patrones REALES calibrados contra GSE39582
       (no los placeholders hechos a mano del esqueleto original)
    3. prognosis.py -- el modulo de trayectoria/alerta de recurrencia

Simula un escenario clinico post-quirurgico:
    - Paciente sin enfermedad residual detectable en las primeras
      mediciones (vector de estado cerca de cero)
    - En algun punto de seguimiento, aparece senal molecular real
      (el paciente recae) empujando hacia el atractor de peor
      pronostico (CMS4, consistente con lo que ya validamos en
      GSE39582/GSE17536: CMS4 es sistematicamente el peor)
    - Verifica que hazard_from_trajectory + detect_recurrence_signal
      detectan la alerta, y en que punto del seguimiento

USO:
    python3 src/prognosis_demo.py --patterns results_gse39582_v2/calibrated_patterns.tsv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attractor_model import build_model_from_patterns, dynamics
from calibration import load_calibrated_patterns
from prognosis import detect_recurrence_signal, hazard_from_trajectory
from scipy.integrate import solve_ivp

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def simulate_longitudinal_patient(
    W, gene_order, recurrence_pattern, n_genes,
    n_timepoints=8, months_between_checks=3,
    recurrence_onset_month=15, beta=2.0,
):
    """
    Simula mediciones periodicas post-quirurgicas (ej. cada 3 meses).
    Antes de recurrence_onset_month, sin forzamiento (MRD negativo).
    Despues, un termino de sesgo hacia recurrence_pattern representa
    la reaparicion de enfermedad.
    """
    t_checks = np.arange(0, n_timepoints * months_between_checks, months_between_checks)
    x_series = np.zeros((n_genes, n_timepoints))
    x_current = np.zeros(n_genes)

    for i, t in enumerate(t_checks):
        if t >= recurrence_onset_month:
            # sesgo hacia el atractor de recurrencia, escalado por
            # cuanto tiempo ha pasado desde el inicio de la recaida
            months_since_onset = t - recurrence_onset_month
            strength = min(0.15 * months_since_onset, 0.7)
            I_driver = strength * recurrence_pattern
        else:
            I_driver = np.zeros(n_genes)

        # integrar un intervalo corto entre chequeos, partiendo del
        # estado actual (la dinamica es continua, las mediciones son
        # muestreos periodicos de una trayectoria subyacente continua)
        sol = solve_ivp(
            dynamics, (0, months_between_checks), x_current,
            args=(W, I_driver, beta, 0.0, None), method="RK45",
            rtol=1e-8, atol=1e-10,
        )
        x_current = sol.y[:, -1]
        x_series[:, i] = x_current

    return t_checks, x_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True, help="calibrated_patterns.tsv (patrones REALES, no placeholders)")
    parser.add_argument("--recurrence-target", default="CMS4_mesenchymal",
                         help="Atractor hacia el que se dirige la recaida simulada")
    parser.add_argument("--output", default="figures/prognosis_demo.png")
    args = parser.parse_args()

    print(f"Cargando patrones reales calibrados: {args.patterns}")
    patterns, gene_order = load_calibrated_patterns(args.patterns)
    print(f"Genes ({len(gene_order)}): {gene_order}")
    print(f"Subtipos disponibles: {list(patterns.keys())}")

    if args.recurrence_target not in patterns:
        raise ValueError(f"'{args.recurrence_target}' no esta en los patrones. Opciones: {list(patterns.keys())}")

    W, labels, _ = build_model_from_patterns(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns[args.recurrence_target]

    print(f"\nSimulando trayectoria post-quirurgica (recaida simulada hacia {args.recurrence_target})...")
    t_checks, x_series = simulate_longitudinal_patient(W, gene_order, recurrence_pattern, n_genes)

    hazard = hazard_from_trajectory(x_series)
    alert, alert_idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)

    print(f"\nSerie de riesgo (hazard ordinal) por chequeo:")
    for t, h in zip(t_checks, hazard):
        marker = " <-- ALERTA" if alert and t == t_checks[alert_idx] else ""
        print(f"  mes {t:3d}: hazard={h:.3f}{marker}")

    if alert:
        print(f"\nAlerta de recurrencia detectada en el mes {t_checks[alert_idx]} "
              f"(recaida simulada empezo en mes 15).")
    else:
        print("\nNo se detecto alerta -- revisar threshold_sigma o la fuerza del sesgo simulado.")

    # Figura: trayectoria de genes + serie de hazard
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i, gene in enumerate(gene_order):
        ax1.plot(t_checks, x_series[i], color=WONG[i % len(WONG)], marker="o", markersize=3, label=gene, linewidth=1.4)
    ax1.axvline(15, color="grey", linestyle="--", linewidth=1, label="inicio recaida (simulado)")
    ax1.set_ylabel("Expresion (z-score)")
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax1.set_title("Trayectoria simulada de un paciente -- panel PCR post-quirurgico")

    ax2.plot(t_checks, hazard, color="#D55E00", marker="o", linewidth=1.8)
    if alert:
        ax2.axvline(t_checks[alert_idx], color="red", linestyle=":", linewidth=1.5, label="alerta detectada")
        ax2.legend(fontsize=8)
    ax2.axvline(15, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("Meses desde cirugia")
    ax2.set_ylabel("Hazard ordinal\n(distancia al origen)")

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en: {out_path}")


if __name__ == "__main__":
    main()
