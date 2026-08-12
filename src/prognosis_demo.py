"""
prognosis_demo.py

Conecta cuatro piezas que hasta ahora vivian por separado:
    1. attractor_model.py -- la dinamica (ODE tipo Hopfield)
    2. calibration.py -- los patrones REALES calibrados contra GSE39582
       (no los placeholders hechos a mano del esqueleto original)
    3. prognosis.py -- el modulo de trayectoria/alerta de recurrencia
    4. treatment_perturbation.py -- que tratamientos tienen mecanismo
       aplicable dado el estado actual del paciente

Simula un escenario clinico post-quirurgico:
    - Paciente sin enfermedad residual detectable en las primeras
      mediciones (vector de estado cerca de cero)
    - En algun punto de seguimiento, aparece senal molecular real
      (el paciente recae) empujando hacia un atractor especifico
    - Verifica que hazard_from_trajectory + detect_recurrence_signal
      detectan la alerta, y en que punto del seguimiento
    - Reporta, junto con la alerta: (a) que tan solida es la evidencia
      externa de ESE atractor especifico (CMS4 solido y robusto al
      ajuste por estadio; CMS1 significativo por primera vez pero
      pendiente de confirmacion; CMS3 sin evidencia -- ver
      PROJECT_STATUS.md) y (b) que tratamientos tienen mecanismo
      aplicable al estado actual del paciente, con su evidencia citada

USO:
    python3 src/prognosis_demo.py --patterns results_gse39582_final/calibrated_patterns.tsv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attractor_model import build_model_from_patterns, dynamics
from calibration import load_calibrated_patterns
from modern_hopfield import patterns_to_matrix, simulate_longitudinal_patient_hopfield_v2
from prognosis import detect_recurrence_signal, hazard_from_trajectory
from treatment_perturbation import TREATMENT_MECHANISMS, apply_treatment_perturbation, describe_treatment
from scipy.integrate import solve_ivp

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# Fuerza de evidencia externa por atractor -- del Cox estratificado
# combinando las 4 cohortes externas GSE17536+GSE17537+GSE14333+GSE33113
# (n=415, 100 eventos; ajustado por estadio: n=388, 80 eventos --
# ver PROJECT_STATUS.md, actualizado agosto 2026).
# Referencia del modelo: CMS2_canonical_WNT (HR=1.0 por definicion).
EVIDENCE_STRENGTH = {
    "CMS1_MSI_immune": {
        "level": "debil",
        "detail": "HR=2.09 vs. CMS2, p=0.016 (ajustado por estadio) -- significativo "
                  "por primera vez tras sumar GSE33113, pero resultado NUEVO sin "
                  "replicacion previa (antes rondaba p=0.07-0.09): tratar con cautela "
                  "hasta confirmarlo en una cohorte adicional.",
    },
    "CMS2_canonical_WNT": {
        "level": "referencia",
        "detail": "Subtipo de referencia del modelo Cox -- sin HR propio que reportar.",
    },
    "CMS3_metabolic": {
        "level": "sin evidencia",
        "detail": "HR sin efecto significativo, p=0.11 (ajustado) -- no distinguible de "
                  "CMS2 en supervivencia externa. Poder estadistico de solo 45%: tampoco "
                  "se puede concluir ausencia de efecto.",
    },
    "CMS4_mesenchymal": {
        "level": "fuerte",
        "detail": "HR=2.06 vs. CMS2, p=0.018 (ajustado por estadio; modelo global "
                  "p<0.001) -- consistente en las 4 cohortes externas (GSE17536, "
                  "GSE17537, GSE14333, GSE33113) y robusto al ajuste: el HR apenas "
                  "cambia entre el modelo crudo restringido (2.34) y el ajustado (2.06).",
    },
}


def classify_current_state(x_current: np.ndarray, patterns: dict) -> tuple[str, float]:
    """Correlacion maxima del estado actual con cada patron -- misma logica
    que risk_score_from_expression en survival_validation.py, reimplementada
    aqui para no acoplar este script a un dataframe de cohorte completo."""
    if np.linalg.norm(x_current) < 1e-8:
        return "none", 0.0
    correlations = {label: float(np.corrcoef(x_current, p)[0, 1]) for label, p in patterns.items()}
    best = max(correlations, key=correlations.get)
    return best, correlations[best]


def applicable_treatments(x_current: np.ndarray, gene_order: list, patterns: dict, efficacy_threshold: float = 0.05) -> list:
    """Lista de tratamientos cuyo mecanismo tiene eficacia no-trivial dado
    el estado actual del paciente (ver treatment_perturbation.py)."""
    applicable = []
    for name in TREATMENT_MECHANISMS:
        try:
            I = apply_treatment_perturbation(x_current, gene_order, name, patterns, ras_braf_wildtype=None)
        except ValueError:
            continue  # el panel no incluye los genes del mecanismo
        efficacy = np.linalg.norm(I)
        if efficacy > efficacy_threshold:
            applicable.append((name, efficacy))
    return sorted(applicable, key=lambda pair: -pair[1])


def simulate_longitudinal_patient(
    model_matrix, gene_order, recurrence_pattern, n_genes,
    n_timepoints=8, months_between_checks=3,
    recurrence_onset_month=15, beta=None,
    dynamics_model="modern_hopfield", max_forcing_strength=1.5,
):
    """
    Simula mediciones periodicas post-quirurgicas (ej. cada 3 meses).

    dynamics_model="modern_hopfield" (predeterminado) usa la version V2:
    beta=3, reposo estabilizado, transicion suave y driver normalizado.
    dynamics_model="projection_legacy" conserva la ecuacion historica
    exclusivamente para comparacion reproducible (beta=2 por defecto).
    Antes de recurrence_onset_month, sin forzamiento (MRD negativo).
    Despues, un termino de sesgo hacia recurrence_pattern representa
    la reaparicion de enfermedad.
    """
    if dynamics_model == "modern_hopfield":
        resolved_beta = 3.0 if beta is None else float(beta)
        return simulate_longitudinal_patient_hopfield_v2(
            model_matrix, recurrence_pattern, n_genes,
            n_timepoints=n_timepoints,
            months_between_checks=months_between_checks,
            recurrence_onset_month=recurrence_onset_month,
            beta=resolved_beta,
            max_forcing_strength=max_forcing_strength,
            smooth_transition=True,
            forcing_ramp_duration_months=12.0,
            normalize_driver=True,
        )
    if dynamics_model != "projection_legacy":
        raise ValueError("dynamics_model debe ser 'modern_hopfield' o 'projection_legacy'")

    resolved_beta = 2.0 if beta is None else float(beta)
    W = model_matrix
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
            args=(W, I_driver, resolved_beta, 0.0, None), method="RK45",
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
    parser.add_argument("--dynamics-model",
                        choices=["modern_hopfield", "projection_legacy"],
                        default="modern_hopfield",
                        help="Motor dinamico; Modern Hopfield V2 es el predeterminado")
    parser.add_argument("--beta", type=float, default=None,
                        help="Default dependiente del motor: 3.0 moderno, 2.0 legacy")
    parser.add_argument("--output", default="figures/prognosis_demo.png")
    args = parser.parse_args()

    print(f"Cargando patrones reales calibrados: {args.patterns}")
    patterns, gene_order = load_calibrated_patterns(args.patterns)
    print(f"Genes ({len(gene_order)}): {gene_order}")
    print(f"Subtipos disponibles: {list(patterns.keys())}")

    if args.recurrence_target not in patterns:
        raise ValueError(f"'{args.recurrence_target}' no esta en los patrones. Opciones: {list(patterns.keys())}")

    if args.dynamics_model == "modern_hopfield":
        model_matrix, _ = patterns_to_matrix(patterns)
    else:
        model_matrix, _, _ = build_model_from_patterns(patterns)
    n_genes = len(gene_order)
    recurrence_pattern = patterns[args.recurrence_target]

    print(f"\nSimulando trayectoria post-quirurgica (recaida simulada hacia {args.recurrence_target})...")
    t_checks, x_series = simulate_longitudinal_patient(
        model_matrix, gene_order, recurrence_pattern, n_genes,
        dynamics_model=args.dynamics_model, beta=args.beta)

    hazard = hazard_from_trajectory(x_series)
    alert, alert_idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)

    print(f"\nSerie de riesgo (hazard ordinal) por chequeo:")
    for t, h in zip(t_checks, hazard):
        marker = " <-- ALERTA" if alert and t == t_checks[alert_idx] else ""
        print(f"  mes {t:3d}: hazard={h:.3f}{marker}")

    if alert:
        x_at_alert = x_series[:, alert_idx]
        attractor, corr = classify_current_state(x_at_alert, patterns)

        print(f"\nAlerta de recurrencia detectada en el mes {t_checks[alert_idx]} "
              f"(recaida simulada empezo en mes 15).")
        print(f"\n--- Direccion de la alerta ---")
        print(f"Estado se dirige hacia: {attractor} (correlacion={corr:.3f})")

        if attractor in EVIDENCE_STRENGTH:
            ev = EVIDENCE_STRENGTH[attractor]
            print(f"Fuerza de evidencia externa de este atractor: {ev['level'].upper()}")
            print(f"  {ev['detail']}")

        print(f"\n--- Tratamientos con mecanismo aplicable a este estado ---")
        treatments = applicable_treatments(x_at_alert, gene_order, patterns)
        if treatments:
            for name, efficacy in treatments:
                print(f"  [{name}] intensidad simulada arbitraria={efficacy:.3f}")
                print(f"    {describe_treatment(name)}")
                if name == "anti_egfr":
                    print(
                        "    AVISO: estatus RAS/BRAF real no disponible en esta simulacion -- "
                        "este numero usa el proxy debil por RNA (cercania a CMS3), NO sustituye "
                        "la prueba de mutacion real (qPCR alelo-especifico/HRM)."
                    )
        else:
            print("  Ninguno de los mecanismos modelados tiene intensidad simulada no-trivial en este estado.")
        print(
            "\n  RECORDATORIO: direccion de efecto fundamentada en literatura clinica; los numeros "
            "de arriba son INTENSIDAD SIMULADA ARBITRARIA, no eficacia clinica ni magnitud "
            "calibrada -- ver treatment_perturbation.py. No usar para decisiones reales."
        )
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
