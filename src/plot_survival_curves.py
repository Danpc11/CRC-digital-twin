"""
plot_survival_curves.py

Grafica las curvas Kaplan-Meier por subtipo CMS (modelo vs. etiqueta
oficial) a partir de scored_cohort.tsv, generado por run_pipeline.py.

Esto es el chequeo de DIRECCION que el log-rank test no da por si
solo: un p-valor significativo confirma que las curvas difieren, pero
no dice cual grupo va peor. La literatura (Guinney et al. 2015)
reporta CMS4 (mesenquimal) con peor pronostico y CMS1/CMS2 mejor --
si las curvas de este analisis muestran lo contrario, es señal de un
problema (etiquetas invertidas, error de mapeo, etc.), no una
confirmacion valida.

USO:
    python3 src/plot_survival_curves.py --input results_gse39582/scored_cohort.tsv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from survival_validation import validate_survival_by_subtype

# Paleta Wong (colorblind-safe), orden fijo por subtipo para
# comparabilidad visual entre paneles
CMS_COLORS = {
    "CMS1_MSI_immune": "#0072B2",
    "CMS2_canonical_WNT": "#E69F00",
    "CMS3_metabolic": "#009E73",
    "CMS4_mesenchymal": "#D55E00",
}


def plot_km_panel(ax, scored_df, group_col, duration_col, event_col, title):
    exclude_none = scored_df[scored_df[group_col] != "none"] if group_col == "cms_label" else scored_df
    result = validate_survival_by_subtype(
        exclude_none, duration_col=duration_col, event_col=event_col, group_col=group_col
    )
    for label, kmf in result["km_fitters"].items():
        color = CMS_COLORS.get(label, "#999999")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False, linewidth=1.8)
    ax.set_title(f"{title}\n(log-rank p = {result['logrank_p_value']:.3g}, n={result['n_patients']})", fontsize=10)
    ax.set_xlabel("Meses")
    ax.set_ylabel("Probabilidad de supervivencia libre de recidiva")
    ax.legend(fontsize=8, loc="lower left")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="scored_cohort.tsv generado por run_pipeline.py")
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--output", default="figures/survival_curves.png")
    args = parser.parse_args()

    scored = pd.read_csv(args.input, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_km_panel(
        axes[0], scored, "predicted_cms", args.duration_col, args.event_col,
        "Reclasificacion del modelo (panel reducido)"
    )
    plot_km_panel(
        axes[1], scored, "cms_label", args.duration_col, args.event_col,
        "Etiqueta CMS oficial del consorcio"
    )

    fig.suptitle(
        "Curvas Kaplan-Meier por subtipo CMS -- verificar que CMS4 (naranja oscuro) "
        "vaya peor que CMS1/CMS2 (azul/naranja claro), consistente con literatura",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
