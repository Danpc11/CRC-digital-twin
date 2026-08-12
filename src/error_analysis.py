"""
error_analysis.py
Análisis detallado de errores de clasificación en cualquier cohorte externa.

Preguntas que responde:
  1. ¿Los errores son de baja confianza (el modelo sabe que no sabe)?
  2. ¿Cuál es el umbral de confianza óptimo para maximizar precisión?
  3. ¿El error CMS1→CMS4 es sistemático o son casos ambiguos?
  4. ¿Qué genes diferencian los casos bien clasificados de los mal clasificados?

USO:
    python3 src/error_analysis.py \
        --input results_external_gse17536/scored_external_cohort.tsv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

GENES = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

# Paleta Wong (daltónica) para los 4 subtipos -- la MISMA asignación de
# colores que usa el resto del proyecto (app.py, plot_survival_curves.py)
COLORS = {
    "CMS1_MSI_immune":      "#0072B2",
    "CMS2_canonical_WNT":   "#E69F00",
    "CMS3_metabolic":       "#009E73",
    "CMS4_mesenchymal":     "#D55E00",
}
SHORT = {
    "CMS1_MSI_immune":    "CMS1",
    "CMS2_canonical_WNT": "CMS2",
    "CMS3_metabolic":     "CMS3",
    "CMS4_mesenchymal":   "CMS4",
}


def load(
    path: str, prediction_col: str = "predicted_cms",
    confidence_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    confidence_col = confidence_col or (
        "modern_hopfield_input_margin"
        if prediction_col == "modern_hopfield_cms" else "classification_confidence")
    missing = {"cms_label", prediction_col, confidence_col} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para analizar errores: {missing}")
    df = df[df["cms_label"] != "none"].copy()
    if prediction_col == "modern_hopfield_cms":
        n_indeterminate = int((df[prediction_col] == "indeterminado").sum())
        if n_indeterminate:
            print(f"Modern Hopfield: {n_indeterminate} abstenciones excluidas del error condicionado")
        df = df[df[prediction_col] != "indeterminado"].copy()
    # Las funciones internas usan nombres canonicos; se conserva el origen.
    df["analysis_prediction_source"] = prediction_col
    df["predicted_cms"] = df[prediction_col]
    df["classification_confidence"] = df[confidence_col]
    df["correct"] = df["cms_label"] == df["predicted_cms"]
    df["error_type"] = df.apply(
        lambda r: "correct" if r["correct"]
        else f"{SHORT[r['cms_label']]}→{SHORT[r['predicted_cms']]}",
        axis=1
    )
    return df


# ── 1. Confianza: correctos vs errores ──────────────────────────────────────
def confidence_analysis(df: pd.DataFrame) -> dict:
    correct   = df[df["correct"]]["classification_confidence"]
    incorrect = df[~df["correct"]]["classification_confidence"]

    print("\n── 1. CONFIANZA: correctos vs errores ──────────────────────────")
    print(f"  Correctos  (n={len(correct):3d})  media={correct.mean():.3f}  "
          f"mediana={correct.median():.3f}  p25={correct.quantile(0.25):.3f}")
    print(f"  Errores    (n={len(incorrect):3d})  media={incorrect.mean():.3f}  "
          f"mediana={incorrect.median():.3f}  p25={incorrect.quantile(0.25):.3f}")

    if len(correct) and len(incorrect):
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(correct, incorrect, alternative="greater")
        print(f"  Mann-Whitney U (correct > error): U={stat:.0f}, p={p:.2e}")
    else:
        stat = p = float("nan")
        print("  Mann-Whitney U: no estimable (falta al menos uno de los grupos)")

    return {"correct_mean": correct.mean(), "error_mean": incorrect.mean(), "p": p}


# ── 2. Threshold sweep ───────────────────────────────────────────────────────
def threshold_sweep(df: pd.DataFrame, out_dir: Path, cohort_name: str = "cohorte externa"):
    if df.empty:
        return pd.DataFrame(), None
    max_confidence = float(df["classification_confidence"].max())
    thresholds = np.linspace(0.0, max(0.9, max_confidence), 91)
    rows = []
    for t in thresholds:
        hi = df[df["classification_confidence"] >= t]
        if len(hi) == 0:
            continue
        acc   = (hi["correct"]).mean()
        cov   = len(hi) / len(df)
        rows.append({"threshold": t, "accuracy": acc,
                     "coverage": cov, "n": len(hi)})
    sweep = pd.DataFrame(rows)

    # F-score de accuracy×coverage (elige el threshold de máximo F1-like)
    sweep["f"] = 2 * sweep["accuracy"] * sweep["coverage"] / (
        sweep["accuracy"] + sweep["coverage"] + 1e-9)
    best = sweep.loc[sweep["f"].idxmax()]

    print(f"\n── 2. THRESHOLD SWEEP ──────────────────────────────────────────")
    print(f"  Threshold exploratorio in-sample (max F_acc×cov): {best['threshold']:.2f}")
    print(f"    → accuracy={best['accuracy']:.1%}  cobertura={best['coverage']:.1%}  "
          f"n={best['n']:.0f}/{len(df)}")
    print(f"  A conf ≥ 0.60: "
          + _at_threshold(sweep, 0.60))
    print(f"  A conf ≥ 0.70: "
          + _at_threshold(sweep, 0.70))
    print(f"  A conf ≥ 0.80: "
          + _at_threshold(sweep, 0.80))

    # Figura
    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax2 = ax1.twinx()
    ax1.plot(sweep["threshold"], sweep["accuracy"], color="#0D6E6E", lw=2, label="Accuracy")
    ax2.plot(sweep["threshold"], sweep["coverage"], color="#C94040", lw=2,
             ls="--", label="Cobertura")
    ax1.axvline(best["threshold"], color="gray", lw=1.2, ls=":")
    ax1.set_xlabel("Umbral de confianza")
    ax1.set_ylabel("Accuracy", color="#0D6E6E")
    ax2.set_ylabel("Cobertura", color="#C94040")
    ax1.set_ylim(0.5, 1.01)
    ax2.set_ylim(0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9,
               loc="lower left")
    ax1.set_title(f"Accuracy vs cobertura por umbral de confianza\n({cohort_name})",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=180)
    plt.close(fig)
    print(f"  → figura: {out_dir}/threshold_sweep.png")

    return sweep, best


def _at_threshold(sweep, t):
    row = sweep[sweep["threshold"].round(2) == round(t, 2)]
    if row.empty:
        return "N/A"
    r = row.iloc[0]
    return (f"accuracy={r['accuracy']:.1%}  cobertura={r['coverage']:.1%}  "
            f"n={r['n']:.0f}")


# ── 3. Error breakdown por subtipo ───────────────────────────────────────────
def error_breakdown(
    df: pd.DataFrame, out_dir: Path, cohort_name: str = "cohorte externa"
) -> pd.Series:
    errors = df[~df["correct"]].copy()

    print(f"\n── 3. BREAKDOWN DE ERRORES (n={len(errors)}) ───────────────────")
    counts = errors["error_type"].value_counts()
    for err_type, n in counts.items():
        subset = errors[errors["error_type"] == err_type]
        conf_med = subset["classification_confidence"].median()
        print(f"  {err_type:20s}  n={n:2d}  conf_mediana={conf_med:.3f}")

    # Foco: CMS1→CMS4 (el error más preocupante)
    cms1_4 = errors[errors["error_type"] == "CMS1→CMS4"]
    if len(cms1_4):
        print(f"\n  FOCO CMS1→CMS4 (n={len(cms1_4)}):")
        print(f"    conf  media={cms1_4['classification_confidence'].mean():.3f}  "
              f"min={cms1_4['classification_confidence'].min():.3f}  "
              f"max={cms1_4['classification_confidence'].max():.3f}")
        print(f"    genes (media z-score de estos {len(cms1_4)} pacientes):")
        available_genes = [gene for gene in GENES if gene in cms1_4.columns]
        gene_means = cms1_4[available_genes].mean().sort_values()
        for g, v in gene_means.items():
            bar = "█" * int(abs(v) * 5) if abs(v) > 0.1 else "·"
            sign = "+" if v > 0 else "-"
            print(f"      {g:8s}  {sign}{abs(v):.3f}  {bar}")

    # Figura: distribución de confianza por tipo de error
    top_errors = counts[counts >= 2].index.tolist()[:6]
    if top_errors:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        data_plot = [errors[errors["error_type"] == e]["classification_confidence"].values
                     for e in top_errors]
        bp = ax.boxplot(data_plot, patch_artist=True, vert=True,
                        medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#B0C4DE")
        ax.set_xticks(range(1, len(top_errors) + 1))
        ax.set_xticklabels(top_errors, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Confianza de clasificación")
        ax.set_title(f"Distribución de confianza por tipo de error\n({cohort_name})",
                     fontsize=10)
        ax.axhline(0.5, ls="--", color="gray", lw=1, label="conf=0.5")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "error_confidence_by_type.png", dpi=180)
        plt.close(fig)
        print(f"  → figura: {out_dir}/error_confidence_by_type.png")
    return counts


# ── 4. Heatmap de expresión génica: CMS1 correctos vs CMS1→CMS4 ─────────────
def cms1_gene_heatmap(
    df: pd.DataFrame, out_dir: Path, cohort_name: str = "cohorte externa"
):
    available_genes = [gene for gene in GENES if gene in df.columns]
    if not available_genes:
        print("\n── 4. EXPRESIÓN GÉNICA omitida: no hay genes del panel en el TSV ──")
        return
    cms1_all = df[df["cms_label"] == "CMS1_MSI_immune"].copy()
    cms1_all["group"] = cms1_all["predicted_cms"].map(
        lambda x: "Correcto (CMS1)" if x == "CMS1_MSI_immune" else
                  ("Error→CMS4" if x == "CMS4_mesenchymal" else f"Error→{SHORT[x]}")
    )

    print(f"\n── 4. EXPRESIÓN GÉNICA: CMS1 (n={len(cms1_all)}) ──────────────")
    for grp, sub in cms1_all.groupby("group"):
        print(f"\n  [{grp}  n={len(sub)}]")
        gene_m = sub[available_genes].mean().sort_values(ascending=False)
        for g, v in gene_m.items():
            bar = "█" * max(0, int(v * 3))
            print(f"    {g:8s}  {v:+.3f}  {bar}")

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    groups_plot = ["Correcto (CMS1)", "Error→CMS4"]
    vmin, vmax = -2.5, 2.5

    for ax, grp in zip(axes, groups_plot):
        sub = cms1_all[cms1_all["group"] == grp]
        if sub.empty:
            ax.set_title(f"{grp}\n(sin datos)")
            continue
        mat = sub[available_genes].values
        im = ax.imshow(mat.T, aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(available_genes)))
        ax.set_yticklabels(available_genes, fontsize=9)
        ax.set_xlabel("Pacientes")
        ax.set_title(f"{grp}\n(n={len(sub)})", fontsize=10)
        ax.set_xticks([])

    fig.colorbar(im, ax=axes, shrink=0.7, label="z-score expresión")
    fig.suptitle(f"Expresión génica: CMS1 correcto vs CMS1→CMS4 — {cohort_name}",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cms1_vs_cms4_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → figura: {out_dir}/cms1_vs_cms4_heatmap.png")


# ── 5. Resumen ejecutivo ─────────────────────────────────────────────────────
def cohen_kappa_from_labels(observed: pd.Series, predicted: pd.Series) -> float:
    """Calcula kappa de Cohen a partir de las etiquetas observadas/predichas."""
    labels = sorted(set(observed.dropna()) | set(predicted.dropna()))
    if not labels or len(observed) == 0:
        return float("nan")
    observed_cat = pd.Categorical(observed, categories=labels)
    predicted_cat = pd.Categorical(predicted, categories=labels)
    table = pd.crosstab(observed_cat, predicted_cat, dropna=False).reindex(
        index=labels, columns=labels, fill_value=0)
    matrix = table.to_numpy(dtype=float)
    n = matrix.sum()
    agreement = np.trace(matrix) / n
    expected = np.dot(matrix.sum(axis=1), matrix.sum(axis=0)) / n**2
    return float((agreement - expected) / (1.0 - expected)) if expected < 1 else float("nan")


def summary(
    df: pd.DataFrame, best_threshold: pd.Series,
    cohort_name: str = "cohorte externa",
) -> dict:
    n_total   = len(df)
    n_correct = df["correct"].sum()
    n_error   = n_total - n_correct

    print("\n" + "=" * 65)
    print("RESUMEN EJECUTIVO")
    print("=" * 65)
    kappa = cohen_kappa_from_labels(df["cms_label"], df["predicted_cms"])
    errors = df.loc[~df["correct"], "error_type"].value_counts()
    top_error = errors.index[0] if not errors.empty else "ninguno"
    top_count = int(errors.iloc[0]) if not errors.empty else 0

    print(f"  Cohorte externa: {cohort_name}  (n={n_total})")
    print(f"  Accuracy global:  {n_correct/n_total:.1%}  ({n_correct}/{n_total})")
    print(f"  Errores totales:  {n_error}")
    print(f"  Kappa de Cohen: {kappa:.3f}")
    print()
    print(f"  Threshold exploratorio in-sample: {best_threshold['threshold']:.2f}")
    print(f"    → accuracy {best_threshold['accuracy']:.1%}  "
          f"con cobertura {best_threshold['coverage']:.1%}")
    print()
    print(f"  Error más frecuente: {top_error} (n={top_count})")
    if not errors.empty and (errors == top_count).sum() > 1:
        tied = ", ".join(errors[errors == top_count].index)
        print(f"  Empate entre tipos de error: {tied}")
    print("  AVISO: el umbral fue seleccionado y evaluado en la misma cohorte;")
    print("  debe congelarse y validarse en otra cohorte antes de reportarlo.")
    print("=" * 65)
    return {
        "cohort": cohort_name,
        "n": n_total,
        "n_correct": int(n_correct),
        "n_errors": int(n_error),
        "accuracy": float(n_correct / n_total),
        "cohen_kappa": kappa,
        "exploratory_threshold": float(best_threshold["threshold"]),
        "threshold_accuracy": float(best_threshold["accuracy"]),
        "threshold_coverage": float(best_threshold["coverage"]),
        "most_frequent_error": top_error,
        "most_frequent_error_n": top_count,
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="scored_external_cohort.tsv")
    parser.add_argument("--out-dir", default=None,
                        help="directorio de salida (default: mismo dir que --input)")
    parser.add_argument("--cohort-name", default=None,
                        help="Nombre mostrado en tablas/figuras; default: directorio padre")
    parser.add_argument("--prediction-col", default="predicted_cms",
                        choices=["predicted_cms", "modern_hopfield_cms"])
    parser.add_argument("--confidence-col", default=None,
                        help="Default automatico segun --prediction-col")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_name = args.cohort_name or in_path.parent.name.removeprefix("results_external_")

    df = load(args.input, args.prediction_col, args.confidence_col)
    if df.empty:
        message = (
            "No hay muestras evaluables con etiqueta CMS oficial y prediccion aceptada; "
            "se omite el analisis de errores. Esto es esperado en cohortes con cms_label='none'."
        )
        print(message)
        pd.DataFrame([{
            "cohort": cohort_name, "status": "no_evaluable",
            "reason": "sin_etiquetas_oficiales_o_sin_predicciones_aceptadas",
            "n": 0,
        }]).to_csv(out_dir / "classification_error_summary.tsv", sep="\t", index=False)
        (out_dir / "classification_error_report.txt").write_text(message + "\n")
        return

    confidence_analysis(df)
    sweep, best = threshold_sweep(df, out_dir, cohort_name)
    error_breakdown(df, out_dir, cohort_name)
    cms1_gene_heatmap(df, out_dir, cohort_name)
    metrics = summary(df, best, cohort_name)
    sweep.to_csv(out_dir / "classification_threshold_sweep.tsv", sep="\t", index=False)
    pd.DataFrame([metrics]).to_csv(
        out_dir / "classification_error_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
