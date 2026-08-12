"""
external_validation.py

Validacion externa REAL: carga los patrones YA CALIBRADOS contra
GSE39582 (calibrated_patterns.tsv) y los aplica, congelados, a una
cohorte externa distinta -- sin recalibrar. Recalibrar sobre la
cohorte externa invalidaria la validacion (dejaria de ser "externa").

Cada cohorte se z-scorea dentro de si misma (practica estandar para
robustez entre plataformas/lotes -- ver metodologia de Nearest
Template Prediction, en la que se basa CMS classifier original), pero
los patrones/centroides contra los que se compara son fijos, tomados
de GSE39582.

USO:
    python3 src/external_validation.py \\
        --patterns results_gse39582_v2/calibrated_patterns.tsv \\
        --input data/gse17536_cms_labeled.tsv \\
        --output results_external_gse17536/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import infer_gene_columns, load_calibrated_patterns, load_labeled_dataset, zscore_genes
from modern_hopfield import score_cohort_modern_hopfield, validate_modern_hopfield_beta
from survival_validation import (
    interpret_validation_result,
    score_cohort,
    validate_survival_by_subtype,
)


def main():
    parser = argparse.ArgumentParser(description="Validacion externa con patrones congelados")
    parser.add_argument("--patterns", required=True, help="calibrated_patterns.tsv de la cohorte de entrenamiento")
    parser.add_argument("--input", required=True, help="TSV de la cohorte externa (esquema de calibration.py)")
    parser.add_argument("--output", default="results_external", help="Directorio de salida")
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--endpoint-label", default="supervivencia libre de recidiva (RFS)")
    parser.add_argument(
        "--classifier", choices=["correlation", "modern_hopfield", "both"],
        default="correlation",
        help="Clasificador externo a validar; 'both' conserva correlacion y agrega Modern Hopfield",
    )
    parser.add_argument("--beta", type=float, default=3.0,
                        help="Beta de Modern Hopfield (solo para modern_hopfield/both)")
    parser.add_argument("--modern-corr-threshold", type=float, default=0.8,
                        help="Correlacion minima para aceptar una recuperacion Modern Hopfield")
    parser.add_argument("--modern-integration-time", type=float, default=30.0)
    parser.add_argument("--require-valid-beta", action="store_true",
                        help="Abortar si beta no conserva estables los patrones congelados")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando patrones congelados: {args.patterns}")
    patterns, pattern_genes = load_calibrated_patterns(args.patterns)
    print(f"Genes en los patrones ({len(pattern_genes)}): {pattern_genes}")

    print(f"\nCargando cohorte externa: {args.input}")
    df = load_labeled_dataset(args.input)
    cohort_genes = infer_gene_columns(df)

    missing = set(pattern_genes) - set(cohort_genes)
    if missing:
        raise ValueError(
            f"La cohorte externa no tiene estos genes de los patrones congelados: {missing}. "
            "No se puede aplicar el modelo sin ellos."
        )
    extra = set(cohort_genes) - set(pattern_genes)
    if extra:
        print(f"AVISO: la cohorte externa tiene genes extra no usados por los patrones ({extra}) -- se ignoran.")

    # Usar SOLO los genes de los patrones congelados, en el mismo orden
    use_genes = pattern_genes

    print(f"\nz-score dentro de la cohorte externa (no se toca la calibracion)...")
    z = zscore_genes(df, use_genes)

    print("Clasificando con patrones congelados (SIN recalibrar)...")
    scored = score_cohort(z, use_genes, patterns)

    if args.classifier in {"modern_hopfield", "both"}:
        if args.modern_integration_time <= 0:
            raise ValueError("modern_integration_time debe ser > 0")
        beta_check = validate_modern_hopfield_beta(patterns, args.beta)
        beta_check["table"].to_csv(
            out_dir / "modern_hopfield_beta_validation.tsv", sep="\t", index=False)
        if not beta_check["valid"]:
            message = (
                f"beta={args.beta:g} no conserva como atractores estables a: "
                f"{beta_check['invalid_patterns']}"
            )
            if args.require_valid_beta:
                raise ValueError(message)
            print(f"AVISO: {message}. Los resultados se guardan como exploratorios.")
        print(f"Recuperando estados con Modern Hopfield (beta={args.beta:g})...")
        scored = score_cohort_modern_hopfield(
            scored, use_genes, patterns, beta=args.beta,
            corr_threshold=args.modern_corr_threshold,
            integration_time=args.modern_integration_time)
        accepted = scored["modern_hopfield_cms"] != "indeterminado"
        print(
            f"Cobertura Modern Hopfield: {accepted.sum()}/{len(scored)} "
            f"({accepted.mean():.1%}); indeterminadas={(~accepted).sum()}"
        )
        if "cms_label" in scored.columns:
            labeled = scored[scored["cms_label"] != "none"].copy()
            accepted_labeled = labeled[labeled["modern_hopfield_cms"] != "indeterminado"]
            scored["modern_hopfield_matches_official"] = (
                scored["modern_hopfield_cms"] == scored["cms_label"])
            if len(accepted_labeled):
                y_true = accepted_labeled["cms_label"]
                y_pred = accepted_labeled["modern_hopfield_cms"]
                metrics = {
                    "n_labeled": len(labeled), "n_accepted": len(accepted_labeled),
                    "coverage": len(accepted_labeled) / len(labeled),
                    "accuracy_accepted": float((y_true == y_pred).mean()),
                    "accuracy_all_labeled_counting_abstention_as_error": float(
                        (labeled["cms_label"] == labeled["modern_hopfield_cms"]).mean()),
                    "balanced_accuracy_accepted": balanced_accuracy_score(y_true, y_pred),
                    "macro_f1_accepted": f1_score(y_true, y_pred, average="macro"),
                    "cohen_kappa_accepted": cohen_kappa_score(y_true, y_pred),
                }
                pd.DataFrame([metrics]).to_csv(
                    out_dir / "modern_hopfield_classification_metrics.tsv",
                    sep="\t", index=False)
                confusion = pd.crosstab(
                    y_true, y_pred, rownames=["cms_oficial"],
                    colnames=["modern_hopfield_cms"], dropna=False)
                confusion.to_csv(
                    out_dir / "modern_hopfield_confusion_matrix.tsv", sep="\t")
                print(
                    f"Concordancia Modern vs CMS oficial (aceptadas): "
                    f"accuracy={metrics['accuracy_accepted']:.1%}, "
                    f"balanced_accuracy={metrics['balanced_accuracy_accepted']:.1%}, "
                    f"macro-F1={metrics['macro_f1_accepted']:.3f}, "
                    f"kappa={metrics['cohen_kappa_accepted']:.3f}")
    scored.to_csv(out_dir / "scored_external_cohort.tsv", sep="\t", index=False)

    if args.duration_col not in df.columns or args.event_col not in df.columns:
        print(
            f"\nAVISO: no se encontraron columnas '{args.duration_col}'/'{args.event_col}' -- "
            "solo se guardo la clasificacion, sin validacion de supervivencia."
        )
        return

    print(f"\n--- Validacion de supervivencia externa: {args.endpoint_label} ---")
    model_group_cols = []
    if args.classifier in {"correlation", "both"}:
        model_group_cols.append("predicted_cms")
    if args.classifier in {"modern_hopfield", "both"}:
        model_group_cols.append("modern_hopfield_cms")

    reports = []
    model_results = {}
    for group_col in model_group_cols:
        analysis_df = scored
        if group_col == "modern_hopfield_cms":
            analysis_df = scored[scored[group_col] != "indeterminado"]
        print(f"\nValidacion del clasificador: {group_col}")
        result = validate_survival_by_subtype(
            analysis_df, duration_col=args.duration_col, event_col=args.event_col,
            endpoint_label=args.endpoint_label, group_col=group_col,
        )
        report = interpret_validation_result(result)
        print(report)
        reports.append(report)
        model_results[group_col] = result
    print(
        "\nIMPORTANTE: este resultado es sobre una cohorte que el modelo NUNCA vio "
        "durante la calibracion -- si sale significativo, es evidencia bastante mas "
        "fuerte que la validacion en la misma cohorte de entrenamiento."
    )

    report_full = ("\n\n" + "=" * 60 + "\n\n").join(reports)

    if "cms_label" in scored.columns:
        print("\nLinea base: etiqueta CMS oficial del consorcio (en esta misma cohorte externa)")
        baseline_df = scored[scored["cms_label"] != "none"]
        n_excluded = len(scored) - len(baseline_df)
        if n_excluded > 0:
            print(f"  ({n_excluded} muestras 'none' excluidas de la linea base)")
        try:
            result_baseline = validate_survival_by_subtype(
                baseline_df, duration_col=args.duration_col, event_col=args.event_col,
                endpoint_label=args.endpoint_label, group_col="cms_label",
            )
            report_baseline = interpret_validation_result(result_baseline)
            print(report_baseline)
            report_full += "\n\n" + "=" * 60 + "\n\n" + report_baseline

            primary_group = model_group_cols[0]
            p_model = model_results[primary_group]["logrank_p_value"]
            p_baseline = result_baseline["logrank_p_value"]
            if p_baseline >= 0.05 and p_model >= 0.05:
                diag = (
                    "\nDIAGNOSTICO: NI la etiqueta oficial (p={:.4g}) NI el panel del modelo "
                    "(p={:.4g}) separan supervivencia en esta cohorte externa. Esto apunta a "
                    "que esta cohorte especificamente tiene una asociacion CMS-supervivencia "
                    "mas debil o un tamano de muestra insuficiente, no necesariamente a un "
                    "fallo de generalizacion especifico del panel reducido."
                ).format(p_baseline, p_model)
                print(diag)
                report_full += "\n" + diag
            elif p_baseline < 0.05 and p_model >= 0.05:
                diag = (
                    "\nDIAGNOSTICO: la etiqueta oficial SI separa supervivencia en esta cohorte "
                    "externa (p={:.4g}) pero el panel congelado del modelo NO (p={:.4g}). Esto "
                    "SI apunta a un problema de generalizacion del panel reducido especificamente."
                ).format(p_baseline, p_model)
                print(diag)
                report_full += "\n" + diag
        except ValueError as e:
            print(f"No se pudo correr la linea base: {e}")

    with open(out_dir / "external_validation_report.txt", "w") as f:
        f.write(report_full)
    print(f"\nReporte guardado en: {out_dir / 'external_validation_report.txt'}")


if __name__ == "__main__":
    main()
