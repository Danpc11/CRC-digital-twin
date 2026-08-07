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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import infer_gene_columns, load_calibrated_patterns, load_labeled_dataset, zscore_genes
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
    scored.to_csv(out_dir / "scored_external_cohort.tsv", sep="\t", index=False)

    if args.duration_col not in df.columns or args.event_col not in df.columns:
        print(
            f"\nAVISO: no se encontraron columnas '{args.duration_col}'/'{args.event_col}' -- "
            "solo se guardo la clasificacion, sin validacion de supervivencia."
        )
        return

    print(f"\n--- Validacion de supervivencia externa: {args.endpoint_label} ---")
    print("\n[1/2] Reclasificacion del modelo (patrones congelados de GSE39582)")
    result_model = validate_survival_by_subtype(
        scored, duration_col=args.duration_col, event_col=args.event_col,
        endpoint_label=args.endpoint_label, group_col="predicted_cms",
    )
    report_model = interpret_validation_result(result_model)
    print(report_model)
    print(
        "\nIMPORTANTE: este resultado es sobre una cohorte que el modelo NUNCA vio "
        "durante la calibracion -- si sale significativo, es evidencia bastante mas "
        "fuerte que la validacion en la misma cohorte de entrenamiento."
    )

    report_full = report_model

    if "cms_label" in scored.columns:
        print("\n[2/2] Linea base: etiqueta CMS oficial del consorcio (en esta misma cohorte externa)")
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
            report_full = report_model + "\n\n" + "=" * 60 + "\n\n" + report_baseline

            p_model = result_model["logrank_p_value"]
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
