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
    result = validate_survival_by_subtype(
        scored, duration_col=args.duration_col, event_col=args.event_col,
        endpoint_label=args.endpoint_label, group_col="predicted_cms",
    )
    report = interpret_validation_result(result)
    print(report)
    print(
        "\nIMPORTANTE: este resultado es sobre una cohorte que el modelo NUNCA vio "
        "durante la calibracion -- si sale significativo, es evidencia bastante mas "
        "fuerte que la validacion en la misma cohorte de entrenamiento."
    )

    with open(out_dir / "external_validation_report.txt", "w") as f:
        f.write(report)
    print(f"\nReporte guardado en: {out_dir / 'external_validation_report.txt'}")


if __name__ == "__main__":
    main()
