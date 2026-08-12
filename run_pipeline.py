"""
run_pipeline.py

Orquestador de linea de comandos: calibracion + validacion de
supervivencia en un solo paso, sobre cualquier TSV con el esquema
descrito en calibration.py.

Uso:
    python run_pipeline.py --input data/synthetic_cohort.tsv --output results/

Con datos reales (una vez descargados de GSE39582 + etiquetas CMS de
Synapse, formateados al esquema TSV esperado):
    python run_pipeline.py --input data/gse39582_cms_labeled.tsv --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration import (
    calibrate_patterns_from_data,
    compute_gene_stats,
    infer_gene_columns,
    load_labeled_dataset,
    save_calibrated_patterns,
    zscore_genes,
)
from survival_validation import (
    interpret_validation_result,
    score_cohort,
    validate_survival_by_subtype,
)

# Pares de columnas de supervivencia reconocidos, en orden de preferencia.
# RFS (recidiva) es preferible a OS (global) cuando ambos estan
# disponibles porque es el endpoint mas directamente accionable
# clinicamente, pero no siempre esta curado en la fuente (ej. TCGA-COAD
# no tiene dfsMo poblado, solo osMo -- ver format_to_schema.py).
SURVIVAL_ENDPOINTS = [
    ("relapse_free_months", "relapse_event", "supervivencia libre de recidiva (RFS)"),
    ("overall_survival_months", "death_event", "supervivencia global (OS)"),
]


def detect_survival_endpoint(df):
    """Detecta cual par de columnas de supervivencia esta presente en el TSV."""
    for duration_col, event_col, label in SURVIVAL_ENDPOINTS:
        if duration_col in df.columns and event_col in df.columns:
            return duration_col, event_col, label
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Pipeline de calibracion + validacion CRC digital twin")
    parser.add_argument("--input", required=True, help="TSV de entrada (ver esquema en calibration.py)")
    parser.add_argument("--output", default="results", help="Directorio de salida")
    parser.add_argument(
        "--skip-survival", action="store_true",
        help="Omitir validacion de supervivencia (usar si el TSV no tiene columnas de supervivencia)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset: {args.input}")
    df = load_labeled_dataset(args.input)
    gene_cols = infer_gene_columns(df)
    print(f"Genes detectados ({len(gene_cols)}): {gene_cols}")

    print("\n--- Calibracion ---")
    patterns, gene_cols = calibrate_patterns_from_data(df, gene_cols)
    gene_stats = compute_gene_stats(df, gene_cols)
    save_calibrated_patterns(patterns, gene_cols, out_dir / "calibrated_patterns.tsv", gene_stats)

    if args.skip_survival:
        print("\nValidacion de supervivencia omitida (--skip-survival).")
        return

    duration_col, event_col, endpoint_label = detect_survival_endpoint(df)
    if duration_col is None:
        print(
            "\nAVISO: el TSV no tiene ninguno de los pares de columnas de "
            f"supervivencia reconocidos ({[e[:2] for e in SURVIVAL_ENDPOINTS]}) -- "
            "no se puede correr validacion. Usa --skip-survival para silenciar "
            "este aviso, o agrega esas columnas."
        )
        return

    print(f"\n--- Validacion de supervivencia: {endpoint_label} ---")
    z = zscore_genes(df, gene_cols)
    scored = score_cohort(z, gene_cols, patterns)
    scored.to_csv(out_dir / "scored_cohort.tsv", sep="\t", index=False)

    print("\n[1/2] Reclasificacion del modelo (panel reducido)")
    result_model = validate_survival_by_subtype(
        scored, duration_col=duration_col, event_col=event_col,
        endpoint_label=endpoint_label, group_col="predicted_cms",
    )
    report_model = interpret_validation_result(result_model)
    print(report_model)

    report_full = report_model

    if "cms_label" in scored.columns:
        print("\n[2/2] Linea base: etiqueta CMS oficial del consorcio")
        # Excluir 'none' (muestras no clasificadas por el consorcio) --
        # no es un grupo biologico, es "sin clasificar", y agrega ruido
        # a un test multivariado sin aportar nada interpretable.
        baseline_df = scored[scored["cms_label"] != "none"]
        n_excluded = len(scored) - len(baseline_df)
        if n_excluded > 0:
            print(f"  ({n_excluded} muestras 'none' excluidas de la linea base)")
        try:
            result_baseline = validate_survival_by_subtype(
                baseline_df, duration_col=duration_col, event_col=event_col,
                endpoint_label=endpoint_label, group_col="cms_label",
            )
            report_baseline = interpret_validation_result(result_baseline)
            print(report_baseline)
            report_full = report_model + "\n\n" + "=" * 60 + "\n\n" + report_baseline

            p_model = result_model["logrank_p_value"]
            p_baseline = result_baseline["logrank_p_value"]
            if p_baseline < 0.05 and p_model >= 0.05:
                diag = (
                    "\nDIAGNOSTICO: la etiqueta CMS oficial SI separa supervivencia "
                    "(p={:.4g}) pero el panel reducido del modelo NO (p={:.4g}). Esto "
                    "apunta a perdida de senal por reduccion de panel -- no a ausencia "
                    "real de asociacion CMS-supervivencia en esta cohorte/endpoint. "
                    "Revisar si los genes sustitutos (fallback) usados en "
                    "format_to_schema.py capturan bien el eje biologico original."
                ).format(p_baseline, p_model)
                print(diag)
                report_full += "\n" + diag
        except ValueError as e:
            print(f"No se pudo correr la linea base: {e}")

    with open(out_dir / "validation_report.txt", "w") as f:
        f.write(report_full)
    print(f"\nReporte guardado en: {out_dir / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
