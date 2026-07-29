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
    save_calibrated_patterns(patterns, gene_cols, out_dir / "calibrated_patterns.tsv")

    if args.skip_survival:
        print("\nValidacion de supervivencia omitida (--skip-survival).")
        return

    if "relapse_free_months" not in df.columns or "relapse_event" not in df.columns:
        print(
            "\nAVISO: el TSV no tiene columnas 'relapse_free_months'/'relapse_event' -- "
            "no se puede correr validacion de supervivencia. Usa --skip-survival para "
            "silenciar este aviso, o agrega esas columnas."
        )
        return

    print("\n--- Validacion de supervivencia ---")
    z = zscore_genes(df, gene_cols)
    scored = score_cohort(z, gene_cols, patterns)
    scored.to_csv(out_dir / "scored_cohort.tsv", sep="\t", index=False)

    result = validate_survival_by_subtype(scored)
    print(interpret_validation_result(result))

    with open(out_dir / "validation_report.txt", "w") as f:
        f.write(interpret_validation_result(result))
    print(f"\nReporte guardado en: {out_dir / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
