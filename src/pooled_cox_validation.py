"""
pooled_cox_validation.py

Combina multiples cohortes de validacion externa en un solo analisis
de supervivencia con mas poder estadistico que log-rank tests
separados por cohorte, cada uno subpotenciado por si solo (GSE17536
n=145, GSE17537 n=55 -- ninguno alcanza para detectar un efecto
moderado con confianza).

Usa un modelo de Cox de riesgos proporcionales ESTRATIFICADO por
cohorte: cada cohorte puede tener su propia funcion de riesgo basal
(dan cuenta de diferencias tecnicas/de poblacion entre estudios), pero
el efecto del subtipo CMS predicho se estima de forma conjunta sobre
todos los pacientes combinados. Esto es mas correcto que promediar
p-valores o concatenar sin ajustar por cohorte.

USO:
    python3 src/pooled_cox_validation.py \\
        --cohort GSE17536 results_external_gse17536_final/scored_external_cohort.tsv \\
        --cohort GSE17537 results_external_gse17537/scored_external_cohort.tsv \\
        --output results_pooled_cox/
"""

import argparse
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", action="append", nargs=2, metavar=("NOMBRE", "SCORED_TSV"), required=True,
        help="Nombre de la cohorte y ruta a su scored_external_cohort.tsv (de external_validation.py). "
             "Repetir para cada cohorte a combinar."
    )
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--reference", default=None,
                         help="Subtipo CMS a usar como referencia (default: el mas frecuente en el pool)")
    parser.add_argument("--output", default="results_pooled_cox")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, path in args.cohort:
        df = pd.read_csv(path, sep="\t")
        missing = {args.duration_col, args.event_col, "predicted_cms"} - set(df.columns)
        if missing:
            raise ValueError(f"'{path}' no tiene las columnas requeridas: {missing}")
        df["cohort"] = name
        frames.append(df)
        print(f"{name}: {len(df)} muestras cargadas desde {path}")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.dropna(subset=[args.duration_col, args.event_col, "predicted_cms"])

    print(f"\nn total combinado (con datos completos): {len(pooled)}")
    print("\nMuestras por cohorte:")
    print(pooled.groupby("cohort").size())
    print("\nMuestras por subtipo predicho:")
    print(pooled["predicted_cms"].value_counts())

    if pooled["cohort"].nunique() < 2:
        print(
            "\nAVISO: solo hay 1 cohorte con datos completos -- el modelo estratificado "
            "no aporta nada sobre un log-rank simple en ese caso, pero se corre igual."
        )

    reference = args.reference or pooled["predicted_cms"].value_counts().idxmax()
    print(f"\nSubtipo de referencia (hazard ratio = 1.0 para este grupo): {reference}")

    dummies = pd.get_dummies(pooled["predicted_cms"], prefix="cms", dtype=float)
    ref_col = f"cms_{reference}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    else:
        print(f"AVISO: '{reference}' no encontrado entre los subtipos predichos, no se elimina ninguna columna de referencia")

    cox_df = pd.concat([
        pooled[[args.duration_col, args.event_col, "cohort"]].reset_index(drop=True),
        dummies.reset_index(drop=True),
    ], axis=1)
    cox_df = cox_df.rename(columns={args.duration_col: "duration", args.event_col: "event"})

    print("\n--- Ajustando Cox estratificado por cohorte ---")
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event", strata=["cohort"])

    print("\n--- Resumen del modelo ---")
    cph.print_summary()

    summary = cph.summary
    summary.to_csv(out_dir / "cox_summary.tsv", sep="\t")

    lr_test = cph.log_likelihood_ratio_test()
    print(
        f"\nTest de razon de verosimilitud (significancia GLOBAL de que el subtipo CMS "
        f"predicho aporte informacion de supervivencia, sobre {len(pooled)} pacientes "
        f"combinados de {pooled['cohort'].nunique()} cohortes): "
        f"p={lr_test.p_value:.4g}, chi2={lr_test.test_statistic:.3f}, df={lr_test.degrees_freedom}"
    )

    with open(out_dir / "cox_report.txt", "w") as f:
        f.write(f"n total combinado: {len(pooled)}\n")
        f.write(f"Cohortes: {dict(pooled.groupby('cohort').size())}\n")
        f.write(f"Subtipo de referencia: {reference}\n\n")
        f.write(str(summary))
        f.write(
            f"\n\nLog-likelihood ratio test (significancia global): "
            f"p={lr_test.p_value:.4g}, chi2={lr_test.test_statistic:.3f}, df={lr_test.degrees_freedom}\n"
        )
        f.write(
            "\nNOTA: p<0.05 aqui es evidencia mas fuerte que log-rank tests separados por "
            "cohorte porque combina el poder estadistico de ambas sin ignorar diferencias "
            "de linea base entre estudios (estratificacion). Sigue sin ser validacion "
            "clinica -- eso requeriria cohortes prospectivas independientes adicionales."
        )

    print(f"\nReporte completo guardado en: {out_dir}")


if __name__ == "__main__":
    main()
