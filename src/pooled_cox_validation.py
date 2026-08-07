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
import sys

import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_covariates import prepare_covariates


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
    parser.add_argument("--adjust-stage", action="store_true",
                         help="Ademas del modelo crudo, ajustar por estadio clinico armonizado "
                              "(requiere columna 'stage' en los TSV de entrada)")
    parser.add_argument("--keep-stage-iv", action="store_true",
                         help="No excluir pacientes en estadio IV del modelo ajustado "
                              "(por default se excluyen: ya tienen metastasis al diagnostico)")
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

    def fit_cox(data: pd.DataFrame, covariates: list, label: str):
        """Ajusta un Cox estratificado por cohorte con las covariables dadas."""
        dummies = pd.get_dummies(data["predicted_cms"], prefix="cms", dtype=float)
        ref_col = f"cms_{reference}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])

        base = data[[args.duration_col, args.event_col, "cohort"] + covariates].reset_index(drop=True)
        cox_df = pd.concat([base, dummies.reset_index(drop=True)], axis=1)
        cox_df = cox_df.rename(columns={args.duration_col: "duration", args.event_col: "event"})
        cox_df = cox_df.dropna()

        print(f"\n{'=' * 66}\n{label}  (n = {len(cox_df)})\n{'=' * 66}")
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="duration", event_col="event", strata=["cohort"])
        cph.print_summary()
        lr = cph.log_likelihood_ratio_test()
        print(f"\nTest de razon de verosimilitud: p={lr.p_value:.4g}, "
              f"chi2={lr.test_statistic:.3f}, df={lr.degrees_freedom}")
        print(f"Concordance: {cph.concordance_index_:.3f}")
        return cph, lr, len(cox_df)

    # --- Modelo 1: CRUDO (solo subtipo CMS) ---
    cph_crude, lr_crude, n_crude = fit_cox(pooled, [], "MODELO CRUDO -- solo subtipo CMS")
    cph_crude.summary.to_csv(out_dir / "cox_summary_crude.tsv", sep="\t")

    # --- Modelo 2: AJUSTADO por estadio (si hay datos) ---
    cph_adj = lr_adj = None
    n_adj = 0
    if args.adjust_stage:
        if "stage" not in pooled.columns:
            print(
                "\nAVISO: se pidio --adjust-stage pero las cohortes no traen columna 'stage'. "
                "Reconstruye los datasets con los scripts build_*_dataset.py actualizados "
                "(incluyen el estadio) y vuelve a correr external_validation.py."
            )
        else:
            print(f"\n{'=' * 66}\nARMONIZACION DE ESTADIO ENTRE COHORTES\n{'=' * 66}")
            adj_data = prepare_covariates(
                pooled, stage_col="stage", drop_stage_iv=not args.keep_stage_iv)
            if adj_data["stage_harmonized"].notna().sum() < 20:
                print("\nDemasiados pocos pacientes con estadio utilizable -- se omite el "
                      "modelo ajustado.")
            else:
                cph_adj, lr_adj, n_adj = fit_cox(
                    adj_data, ["stage_harmonized"],
                    "MODELO AJUSTADO -- subtipo CMS + estadio")
                cph_adj.summary.to_csv(out_dir / "cox_summary_adjusted.tsv", sep="\t")

    # --- Comparacion crudo vs. ajustado: la pregunta que importa ---
    if cph_adj is not None:
        print(f"\n{'=' * 66}\nCRUDO vs. AJUSTADO POR ESTADIO\n{'=' * 66}")
        rows = []
        for cov in cph_crude.summary.index:
            hr_c = cph_crude.summary.loc[cov, "exp(coef)"]
            p_c = cph_crude.summary.loc[cov, "p"]
            if cov in cph_adj.summary.index:
                hr_a = cph_adj.summary.loc[cov, "exp(coef)"]
                p_a = cph_adj.summary.loc[cov, "p"]
            else:
                hr_a = p_a = float("nan")
            rows.append({"covariable": cov, "HR_crudo": hr_c, "p_crudo": p_c,
                          "HR_ajustado": hr_a, "p_ajustado": p_a})
        comp = pd.DataFrame(rows)
        print(comp.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        comp.to_csv(out_dir / "cox_crude_vs_adjusted.tsv", sep="\t", index=False)

        print(f"\nn crudo = {n_crude}, n ajustado = {n_adj} "
              f"(la diferencia son pacientes sin estadio utilizable o en estadio IV).")
        print(
            "\nCOMO LEER ESTO: si un subtipo mantiene su HR y su significancia al ajustar "
            "por estadio, aporta informacion pronostica INDEPENDIENTE de la estadificacion "
            "clinica -- que es lo que justifica hacer una prueba molecular adicional. Si el "
            "efecto desaparece, el panel podria estar detectando indirectamente el estadio, "
            "no valor pronostico propio."
        )

    cph, lr_test, summary = cph_crude, lr_crude, cph_crude.summary

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
