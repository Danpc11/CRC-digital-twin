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
    cph_adj = lr_adj = cph_restr = None
    n_adj = n_restr = 0
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
                # MODELO 2a -- crudo RESTRINGIDO a la misma muestra del ajustado.
                # Esta es la comparacion que separa dos explicaciones que de
                # otro modo se confunden: si el HR cae aqui (sin incluir
                # estadio), la atenuacion viene de la PERDIDA DE MUESTRA
                # (menos eventos); si se mantiene aqui y solo cae en el
                # ajustado, la atenuacion viene del AJUSTE real por estadio.
                cph_restr, lr_restr, n_restr = fit_cox(
                    adj_data, [],
                    "MODELO CRUDO RESTRINGIDO -- misma muestra que el ajustado, SIN estadio")
                cph_restr.summary.to_csv(out_dir / "cox_summary_crude_restricted.tsv", sep="\t")

                cph_adj, lr_adj, n_adj = fit_cox(
                    adj_data, ["stage_harmonized"],
                    "MODELO AJUSTADO -- subtipo CMS + estadio")
                cph_adj.summary.to_csv(out_dir / "cox_summary_adjusted.tsv", sep="\t")

    # --- Comparacion de los tres modelos: la pregunta que importa ---
    if cph_adj is not None:
        print(f"\n{'=' * 78}\nCOMPARACION DE MODELOS\n{'=' * 78}")
        rows = []
        for cov in cph_crude.summary.index:
            row = {"covariable": cov,
                    "HR_crudo_full": cph_crude.summary.loc[cov, "exp(coef)"],
                    "p_crudo_full": cph_crude.summary.loc[cov, "p"]}
            if cph_restr is not None and cov in cph_restr.summary.index:
                row["HR_crudo_restr"] = cph_restr.summary.loc[cov, "exp(coef)"]
                row["p_crudo_restr"] = cph_restr.summary.loc[cov, "p"]
            if cov in cph_adj.summary.index:
                row["HR_ajustado"] = cph_adj.summary.loc[cov, "exp(coef)"]
                row["p_ajustado"] = cph_adj.summary.loc[cov, "p"]
            rows.append(row)
        comp = pd.DataFrame(rows)
        print(comp.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        comp.to_csv(out_dir / "cox_model_comparison.tsv", sep="\t", index=False)

        ev_full = int(cph_crude.event_observed.sum())
        ev_adj = int(cph_adj.event_observed.sum())
        print(f"\nn crudo completo = {n_crude} ({ev_full} eventos)")
        print(f"n restringido/ajustado = {n_adj} ({ev_adj} eventos, "
              f"{100*(ev_full-ev_adj)/ev_full:.0f}% menos que el completo)")

        # --- Diagnostico automatico por covariable ---
        if cph_restr is not None:
            print(f"\n{'-' * 78}\nDIAGNOSTICO: origen de la atenuacion\n{'-' * 78}")
            for cov in cph_crude.summary.index:
                if cov not in cph_restr.summary.index or cov not in cph_adj.summary.index:
                    continue
                hr_f = cph_crude.summary.loc[cov, "exp(coef)"]
                hr_r = cph_restr.summary.loc[cov, "exp(coef)"]
                hr_a = cph_adj.summary.loc[cov, "exp(coef)"]

                d_restr = hr_r - hr_f     # cambio por restriccion de muestra
                d_adj = hr_a - hr_r       # cambio por el ajuste en si
                d_total = hr_a - hr_f

                # cambio relativo respecto al efecto crudo, para no
                # sobreinterpretar movimientos triviales
                rel = abs(d_total) / max(abs(hr_f - 1.0), 1e-9)
                if rel < 0.15:
                    verdict = "el HR practicamente no cambia -- robusto al ajuste y a la restriccion"
                elif abs(d_adj) > abs(d_restr) * 1.5:
                    direction = "se atenua" if abs(hr_a - 1) < abs(hr_r - 1) else "se refuerza"
                    verdict = (f"el efecto {direction} sobre todo por el AJUSTE POR ESTADIO "
                               f"(cambio por ajuste: {d_adj:+.2f} vs. por restriccion: {d_restr:+.2f})")
                elif abs(d_restr) > abs(d_adj) * 1.5:
                    direction = "se atenua" if abs(hr_r - 1) < abs(hr_f - 1) else "se refuerza"
                    verdict = (f"el efecto {direction} sobre todo por la RESTRICCION DE MUESTRA, "
                               f"no por el ajuste (cambio por restriccion: {d_restr:+.2f} vs. "
                               f"por ajuste: {d_adj:+.2f})")
                else:
                    verdict = (f"el cambio se reparte entre restriccion ({d_restr:+.2f}) "
                               f"y ajuste ({d_adj:+.2f})")

                print(f"\n{cov}")
                print(f"  HR: {hr_f:.2f} (completo) -> {hr_r:.2f} (restringido) -> {hr_a:.2f} (ajustado)")
                print(f"  {verdict}")

        print(
            "\nCOMO LEER ESTO: comparar el crudo COMPLETO con el ajustado confunde dos "
            "efectos (menos pacientes y control por estadio). El crudo RESTRINGIDO usa "
            "exactamente la misma muestra que el ajustado, asi que la diferencia entre "
            "esos dos se debe UNICAMENTE al ajuste por estadio."
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
