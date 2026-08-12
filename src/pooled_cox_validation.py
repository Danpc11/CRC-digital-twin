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

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_covariates import prepare_covariates


def build_cox_frame(
    data: pd.DataFrame, duration_col: str, event_col: str,
    reference: str, clinical_covariates: list[str] | None = None,
    include_cms: bool = True, cms_levels: list[str] | None = None,
    group_col: str = "predicted_cms",
) -> pd.DataFrame:
    """Construye exactamente la misma muestra para modelos Cox anidados."""
    clinical_covariates = clinical_covariates or []
    base_cols = [duration_col, event_col, "cohort"] + clinical_covariates
    base = data[base_cols].reset_index(drop=True).copy()
    base = base.rename(columns={duration_col: "duration", event_col: "event"})
    if include_cms:
        levels = cms_levels or sorted(data[group_col].dropna().unique())
        if reference not in levels:
            raise ValueError(f"La referencia '{reference}' no aparece en {group_col}")
        groups = pd.Categorical(data[group_col], categories=levels)
        dummies = pd.get_dummies(groups, prefix="cms", dtype=float)
        dummies = dummies.drop(columns=[f"cms_{reference}"], errors="ignore")
        base = pd.concat([base, dummies.reset_index(drop=True)], axis=1)
    return base.dropna()


def nested_model_increment(reduced: CoxPHFitter, full: CoxPHFitter) -> dict:
    """Prueba LRT del aporte conjunto de las covariables añadidas."""
    df_added = len(full.params_) - len(reduced.params_)
    if df_added <= 0:
        raise ValueError("El modelo completo debe contener mas parametros que el reducido")
    statistic = max(0.0, 2.0 * (full.log_likelihood_ - reduced.log_likelihood_))
    return {
        "lr_chi2": float(statistic), "df": int(df_added),
        "p_incremental": float(chi2.sf(statistic, df_added)),
        "c_index_stage_only": float(reduced.concordance_index_),
        "c_index_stage_plus_cms": float(full.concordance_index_),
        "delta_c_index": float(full.concordance_index_ - reduced.concordance_index_),
        "aic_partial_stage_only": float(reduced.AIC_partial_),
        "aic_partial_stage_plus_cms": float(full.AIC_partial_),
    }


def bootstrap_cindex_increment(
    data: pd.DataFrame, duration_col: str, event_col: str, reference: str,
    iterations: int = 200, seed: int = 2026,
    group_col: str = "predicted_cms",
) -> pd.DataFrame:
    """Bootstrap estratificado por cohorte del cambio de C-index al añadir CMS."""
    if iterations < 0:
        raise ValueError("bootstrap_iterations debe ser >= 0")
    if iterations == 0:
        return pd.DataFrame(columns=["iteration", "delta_c_index"])
    rng = np.random.default_rng(seed)
    levels = sorted(data[group_col].dropna().unique())
    rows = []
    for iteration in range(iterations):
        pieces = []
        for _, cohort_df in data.groupby("cohort", sort=False):
            sampled_positions = rng.integers(0, len(cohort_df), size=len(cohort_df))
            pieces.append(cohort_df.iloc[sampled_positions].copy())
        sample = pd.concat(pieces, ignore_index=True)
        try:
            stage_df = build_cox_frame(
                sample, duration_col, event_col, reference,
                ["stage_harmonized"], include_cms=False, group_col=group_col)
            full_df = build_cox_frame(
                sample, duration_col, event_col, reference,
                ["stage_harmonized"], include_cms=True, cms_levels=levels,
                group_col=group_col)
            stage_model = CoxPHFitter().fit(
                stage_df, "duration", "event", strata=["cohort"])
            full_model = CoxPHFitter().fit(
                full_df, "duration", "event", strata=["cohort"])
            rows.append({
                "iteration": iteration,
                "delta_c_index": full_model.concordance_index_ - stage_model.concordance_index_,
            })
        except Exception:
            # Algunos remuestreos con muy pocos eventos pueden ser singulares.
            rows.append({"iteration": iteration, "delta_c_index": np.nan})
    return pd.DataFrame(rows)


def leave_one_cohort_out_validation(
    data: pd.DataFrame, duration_col: str, event_col: str, reference: str,
    group_col: str = "predicted_cms",
) -> pd.DataFrame:
    """Entrena omitiendo cada cohorte y evalúa discriminación en la omitida.

    El C-index de prueba usa el predictor lineal, que no necesita estimar una
    función basal para la cohorte nueva. La calibración absoluta no puede
    extrapolarse así desde un Cox estratificado y se reporta por separado.
    """
    levels = sorted(data[group_col].dropna().unique())
    rows = []
    for omitted in sorted(data["cohort"].unique()):
        train = data[data["cohort"] != omitted]
        test = data[data["cohort"] == omitted]
        row = {
            "cohort_omitted": omitted, "n_train": len(train),
            "events_train": int(train[event_col].sum()), "n_test": len(test),
            "events_test": int(test[event_col].sum()),
        }
        try:
            train_stage = build_cox_frame(
                train, duration_col, event_col, reference, ["stage_harmonized"],
                include_cms=False, group_col=group_col)
            train_full = build_cox_frame(
                train, duration_col, event_col, reference, ["stage_harmonized"],
                include_cms=True, cms_levels=levels, group_col=group_col)
            stage_model = CoxPHFitter().fit(
                train_stage, "duration", "event", strata=["cohort"])
            full_model = CoxPHFitter().fit(
                train_full, "duration", "event", strata=["cohort"])
            row.update(nested_model_increment(stage_model, full_model))

            test_stage = build_cox_frame(
                test, duration_col, event_col, reference, ["stage_harmonized"],
                include_cms=False, group_col=group_col)
            test_full = build_cox_frame(
                test, duration_col, event_col, reference, ["stage_harmonized"],
                include_cms=True, cms_levels=levels, group_col=group_col)
            stage_lp = test_stage[stage_model.params_.index] @ stage_model.params_
            full_lp = test_full[full_model.params_.index] @ full_model.params_
            row["c_index_test_stage_only"] = concordance_index(
                test_stage["duration"], -stage_lp, test_stage["event"])
            row["c_index_test_stage_plus_cms"] = concordance_index(
                test_full["duration"], -full_lp, test_full["event"])
            row["delta_c_index_test"] = (
                row["c_index_test_stage_plus_cms"] - row["c_index_test_stage_only"])
            for covariate in full_model.summary.index:
                if covariate.startswith("cms_"):
                    row[f"HR_{covariate}"] = float(
                        full_model.summary.loc[covariate, "exp(coef)"])
                    row[f"p_{covariate}"] = float(full_model.summary.loc[covariate, "p"])
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def apparent_calibration_at_horizons(
    cph: CoxPHFitter, cox_df: pd.DataFrame,
    horizons: list[float] | tuple[float, ...] = (36.0, 60.0),
) -> pd.DataFrame:
    """Compara riesgo medio predicho y riesgo KM dentro de cada estrato.

    Es calibración aparente porque coeficientes y riesgos basales se estiman en
    estas mismas cohortes. Se guarda para detectar desajustes gruesos, no como
    validación externa de riesgo absoluto.
    """
    rows = []
    for cohort, sub in cox_df.groupby("cohort"):
        km = KaplanMeierFitter().fit(sub["duration"], sub["event"])
        for horizon in horizons:
            survival = cph.predict_survival_function(sub, times=[float(horizon)])
            predicted_risk = float(1.0 - survival.iloc[0].mean())
            observed_risk = float(1.0 - km.predict(float(horizon)))
            rows.append({
                "cohort": cohort, "horizon_months": float(horizon),
                "n": len(sub), "events_total": int(sub["event"].sum()),
                "predicted_risk_mean": predicted_risk,
                "observed_risk_km": observed_risk,
                "calibration_error_predicted_minus_observed": predicted_risk - observed_risk,
                "calibration_type": "aparente_dentro_del_estrato",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", action="append", nargs=2, metavar=("NOMBRE", "SCORED_TSV"), required=True,
        help="Nombre de la cohorte y ruta a su scored_external_cohort.tsv (de external_validation.py). "
             "Repetir para cada cohorte a combinar."
    )
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--group-col", default="predicted_cms",
                        help="Columna CMS: predicted_cms o modern_hopfield_cms")
    parser.add_argument("--reference", default=None,
                         help="Subtipo CMS a usar como referencia (default: el mas frecuente en el pool)")
    parser.add_argument("--adjust-stage", action="store_true",
                         help="Ademas del modelo crudo, ajustar por estadio clinico armonizado "
                              "(requiere columna 'stage' en los TSV de entrada)")
    parser.add_argument("--keep-stage-iv", action="store_true",
                         help="No excluir pacientes en estadio IV del modelo ajustado "
                              "(por default se excluyen: ya tienen metastasis al diagnostico)")
    parser.add_argument("--bootstrap-iterations", type=int, default=200,
                        help="Remuestreos para IC del cambio de C-index; 0 lo desactiva")
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--calibration-horizons", nargs="+", type=float,
                        default=[36.0, 60.0],
                        help="Horizontes para calibracion aparente dentro de cada cohorte")
    parser.add_argument("--no-loco", action="store_true",
                        help="Omitir validacion leave-one-cohort-out")
    parser.add_argument("--output", default="results_pooled_cox")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, path in args.cohort:
        df = pd.read_csv(path, sep="\t")
        missing = {args.duration_col, args.event_col, args.group_col} - set(df.columns)
        if missing:
            raise ValueError(f"'{path}' no tiene las columnas requeridas: {missing}")
        df["cohort"] = name
        frames.append(df)
        print(f"{name}: {len(df)} muestras cargadas desde {path}")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.dropna(subset=[args.duration_col, args.event_col, args.group_col])
    if args.group_col == "modern_hopfield_cms":
        n_before = len(pooled)
        pooled = pooled[pooled[args.group_col] != "indeterminado"]
        print(f"Modern Hopfield: {n_before-len(pooled)} muestras indeterminadas excluidas")

    print(f"\nn total combinado (con datos completos): {len(pooled)}")
    print("\nMuestras por cohorte:")
    print(pooled.groupby("cohort").size())
    print("\nMuestras por subtipo predicho:")
    print(pooled[args.group_col].value_counts())

    if pooled["cohort"].nunique() < 2:
        print(
            "\nAVISO: solo hay 1 cohorte con datos completos -- el modelo estratificado "
            "no aporta nada sobre un log-rank simple en ese caso, pero se corre igual."
        )

    reference = args.reference or pooled[args.group_col].value_counts().idxmax()
    print(f"\nSubtipo de referencia (hazard ratio = 1.0 para este grupo): {reference}")

    cms_levels = sorted(pooled[args.group_col].dropna().unique())

    def fit_cox(
        data: pd.DataFrame, covariates: list, label: str,
        include_cms: bool = True,
    ):
        """Ajusta un Cox estratificado por cohorte con las covariables dadas."""
        cox_df = build_cox_frame(
            data, args.duration_col, args.event_col, reference,
            covariates, include_cms=include_cms, cms_levels=cms_levels,
            group_col=args.group_col)

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
    cph_adj = lr_adj = cph_restr = cph_stage_only = None
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

                cph_stage_only, _, _ = fit_cox(
                    adj_data, ["stage_harmonized"],
                    "MODELO CLINICO BASE -- solo estadio", include_cms=False)
                cph_stage_only.summary.to_csv(
                    out_dir / "cox_summary_stage_only.tsv", sep="\t")

                cph_adj, lr_adj, n_adj = fit_cox(
                    adj_data, ["stage_harmonized"],
                    "MODELO AJUSTADO -- subtipo CMS + estadio")
                cph_adj.summary.to_csv(out_dir / "cox_summary_adjusted.tsv", sep="\t")

                incremental = nested_model_increment(cph_stage_only, cph_adj)
                print(f"\n{'=' * 78}\nAPORTE INCREMENTAL DE CMS SOBRE ESTADIO\n{'=' * 78}")
                print(
                    f"LRT estadio vs. estadio+CMS: chi2={incremental['lr_chi2']:.3f}, "
                    f"df={incremental['df']}, p={incremental['p_incremental']:.4g}")
                print(
                    f"C-index: estadio={incremental['c_index_stage_only']:.3f}, "
                    f"estadio+CMS={incremental['c_index_stage_plus_cms']:.3f}, "
                    f"delta={incremental['delta_c_index']:+.3f}")

                boot = bootstrap_cindex_increment(
                    adj_data, args.duration_col, args.event_col, reference,
                    iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
                    group_col=args.group_col)
                valid_delta = boot["delta_c_index"].dropna()
                incremental["bootstrap_iterations_requested"] = args.bootstrap_iterations
                incremental["bootstrap_iterations_valid"] = len(valid_delta)
                if len(valid_delta):
                    low, high = np.quantile(valid_delta, [0.025, 0.975])
                    incremental["delta_c_index_bootstrap_low95"] = float(low)
                    incremental["delta_c_index_bootstrap_high95"] = float(high)
                    print(
                        f"IC95% bootstrap del delta C-index: [{low:+.3f}, {high:+.3f}] "
                        f"({len(valid_delta)}/{args.bootstrap_iterations} remuestreos validos)")
                pd.DataFrame([incremental]).to_csv(
                    out_dir / "cox_incremental_value.tsv", sep="\t", index=False)
                boot.to_csv(out_dir / "cox_incremental_cindex_bootstrap.tsv", sep="\t", index=False)

                if not args.no_loco:
                    print(f"\n{'=' * 78}\nVALIDACION LEAVE-ONE-COHORT-OUT\n{'=' * 78}")
                    loco = leave_one_cohort_out_validation(
                        adj_data, args.duration_col, args.event_col, reference,
                        group_col=args.group_col)
                    print(loco.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
                    loco.to_csv(
                        out_dir / "cox_leave_one_cohort_out.tsv", sep="\t", index=False)

                adjusted_cox_df = build_cox_frame(
                    adj_data, args.duration_col, args.event_col, reference,
                    ["stage_harmonized"], include_cms=True,
                    cms_levels=cms_levels, group_col=args.group_col)
                calibration = apparent_calibration_at_horizons(
                    cph_adj, adjusted_cox_df, args.calibration_horizons)
                print(f"\n{'=' * 78}\nCALIBRACION APARENTE POR COHORTE\n{'=' * 78}")
                print(calibration.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
                print("AVISO: esta calibracion reutiliza las cohortes del ajuste. Un Cox "
                      "estratificado no puede transferir riesgo absoluto a una cohorte nueva "
                      "sin estimar su riesgo basal; no reportar esto como calibracion externa.")
                calibration.to_csv(
                    out_dir / "cox_apparent_calibration.tsv", sep="\t", index=False)

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
        f.write(f"Columna de clasificacion: {args.group_col}\n\n")
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
