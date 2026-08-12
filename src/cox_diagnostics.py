"""
cox_diagnostics.py

Diagnosticos formales del modelo de Cox, ausentes en pooled_cox_validation.py
hasta ahora -- el modelo ajustaba riesgos proporcionales y reportaba HR/p,
pero nunca verificaba si el supuesto central (riesgos proporcionales)
realmente se sostiene, si hay observaciones que dominan el resultado, o
si el efecto es consistente entre cohortes en vez de ser un artefacto
de una sola de ellas dominando el pool.

Tres diagnosticos, cada uno con evidencia de que la mecanica funciona
(ver tests):

  1. Supuesto de riesgos proporcionales (test de Schoenfeld) --
     lifelines.statistics.proportional_hazard_test. Un p-valor
     significativo indica que el efecto de esa covariable SI cambia en
     el tiempo, violando el supuesto que todo el modelo de Cox asume.
  2. Observaciones influyentes -- residuos delta-beta
     (CoxPHFitter.compute_residuals(kind="delta_beta")). Pacientes
     cuyo delta-beta es grande estan "cargando" el resultado
     desproporcionadamente -- vale la pena revisarlos a mano.
  3. Heterogeneidad del efecto entre cohortes -- ajusta el modelo por
     separado en cada cohorte y compara los HR (estilo forest plot),
     mas un termino de interaccion cohorte×covariable como prueba
     formal de heterogeneidad.

USO
    python3 src/cox_diagnostics.py \\
        --input results_external_gse17536/scored_external_cohort.tsv \\
        --input results_external_gse17537/scored_external_cohort.tsv \\
        --input results_external_gse14333/scored_external_cohort.tsv \\
        --input results_external_gse33113/scored_external_cohort.tsv \\
        --reference CMS2_canonical_WNT --output results_cox_diagnostics/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import norm


def check_proportional_hazards(cph: CoxPHFitter, df: pd.DataFrame, p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Test de Schoenfeld por covariable -- p < p_threshold indica que el
    efecto de esa covariable cambia en el tiempo, violando el supuesto
    de riesgos proporcionales que todo el modelo de Cox asume. No es
    un detalle tecnico menor: si se viola, el HR reportado es un
    promedio temporal que puede esconder un efecto que en realidad
    crece, decae, o se invierte con el tiempo.
    """
    result = proportional_hazard_test(cph, df, time_transform="rank")
    out = result.summary.copy()
    out["viola_supuesto"] = out["p"] < p_threshold
    return out


def check_influential_observations(
    cph: CoxPHFitter, df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Residuos delta-beta: cuanto cambiaria cada coeficiente si se
    quitara esa observacion. Valores grandes senalan pacientes que
    dominan desproporcionadamente el resultado -- vale la pena
    revisarlos a mano (¿son datos correctos? ¿un caso atipico real?).
    """
    residuals = cph.compute_residuals(df, kind="delta_beta")
    magnitud = residuals.abs().sum(axis=1)
    idx_top = magnitud.sort_values(ascending=False).head(top_n).index
    out = residuals.loc[idx_top].copy()
    out["magnitud_total"] = magnitud.loc[idx_top]
    return out.sort_values("magnitud_total", ascending=False)


def fit_piecewise_cms4_effect(
    df: pd.DataFrame, duration_col: str, event_col: str,
    covariate_cols: list[str], cutoff: float = 36.0,
    cohort_col: str = "cohort", stratify: bool = True,
) -> dict:
    """Estima HR de CMS4 antes y despues de un corte preespecificado.

    Cada paciente se transforma al formato start-stop. El indicador CMS4
    constante se reemplaza por dos covariables dependientes del tiempo,
    evitando interpretar un unico HR promedio si el efecto no es proporcional.
    """
    cms4_col = "cms_CMS4_mesenchymal"
    if cms4_col not in covariate_cols:
        raise ValueError(f"No se encontro la covariable requerida '{cms4_col}'")
    if cutoff <= 0:
        raise ValueError("time_cutoff debe ser > 0")

    constant_covariates = [c for c in covariate_cols if c != cms4_col]
    required = [duration_col, event_col, cms4_col] + constant_covariates
    if stratify:
        required.append(cohort_col)
    clean = df[required].dropna().copy()
    clean = clean[clean[duration_col] > 0]
    rows = []
    for patient_id, (_, row) in enumerate(clean.iterrows()):
        duration = float(row[duration_col])
        event = int(row[event_col])
        intervals = [(0.0, min(duration, cutoff), "early")]
        if duration > cutoff:
            intervals.append((cutoff, duration, "late"))
        for start, stop, period in intervals:
            if stop <= start:
                continue
            record = {
                "patient_id": patient_id, "start": start, "stop": stop,
                "event": int(event and np.isclose(stop, duration)),
                "cms4_early": float(row[cms4_col]) if period == "early" else 0.0,
                "cms4_late": float(row[cms4_col]) if period == "late" else 0.0,
            }
            for covariate in constant_covariates:
                record[covariate] = float(row[covariate])
            if stratify:
                record[cohort_col] = row[cohort_col]
            rows.append(record)
    tv_df = pd.DataFrame(rows)
    if tv_df.empty or tv_df["event"].sum() == 0:
        raise ValueError("No hay intervalos/eventos suficientes para el Cox temporal")

    ctv = CoxTimeVaryingFitter()
    ctv.fit(
        tv_df, id_col="patient_id", start_col="start", stop_col="stop",
        event_col="event", strata=[cohort_col] if stratify else None,
    )
    covariance = ctv.variance_matrix_.to_numpy()
    param_names = list(ctv.params_.index)
    early_idx = param_names.index("cms4_early")
    late_idx = param_names.index("cms4_late")
    beta_early = float(ctv.params_["cms4_early"])
    beta_late = float(ctv.params_["cms4_late"])
    variance_diff = float(
        covariance[early_idx, early_idx]
        + covariance[late_idx, late_idx]
        - 2 * covariance[early_idx, late_idx])
    z_difference = (beta_early - beta_late) / np.sqrt(max(variance_diff, 1e-15))
    return {
        "model": ctv, "summary": ctv.summary.copy(), "cutoff": float(cutoff),
        "n_patients": int(len(clean)), "n_events": int(tv_df["event"].sum()),
        "hr_early": float(np.exp(beta_early)), "hr_late": float(np.exp(beta_late)),
        "z_early_vs_late": float(z_difference),
        "p_early_vs_late": float(2 * norm.sf(abs(z_difference))),
    }


def check_heterogeneity_across_cohorts(
    df: pd.DataFrame, duration_col: str, event_col: str,
    covariate_cols: list, cohort_col: str = "cohort",
) -> dict:
    """
    Ajusta el modelo POR SEPARADO en cada cohorte (sin estratificar) y
    compara los HR -- si una sola cohorte tiene un HR muy distinto a
    las demas, el efecto pooled puede estar dominado por esa cohorte
    en vez de ser un patron consistente. Complementa (no reemplaza) el
    modelo estratificado de pooled_cox_validation.py.

    Tambien corre un termino de interaccion cohorte×covariable como
    prueba formal: si la interaccion es significativa, hay evidencia
    estadistica de heterogeneidad real, no solo diferencias visuales
    en la tabla.
    """
    per_cohort = {}
    for cohort, sub in df.groupby(cohort_col):
        sub = sub[[duration_col, event_col] + covariate_cols].dropna()
        n_events = int(sub[event_col].sum())
        if n_events < 5 or sub[covariate_cols].nunique().min() < 2:
            per_cohort[cohort] = {"n": len(sub), "eventos": n_events, "HR": None, "p": None,
                                    "nota": "insuficiente para ajustar por separado"}
            continue
        cph_c = CoxPHFitter()
        try:
            cph_c.fit(sub, duration_col=duration_col, event_col=event_col)
            for cov in covariate_cols:
                per_cohort.setdefault(cohort, {})[f"HR_{cov}"] = float(cph_c.summary.loc[cov, "exp(coef)"])
                per_cohort[cohort][f"p_{cov}"] = float(cph_c.summary.loc[cov, "p"])
            per_cohort[cohort]["n"] = len(sub)
            per_cohort[cohort]["eventos"] = n_events
        except Exception as e:
            per_cohort[cohort] = {"n": len(sub), "eventos": n_events, "error": str(e)}

    tabla = pd.DataFrame(per_cohort).T

    # prueba formal de interaccion cohorte x covariable (una covariable
    # a la vez, para no saturar el modelo con pocas cohortes)
    interaction_tests = {}
    cohort_dummies = pd.get_dummies(df[cohort_col], prefix="cohort", drop_first=True, dtype=float)
    for cov in covariate_cols:
        # Deteccion proactiva: si alguna cohorte tiene 'cov' CONSTANTE
        # dentro de ella, el termino de interaccion cov*dummy_cohorte
        # queda perfectamente colineal con el dummy solo (si cov=k
        # constante, cov*dummy = k*dummy) -- causa matriz singular.
        # Confirmado real: pasa exactamente esto con GSE33113 (estadio
        # constante, cohorte de estadio II homogeneo por diseno) y
        # stage_harmonized. Detectar ANTES de intentar el fit, con un
        # mensaje que explique la causa real -- no es un bug de los
        # datos, es un problema de identificabilidad: no se puede
        # probar heterogeneidad de un efecto en una cohorte que no
        # tiene ninguna variacion de ese covariable.
        cohortes_constantes = [
            c for c, sub in df.groupby(cohort_col) if sub[cov].nunique(dropna=True) <= 1
        ]
        if cohortes_constantes:
            interaction_tests[cov] = {
                "error": (
                    f"No estimable: {cohortes_constantes} tiene(n) '{cov}' constante dentro "
                    "de la cohorte -- el termino de interaccion queda perfectamente colineal "
                    "con el dummy de esa cohorte (matriz singular, no es un bug). No se puede "
                    "probar heterogeneidad de este efecto en una cohorte sin variacion de "
                    "este covariable. Considerar excluir esa cohorte de este test especifico, "
                    "o interpretar la heterogeneidad solo entre las cohortes que si varian."
                )
            }
            continue

        inter_df = df[[duration_col, event_col, cov]].copy()
        inter_df = pd.concat([inter_df, cohort_dummies], axis=1)
        for cdum in cohort_dummies.columns:
            inter_df[f"{cov}_x_{cdum}"] = df[cov] * cohort_dummies[cdum]
        inter_df = inter_df.dropna()
        try:
            cph_int = CoxPHFitter()
            cph_int.fit(inter_df, duration_col=duration_col, event_col=event_col)
            inter_cols = [c for c in inter_df.columns if c.startswith(f"{cov}_x_")]
            # prueba de razon de verosimilitud: con vs. sin terminos de interaccion
            cph_no_int = CoxPHFitter()
            cph_no_int.fit(inter_df.drop(columns=inter_cols), duration_col=duration_col, event_col=event_col)
            lr_stat = 2 * (cph_int.log_likelihood_ - cph_no_int.log_likelihood_)
            from scipy.stats import chi2
            p_het = 1 - chi2.cdf(lr_stat, df=len(inter_cols))
            interaction_tests[cov] = {"chi2": lr_stat, "df": len(inter_cols), "p_heterogeneidad": p_het}
        except Exception as e:
            interaction_tests[cov] = {"error": str(e)}

    return {"tabla_por_cohorte": tabla, "test_interaccion": pd.DataFrame(interaction_tests).T}


def run_full_diagnostics(
    df: pd.DataFrame, duration_col: str, event_col: str, covariate_cols: list,
    cohort_col: str = "cohort", stratify: bool = True,
    output_dir: str | Path | None = None,
    time_varying_cms4: bool = True, time_cutoff: float = 36.0,
) -> dict:
    """
    Corre los tres diagnosticos sobre EL MISMO MODELO que produce los HR
    principales -- estratificado por cohorte (strata=[cohort_col]) por
    default, para que Schoenfeld/delta-beta diagnostiquen el modelo
    real (pooled_cox_validation.py), no uno distinto. stratify=False
    solo para debug/comparacion explicita.
    """
    cph = CoxPHFitter()
    cols_needed = [duration_col, event_col] + covariate_cols
    if stratify:
        cols_needed = cols_needed + [cohort_col]
    fit_df = df[cols_needed].dropna()

    if stratify:
        cph.fit(fit_df, duration_col=duration_col, event_col=event_col, strata=[cohort_col])
    else:
        cph.fit(fit_df, duration_col=duration_col, event_col=event_col)

    print("=" * 78)
    modo = "ESTRATIFICADO por cohorte (mismo modelo que produce los HR principales)" \
           if stratify else "SIN estratificar (NO es el modelo principal -- solo comparacion)"
    print(f"MODELO: {modo}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("1. SUPUESTO DE RIESGOS PROPORCIONALES (test de Schoenfeld)")
    print("=" * 78)
    ph_result = check_proportional_hazards(cph, fit_df)
    print(ph_result.to_string())
    if ph_result["viola_supuesto"].any():
        violadas = ph_result[ph_result["viola_supuesto"]].index.tolist()
        print(f"\nAVISO: el supuesto de riesgos proporcionales se viola para: {violadas}. "
              "El HR reportado para esas covariables es un promedio temporal -- "
              "puede esconder un efecto que cambia con el tiempo.")
    else:
        print("\nSupuesto de riesgos proporcionales no rechazado para ninguna covariable.")

    temporal = None
    temporal_error = None
    if time_varying_cms4 and "cms_CMS4_mesenchymal" in covariate_cols:
        print("\n" + "=" * 78)
        print(f"1b. SENSIBILIDAD TEMPORAL CMS4 (corte preespecificado={time_cutoff:g} meses)")
        print("=" * 78)
        try:
            temporal = fit_piecewise_cms4_effect(
                df, duration_col, event_col, covariate_cols,
                cutoff=time_cutoff, cohort_col=cohort_col, stratify=stratify)
            print(temporal["summary"].to_string())
            print(
                f"\nHR CMS4 temprano (<= {time_cutoff:g} meses): {temporal['hr_early']:.3f}\n"
                f"HR CMS4 tardio  (>  {time_cutoff:g} meses): {temporal['hr_late']:.3f}\n"
                f"Contraste temprano vs tardio: p={temporal['p_early_vs_late']:.4g}")
            print("El corte debe declararse antes de inspeccionar resultados; probar muchos "
                  "cortes y elegir el mas significativo inflaria el error tipo I.")
        except Exception as exc:
            temporal_error = str(exc)
            print(f"No se pudo estimar el modelo temporal CMS4: {temporal_error}")

    print("\n" + "=" * 78)
    print("2. OBSERVACIONES INFLUYENTES (residuos delta-beta)")
    print("=" * 78)
    influential = check_influential_observations(cph, fit_df)
    print(influential.to_string())
    print("\nEstas son las observaciones que mas cambiarian el resultado si se "
          "quitaran -- no implica que esten mal, pero vale la pena revisarlas.")

    print("\n" + "=" * 78)
    print("3. HETEROGENEIDAD DEL EFECTO ENTRE COHORTES")
    print("=" * 78)
    print("(Este diagnostico SI ajusta por cohorte por separado a proposito --")
    print(" es justo lo que compara: si el efecto es consistente entre cohortes")
    print(" o si el modelo estratificado esconde diferencias reales.)")
    het = None
    if cohort_col in df.columns:
        het = check_heterogeneity_across_cohorts(df, duration_col, event_col, covariate_cols, cohort_col)
        print("\nHR por cohorte, ajustado por separado:")
        print(het["tabla_por_cohorte"].to_string())
        print("\nTest de interaccion cohorte x covariable (heterogeneidad formal):")
        print(het["test_interaccion"].to_string())
    else:
        print(f"Sin columna '{cohort_col}' -- diagnostico de heterogeneidad omitido.")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ph_result.to_csv(out / "cox_diag_proportional_hazards.tsv", sep="\t")
        influential.to_csv(out / "cox_diag_influential_observations.tsv", sep="\t")
        if het is not None:
            het["tabla_por_cohorte"].to_csv(out / "cox_diag_heterogeneity_by_cohort.tsv", sep="\t")
            het["test_interaccion"].to_csv(out / "cox_diag_heterogeneity_test.tsv", sep="\t")
        if temporal is not None:
            temporal["summary"].to_csv(out / "cox_diag_cms4_time_varying.tsv", sep="\t")
            pd.DataFrame([{
                "cutoff_months": temporal["cutoff"],
                "n_patients": temporal["n_patients"], "n_events": temporal["n_events"],
                "hr_early": temporal["hr_early"], "hr_late": temporal["hr_late"],
                "z_early_vs_late": temporal["z_early_vs_late"],
                "p_early_vs_late": temporal["p_early_vs_late"],
            }]).to_csv(out / "cox_diag_cms4_time_contrast.tsv", sep="\t", index=False)
        print(f"\nTablas guardadas en: {out}")

    return {
        "cph": cph, "proportional_hazards": ph_result,
        "time_varying_cms4": temporal, "time_varying_cms4_error": temporal_error,
        "influential": influential, "heterogeneity": het,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                         help="scored_*.tsv (repetir para combinar cohortes)")
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--group-col", default="predicted_cms")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--adjust-stage", action="store_true",
                         help="Incluir estadio armonizado como covariable (requiere columna "
                              "'stage' en los TSV de entrada -- ver clinical_covariates.py)")
    parser.add_argument("--no-stratify", action="store_true",
                         help="NO estratificar por cohorte (solo para comparacion/debug -- "
                              "el modelo principal SI estratifica, ver pooled_cox_validation.py)")
    parser.add_argument("--time-cutoff", type=float, default=36.0,
                        help="Corte preespecificado para HR temprano/tardio de CMS4")
    parser.add_argument("--no-time-varying-cms4", action="store_true",
                        help="Omitir el analisis temporal de sensibilidad de CMS4")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frames = []
    for p in args.input:
        d = pd.read_csv(p, sep="\t")
        if "cohort" not in d.columns:
            d["cohort"] = Path(p).parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[args.duration_col, args.event_col, args.group_col])

    reference = args.reference or df[args.group_col].value_counts().idxmax()
    dummies = pd.get_dummies(df[args.group_col], prefix="cms", dtype=float)
    ref_col = f"cms_{reference}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    df = pd.concat([df, dummies], axis=1)
    covariate_cols = list(dummies.columns)

    if args.adjust_stage:
        if "stage" not in df.columns:
            print("AVISO: se pidio --adjust-stage pero no hay columna 'stage' en los datos "
                  "de entrada -- se omite. Reconstruye las cohortes con --stage-col "
                  "(ver build_external_cohort_generic.py).")
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from clinical_covariates import prepare_covariates
            df = prepare_covariates(df, stage_col="stage", cohort_col="cohort")
            df = df.dropna(subset=["stage_harmonized"])
            covariate_cols = covariate_cols + ["stage_harmonized"]
            print(f"Estadio armonizado incluido -- n tras excluir sin estadio: {len(df)}")

    print(f"Referencia: {reference} | covariables: {covariate_cols} | n={len(df)}\n")
    run_full_diagnostics(df, args.duration_col, args.event_col, covariate_cols,
                          cohort_col="cohort", stratify=not args.no_stratify,
                          output_dir=args.output,
                          time_varying_cms4=not args.no_time_varying_cms4,
                          time_cutoff=args.time_cutoff)


if __name__ == "__main__":
    main()
