"""
cox_clinical_adjustment.py

¿Conserva el subtipo CMS su valor pronostico tras ajustar por las
covariables clinicas que YA se miden en la rutina?

MOTIVACION
----------
El eje CMS1 se solapa en gran medida con dMMR/MSI-H, que ya tiene una
prueba clinica estandar (IHC de proteinas MMR o PCR de microsatelites).
Un revisor preguntara, con razon: ¿que anade el panel sobre lo que ya se
hace? La respuesta defendible esta en los otros ejes -- sobre todo CMS4
(mesenquimal, mal pronostico, sin prueba de rutina) y en menor medida
CMS3. Este script responde esa pregunta con modelos de Cox anidados:

    A. CMS solo
    B. CMS + estadio
    C. estadio + covariables clinicas (SIN CMS)      <- linea base clinica
    D. CMS + estadio + covariables clinicas
    E. (opcional) CMS + estadio dentro de un SUBGRUPO, p. ej. solo pMMR

y reporta: HR/IC95%/p por termino, LRT anidado D vs C (aporte conjunto
de CMS sobre la clinica), cambio de C-index, y test de Schoenfeld del
modelo D. Si el HR de CMS4 se conserva en D y en E, el panel aporta algo
que la IHC de MMR no da. Si no, mas vale saberlo antes que despues.

DATOS
-----
Requiere un scored_*.tsv (de run_pipeline.py / external_validation.py)
que traiga las covariables como columnas. Hoy solo GSE39582 anota MMR
(`msi_status` = pMMR/dMMR, arrastrado por build_gse39582_dataset.py);
ninguna de las 5 cohortes externas lo trae, asi que este analisis es
IN-SAMPLE sobre la cohorte de entrenamiento: los HR de CMS estan
optimistamente sesgados por eso, y el ajuste por MMR no lo corrige.
Reportar como tal.

Reutiliza build_cox_frame / nested_model_increment (pooled_cox_validation.py),
harmonize_stage (clinical_covariates.py) y check_proportional_hazards
(cox_diagnostics.py) -- no duplica logica de Cox.

USO
    python3 src/cox_clinical_adjustment.py \\
        --cohort GSE39582 results_gse39582/scored_cohort.tsv \\
        --covariate msi_status=dMMR \\
        --subgroup msi_status=pMMR \\
        --output results_cox_clinical/

    # varias cohortes -> Cox estratificado por cohorte, como pooled-cox
    # varias covariables -> repetir --covariate (p. ej. braf_status=M)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_covariates import harmonize_stage
from cox_diagnostics import check_proportional_hazards
from pooled_cox_validation import build_cox_frame, nested_model_increment, stratified_c_index


def parse_level_spec(spec: str) -> tuple[str, str]:
    """'msi_status=dMMR' -> ('msi_status', 'dMMR'). Falla claro si no trae '='."""
    if "=" not in spec:
        raise ValueError(f"Se esperaba COLUMNA=NIVEL, se recibio '{spec}'")
    col, level = spec.split("=", 1)
    col, level = col.strip(), level.strip()
    if not col or not level:
        raise ValueError(f"Especificacion vacia en '{spec}'")
    return col, level


def binary_indicator(values: pd.Series, positive_level: str) -> pd.Series:
    """1 si el valor es `positive_level`, 0 si es otro valor no faltante, NaN si falta.
    Comparacion insensible a mayusculas/espacios (dMMR == dmmr == ' dMMR ')."""
    raw = values.astype("string").str.strip().str.lower()
    out = (raw == positive_level.strip().lower()).astype("float")
    out[raw.isna() | (raw == "nan") | (raw == "na") | (raw == "")] = np.nan
    return out


def indicator_name(col: str, level: str) -> str:
    return f"{col}_{level}"


def prepare_single_frame(
    df: pd.DataFrame, covariates: list[tuple[str, str]],
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
    group_col: str = "predicted_cms", stage_col: str = "stage",
    drop_stage_iv: bool = True, verbose: bool = True,
) -> pd.DataFrame:
    """Armoniza estadio, construye indicadores binarios y deja solo filas completas.

    Devuelve el dataframe listo para build_cox_frame (con 'cohort' y
    'stage_harmonized'). No decide nada estadistico; solo prepara.
    """
    out = df.copy()
    if "cohort" not in out.columns:
        out["cohort"] = "unica"
    if stage_col not in out.columns:
        raise ValueError(f"Falta la columna de estadio '{stage_col}' en el TSV de entrada")
    pieces = [harmonize_stage(sub[stage_col], str(c), verbose)
              for c, sub in out.groupby("cohort", sort=False)]
    out["stage_harmonized"] = pd.concat(pieces).reindex(out.index)
    if drop_stage_iv:
        n_iv = int((out["stage_harmonized"] == 4).sum())
        out = out[out["stage_harmonized"] != 4]
        if verbose and n_iv:
            print(f"Excluidos {n_iv} pacientes en estadio IV (RFS no comparable).")
    for col, level in covariates:
        if col not in out.columns:
            raise ValueError(f"Falta la covariable '{col}' en el TSV de entrada")
        out[indicator_name(col, level)] = binary_indicator(out[col], level)
    needed = [duration_col, event_col, group_col, "stage_harmonized"] + [
        indicator_name(c, l) for c, l in covariates]
    n_before = len(out)
    out = out.dropna(subset=needed)
    if verbose:
        print(f"n con datos completos para el Cox: {len(out)} (de {n_before}); "
              f"eventos = {int(out[event_col].sum())}")
        for col, level in covariates:
            print(f"  {indicator_name(col, level)} = 1 en {int(out[indicator_name(col, level)].sum())} pacientes")
    return out


def crosstab_group_by_covariate(df: pd.DataFrame, group_col: str, covariate_col: str) -> pd.DataFrame:
    """Tabla CMS x covariable cruda (con margenes) -- muestra el solapamiento
    que motiva todo el analisis (p. ej. cuantos CMS1 son dMMR)."""
    cov = df[covariate_col].astype("string").fillna("sin_dato")
    return pd.crosstab(df[group_col], cov, margins=True)


MIN_N_PER_LEVEL = 10
MIN_EVENTS_PER_LEVEL = 5


def level_counts(data: pd.DataFrame, group_col: str, event_col: str) -> pd.DataFrame:
    """n y eventos por nivel de CMS en la muestra que entra al modelo."""
    g = data.groupby(group_col, observed=True)[event_col]
    return pd.DataFrame({"n": g.size(), "eventos": g.sum().astype(int)})


def sparse_level_warnings(data: pd.DataFrame, group_col: str, event_col: str,
                          label: str = "") -> list[str]:
    """
    Avisa cuando un nivel de CMS tiene pocas observaciones o pocos eventos:
    el HR de ese nivel es inestable (IC95% de varios ordenes de magnitud,
    p. ej. CMS1 dentro de pMMR con ~15 pacientes) y no debe interpretarse.
    Devuelve la lista de mensajes (vacia si todo esta bien).
    """
    msgs = []
    for level, row in level_counts(data, group_col, event_col).iterrows():
        if row["n"] < MIN_N_PER_LEVEL or row["eventos"] < MIN_EVENTS_PER_LEVEL:
            msgs.append(
                f"[{label}] nivel {level}: n={int(row['n'])}, eventos={int(row['eventos'])} "
                f"(<{MIN_N_PER_LEVEL} obs o <{MIN_EVENTS_PER_LEVEL} eventos) -- HR INESTABLE, "
                "no interpretar; considerar agrupar niveles o un Cox penalizado.")
    return msgs


def _summary_table(cph: CoxPHFitter, label: str, n: int, events: int,
                   cox_df: pd.DataFrame | None = None) -> pd.DataFrame:
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["HR", "IC95_inf", "IC95_sup", "p"]
    s.insert(0, "modelo", label)
    s["n"] = n
    s["eventos"] = events
    s["c_index"] = cph.concordance_index_
    if cox_df is not None and "cohort" in cox_df.columns and cox_df["cohort"].nunique() > 1:
        s["c_index_estratificado"] = stratified_c_index(cph, cox_df)
    # marca de inestabilidad: IC95 que abarca mas de 2 ordenes de magnitud
    s["hr_inestable"] = (s["IC95_sup"] / s["IC95_inf"]) > 100
    return s


def fit_cox(
    data: pd.DataFrame, covariate_cols: list[str], reference: str,
    include_cms: bool, group_col: str = "predicted_cms",
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
    cms_levels: list[str] | None = None,
) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Ajusta un Cox con las covariables dadas; estratifica por cohorte solo si hay >1."""
    cox_df = build_cox_frame(
        data, duration_col, event_col, reference, covariate_cols,
        include_cms=include_cms, cms_levels=cms_levels, group_col=group_col)
    strata = ["cohort"] if cox_df["cohort"].nunique() > 1 else None
    fit_df = cox_df if strata else cox_df.drop(columns=["cohort"])
    cph = CoxPHFitter().fit(fit_df, "duration", "event", strata=strata)
    return cph, cox_df


def fit_nested_clinical_models(
    data: pd.DataFrame, indicator_cols: list[str], reference: str,
    group_col: str = "predicted_cms",
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
) -> dict:
    """Ajusta los modelos A-D sobre EXACTAMENTE la misma muestra y calcula el
    LRT anidado D vs C (aporte conjunto de CMS sobre estadio + clinica)."""
    cms_levels = sorted(data[group_col].dropna().unique())
    n, events = len(data), int(data[event_col].sum())
    spec = {
        "A_cms": ([], True),
        "B_cms_estadio": (["stage_harmonized"], True),
        "C_estadio_clinica": (["stage_harmonized"] + indicator_cols, False),
        "D_cms_estadio_clinica": (["stage_harmonized"] + indicator_cols, True),
    }
    print(f"\nn y eventos por nivel de {group_col} (muestra de los modelos A-D):")
    print(level_counts(data, group_col, event_col).to_string())
    for msg in sparse_level_warnings(data, group_col, event_col, "A-D"):
        print("AVISO: " + msg)
    models, tables = {}, []
    for label, (covs, with_cms) in spec.items():
        cph, cox_df = fit_cox(data, covs, reference, with_cms, group_col,
                              duration_col, event_col, cms_levels)
        models[label] = (cph, cox_df)
        tables.append(_summary_table(cph, label, n, events, cox_df))
    increment = nested_model_increment(models["C_estadio_clinica"][0],
                                       models["D_cms_estadio_clinica"][0],
                                       models["C_estadio_clinica"][1],
                                       models["D_cms_estadio_clinica"][1])
    # nested_model_increment nombra las columnas pensando en estadio;
    # aqui la linea base es estadio + clinica -- renombrar para no confundir.
    increment = {
        "lr_chi2": increment["lr_chi2"], "df": increment["df"],
        "p_incremental_cms_sobre_clinica": increment["p_incremental"],
        "c_index_estadio_clinica": increment["c_index_stage_only"],
        "c_index_mas_cms": increment["c_index_stage_plus_cms"],
        "delta_c_index": increment["delta_c_index"],
        "c_index_estratificado_estadio_clinica": increment.get("c_index_stratified_stage_only"),
        "c_index_estratificado_mas_cms": increment.get("c_index_stratified_stage_plus_cms"),
        "delta_c_index_estratificado": increment.get("delta_c_index_stratified"),
        "n": n, "eventos": events,
    }
    return {"models": models, "summary": pd.concat(tables), "increment": increment}


def fit_subgroup_model(
    data: pd.DataFrame, subgroup_col: str, subgroup_level: str, reference: str,
    group_col: str = "predicted_cms",
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
) -> tuple[CoxPHFitter, pd.DataFrame] | None:
    """Modelo E: CMS + estadio SOLO dentro del subgrupo (p. ej. pMMR) -- la
    pregunta 'entre los pacientes que la prueba de rutina llama iguales,
    ¿el subtipo aun separa?'. Devuelve None si el subgrupo es muy chico."""
    mask = binary_indicator(data[subgroup_col], subgroup_level) == 1.0
    sub = data[mask]
    if len(sub) < 30 or sub[event_col].sum() < 10:
        print(f"AVISO: subgrupo {subgroup_col}={subgroup_level} muy chico "
              f"(n={len(sub)}, eventos={int(sub[event_col].sum())}); se omite el modelo E.")
        return None
    label = f"E_cms_estadio_solo_{subgroup_col}={subgroup_level}"
    counts = level_counts(sub, group_col, event_col)
    print(f"\n[{label}] n y eventos por nivel de {group_col}:")
    print(counts.to_string())
    for msg in sparse_level_warnings(sub, group_col, event_col, label):
        print("AVISO: " + msg)
    cph, cox_df = fit_cox(sub, ["stage_harmonized"], reference, True, group_col,
                          duration_col, event_col)
    return cph, _summary_table(cph, label, len(sub), int(sub[event_col].sum()), cox_df)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort", action="append", nargs=2, metavar=("NOMBRE", "SCORED_TSV"),
                        required=True, help="Repetir para varias cohortes (Cox estratificado)")
    parser.add_argument("--covariate", action="append", required=True, metavar="COLUMNA=NIVEL",
                        help="Covariable binaria: 1 si la columna vale NIVEL. Ej. msi_status=dMMR")
    parser.add_argument("--subgroup", metavar="COLUMNA=NIVEL", default=None,
                        help="Repetir CMS+estadio solo dentro de este subgrupo. Ej. msi_status=pMMR")
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--group-col", default="predicted_cms",
                        help="predicted_cms, modern_hopfield_cms o cms_label (etiqueta oficial)")
    parser.add_argument("--stage-col", default="stage")
    parser.add_argument("--reference", default="CMS2_canonical_WNT")
    parser.add_argument("--keep-stage-iv", action="store_true")
    parser.add_argument("--output", default="results_cox_clinical")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    covariates = [parse_level_spec(s) for s in args.covariate]

    frames = []
    for name, path in args.cohort:
        df = pd.read_csv(path, sep="\t")
        df["cohort"] = name
        frames.append(df)
        print(f"{name}: {len(df)} muestras desde {path}")
    raw = pd.concat(frames, ignore_index=True)
    if args.group_col == "cms_label":
        raw = raw[raw[args.group_col] != "none"]

    # 1. Solapamiento CMS x covariable (con TODOS los pacientes, antes de filtrar)
    for col, _ in covariates:
        ct = crosstab_group_by_covariate(raw, args.group_col, col)
        print(f"\n=== {args.group_col} x {col} ===\n{ct}")
        ct.to_csv(out_dir / f"crosstab_{args.group_col}_x_{col}.tsv", sep="\t")

    # 2. Muestra analitica
    data = prepare_single_frame(
        raw, covariates, args.duration_col, args.event_col, args.group_col,
        args.stage_col, drop_stage_iv=not args.keep_stage_iv)
    if args.reference not in set(data[args.group_col]):
        raise ValueError(f"La referencia '{args.reference}' no aparece en {args.group_col}")
    indicator_cols = [indicator_name(c, l) for c, l in covariates]

    # 3. Modelos anidados A-D
    res = fit_nested_clinical_models(
        data, indicator_cols, args.reference, args.group_col, args.duration_col, args.event_col)
    pd.set_option("display.width", 160)
    for label, (cph, cox_df) in res["models"].items():
        print(f"\n--- {label}  (n={len(cox_df)}, C={cph.concordance_index_:.3f})")
        print(res["summary"][res["summary"]["modelo"] == label]
              .drop(columns=["modelo", "n", "eventos", "c_index"]).round(4).to_string())
    inc = res["increment"]
    print(f"\nLRT (estadio + clinica) vs (+ CMS): chi2={inc['lr_chi2']:.3f}, df={inc['df']}, "
          f"p={inc['p_incremental_cms_sobre_clinica']:.4g}")
    print(f"C-index: {inc['c_index_estadio_clinica']:.3f} -> {inc['c_index_mas_cms']:.3f} "
          f"(delta {inc['delta_c_index']:+.3f})")

    summary = res["summary"]

    # 4. Modelo E: subgrupo
    if args.subgroup:
        s_col, s_level = parse_level_spec(args.subgroup)
        sub = fit_subgroup_model(data, s_col, s_level, args.reference, args.group_col,
                                 args.duration_col, args.event_col)
        if sub is not None:
            cph_e, table_e = sub
            print(f"\n--- {table_e['modelo'].iloc[0]}  (n={table_e['n'].iloc[0]}, "
                  f"eventos={table_e['eventos'].iloc[0]}, C={cph_e.concordance_index_:.3f})")
            print(table_e.drop(columns=["modelo", "n", "eventos", "c_index"]).round(4).to_string())
            summary = pd.concat([summary, table_e])

    # 5. Riesgos proporcionales del modelo D
    cph_d, df_d = res["models"]["D_cms_estadio_clinica"]
    ph = check_proportional_hazards(cph_d, df_d.drop(columns=["cohort"])
                                    if df_d["cohort"].nunique() == 1 else df_d)
    print("\n=== Schoenfeld, modelo D ===")
    print(ph.round(4).to_string())
    if ph["viola_supuesto"].any():
        print("AVISO: alguna covariable viola riesgos proporcionales; el HR es un promedio temporal.")

    summary.to_csv(out_dir / "cox_clinical_summary.tsv", sep="\t")
    pd.DataFrame([inc]).to_csv(out_dir / "cox_clinical_incremental.tsv", sep="\t", index=False)
    ph.to_csv(out_dir / "cox_clinical_schoenfeld_modelD.tsv", sep="\t")
    print(f"\nSalidas en {out_dir}/")
    print("RECORDATORIO: si la unica cohorte con la covariable es la de entrenamiento, "
          "esto es in-sample -- los HR de CMS estan optimistamente sesgados.")


if __name__ == "__main__":
    main()
