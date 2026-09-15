"""
cox_treatment_interaction.py

¿La quimioterapia adyuvante se asocia con distinto beneficio segun el
subtipo CMS? -- la pregunta PREDICTIVA que el resto del pipeline no
responde.

POR QUE ESTE MODULO EXISTE
--------------------------
Un HR de CMS4 = 1.8 dice que CMS4 recae mas (efecto PRONOSTICO). No dice
que la quimio le sirva mas o menos (efecto PREDICTIVO). Para lo segundo
hace falta la INTERACCION CMS x quimio en un Cox que incluya ambos
efectos principales. GSE39582 anota `chemotherapy.adjuvant` (Y/N) y
GSE14333 `AdjCTX`; hasta ahora se descartaban.

MODELOS (misma muestra, estadio II-III, estratificado por cohorte)
    M0: estadio(cat) + CMS + quimio                  (efectos principales)
    M1: estadio(cat) + CMS + quimio + CMS x quimio    (interaccion)
    LRT M1 vs M0, df = (#niveles CMS - 1)
    + HR de quimio DENTRO de cada CMS (modelo M1 reparametrizado por
      subgrupo), con n/eventos por celda.

ADVERTENCIA METODOLOGICA (leer antes de interpretar)
----------------------------------------------------
Es un analisis OBSERVACIONAL. La quimio adyuvante se indica por estadio
(III casi siempre, II de alto riesgo, I nunca), edad y comorbilidad:
hay confusion por indicacion que el ajuste por estadio solo corrige en
parte. Un HR de quimio > 1 dentro de un CMS NO significa que la quimio
dane; significa que a ese grupo se le dio quimio por tener peor
pronostico. Por eso:

  * el resultado util es la INTERACCION (¿el efecto de quimio DIFIERE
    entre CMS?), no el HR marginal de quimio;
  * incluso la interaccion solo genera hipotesis. La evidencia
    predictiva real viene de ensayos aleatorizados con estratificacion
    por CMS (p. ej. Song et al. 2016 JAMA Oncol, NSABP C-07;
    PETACC-3), no de cohortes retrospectivas;
  * la potencia para interacciones es baja (regla practica: se
    necesitan ~4x los eventos del efecto principal). Reportar el IC del
    termino de interaccion, no solo el p.

Y como en todo el proyecto: si `predicted_cms` viene de centroides
calibrados en la misma cohorte, el analisis es in-sample.

USO
    python3 src/cox_treatment_interaction.py \\
        --cohort GSE39582 results_gse39582/scored_cohort.tsv \\
        --cohort GSE14333 results_ext/gse14333/scored_external_cohort.tsv \\
        --chemo-col adjuvant_chemo --chemo-yes Y \\
        --output results_cox_chemo/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_covariates import expand_stage_categorical, prepare_covariates  # noqa: E402
from cox_clinical_adjustment import binary_indicator  # noqa: E402
from pooled_cox_validation import DEFAULT_CMS_REFERENCE, stratified_c_index  # noqa: E402

MIN_N_PER_CELL = 10
MIN_EVENTS_PER_CELL = 5


def prepare_interaction_frame(
    data: pd.DataFrame, chemo_col: str, chemo_yes: str,
    group_col: str = "predicted_cms", stage_col: str = "stage",
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
    stages_kept: tuple[int, ...] = (2, 3), verbose: bool = True,
) -> pd.DataFrame:
    """
    Deja el dataframe listo: estadio armonizado y restringido a II-III
    (estadio I casi nunca recibe quimio -> celdas vacias; IV excluido
    de RFS), indicador de quimio 0/1, sin faltantes en nada de lo que
    entra al modelo.
    """
    out = prepare_covariates(data, stage_col=stage_col, drop_stage_iv=True, verbose=verbose)
    out = out[out["stage_harmonized"].isin(stages_kept)]
    out["chemo"] = binary_indicator(out[chemo_col], chemo_yes)
    needed = [duration_col, event_col, group_col, "stage_harmonized", "chemo", "cohort"]
    before = len(out)
    out = out.dropna(subset=needed)
    out = out[out[group_col] != "none"]
    if verbose:
        print(f"Muestra para interaccion: {len(out)} (de {before} en estadio {stages_kept}); "
              f"quimio=1 en {int(out['chemo'].sum())}")
    return out.reset_index(drop=True)


def cell_counts(df: pd.DataFrame, group_col: str, event_col: str) -> pd.DataFrame:
    """n y eventos por celda CMS x quimio -- imprescindible antes de leer HR."""
    g = df.groupby([group_col, "chemo"], observed=True)[event_col]
    tab = pd.DataFrame({"n": g.size(), "eventos": g.sum().astype(int)}).reset_index()
    tab["chemo"] = tab["chemo"].map({0.0: "sin_quimio", 1.0: "quimio"})
    tab["celda_escasa"] = (tab["n"] < MIN_N_PER_CELL) | (tab["eventos"] < MIN_EVENTS_PER_CELL)
    return tab


def build_frames(
    df: pd.DataFrame, reference: str, group_col: str,
    duration_col: str, event_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Devuelve (frame_M0, frame_M1, nombres_dummies_cms)."""
    levels = sorted(df[group_col].dropna().unique())
    if reference not in levels:
        raise ValueError(f"La referencia '{reference}' no aparece en {group_col}: {levels}")
    work, stage_cols = expand_stage_categorical(df)
    base = pd.DataFrame({
        "duration": work[duration_col].astype(float),
        "event": work[event_col].astype(float),
        "cohort": work["cohort"].astype(str),
        "chemo": work["chemo"].astype(float),
    })
    for c in stage_cols:
        base[c] = work[c].astype(float)
    cms_dummies = pd.get_dummies(pd.Categorical(work[group_col], categories=levels),
                                 prefix="cms", dtype=float)
    cms_dummies = cms_dummies.drop(columns=[f"cms_{reference}"])
    cms_cols = list(cms_dummies.columns)
    base = pd.concat([base.reset_index(drop=True), cms_dummies.reset_index(drop=True)], axis=1)
    m1 = base.copy()
    for c in cms_cols:
        m1[f"{c}_x_chemo"] = m1[c] * m1["chemo"]
    return base.dropna(), m1.dropna(), cms_cols


def _fit(frame: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame]:
    strata = ["cohort"] if frame["cohort"].nunique() > 1 else None
    fit_df = frame if strata else frame.drop(columns=["cohort"])
    cph = CoxPHFitter().fit(fit_df, "duration", "event", strata=strata)
    return cph, frame


def interaction_test(
    df: pd.DataFrame, reference: str = DEFAULT_CMS_REFERENCE,
    group_col: str = "predicted_cms",
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
) -> dict:
    """LRT del bloque de interaccion CMS x quimio (M1 vs M0)."""
    f0, f1, cms_cols = build_frames(df, reference, group_col, duration_col, event_col)
    cph0, _ = _fit(f0)
    cph1, _ = _fit(f1)
    stat = max(0.0, 2.0 * (cph1.log_likelihood_ - cph0.log_likelihood_))
    dof = len(cms_cols)
    summ = cph1.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summ.columns = ["HR", "IC95_inf", "IC95_sup", "p"]
    return {
        "lr_chi2": float(stat), "df": int(dof), "p_interaction": float(chi2.sf(stat, dof)),
        "n": int(len(f1)), "eventos": int(f1["event"].sum()),
        "c_index_M0": float(cph0.concordance_index_), "c_index_M1": float(cph1.concordance_index_),
        "c_index_estratificado_M0": stratified_c_index(cph0, f0) if f0["cohort"].nunique() > 1 else np.nan,
        "c_index_estratificado_M1": stratified_c_index(cph1, f1) if f1["cohort"].nunique() > 1 else np.nan,
        "summary_M0": cph0.summary, "summary_M1": summ,
        "cph_M0": cph0, "cph_M1": cph1, "frame_M1": f1, "cms_cols": cms_cols,
    }


def chemo_hr_within_cms(
    df: pd.DataFrame, group_col: str = "predicted_cms",
    duration_col: str = "relapse_free_months", event_col: str = "relapse_event",
) -> pd.DataFrame:
    """
    HR de quimio (1 vs 0) DENTRO de cada CMS: Cox con estadio(cat) + quimio
    ajustado por separado en cada subgrupo, estratificado por cohorte.
    Equivale a reparametrizar M1 por subgrupo, pero deja el estadio
    libre dentro de cada CMS. Las celdas escasas se marcan; sus HR no se
    interpretan.
    """
    rows = []
    for level, sub in df.groupby(group_col, observed=True):
        n, ev = len(sub), int(sub[event_col].sum())
        n_chemo = int(sub["chemo"].sum())
        row = {"cms": level, "n": n, "eventos": ev, "n_quimio": n_chemo, "n_sin_quimio": n - n_chemo}
        ev_chemo = int(sub.loc[sub["chemo"] == 1.0, event_col].sum())
        escaso = (min(n_chemo, n - n_chemo) < MIN_N_PER_CELL
                  or min(ev_chemo, ev - ev_chemo) < MIN_EVENTS_PER_CELL)
        row["celda_escasa"] = bool(escaso)
        try:
            work, stage_cols = expand_stage_categorical(sub)
            frame = pd.DataFrame({
                "duration": work[duration_col].astype(float), "event": work[event_col].astype(float),
                "cohort": work["cohort"].astype(str), "chemo": work["chemo"].astype(float),
            })
            for c in stage_cols:
                frame[c] = work[c].astype(float)
            frame = frame.dropna()
            # quitar columnas constantes (p. ej. un solo estadio en el subgrupo)
            const = [c for c in stage_cols if frame[c].nunique() < 2]
            frame = frame.drop(columns=const)
            cph, _ = _fit(frame)
            row.update({
                "HR_quimio": float(cph.summary.loc["chemo", "exp(coef)"]),
                "IC95_inf": float(cph.summary.loc["chemo", "exp(coef) lower 95%"]),
                "IC95_sup": float(cph.summary.loc["chemo", "exp(coef) upper 95%"]),
                "p": float(cph.summary.loc["chemo", "p"]),
            })
        except Exception as exc:  # singularidad, celdas vacias...
            row["error"] = str(exc)[:120]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", action="append", nargs=2, metavar=("NOMBRE", "SCORED_TSV"), required=True)
    ap.add_argument("--chemo-col", default="adjuvant_chemo")
    ap.add_argument("--chemo-yes", default="Y", help="Valor que significa 'recibio quimio' (Y, yes, 1...)")
    ap.add_argument("--group-col", default="predicted_cms",
                    choices=["predicted_cms", "modern_hopfield_cms", "cms_label"])
    ap.add_argument("--reference", default=DEFAULT_CMS_REFERENCE)
    ap.add_argument("--stage-col", default="stage")
    ap.add_argument("--stages", default="2,3", help="Estadios a incluir (default II-III)")
    ap.add_argument("--duration-col", default="relapse_free_months")
    ap.add_argument("--event-col", default="relapse_event")
    ap.add_argument("--output", default="results_cox_chemo")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pieces = []
    for name, path in args.cohort:
        d = pd.read_csv(path, sep="\t")
        if args.chemo_col not in d.columns:
            print(f"AVISO: {name} no trae '{args.chemo_col}'; se omite de este analisis.")
            continue
        d["cohort"] = name
        pieces.append(d)
    if not pieces:
        sys.exit("Ninguna cohorte trae la columna de quimioterapia. Reconstruye con "
                 "build_gse39582_dataset.py actual o --chemo-col en build_external_cohort_generic.py.")
    data = pd.concat(pieces, ignore_index=True)
    stages = tuple(int(x) for x in args.stages.split(","))

    df = prepare_interaction_frame(data, args.chemo_col, args.chemo_yes, args.group_col,
                                   args.stage_col, args.duration_col, args.event_col, stages)

    print("\n=== n y eventos por celda CMS x quimio ===")
    cells = cell_counts(df, args.group_col, args.event_col)
    print(cells.to_string(index=False))
    cells.to_csv(out / "chemo_cells.tsv", sep="\t", index=False)
    if cells["celda_escasa"].any():
        print("AVISO: hay celdas escasas; los HR que dependan de ellas no se interpretan.")

    res = interaction_test(df, args.reference, args.group_col, args.duration_col, args.event_col)
    print(f"\n=== M1: efectos principales + CMS x quimio (ref. CMS = {args.reference}, "
          f"estadio II ref., n={res['n']}, eventos={res['eventos']}) ===")
    print(res["summary_M1"].to_string(float_format=lambda x: f"{x:.3g}"))
    print(f"\nLRT interaccion CMS x quimio (M1 vs M0): chi2={res['lr_chi2']:.3f}, "
          f"df={res['df']}, p={res['p_interaction']:.4g}")
    if not np.isnan(res["c_index_estratificado_M1"]):
        print(f"C-index estratificado: M0={res['c_index_estratificado_M0']:.3f}, "
              f"M1={res['c_index_estratificado_M1']:.3f}")
    res["summary_M0"].to_csv(out / "cox_M0_main_effects.tsv", sep="\t")
    res["summary_M1"].to_csv(out / "cox_M1_interaction.tsv", sep="\t")
    pd.DataFrame([{k: v for k, v in res.items()
                   if not k.startswith(("summary", "cph", "frame", "cms_cols"))}]
                 ).to_csv(out / "interaction_test.tsv", sep="\t", index=False)

    print("\n=== HR de quimio DENTRO de cada CMS (estadio categorico, estratificado) ===")
    within = chemo_hr_within_cms(df, args.group_col, args.duration_col, args.event_col)
    print(within.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    within.to_csv(out / "chemo_hr_within_cms.tsv", sep="\t", index=False)

    print("\nCOMO LEER ESTO: analisis observacional con confusion por indicacion. "
          "El dato relevante es si el efecto de la quimio DIFIERE entre CMS (LRT de interaccion "
          "y su IC), no el HR marginal de quimio. Un HR>1 de quimio dentro de un CMS refleja "
          "que se trato a los de peor pronostico, no que la quimio dane. Genera hipotesis; "
          "la evidencia predictiva viene de ensayos aleatorizados estratificados por CMS.")
    if args.group_col != "cms_label":
        print("RECORDATORIO: si predicted_cms se calibro en alguna de estas cohortes, es in-sample.")
    print(f"\nSalidas en {out}/")


if __name__ == "__main__":
    main()
