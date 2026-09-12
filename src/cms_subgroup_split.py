"""
cms_subgroup_split.py

Divide un subtipo CMS por una covariable clinica binaria y pregunta si los
dos subgrupos tienen pronostico distinto -- p. ej. CMS1 dividido por MMR:
¿los CMS1 pMMR ("inmunes" sin MSI) arrastran el mal pronostico que CMS1
muestra en otras cohortes, mientras los CMS1 dMMR van bien?

Motivacion concreta (2026-09-11): el HR de CMS1 es 0.7-0.8 en GSE39582 pero
~2.1 en el Cox agrupado de las 5 cohortes externas. Una hipotesis era que
la mezcla dMMR/pMMR dentro de CMS1 explicara la discrepancia. Resultado en
GSE39582: direccion coherente (CMS1-pMMR peor que CMS1-dMMR) pero sin poder
(17 eventos) y AMBOS subgrupos con HR<1 vs CMS2 -- la division no explica
la heterogeneidad. Ver PROJECT_STATUS.md.

Que hace:
  1. Crea la variable de grupo con el subtipo elegido partido por la
     covariable: "CMS1_MSI_immune|dMMR", "CMS1_MSI_immune|pMMR", y el resto
     de subtipos intactos.
  2. Cox + estadio armonizado (referencia configurable) con esos grupos.
  3. Dentro del subtipo partido: log-rank y Cox+estadio nivel A vs nivel B,
     mas RFS a 3 y 5 anios por Kaplan-Meier.
  4. Tablas cruzadas de los subgrupos contra otras columnas descriptivas
     (p. ej. braf_status, kras_status, estadio) para caracterizarlos.
  5. Opcional: repite el Cox anadiendo una covariable binaria extra
     (--extra-covariate braf_status=M) para ver si explica el efecto.

Reutiliza harmonize_stage (clinical_covariates.py) y binary_indicator
(cox_clinical_adjustment.py). Misma muestra analitica que cox-clinical:
RFS, estadio I-III (IV excluido salvo --keep-stage-iv), covariable conocida.

USO
    python3 src/cms_subgroup_split.py \\
        --input results_gse39582/scored_cohort.tsv \\
        --split-cms CMS1_MSI_immune --by msi_status \\
        --describe braf_status kras_status \\
        --extra-covariate braf_status=M \\
        --output results_cms1_mmr/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_covariates import harmonize_stage
from cox_clinical_adjustment import (
    binary_indicator,
    check_indicator_varies,
    drop_unclassified,
    parse_level_spec,
)

MIN_SUBGROUP_N = 10


def split_group_labels(df: pd.DataFrame, group_col: str, split_cms: str, by_col: str) -> pd.Series:
    """Etiqueta de grupo: `split_cms` se parte en '{split_cms}|{valor de by_col}';
    el resto de subtipos queda igual. Filas del subtipo partido sin valor en
    by_col quedan como NaN (se excluyen despues)."""
    g = df[group_col].astype("object").copy()
    is_split = g == split_cms
    by = df[by_col].astype("string").str.strip()
    g[is_split] = np.where(by[is_split].notna(), split_cms + "|" + by[is_split].astype(str), np.nan)
    return g


def prepare_frame(
    df: pd.DataFrame, group_col: str, split_cms: str, by_col: str,
    duration_col: str, event_col: str, stage_col: str = "stage",
    drop_stage_iv: bool = True, require_by_all: bool = True, verbose: bool = True,
) -> pd.DataFrame:
    """Muestra analitica: estadio armonizado (IV fuera por default), covariable
    conocida en todos (por default), grupo partido, RFS completo."""
    out = drop_unclassified(df, group_col).copy()
    if split_cms not in set(out[group_col]):
        raise ValueError(
            f"'{split_cms}' no aparece en '{group_col}' (valores: {sorted(set(out[group_col]))}). "
            "Sin esto el analisis degenera en un Cox por CMS sin particion.")
    if stage_col not in out.columns:
        raise ValueError(f"Falta la columna de estadio '{stage_col}'")
    out["stage_harmonized"] = harmonize_stage(out[stage_col], verbose=verbose)
    if drop_stage_iv:
        out = out[out["stage_harmonized"] != 4]
    if require_by_all:
        # Misma muestra analitica que cox-clinical: la covariable debe ser
        # conocida en TODOS los pacientes, no solo en el subtipo partido.
        # Asi los HR son comparables entre ambos analisis.
        out = out[out[by_col].notna()]
    out["group"] = split_group_labels(out, group_col, split_cms, by_col)
    out = out.dropna(subset=[duration_col, event_col, "stage_harmonized", "group"]).copy()
    out["duration"] = out[duration_col].astype(float)
    out["event"] = out[event_col].astype(int)
    if verbose:
        print(f"n={len(out)}, eventos={int(out['event'].sum())}")
        print("grupos:", out["group"].value_counts().to_dict())
    return out


def fit_group_cox(data: pd.DataFrame, reference: str, extra_indicator: str | None = None) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Cox + estadio con los grupos partidos (dummies, referencia excluida)."""
    if reference not in set(data["group"]):
        raise ValueError(f"La referencia '{reference}' no aparece entre los grupos: {sorted(set(data['group']))}")
    X = pd.DataFrame({"duration": data["duration"], "event": data["event"],
                      "stage_harmonized": data["stage_harmonized"]})
    levels = [lv for lv in sorted(data["group"].unique()) if lv != reference]
    for lv in levels:
        X[f"grp_{lv}"] = (data["group"] == lv).astype(float)
    if extra_indicator is not None:
        X[extra_indicator] = data[extra_indicator]
    X = X.dropna()
    cph = CoxPHFitter().fit(X, "duration", "event")
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["HR", "IC95_inf", "IC95_sup", "p"]
    # Conteos sobre la MISMA muestra que ajusto el modelo (X.dropna() puede
    # haber quitado pacientes sin la covariable extra).
    fitted = data.loc[X.index]
    n_by = fitted["group"].value_counts()
    ev_by = fitted.groupby("group")["event"].sum()
    s["n_grupo"] = [int(n_by.get(c[4:], len(X))) if c.startswith("grp_") else len(X) for c in s.index]
    s["eventos_grupo"] = [int(ev_by.get(c[4:], X["event"].sum())) if c.startswith("grp_") else int(X["event"].sum()) for c in s.index]
    s["c_index"] = cph.concordance_index_
    return cph, s


def within_split_comparison(data: pd.DataFrame, split_cms: str, by_col: str) -> dict:
    """Dentro del subtipo partido: nivel A vs nivel B (log-rank, Cox+estadio, KM a 36/60 m)."""
    sub = data[data["group"].str.startswith(split_cms + "|")].copy()
    levels = sorted(sub["group"].unique())
    out = {"n": len(sub), "events": int(sub["event"].sum()), "levels": levels}
    if len(levels) != 2 or (sub["group"] == levels[0]).sum() < MIN_SUBGROUP_N \
            or (sub["group"] == levels[1]).sum() < MIN_SUBGROUP_N:
        out["note"] = f"se necesitan exactamente 2 niveles con >= {MIN_SUBGROUP_N} cada uno; hay {levels}"
        return out
    a, b = levels
    ma, mb = sub["group"] == a, sub["group"] == b
    lr = logrank_test(sub.loc[ma, "duration"], sub.loc[mb, "duration"], sub.loc[ma, "event"], sub.loc[mb, "event"])
    X = pd.DataFrame({"duration": sub["duration"], "event": sub["event"],
                      "stage_harmonized": sub["stage_harmonized"], "is_" + b.split("|")[1]: mb.astype(float)})
    cph = CoxPHFitter().fit(X, "duration", "event")
    row = cph.summary.iloc[-1]
    out.update({
        "logrank_p": float(lr.p_value), "cox_HR_B_vs_A": float(row["exp(coef)"]),
        "cox_IC95_inf": float(row["exp(coef) lower 95%"]), "cox_IC95_sup": float(row["exp(coef) upper 95%"]),
        "cox_p": float(row["p"]), "A": a, "B": b,
    })
    for lv, m in [(a, ma), (b, mb)]:
        km = KaplanMeierFitter().fit(sub.loc[m, "duration"], sub.loc[m, "event"])
        out[f"rfs36_{lv}"] = float(km.predict(36.0))
        out[f"rfs60_{lv}"] = float(km.predict(60.0))
        out[f"n_{lv}"] = int(m.sum())
        out[f"events_{lv}"] = int(sub.loc[m, "event"].sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="scored_*.tsv con la covariable, estadio y RFS")
    parser.add_argument("--split-cms", default="CMS1_MSI_immune", help="Subtipo a partir")
    parser.add_argument("--by", default="msi_status", help="Columna binaria por la que se parte")
    parser.add_argument("--group-col", default="predicted_cms",
                        help="predicted_cms, modern_hopfield_cms o cms_label")
    parser.add_argument("--reference", default="CMS2_canonical_WNT")
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--stage-col", default="stage")
    parser.add_argument("--keep-stage-iv", action="store_true")
    parser.add_argument("--keep-missing-by", action="store_true",
                        help="Conservar pacientes de OTROS subtipos sin valor en --by "
                             "(por default se excluyen, para usar la misma muestra que cox-clinical)")
    parser.add_argument("--describe", nargs="*", default=[],
                        help="Columnas a cruzar contra los subgrupos (ej. braf_status kras_status)")
    parser.add_argument("--extra-covariate", metavar="COLUMNA=NIVEL", default=None,
                        help="Repetir el Cox anadiendo este indicador (ej. braf_status=M)")
    parser.add_argument("--output", default="results_cms_split")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, sep="\t")
    for col in [args.group_col, args.by, args.stage_col, args.duration_col, args.event_col]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en {args.input}")

    data = prepare_frame(df, args.group_col, args.split_cms, args.by,
                         args.duration_col, args.event_col, args.stage_col,
                         drop_stage_iv=not args.keep_stage_iv,
                         require_by_all=not args.keep_missing_by)

    pd.set_option("display.width", 160)
    cph, summary = fit_group_cox(data, args.reference)
    print(f"\n--- Cox {args.split_cms} partido por {args.by} + estadio (ref {args.reference}), "
          f"C={cph.concordance_index_:.3f}")
    print(summary.round(4).to_string())
    summary.to_csv(out_dir / "cox_split_groups.tsv", sep="\t")

    mlr = multivariate_logrank_test(data["duration"], data["group"], data["event"])
    print(f"log-rank global ({data['group'].nunique()} grupos): p={mlr.p_value:.4g}")

    within = within_split_comparison(data, args.split_cms, args.by)
    print(f"\n--- Dentro de {args.split_cms}: {within.get('levels')}  (n={within['n']}, eventos={within['events']})")
    if "note" in within:
        print("  ", within["note"])
    else:
        print(f"  log-rank p={within['logrank_p']:.4g} | Cox+estadio {within['B']} vs {within['A']}: "
              f"HR={within['cox_HR_B_vs_A']:.2f} ({within['cox_IC95_inf']:.2f}-{within['cox_IC95_sup']:.2f}), p={within['cox_p']:.4g}")
        for lv in within["levels"]:
            print(f"  {lv}: n={within[f'n_{lv}']}, eventos={within[f'events_{lv}']}, "
                  f"RFS 3a={within[f'rfs36_{lv}']:.2f}, RFS 5a={within[f'rfs60_{lv}']:.2f}")
    pd.DataFrame([within]).to_csv(out_dir / "within_split_comparison.tsv", sep="\t", index=False)

    sub = data[data["group"].str.startswith(args.split_cms + "|")]
    for col in list(args.describe) + ["stage_harmonized"]:
        if col not in sub.columns:
            print(f"AVISO: columna descriptiva '{col}' no existe; se omite")
            continue
        ct = pd.crosstab(sub["group"], sub[col].astype("string").fillna("sin_dato"))
        print(f"\n{col} por subgrupo:\n{ct}")
        ct.to_csv(out_dir / f"crosstab_{col}_by_subgroup.tsv", sep="\t")

    if args.extra_covariate:
        col, level = parse_level_spec(args.extra_covariate)
        if col not in data.columns:
            raise ValueError(f"Falta la covariable extra '{col}'")
        name = f"{col}_{level}"
        data[name] = binary_indicator(data[col], level)
        check_indicator_varies(data[name], name)
        cph2, summary2 = fit_group_cox(data, args.reference, extra_indicator=name)
        print(f"\n--- + {name}  (n={int(summary2['n_grupo'].iloc[0])}, C={cph2.concordance_index_:.3f})")
        print(summary2.round(4).to_string())
        summary2.to_csv(out_dir / f"cox_split_groups_plus_{name}.tsv", sep="\t")

    print(f"\nSalidas en {out_dir}/")


if __name__ == "__main__":
    main()
