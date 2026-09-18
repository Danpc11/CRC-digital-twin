"""Figura 4 — Valor añadido sobre la clínica de rutina (180 x 80 mm).

A. Kaplan-Meier de RFS por subtipo predicho (panel v0.3, TGFB1->EFEMP2) en los pacientes pMMR de GSE39582 con
   estadio I–III y RFS disponible (n=380, 122 eventos): cuatro curvas con IC95% sombreado, tabla de
   números en riesgo a 0, 24, 48, 72 y 96 meses, log-rank CMS4 vs resto anotado. Eje truncado a
   100 meses (el seguimiento máximo es 192 meses, con muy pocos pacientes en riesgo).
   Fuente: results_gse39582/scored_cohort.tsv (scored_cohort con anotación MMR);
   la muestra reproduce la del modelo E de results_cox_clinical/cox_clinical_summary.tsv.
B. Forest plot del modelo D (CMS + estadio + MMR; n=449, 132 eventos, 69 dMMR): estadio (por nivel),
   dMMR, CMS1, CMS3, CMS4, referencia CMS2. Fuente: results_cox_clinical/cox_clinical_summary.tsv
   y cox_clinical_incremental.tsv (estadio categorico III vs I+II).

Las cifras de resumen del modelo D (n, eventos, LRT, ΔC-index) NO se dibujan: van en el pie de figura
(figuras/Fig4_pie_de_figura.md, generado por este script).

Uso: python3 figures/fig4_clinica.py [--outdir DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CMS_COLORS, CMS_LONG, CMS_ORDER, GRAY, GRAY_DARK, INK, FONT_MIN, mm, panel_label,
                     add_axes_mm, save_figure, setup_style, base_parser, resolve_paths, forest_plot, fmt_p)

FIG_W, FIG_H = 180.0, 80.0
RISK_TIMES = [0, 24, 48, 72, 96]
XMAX = 100.0


def load_km_cohort(repo: Path):
    d = pd.read_csv(repo / "results_gse39582/scored_cohort.tsv", sep="\t")
    d["cms"] = d["predicted_cms"].map(CMS_LONG)
    k = d[d["stage"].isin([1.0, 2.0, 3.0]) & d["msi_status"].notna()].dropna(subset=["relapse_free_months", "relapse_event"])
    n_all, ev_all, n_dmmr = len(k), int(k["relapse_event"].sum()), int((k["msi_status"] == "dMMR").sum())
    p = k[k["msi_status"] == "pMMR"].copy()
    return p, (n_all, ev_all, n_dmmr)


def load_cox(repo: Path):
    s = pd.read_csv(repo / "results_cox_clinical/cox_clinical_summary.tsv", sep="\t")
    s = s[s["modelo"] == "D_cms_estadio_clinica"].set_index("covariate")
    e = pd.read_csv(repo / "results_cox_clinical/cox_clinical_summary.tsv", sep="\t")
    e = e[e["modelo"].str.startswith("E_")].set_index("covariate")
    inc = pd.read_csv(repo / "results_cox_clinical/cox_clinical_incremental.tsv", sep="\t").iloc[0]
    def row(cov, label, color=GRAY_DARK):
        r = s.loc[cov]
        return {"label": label, "hr": r["HR"], "lo": r["IC95_inf"], "hi": r["IC95_sup"], "p": r["p"], "color": color}
    rows = [row("stage_III", "Estadio III"),
            row("msi_status_dMMR", "dMMR"),
            row("cms_CMS1_MSI_immune", "CMS1", CMS_COLORS["CMS1"]),
            {"label": "CMS2", "ref": True, "color": CMS_COLORS["CMS2"]},
            row("cms_CMS3_metabolic", "CMS3", CMS_COLORS["CMS3"]),
            row("cms_CMS4_mesenchymal", "CMS4", CMS_COLORS["CMS4"])]
    meta = {"n": int(s["n"].iloc[0]), "eventos": int(s["eventos"].iloc[0]), "c_index": float(s["c_index"].iloc[0])}
    return rows, inc, meta, e


def draw_km(fig, p: pd.DataFrame):
    ax = add_axes_mm(fig, left=14, top=6, width=78, height=44, fig_w=FIG_W, fig_h=FIG_H)
    kmf = KaplanMeierFitter()
    at_risk = {}
    end_y = {}
    for c in CMS_ORDER:
        sub = p[p["cms"] == c]
        kmf.fit(sub["relapse_free_months"], sub["relapse_event"], label=c)
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12, color=CMS_COLORS[c], lw=1.1,
                                   show_censors=False, legend=False)
        at_risk[c] = [int((sub["relapse_free_months"] >= t).sum()) for t in RISK_TIMES]
        sf = kmf.survival_function_at_times([XMAX]).values[0]
        end_y[c] = sf
    # etiquetas directas al final de las curvas, separadas si se solapan
    ys = sorted(end_y.items(), key=lambda kv: kv[1])
    placed = []
    for c, y in ys:
        yy = y
        for py in placed:
            if abs(yy - py) < 0.045:
                yy = py + 0.045
        placed.append(yy)
        ax.text(XMAX + 1.5, yy, c, color=CMS_COLORS[c], fontsize=FONT_MIN, fontweight="bold", va="center", ha="left",
                clip_on=False)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xticks(RISK_TIMES)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("Meses desde la cirugía", labelpad=2)
    ax.set_ylabel("Supervivencia libre de recaída", labelpad=2)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5); ax.set_axisbelow(True)
    # log-rank
    g = p["cms"] == "CMS4"
    lr = logrank_test(p.loc[g, "relapse_free_months"], p.loc[~g, "relapse_free_months"],
                      p.loc[g, "relapse_event"], p.loc[~g, "relapse_event"])
    lr4 = multivariate_logrank_test(p["relapse_free_months"], p["cms"], p["relapse_event"])
    ax.text(0.03, 0.06, f"pMMR, estadio I–III: n = {len(p)}, {int(p['relapse_event'].sum())} eventos\n"
                        f"log-rank CMS4 vs resto {fmt_p(lr.p_value)}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_MIN, color=INK, linespacing=1.2)
    # tabla de números en riesgo
    tab = add_axes_mm(fig, left=14, top=60.5, width=78, height=4 * 3.0, fig_w=FIG_W, fig_h=FIG_H)
    tab.set_xlim(0, XMAX); tab.set_ylim(0, 4)
    for i, c in enumerate(CMS_ORDER):
        y = 4 - i - 0.5
        tab.text(-4.0, y, c, ha="right", va="center", fontsize=FONT_MIN, fontweight="bold", color=CMS_COLORS[c],
                 clip_on=False)
        for t, n in zip(RISK_TIMES, at_risk[c]):
            tab.text(t, y, str(n), ha="center", va="center", fontsize=FONT_MIN, color=INK, clip_on=False)
    tab.axis("off")
    fig.text(14 / FIG_W, 1 - 60.0 / FIG_H, "Número en riesgo", ha="left", va="bottom", fontsize=FONT_MIN, color=GRAY_DARK)
    return ax, at_risk, lr.p_value, lr4.p_value


def draw_forest(fig, rows, inc, meta):
    ax = add_axes_mm(fig, left=131, top=6, width=25, height=44, fig_w=FIG_W, fig_h=FIG_H)  # misma altura que el KM
    forest_plot(ax, rows, x_label="HR (escala log)", xlim=(0.25, 4.0), text_x=4.6)
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.text(1.02, 1.02, "HR (IC95%)", transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_MIN,
            color=GRAY_DARK, clip_on=False)
    return ax


def main(argv=None):
    p = base_parser(__doc__.split("\n")[0])
    args = p.parse_args(argv)
    paths = resolve_paths(args)
    setup_style()

    pmmr, (n_all, ev_all, n_dmmr) = load_km_cohort(paths["repo"])
    rows, inc, meta, modelE = load_cox(paths["repo"])
    inc = inc.copy(); inc["n_dMMR"] = n_dmmr

    fig = plt.figure(figsize=(mm(FIG_W), mm(FIG_H)))
    _, at_risk, p_lr, p_lr4 = draw_km(fig, pmmr)
    draw_forest(fig, rows, inc, meta)
    panel_label(fig, 1.5, 1.0, "A", FIG_W, FIG_H)
    panel_label(fig, 108, 1.0, "B", FIG_W, FIG_H)
    written = save_figure(fig, "Fig4_valor_clinico", paths["outdir"], also_svg=args.svg)

    print("== Figura 4: cifras usadas ==")
    print(f"A: muestra estadio I-III con MMR: n={n_all}, eventos={ev_all}, dMMR={n_dmmr}")
    print(f"   pMMR: n={len(pmmr)}, eventos={int(pmmr['relapse_event'].sum())}; por CMS: {pmmr['cms'].value_counts().to_dict()}")
    print(f"   modelo E (archivo): n={int(modelE['n'].iloc[0])}, eventos={int(modelE['eventos'].iloc[0])}")
    print(f"   log-rank CMS4 vs resto p={p_lr:.4f}; 4 grupos p={p_lr4:.4f}")
    print("   números en riesgo:", {c: dict(zip(RISK_TIMES, v)) for c, v in at_risk.items()})
    print("B:")
    for r in rows:
        if not r.get("ref"):
            print(f"  {r['label']}: HR {r['hr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) {fmt_p(r['p'])}")
    print(f"  n={meta['n']}, eventos={meta['eventos']}, C-index={meta['c_index']:.3f}; LRT chi2={inc['lr_chi2']:.2f} "
          f"df={inc['df']:.0f} p={inc['p_incremental_cms_sobre_clinica']:.4f}; dC={inc['delta_c_index']:+.3f} "
          f"({inc['c_index_estadio_clinica']:.3f} -> {inc['c_index_mas_cms']:.3f})")
    pie = (
        "Figura 4. Valor añadido sobre la clínica de rutina. (A) Curvas de Kaplan-Meier de supervivencia libre de recaída "
        f"por subtipo predicho en los {len(pmmr)} pacientes pMMR de GSE39582 con estadio I–III ({int(pmmr['relapse_event'].sum())} "
        "eventos), con IC95% sombreado y número de pacientes en riesgo a 0, 24, 48, 72 y 96 meses; eje truncado a 100 meses "
        f"(log-rank CMS4 vs. resto p={p_lr:.3f}). (B) Forest plot (HR e IC95%, escala logarítmica) del modelo estadio + MMR + CMS "
        f"(modelo D, Tabla 4) en {meta['n']} pacientes ({meta['eventos']} eventos, {n_dmmr} dMMR); referencia CMS2 y pMMR, "
        "estadio III frente a I+II. Aporte conjunto del subtipo sobre estadio + MMR: "
        f"LRT χ²={inc['lr_chi2']:.1f}, {inc['df']:.0f} g.l., p={inc['p_incremental_cms_sobre_clinica']:.3f}; "
        f"ΔC-index +{inc['delta_c_index']:.3f}."
    )
    pie_path = Path(paths["outdir"]) / "Fig4_pie_de_figura.md"
    pie_path.write_text(pie + "\n", encoding="utf-8")
    written.append(pie_path)
    print("Escrito:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
