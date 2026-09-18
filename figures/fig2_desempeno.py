"""Figura 2 — Desempeño del clasificador (3 paneles, 180 x 70 mm).

A. Matriz de confusión subtipo predicho (filas) vs etiqueta oficial CRCSC (columnas), GSE39582 n=519.
   Fuente: results_gse39582/scored_cohort.tsv (panel v0.3, TGFB1->EFEMP2; mismo insumo que concordance_analysis.py).
   Decisión sobre la clase `none` (47 muestras sin etiqueta oficial): se muestra como QUINTA COLUMNA
   GRIS con conteos solamente; NO entra en kappa, exactitud ni en los porcentajes por fila, que se
   calculan sobre las 519 muestras etiquetadas (igual que concordance_analysis.py). Así el lector ve
   dónde caen las muestras sin etiqueta sin que alteren las métricas.
   Porcentaje por columna (default desde 2026-09-14) = sensibilidad del panel para cada subtipo
   oficial, que es la cifra que reporta el manuscrito (89.0 / 76.7 / 91.3 / 83.5 % con el panel v0.3). Con
   `--pct-by row` se muestra el porcentaje por fila (precisión del subtipo predicho).
B. Información mutua gen–eje CMS (uno-contra-resto, estimador de Kraskov, TCGA n=512 etiquetadas) y
   AUC uno-contra-resto en GSE39582 para los 10 genes del panel + FABP1, SI y TGFB1 (paneles previos, gris).
   Fuente: network_analysis/results/crc_net_577/predictive_panel/mi_onevsrest_<eje>.tsv y
   gse39582_feature_selection/gene_ranking_full.tsv. Cada gen se evalúa contra SU eje CMS.
   El AUC se dibuja desde 0.5: barras a la izquierda = marcador inverso (MLH1 baja en CMS1).
C. Forest plot del Cox estratificado por cohorte ajustado por estadio categorico (III vs I+II), 4 cohortes
   externas con etiqueta oficial, misma muestra (n=428, 95 eventos; analisis principal de PROJECT_STATUS.md).
   Fuente: results_pooled_cox_mismamuestra/cox_summary_adjusted.tsv, cox_incremental_value.tsv.

Las cifras de resumen (kappa, exactitud, n, LRT, ΔC-index) NO se dibujan: van en el pie de figura
(figuras/Fig2_pie_de_figura.md, generado por este script con las cifras leídas de los archivos).

Uso: python3 figures/fig2_desempeno.py [--outdir DIR] [--pct-by row|col]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CMS_COLORS, CMS_LONG, CMS_ORDER, PANEL_GENES, REPLACED_GENES, GRAY, GRAY_LIGHT,
                     GRAY_DARK, INK, FONT_MIN, mm, panel_label, add_axes_mm, save_figure, setup_style,
                     base_parser, resolve_paths, forest_plot, fmt_p)

FIG_W, FIG_H = 180.0, 60.0
PANEL_TOP, PANEL_H = 10.0, 38.0   # todos los paneles con la misma altura
AXIS_FILES = {"CMS1": "mi_onevsrest_CMS1_MSI_immune.tsv", "CMS2": "mi_onevsrest_CMS2_canonical_WNT.tsv",
              "CMS3": "mi_onevsrest_CMS3_metabolic.tsv", "CMS4": "mi_onevsrest_CMS4_mesenchymal.tsv"}
AXIS_AUC_COL = {"CMS1": "CMS1_MSI_immune", "CMS2": "CMS2_canonical_WNT", "CMS3": "CMS3_metabolic",
                "CMS4": "CMS4_mesenchymal"}


# ------------------------------------------------------------------ datos
def load_confusion(repo: Path):
    from sklearn.metrics import cohen_kappa_score
    d = pd.read_csv(repo / "results_gse39582/scored_cohort.tsv", sep="\t")
    d["pred"] = d["predicted_cms"].map(CMS_LONG)
    d["ofic"] = d["cms_label"].map(CMS_LONG).fillna("none")
    lab = d[d["ofic"] != "none"]
    ct = pd.crosstab(d["pred"], d["ofic"]).reindex(index=CMS_ORDER, columns=CMS_ORDER + ["none"]).fillna(0).astype(int)
    kappa = cohen_kappa_score(lab["ofic"], lab["pred"])
    acc = (lab["ofic"] == lab["pred"]).mean()
    return ct, kappa, acc, len(lab), int((d["ofic"] == "none").sum())


def load_genes(pp: Path):
    rows = []
    auc = pd.read_csv(pp / "gse39582_feature_selection/gene_ranking_full.tsv", sep="\t", index_col=0)
    mi_tables = {ax: pd.read_csv(pp / f, sep="\t").set_index("gene") for ax, f in AXIS_FILES.items()}
    order = []
    for ax in CMS_ORDER:
        order += [(g, ax, False) for g in PANEL_GENES[ax]]
        order += [(g, gx, True) for g, gx in REPLACED_GENES.items() if gx == ax]
    for g, ax, prev in order:
        mi = mi_tables[ax]
        rows.append({"gene": g, "axis": ax, "previous": prev,
                     "mi": float(mi.loc[g, "mi_vs_axis"]), "mi_pct": float(mi.loc[g, "percentile"]),
                     "auc": float(auc.loc[g, AXIS_AUC_COL[ax]])})
    return pd.DataFrame(rows)


def load_cox(repo: Path):
    d = repo / "results_pooled_cox_mismamuestra"
    s = pd.read_csv(d / "cox_summary_adjusted.tsv", sep="\t").set_index("covariate")
    inc = pd.read_csv(d / "cox_incremental_value.tsv", sep="\t").iloc[0]
    def row(cov, label, color=GRAY_DARK):
        r = s.loc[cov]
        return {"label": label, "hr": r["exp(coef)"], "lo": r["exp(coef) lower 95%"], "hi": r["exp(coef) upper 95%"],
                "p": r["p"], "color": color}
    rows = [row("stage_III", "Estadio III"),
            row("cms_CMS1_MSI_immune", "CMS1", CMS_COLORS["CMS1"]),
            {"label": "CMS2", "ref": True, "color": CMS_COLORS["CMS2"]},
            row("cms_CMS3_metabolic", "CMS3", CMS_COLORS["CMS3"]),
            row("cms_CMS4_mesenchymal", "CMS4", CMS_COLORS["CMS4"])]
    return rows, inc


# ------------------------------------------------------------------ paneles
def draw_confusion(fig, ct, kappa, acc, n_lab, n_none, pct_by="row"):
    ncol = ct.shape[1]
    CELL_W, CELL_H = 9.0, PANEL_H / 4
    ax = add_axes_mm(fig, left=15, top=PANEL_TOP, width=CELL_W * ncol, height=PANEL_H, fig_w=FIG_W, fig_h=FIG_H)
    lab = ct[CMS_ORDER]
    if pct_by == "row":
        pct = lab.div(lab.sum(axis=1), axis=0) * 100
    else:
        pct = lab.div(lab.sum(axis=0), axis=1) * 100
    cmap = plt.get_cmap("Greys")
    for i, r in enumerate(CMS_ORDER):
        for j, c in enumerate(ct.columns):
            v = int(ct.loc[r, c])
            if c == "none":
                face = "#EFEFEF"
                txt, tcol = f"{v}", GRAY_DARK
            else:
                p = pct.loc[r, c]
                face = cmap(0.08 + 0.72 * p / 100)
                txt, tcol = f"{v}\n{p:.1f}%", ("white" if p > 55 else INK)
            ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=face, edgecolor="white", lw=0.8))
            ax.text(j + 0.5, 3 - i + 0.5, txt, ha="center", va="center", fontsize=FONT_MIN, color=tcol,
                    fontweight="bold" if (r == c) else "normal", linespacing=1.1)
    ax.set_xlim(0, ncol); ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(ncol) + 0.5)
    ax.set_xticklabels(CMS_ORDER + ["Ninguna"])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels(CMS_ORDER[::-1])
    for t, c in zip(ax.get_xticklabels(), CMS_ORDER + ["none"]):
        t.set_color(CMS_COLORS.get(c, GRAY)); t.set_fontweight("bold" if c in CMS_COLORS else "normal")
    for t, c in zip(ax.get_yticklabels(), CMS_ORDER[::-1]):
        t.set_color(CMS_COLORS[c]); t.set_fontweight("bold")
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Etiqueta oficial CRCSC", labelpad=2)
    ax.set_ylabel("Subtipo predicho", labelpad=3)
    return ax


def draw_genes(fig, g: pd.DataFrame):
    n = len(g)
    ys = np.arange(n)[::-1]
    colors = [GRAY if p else CMS_COLORS[a] for a, p in zip(g["axis"], g["previous"])]
    ax1 = add_axes_mm(fig, left=76, top=PANEL_TOP, width=17, height=PANEL_H, fig_w=FIG_W, fig_h=FIG_H)
    ax2 = add_axes_mm(fig, left=98, top=PANEL_TOP, width=17, height=PANEL_H, fig_w=FIG_W, fig_h=FIG_H, sharey=ax1)
    ax1.barh(ys, g["mi"], color=colors, height=0.72)
    ax1.set_xlim(0, 0.42); ax1.set_xticks([0, 0.2, 0.4])
    ax1.set_xlabel("Información mutua\ngen–eje (TCGA)", labelpad=2)
    # AUC desde 0.5
    ax2.barh(ys, g["auc"] - 0.5, left=0.5, color=colors, height=0.72)
    ax2.axvline(0.5, color=GRAY_DARK, lw=0.6)
    ax2.set_xlim(0.15, 0.95); ax2.set_xticks([0.25, 0.5, 0.75])
    ax2.set_xticklabels(["0.25", "0.5", "0.75"])
    ax2.set_xlabel("AUC 1-vs-resto\n(GSE39582)", labelpad=2)
    ax2.tick_params(axis="y", length=0, labelleft=False)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(g["gene"], fontstyle="italic")
    for t, c in zip(ax1.get_yticklabels(), colors):
        t.set_color(c)
    ax1.tick_params(axis="y", length=0, pad=2)
    ax1.set_ylim(-0.7, n - 0.3)
    for a in (ax1, ax2):
        a.spines["left"].set_visible(False)
        a.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
        a.set_axisbelow(True)
    # etiquetas de eje CMS a la izquierda (corchete de color)
    for axname in CMS_ORDER:
        idx = np.where(g["axis"].values == axname)[0]
        y0, y1 = ys[idx].min() - 0.36, ys[idx].max() + 0.36
        # corchete en mm: 13.5 mm a la izquierda del eje (los nombres ocupan ~11 mm)
        xb = -13.5 / 17.0
        ax1.plot([xb, xb], [y0, y1], color=CMS_COLORS[axname], lw=1.6, transform=ax1.get_yaxis_transform(),
                 clip_on=False, solid_capstyle="butt")
        ax1.text(xb - 0.6 / 17.0, (y0 + y1) / 2, axname, color=CMS_COLORS[axname], fontsize=FONT_MIN,
                 fontweight="bold", rotation=90, ha="right", va="center", transform=ax1.get_yaxis_transform(),
                 clip_on=False)
    # leyenda gris
    return ax1, ax2


def draw_forest(fig, rows, inc):
    ax = add_axes_mm(fig, left=134, top=PANEL_TOP, width=22, height=PANEL_H, fig_w=FIG_W, fig_h=FIG_H)
    forest_plot(ax, rows, x_label="HR (escala log)", xlim=(0.7, 5.5), text_x=6.3)
    ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"])
    ax.text(1.02, 1.02, "HR (IC95%)", transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_MIN,
            color=GRAY_DARK, clip_on=False)
    return ax


def main(argv=None):
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--pct-by", choices=["row", "col"], default="col")
    args = p.parse_args(argv)
    paths = resolve_paths(args)
    setup_style()

    ct, kappa, acc, n_lab, n_none = load_confusion(paths["repo"])
    genes = load_genes(paths["predictive_panel"])
    rows, inc = load_cox(paths["repo"])

    fig = plt.figure(figsize=(mm(FIG_W), mm(FIG_H)))
    draw_confusion(fig, ct, kappa, acc, n_lab, n_none, pct_by=args.pct_by)
    draw_genes(fig, genes)
    draw_forest(fig, rows, inc)
    for x, letter in ((1.5, "A"), (60, "B"), (118.5, "C")):
        panel_label(fig, x, 1.0, letter, FIG_W, FIG_H)
    stem = "Fig2_desempeno_clasificador" + ("" if args.pct_by == "col" else "_pctfila")
    written = save_figure(fig, stem, paths["outdir"], also_svg=args.svg)

    # ------------------------------------------------ reporte de cifras usadas
    print("== Figura 2: cifras usadas ==")
    print(f"A: n etiquetadas={n_lab}, none={n_none}, kappa={kappa:.4f}, exactitud={acc*100:.2f}%")
    print(ct.to_string())
    lab = ct[CMS_ORDER]
    print("  sumas por columna (etiqueta oficial):", lab.sum(axis=0).to_dict(), "| none:", int(ct["none"].sum()))
    print("  % por columna (exactitud por subtipo oficial):",
          {c: round(lab.loc[c, c] / lab[c].sum() * 100, 1) for c in CMS_ORDER})
    print("  % por fila (precisión del predicho):",
          {c: round(lab.loc[c, c] / lab.loc[c].sum() * 100, 1) for c in CMS_ORDER})
    print("B:"); print(genes.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("C:")
    for r in rows:
        if not r.get("ref"):
            print(f"  {r['label']}: HR {r['hr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) {fmt_p(r['p'])}")
    print(f"  LRT chi2={inc['lr_chi2']:.2f} df={inc['df']:.0f} p={inc['p_incremental']:.4f}; "
          f"dC estrat.={inc['delta_c_index_stratified']:+.3f} [{inc['delta_c_index_stratified_bootstrap_low95']:.3f}, "
          f"{inc['delta_c_index_stratified_bootstrap_high95']:.3f}]; C estrat. {inc['c_index_stratified_stage_plus_cms']:.3f} vs {inc['c_index_stratified_stage_only']:.3f}")
    pie = (
        "Figura 2. Desempeño del clasificador. (A) Matriz de confusión del subtipo predicho (filas) frente a la "
        f"etiqueta oficial del Consorcio (columnas) en GSE39582 (n={n_lab}; kappa de Cohen {kappa:.3f}, exactitud "
        f"{acc*100:.1f}%). "
        + ("Cada celda muestra el conteo y el porcentaje respecto al total de su columna, es decir, la sensibilidad "
           "del panel para cada subtipo oficial; " if args.pct_by == "col" else
           "Cada celda muestra el conteo y el porcentaje por fila (precisión del subtipo predicho); ")
        + 
        f"la columna gris reúne las {n_none} muestras sin etiqueta oficial, que no intervienen en kappa ni en los porcentajes. "
        "(B) Información mutua gen–eje CMS (uno contra el resto, TCGA) y AUC uno contra el resto en GSE39582 de los diez genes "
        "del panel, agrupados por eje CMS; en gris, los tres genes sustituidos (FABP1, SI, TGFB1) como comparación. El AUC se dibuja "
        "desde 0.5: la barra hacia la izquierda de MLH1 corresponde a un marcador inverso (expresión baja en CMS1). "
        "(C) Forest plot (HR e IC95%, escala logarítmica) del modelo de Cox estratificado por cohorte y ajustado por estadio "
        "en las cuatro cohortes externas con etiqueta oficial combinadas (n=428, 95 eventos; Tabla 2); referencia CMS2, estadio III frente a I+II. "
        f"Aporte del subtipo sobre el estadio: LRT χ²={inc['lr_chi2']:.1f}, {inc['df']:.0f} g.l., p={inc['p_incremental']:.4f}; "
        f"ΔC-index estratificado +{inc['delta_c_index_stratified']:.3f} (IC95% {inc['delta_c_index_stratified_bootstrap_low95']:.3f}–"
        f"{inc['delta_c_index_stratified_bootstrap_high95']:.3f})."
    )
    pie_path = Path(paths["outdir"]) / "Fig2_pie_de_figura.md"
    pie_path.write_text(pie + "\n", encoding="utf-8")
    written.append(pie_path)
    print("Escrito:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
