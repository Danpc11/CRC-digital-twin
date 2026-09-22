"""Figura 3 — Arquitectura regulatoria de los subtipos (180 x 110 mm).

A (superior). Heatmap de z_meta (Stouffer, contraste CMS_i vs resto) para la unión de los 20
   reguladores maestros más robustos por CMS (top-20 por |z_meta|, hallmarks/CMS*_vs_rest_robust_tmrs.tsv)
   x 4 columnas CMS (regulators/CMS*_vs_rest_meta.tsv). Escala divergente centrada en 0 saturada a
   ±60 (CDX1 −98 y NR3C1 +84 quedan saturados a propósito). Para que ~65–80 filas quepan con texto
   ≥7 pt en 180 mm, la unión se dibuja en CUATRO BLOQUES lado a lado: cada regulador va al bloque
   del CMS donde es top-20 (si lo es en varios, al de mayor |z_meta|); dentro del bloque se ordena
   por z_meta descendente (activados arriba, reprimidos abajo).
   Un rombo (◆) junto al nombre marca reguladores con genes del panel como blancos replicados en
   ambas redes ARACNe3 (confirmed_both_cohorts=True e is_top20_robust_tmr=True en
   predictive_panel/panel_genes_vs_crc_mra_networks.tsv); el gen blanco se lista en la columna lateral.
   Gris = regulador no medido en ambas cohortes para ese contraste (sin z_meta).
B (inferior). 10 hallmarks x 4 CMS; celda coloreada sólo si el hallmark está replicado (mismo signo y
   padj<0.05 en el GSEA independiente de ambas cohortes; hallmarks/CMS*_vs_rest_gsea_concordance.tsv):
   rojo ↑, azul ↓, blanco no replicado, gris no evaluable (tested_in != both). Se verifican los
   conteos esperados 4/8/7/9.

Uso: python3 figures/fig3_regulacion.py [--outdir DIR] [--mra-dir DIR] [--predictive-panel DIR] [--sat 60]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CMS_COLORS, CMS_ORDER, GRAY, GRAY_LIGHT, GRAY_DARK, INK, FONT_MIN, DIVERGING_CMAP,
                     UP_COLOR, DOWN_COLOR, mm, panel_label, add_axes_mm, save_figure, setup_style,
                     base_parser, resolve_paths, require_path, MRA_CANDIDATES)

FIG_W, FIG_H = 180.0, 110.0
HALLMARK_ES = {
    "GENOME INSTABILITY": "Inestabilidad genómica",
    "EVADING IMMUNE DESTRUCTION": "Evasión de la destrucción inmune",
    "TUMOR-PROMOTING INFLAMMATION": "Inflamación pro-tumoral",
    "RESISTING CELL DEATH": "Resistencia a la muerte celular",
    "SUSTAINED ANGIOGENESIS": "Angiogénesis sostenida",
    "TISSUE INVASION AND METASTASIS": "Invasión tisular y metástasis",
    "SUSTAINING PROLIFERATIVE SIGNALING": "Señalización proliferativa sostenida",
    "EVADING GROWTH SUPPRESSORS": "Evasión de supresores de crecimiento",
    "REPROGRAMMING ENERGY METABOLISM": "Reprogramación del metabolismo",
    "REPLICATIVE IMMORTALITY": "Inmortalidad replicativa",
}
HALLMARK_ORDER = list(HALLMARK_ES.keys())
EXPECTED_REPLICATED = {"CMS1": 4, "CMS2": 8, "CMS3": 7, "CMS4": 9}

# Pie de figura acordado (2026-09-14); reemplaza al pie provisional del docx.
PIE_DE_FIGURA = (
    "Figura 3. Arquitectura regulatoria de los subtipos. (A) Reguladores maestros más robustos de cada subtipo "
    "(los 15–20 de mayor |z| combinado en el contraste subtipo vs. resto; z de Stouffer sobre TCGA y microarreglo, "
    "saturado a ±60). Cada panel muestra la actividad de esos reguladores en los cuatro subtipos; gris, sin z "
    "combinado para ese contraste. El rombo marca a todo regulador robusto que tiene un gen del panel entre sus "
    "blancos replicados en ambas redes, con el gen indicado a la derecha; el texto discute los principales. "
    "(B) Hallmarks del cáncer replicados por subtipo: una celda se colorea solo si el hallmark fue significativo "
    "(padj<0.05) y con el mismo signo en el GSEA independiente de ambas cohortes; rojo, activado en el subtipo; "
    "azul, reprimido; blanco, no replicado; gris, no evaluable por estar presente en una sola cohorte. Los conteos "
    "sobre cada columna indican hallmarks replicados de diez."
)


# ------------------------------------------------------------------ datos
def load_regulators(mra: Path, target_axis: dict | None = None):
    """target_axis: regulador -> conjunto de ejes CMS de sus genes blanco en el panel (para asignar bloque)."""
    target_axis = target_axis or {}
    meta = {c: pd.read_csv(mra / f"regulators/{c}_vs_rest_meta.tsv", sep="\t").set_index("regulator")["z_meta"]
            for c in CMS_ORDER}
    top = {c: pd.read_csv(mra / f"hallmarks/{c}_vs_rest_robust_tmrs.tsv", sep="\t") for c in CMS_ORDER}
    for c in CMS_ORDER:
        t = top[c]
        assert len(t) == 20, f"{c}: esperaba 20 TMR robustos, hay {len(t)}"
        # consistencia: el top-20 del archivo debe coincidir con top-20 por |z_meta| de la tabla meta
        top_by_z = meta[c].abs().sort_values(ascending=False).index[:20]
        missing = set(t["regulator"]) - set(top_by_z)
        if missing:
            print(f"AVISO {c}: robust_tmrs no coincide con top-20 |z_meta| de la tabla meta: {missing}")
    union = sorted({r for c in CMS_ORDER for r in top[c]["regulator"]})
    Z = pd.DataFrame({c: meta[c].reindex(union) for c in CMS_ORDER}, index=union)
    # asignación a bloque
    # Regla: el regulador va al bloque del CMS donde es top-20. Si lo es en varios, se prefiere el eje
    # de su gen blanco en el panel (p. ej. CREB3L1 -> CMS3 por AGR2/GALNT8, aunque |z| sea mayor en CMS2);
    # si no tiene blanco en el panel, el de mayor |z_meta|.
    block = {}
    for r in union:
        cands = [c for c in CMS_ORDER if r in set(top[c]["regulator"])]
        pref = [c for c in cands if c in target_axis.get(r, set())]
        pool = pref or cands
        block[r] = max(pool, key=lambda c: abs(Z.loc[r, c]))
    blocks = {c: sorted([r for r in union if block[r] == c], key=lambda r: -Z.loc[r, c]) for c in CMS_ORDER}
    return Z, blocks, top


def load_targets(pp: Path | None):
    """regulador -> lista de genes del panel que son blancos replicados (ambas redes) y TMR top-20."""
    if pp is None:
        print("AVISO: no encuentro predictive_panel; sin marcas de genes del panel")
        return {}, {}
    t = pd.read_csv(pp / "panel_genes_vs_crc_mra_networks.tsv", sep="\t")
    t = t[(t["confirmed_both_cohorts"].astype(str) == "True") & (t["is_top20_robust_tmr"].astype(str) == "True")]
    out, axes = {}, {}
    for reg, sub in t.groupby("upstream_regulator"):
        out[reg] = sorted(set(sub["panel_gene"]))
        axes[reg] = set(sub["axis"])
    return out, axes


def load_hallmarks(mra: Path):
    """Matriz hallmark x CMS con valores: +1 (↑ replicado), -1 (↓ replicado), 0 (no replicado), NaN (no evaluable)."""
    H = pd.DataFrame(index=HALLMARK_ORDER, columns=CMS_ORDER, dtype=float)
    counts = {}
    for c in CMS_ORDER:
        d = pd.read_csv(mra / f"hallmarks/{c}_vs_rest_gsea_concordance.tsv", sep="\t").set_index("pathway")
        n_rep = 0
        for h in HALLMARK_ORDER:
            r = d.loc[h]
            if str(r["tested_in"]) != "both":
                H.loc[h, c] = np.nan
            elif str(r["replicated"]).upper() == "TRUE":
                assert np.sign(r["NES_tcga"]) == np.sign(r["NES_microarray"])
                H.loc[h, c] = np.sign(r["NES_tcga"])
                n_rep += 1
            else:
                H.loc[h, c] = 0.0
        counts[c] = n_rep
    return H, counts


# ------------------------------------------------------------------ dibujo
def _est_text_mm(txt: str) -> float:
    """Ancho estimado (mm) de texto Arial itálica 7 pt en mayúsculas/dígitos (calibrado sobre el render)."""
    return sum(1.8 if ch.isalnum() else 0.8 for ch in txt)


def draw_heatmap_blocks(fig, Z, blocks, targets, sat):
    cmap = plt.get_cmap(DIVERGING_CMAP)
    norm = Normalize(vmin=-sat, vmax=sat)
    ROW = 2.85          # mm por fila
    CELL_W = 3.4        # mm por columna CMS
    NAME_W = 12.5       # columna de nombres (GTF2IRD1 en negrita itálica mide ~11 mm)
    MARK_W = 1.6        # espacio para el rombo
    GAP = 1.0           # separación entre bloques
    TOP = 18.0          # mm desde arriba donde empieza la primera fila
    # ancho de cada bloque según el texto más largo de su columna de genes blanco
    widths = {}
    for c in CMS_ORDER:
        tg = [_est_text_mm("/".join(targets[r])) for r in blocks[c] if r in targets] or [_est_text_mm("Gen del")]
        widths[c] = NAME_W + 1.0 + 4 * CELL_W + MARK_W + max(max(tg), _est_text_mm("Gen del")) + GAP
    total = sum(widths.values())
    if total > FIG_W - 1.0:
        print(f"AVISO: los bloques suman {total:.1f} mm > {FIG_W} mm; se comprimen proporcionalmente")
        widths = {c: w * (FIG_W - 1.0) / total for c, w in widths.items()}
    x0 = 0.5
    for c in CMS_ORDER:
        regs = blocks[c]
        n = len(regs)
        left_cells = x0 + NAME_W + 1.0
        ax = add_axes_mm(fig, left=left_cells, top=TOP, width=CELL_W * 4, height=ROW * n, fig_w=FIG_W, fig_h=FIG_H)
        x_mark = 4 + (MARK_W / 2) / CELL_W          # en unidades de celda
        x_text = 4 + (MARK_W + 0.3) / CELL_W
        for i, r in enumerate(regs):
            for j, cc in enumerate(CMS_ORDER):
                z = Z.loc[r, cc]
                face = GRAY_LIGHT if pd.isna(z) else cmap(norm(z))
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=face, edgecolor="white", lw=0.4))
            has_t = r in targets
            ax.text(-0.15, n - 1 - i + 0.5, r, ha="right", va="center", fontsize=FONT_MIN, fontstyle="italic",
                    color=INK, fontweight="bold" if has_t else "normal", clip_on=False)
            if has_t:
                # rombo como marcador (Arial no trae el glifo U+25C6)
                ax.plot([x_mark], [n - 1 - i + 0.5], marker="D", ms=2.6, mec="none", mfc=CMS_COLORS[c], clip_on=False)
                ax.text(x_text, n - 1 - i + 0.5, "/".join(targets[r]), ha="left", va="center", fontsize=FONT_MIN,
                        fontstyle="italic", color=INK, clip_on=False)
        ax.set_xlim(0, 4); ax.set_ylim(0, n)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # encabezados de columna (rotados) y de la columna de genes blanco
        for j, cc in enumerate(CMS_ORDER):
            ax.text(j + 0.5, n + 0.25, cc, rotation=90, ha="center", va="bottom", fontsize=FONT_MIN,
                    fontweight="bold", color=CMS_COLORS[cc], clip_on=False)
        ax.text(x_text, n + 0.25, "Gen del\npanel", ha="left", va="bottom", fontsize=FONT_MIN,
                color=GRAY_DARK, clip_on=False, linespacing=1.0)
        # título discreto del bloque, centrado sobre el bloque, encima de los encabezados
        # Sin la sigla TMR (el manuscrito no la define). La expresión completa no cabe en una línea al
        # ancho del bloque (~44 mm), así que va en tres líneas, con el paréntesis en la última;
        # ocupa ~2–10.5 mm y los encabezados rotados empiezan en ~11.5 mm.
        fig.text((x0 + (widths[c] - GAP) / 2) / FIG_W, 1 - 2.0 / FIG_H,
                 f"Reguladores maestros\nrobustos de {c}\n({n} filas)",
                 ha="center", va="top", fontsize=FONT_MIN, color=CMS_COLORS[c], fontweight="bold", linespacing=1.0)
        x0 += widths[c]
    return norm, cmap


def draw_legend_heatmap(fig, norm, cmap, sat):
    cax = add_axes_mm(fig, left=132, top=87.5, width=30, height=2.6, fig_w=FIG_W, fig_h=FIG_H)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-sat, -sat / 2, 0, sat / 2, sat])
    cb.set_ticklabels([f"≤−{sat:.0f}", f"−{sat/2:.0f}", "0", f"{sat/2:.0f}", f"≥{sat:.0f}"])
    cb.ax.tick_params(length=2, width=0.5, labelsize=FONT_MIN, pad=1.5)
    cb.outline.set_linewidth(0.5)
    fig.text(132 / FIG_W, 1 - 86.3 / FIG_H, "z combinado (Stouffer), CMS vs resto\nreprimido ← 0 → activado",
             ha="left", va="bottom", fontsize=FONT_MIN, color=GRAY_DARK, linespacing=1.1)
    import matplotlib.lines as mlines
    fig.add_artist(mlines.Line2D([133.3 / FIG_W], [1 - 96.6 / FIG_H], marker="D", ms=2.6, mec="none",
                                 mfc=GRAY_DARK, transform=fig.transFigure))
    fig.text(135.5 / FIG_W, 1 - 95.5 / FIG_H,
             "regulador con un gen del panel como\n"
             "blanco replicado en ambas redes\n"
             "(gen listado a la derecha). Gris: sin\n"
             "z combinado para ese contraste.",
             ha="left", va="top", fontsize=FONT_MIN, color=GRAY_DARK, linespacing=1.15)


def draw_hallmarks(fig, H, counts):
    n = len(HALLMARK_ORDER)
    ROW = 2.7; CELL_W = 7.0
    ax = add_axes_mm(fig, left=52, top=83, width=CELL_W * 4, height=ROW * n, fig_w=FIG_W, fig_h=FIG_H)
    for i, h in enumerate(HALLMARK_ORDER):
        for j, c in enumerate(CMS_ORDER):
            v = H.loc[h, c]
            if pd.isna(v):
                face, txt, tc = GRAY_LIGHT, "n.e.", GRAY_DARK
            elif v > 0:
                face, txt, tc = UP_COLOR, "↑", "white"
            elif v < 0:
                face, txt, tc = DOWN_COLOR, "↓", "white"
            else:
                face, txt, tc = "white", "", INK
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=face, edgecolor="#BDBDBD", lw=0.4))
            if txt:
                ax.text(j + 0.5, n - 1 - i + 0.5, txt, ha="center", va="center", fontsize=FONT_MIN, color=tc,
                        fontweight="bold" if txt in "↑↓" else "normal")
        ax.text(-0.1, n - 1 - i + 0.5, HALLMARK_ES[h], ha="right", va="center", fontsize=FONT_MIN, color=INK,
                clip_on=False)
    for j, c in enumerate(CMS_ORDER):
        ax.text(j + 0.5, n + 0.15, f"{c}\n{counts[c]}/10", ha="center", va="bottom", fontsize=FONT_MIN,
                fontweight="bold", color=CMS_COLORS[c], clip_on=False, linespacing=1.0)
    ax.set_xlim(0, 4); ax.set_ylim(0, n)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # leyenda
    lx = 84.0
    items = [(UP_COLOR, "↑ replicado, activado en el CMS"), (DOWN_COLOR, "↓ replicado, reprimido en el CMS"),
             ("white", "no replicado"), (GRAY_LIGHT, "n.e.: no evaluable (una sola cohorte)")]
    for k, (col, lab) in enumerate(items):
        y = 84.5 + k * 3.4
        fig.patches.append(Rectangle((lx / FIG_W, 1 - (y + 2.4) / FIG_H), 2.6 / FIG_W, 2.4 / FIG_H,
                                     facecolor=col, edgecolor="#BDBDBD", lw=0.4, transform=fig.transFigure))
        fig.text((lx + 3.6) / FIG_W, 1 - (y + 1.2) / FIG_H, lab, ha="left", va="center", fontsize=FONT_MIN, color=INK)
    fig.text(lx / FIG_W, 1 - 99.5 / FIG_H,
             "Replicado = mismo signo y padj<0.05\nen el GSEA independiente de ambas\ncohortes (TCGA y microarreglo).",
             ha="left", va="top", fontsize=FONT_MIN, color=GRAY_DARK, linespacing=1.15)


def main(argv=None):
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--sat", type=float, default=60.0, help="saturación de la escala de z (±)")
    args = p.parse_args(argv)
    paths = resolve_paths(args)
    setup_style()

    mra = require_path(paths["mra"], "crc_mra_results", MRA_CANDIDATES)
    targets, target_axes = load_targets(paths["predictive_panel"])
    Z, blocks, top = load_regulators(mra, target_axes)
    H, counts = load_hallmarks(mra)

    fig = plt.figure(figsize=(mm(FIG_W), mm(FIG_H)))
    norm, cmap = draw_heatmap_blocks(fig, Z, blocks, targets, args.sat)
    draw_legend_heatmap(fig, norm, cmap, args.sat)
    draw_hallmarks(fig, H, counts)
    panel_label(fig, 1.5, 1.0, "A", FIG_W, FIG_H)
    panel_label(fig, 1.5, 78.0, "B", FIG_W, FIG_H)
    written = save_figure(fig, "Fig3_arquitectura_regulatoria", paths["outdir"], also_svg=args.svg)
    pie_path = Path(paths["outdir"]) / "Fig3_pie_de_figura.md"
    pie_path.write_text(PIE_DE_FIGURA + "\n", encoding="utf-8")
    written.append(pie_path)

    print("== Figura 3: cifras usadas ==")
    print(f"A: unión de top-20 = {len(Z)} reguladores; filas por bloque: " + ", ".join(f"{c}={len(blocks[c])}" for c in CMS_ORDER))
    print(f"   extremos: min z = {Z.min().min():.1f} ({Z.stack().idxmin()}), max z = {Z.max().max():.1f} ({Z.stack().idxmax()}); saturación ±{args.sat:.0f}")
    print(f"   celdas sin z_meta (gris): {int(Z.isna().sum().sum())}")
    print("   reguladores marcados (◆) y sus genes blanco:")
    for c in CMS_ORDER:
        for r in blocks[c]:
            if r in targets:
                print(f"     [{c}] {r:9s} z={Z.loc[r, c]:+6.1f} -> {', '.join(targets[r])}")
    unplaced = set(targets) - set(Z.index)
    if unplaced:
        print("   AVISO: reguladores con blancos en el panel que NO están en la unión top-20:", unplaced)
    genes_covered = sorted({g for r in targets for g in targets[r] if r in Z.index})
    print(f"   genes del panel con al menos un TMR marcado: {len(genes_covered)}/10 -> {genes_covered}")
    print("B: hallmarks replicados por CMS:", counts, "(esperado", EXPECTED_REPLICATED, ")")
    for c in CMS_ORDER:
        if counts[c] != EXPECTED_REPLICATED[c]:
            print(f"   DISCREPANCIA en {c}: {counts[c]} vs esperado {EXPECTED_REPLICATED[c]}")
    print(H.replace({1.0: "↑", -1.0: "↓", 0.0: "·"}).fillna("n.e.").to_string())
    print("Escrito:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
