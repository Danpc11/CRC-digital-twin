"""Figura 1 — Esquema del flujo del estudio (180 x 100 mm), sin datos.

Diagrama de cajas y flechas con tres carriles horizontales y un flujo principal de izquierda a derecha:
- Carril central (flujo principal, borde sólido): transcriptomas públicos -> selección de 10 genes
  (4 criterios) -> centroides CMS1-4 congelados -> clasificación por correlación -> validación pronóstica.
  Los 10 genes se listan bajo la caja de selección, agrupados por eje CMS con el color del subtipo.
- Carril superior (fundamento, borde punteado): reguladores maestros por subtipo -> 8/10 genes son
  blancos de TMR de su eje; flechas hacia "Selección" y "Centroides".
- Carril inferior (hacia la clínica, borde punteado): puente RT-qPCR y marco de simulación, en paralelo,
  ambos etiquetados "en desarrollo".

Implementación: FancyBboxPatch para las cajas, annotate(arrowstyle="-|>") para las flechas, coordenadas
fijas en mm sobre un eje sin marcas (ax.axis("off")). Las coordenadas viven en el diccionario BOXES al
inicio, para reordenar sin tocar el resto. `--numbered` produce un PNG con las cajas numeradas para
revisar el layout.

Uso: python3 figures/fig1_flujo.py [--outdir DIR] [--numbered]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CMS_COLORS, CMS_ORDER, PANEL_GENES, GRAY, GRAY_LIGHT, GRAY_DARK, INK, FONT_MIN,
                     mm, save_figure, setup_style, base_parser, resolve_paths)

FIG_W, FIG_H = 180.0, 100.0

# --------------------------------------------------------------------------------------------------
# COORDENADAS (mm; origen arriba-izquierda; x, y = esquina superior izquierda; w, h = ancho, alto)
# lane: "main" (borde sólido, relleno gris muy claro) | "top" | "bottom" (borde punteado, relleno blanco)
# --------------------------------------------------------------------------------------------------
BOXES = {
    # ---- carril superior: fundamento
    "tmr": dict(x=8, y=4, w=92, h=17, lane="top",
                text="Reguladores maestros por subtipo\nARACNe3 + NaRnEA, TCGA (n=633) y microarreglo (n=1,134);\nmeta-análisis de las dos cohortes"),
    "tmr_out": dict(x=106, y=4, w=72, h=17, lane="top",
                    text="8/10 genes del panel son blancos\nde reguladores maestros de su eje;\nhallmarks del cáncer replicados por subtipo"),
    # ---- carril central: flujo principal
    "datos": dict(x=8, y=36, w=27, h=22, lane="main",
                  text="Transcriptomas\npúblicos\nTCGA (n=577)\nGSE39582 (n=566)"),
    "seleccion": dict(x=39, y=32, w=35, h=30, lane="main",
                      text="Selección de 10 genes:\n4 criterios\n"
                           "• discriminación en 2 cohortes\n• red de coexpresión\n• literatura\n• factibilidad RT-qPCR"),
    "centroides": dict(x=78, y=36, w=27, h=22, lane="main",
                       text="Centroides\nCMS1–4\n(GSE39582,\ncongelados)"),
    "clasif": dict(x=109, y=36, w=24, h=22, lane="main",
                   text="Clasificación\npor correlación\ncon el centroide"),
    "valid": dict(x=137, y=32, w=41, h=30, lane="main",
                  text="Validación pronóstica\n5 cohortes externas (n=545)\n\n• Cox estratificado por cohorte\n• dejando una cohorte fuera\n• ajuste por estadio y MMR"),
    # ---- carril inferior: hacia la clínica
    "qpcr": dict(x=62, y=80, w=56, h=17, lane="bottom", dev=True,
                 text="Puente RT-qPCR\nΔCt con genes de referencia\n(UBB, RPLP0, TBP, ACTB)\n→ escala del modelo"),
    "sim": dict(x=122, y=80, w=56, h=17, lane="bottom", dev=True,
                text="Marco de simulación\nred de Hopfield continua con los 4 centroides\ncomo estados; gemelo digital"),
}
# lista de genes bajo la caja de selección (x, y de la primera línea; una línea por eje CMS)
GENES_POS = dict(x=39.0, y=64.0, dy=2.9)
# flechas: (origen, destino, lado_origen, lado_destino). Lados: "r","l","t","b" (centro de ese lado)
ARROWS = [
    ("datos", "seleccion", "r", "l"), ("seleccion", "centroides", "r", "l"),
    ("centroides", "clasif", "r", "l"), ("clasif", "valid", "r", "l"),
    ("tmr", "tmr_out", "r", "l"),
    ("tmr", "seleccion", "b@56", "t@56"),      # vertical a x=56 mm
    ("tmr", "centroides", "b@91", "t@91"),     # vertical a x=91 mm
    ("clasif", "qpcr", "b@114", "t@114"),   # verticales: salen del borde inferior de "Clasificación"
    ("clasif", "sim", "b@128", "t@128"),    # y entran por arriba a cada caja del carril inferior
]
LANE_LABELS = [  # (y_centro, texto)
    (12.5, "Fundamento\nregulatorio"), (47.0, "Flujo\nprincipal"), (88.5, "Hacia la\nclínica"),
]
LANE_SEP_Y = [27.0, 75.5]  # líneas finas que separan carriles

# Pie de figura acordado (2026-09-14); reemplaza al pie provisional del docx.
PIE_DE_FIGURA = (
    "Figura 1. Flujo del estudio. Carril central: transcriptomas públicos (TCGA, GSE39582), selección de diez genes "
    "con cuatro criterios (discriminación estadística en dos cohortes, red de coexpresión, literatura, factibilidad de "
    "RT-qPCR), centroides por subtipo congelados en GSE39582, clasificación por correlación y validación pronóstica en "
    "cinco cohortes externas (Cox estratificado, validación dejando una cohorte fuera, ajuste por estadio y MMR). "
    "Carril superior: inferencia de reguladores maestros por subtipo (ARACNe3 y NaRnEA en TCGA y microarreglo, "
    "meta-análisis) que fundamenta el panel. Carril inferior, en desarrollo: puente RT-qPCR con genes de referencia y "
    "marco de simulación sobre el mismo espacio de diez dimensiones."
)


def side_point(b, side):
    """Punto medio de un lado de la caja; 'b@56' = lado inferior en x=56 mm."""
    at = None
    if "@" in side:
        side, at = side.split("@"); at = float(at)
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    if side == "r": return (x + w, y + h / 2)
    if side == "l": return (x, y + h / 2)
    if side == "t": return (at if at is not None else x + w / 2, y)
    if side == "b": return (at if at is not None else x + w / 2, y + h)
    raise ValueError(side)


def draw(numbered=False):
    fig = plt.figure(figsize=(mm(FIG_W), mm(FIG_H)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W); ax.set_ylim(FIG_H, 0)  # y hacia abajo, en mm
    ax.set_aspect("equal"); ax.axis("off")

    # separadores y etiquetas de carril
    for y in LANE_SEP_Y:
        ax.plot([1, FIG_W - 1], [y, y], color="#E0E0E0", lw=0.5, zorder=0)
    for y, lab in LANE_LABELS:
        ax.text(2.0, y, lab, rotation=90, ha="left", va="center", fontsize=FONT_MIN, color=GRAY, linespacing=1.0)

    # cajas
    for i, (key, b) in enumerate(BOXES.items(), start=1):
        solid = b["lane"] == "main"
        patch = FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0,rounding_size=1.6",
                               linewidth=0.8 if solid else 0.7, edgecolor=GRAY_DARK if solid else GRAY,
                               facecolor="#F4F4F4" if solid else "white",
                               linestyle="-" if solid else (0, (2.0, 1.6)), zorder=2)
        ax.add_patch(patch)
        lines = b["text"].split("\n")
        # primera línea en negrita, resto normal
        txt = ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["text"], ha="center", va="center",
                      fontsize=FONT_MIN, color=INK, wrap=True, linespacing=1.15, zorder=3)
        # negrita en la primera línea vía un texto superpuesto: matplotlib no mezcla pesos en un mismo Text,
        # así que dibujamos el título aparte y el cuerpo debajo.
        txt.remove()
        title, body = lines[0], "\n".join(lines[1:])
        n_lines = len(lines)
        lh = FONT_MIN * 1.15 * 0.3528  # altura de línea en mm (1 pt = 0.3528 mm)
        y_top = b["y"] + b["h"] / 2 - (n_lines * lh) / 2
        ax.text(b["x"] + b["w"] / 2, y_top + lh / 2, title, ha="center", va="center", fontsize=FONT_MIN,
                fontweight="bold", color=INK, zorder=3)
        if body:
            ax.text(b["x"] + b["w"] / 2, y_top + lh * 1.05, body, ha="center", va="top", fontsize=FONT_MIN,
                    color=INK, wrap=True, linespacing=1.15, zorder=3)
        if b.get("dev"):
            ax.text(b["x"] + b["w"] - 1.2, b["y"] + 1.0, "en desarrollo", ha="right", va="top", fontsize=FONT_MIN,
                    fontstyle="italic", color=GRAY, zorder=3)
        if numbered:
            ax.add_patch(Circle((b["x"] + 2.6, b["y"] + 2.6), 2.2, facecolor="#D55E00", edgecolor="white", lw=0.6, zorder=5))
            ax.text(b["x"] + 2.6, b["y"] + 2.6, str(i), ha="center", va="center", fontsize=FONT_MIN, color="white",
                    fontweight="bold", zorder=6)

    # genes bajo la caja de selección, por eje y color
    for k, c in enumerate(CMS_ORDER):
        y = GENES_POS["y"] + k * GENES_POS["dy"]
        ax.text(GENES_POS["x"], y, f"{c}", ha="left", va="center", fontsize=FONT_MIN, fontweight="bold",
                color=CMS_COLORS[c])
        ax.text(GENES_POS["x"] + 9.5, y, "  ".join(PANEL_GENES[c]), ha="left", va="center", fontsize=FONT_MIN,
                fontstyle="italic", color=CMS_COLORS[c])

    # flechas
    for src, dst, s_side, d_side in ARROWS:
        p0 = side_point(BOXES[src], s_side); p1 = side_point(BOXES[dst], d_side)
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=GRAY_DARK, lw=0.7, shrinkA=0.5, shrinkB=0.5,
                                    mutation_scale=7), zorder=4)
    if numbered:
        ax.text(FIG_W - 1, FIG_H - 0.3, "versión numerada para revisión del layout (no es la figura final)",
                ha="right", va="bottom", fontsize=FONT_MIN, color="#D55E00")
    return fig


def main(argv=None):
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--numbered", action="store_true", help="Sólo el PNG numerado para revisar el layout")
    args = p.parse_args(argv)
    paths = resolve_paths(args)
    setup_style()
    written = []
    if args.numbered:
        fig = draw(numbered=True)
        out = Path(paths["outdir"]); out.mkdir(parents=True, exist_ok=True)
        f = out / "Fig1_flujo_layout_numerado.png"
        fig.savefig(f, dpi=300, facecolor="white"); written.append(f)
        print("Cajas numeradas:")
        for i, (k, b) in enumerate(BOXES.items(), start=1):
            print(f"  {i}. {k:10s} x={b['x']:.0f} y={b['y']:.0f} w={b['w']:.0f} h={b['h']:.0f}  '{b['text'].splitlines()[0]}'")
    else:
        fig = draw(numbered=False)
        written += save_figure(fig, "Fig1_flujo", paths["outdir"], also_svg=args.svg)
        pie_path = Path(paths["outdir"]) / "Fig1_pie_de_figura.md"
        pie_path.write_text(PIE_DE_FIGURA + "\n", encoding="utf-8")
        written.append(pie_path)
    print("Escrito:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
