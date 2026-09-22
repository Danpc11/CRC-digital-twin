"""Estilo y utilidades compartidas por las figuras del trabajo Rosenkranz 2026 (ColoQ).

Convenciones (comunes a todas las figuras):
- Tipografía Arial (Helvetica / Liberation Sans como respaldo), texto >= 7 pt al ancho final.
- Una sola paleta para los cuatro CMS, apta para daltonismo (Okabe-Ito):
  CMS1 naranja, CMS2 azul, CMS3 verde azulado, CMS4 bermellón. Gris neutro para todo lo demás.
- Etiquetas de panel A, B, C en negrita; sin títulos dentro de la figura.
- Ancho de columna doble: 180 mm. Salida: PDF vectorial (fuentes TrueType embebidas) + PNG 300 dpi.

Rutas: los scripts se corren desde cualquier directorio; el repo se resuelve como el padre de
`figures/`. Los insumos que no viven en el repo (respaldo_v0.2.0 del Drive, crc_mra_results)
se buscan en una lista de ubicaciones conocidas y pueden sobreescribirse por CLI.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Paleta
# --------------------------------------------------------------------------------------
CMS_ORDER = ["CMS1", "CMS2", "CMS3", "CMS4"]
CMS_COLORS = {
    "CMS1": "#E69F00",  # naranja
    "CMS2": "#0072B2",  # azul
    "CMS3": "#009E73",  # verde azulado
    "CMS4": "#D55E00",  # bermellón
}
CMS_LONG = {  # nombres internos del pipeline -> etiqueta corta
    "CMS1_MSI_immune": "CMS1",
    "CMS2_canonical_WNT": "CMS2",
    "CMS3_metabolic": "CMS3",
    "CMS4_mesenchymal": "CMS4",
}
CMS_SUBTITLE = {"CMS1": "MSI/inmune", "CMS2": "canónico", "CMS3": "metabólico", "CMS4": "mesenquimal"}
GRAY = "#7F7F7F"
GRAY_LIGHT = "#C8C8C8"
GRAY_DARK = "#404040"
INK = "#1A1A1A"

# Paleta divergente para z (heatmap) y para dirección de hallmarks
DIVERGING_CMAP = "RdBu_r"
UP_COLOR = "#B2182B"
DOWN_COLOR = "#2166AC"

# Panel v0.2.0, agrupado por eje CMS
PANEL_GENES = {
    "CMS1": ["MLH1", "GNLY", "USP18"],
    "CMS2": ["MYC", "AXIN2"],
    "CMS3": ["GALNT8", "CPS1", "AGR2"],
    "CMS4": ["VIM", "EFEMP2"],
}
REPLACED_GENES = {"FABP1": "CMS3", "SI": "CMS3", "TGFB1": "CMS4"}  # panel previo, se muestran en gris

# --------------------------------------------------------------------------------------
# Tipografía y rcParams
# --------------------------------------------------------------------------------------
FONT_MIN = 7.0
FONT_PANEL = 10.0


def setup_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_MIN,
        "axes.labelsize": FONT_MIN,
        "axes.titlesize": FONT_MIN,
        "xtick.labelsize": FONT_MIN,
        "ytick.labelsize": FONT_MIN,
        "legend.fontsize": FONT_MIN,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # TrueType embebido, texto editable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": GRAY_DARK,
    })


def mm(x: float) -> float:
    """milímetros -> pulgadas (para figsize)."""
    return x / 25.4


def panel_label(fig, x_mm: float, y_mm: float, letter: str, width_mm: float, height_mm: float) -> None:
    """Etiqueta de panel en negrita en coordenadas de figura dadas en mm desde la esquina superior izquierda."""
    fig.text(x_mm / width_mm, 1 - y_mm / height_mm, letter, fontsize=FONT_PANEL, fontweight="bold",
             ha="left", va="top", color=INK)


def add_axes_mm(fig, left: float, top: float, width: float, height: float, fig_w: float, fig_h: float, **kw):
    """Crea un eje con posición en mm (origen arriba-izquierda) sobre una figura de fig_w x fig_h mm."""
    return fig.add_axes([left / fig_w, 1 - (top + height) / fig_h, width / fig_w, height / fig_h], **kw)


def save_figure(fig, stem: str, outdir: Path, also_svg: bool = False) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in (["pdf", "png"] + (["svg"] if also_svg else [])):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, facecolor="white")
        written.append(p)
    return written


# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT.parent  # CRC-dt/ (contiene CRC-digital-twin/ y network_analysis/)

DRIVE_PREMIO = Path.home() / "Library/CloudStorage/GoogleDrive-hachepunto@gmail.com/Mi unidad/premio_Roche"

RESPALDO_CANDIDATES = [
    DRIVE_PREMIO / "respaldo_v0.2.0",
    REPO_ROOT / "respaldo_v0.2.0",
]
MRA_CANDIDATES = [
    Path("/STORAGE/genut/calixto/crc_mra_results"),
    DRIVE_PREMIO / "respaldo_v0.2.0/crc_mra_results",
    REPO_ROOT / "respaldo_v0.2.0/crc_mra_results",
]
PREDICTIVE_PANEL_CANDIDATES = [
    WORK_ROOT / "network_analysis/results/crc_net_577/predictive_panel",
    REPO_ROOT / "network_analysis/results/crc_net_577/predictive_panel",
]
OUTDIR_CANDIDATES = [
    DRIVE_PREMIO / "figuras",
    REPO_ROOT / "figures" / "out",
]


def first_existing(candidates, what: str) -> Path:
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError(f"No encuentro {what}; probé: " + "; ".join(str(c) for c in candidates))


def first_existing_optional(candidates):
    """Como first_existing pero devuelve None en vez de reventar.

    resolve_paths() se llama igual para las cuatro figuras, pero cada una
    consume insumos distintos: solo fig3 lee crc_mra_results y ninguna lee
    respaldo_v0.2.0. Exigirlos a todas hacía que fig2 abortara en una
    máquina sin el Drive montado por un directorio que no iba a abrir.
    Quien de verdad necesite uno debe comprobar que no es None.
    """
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--outdir", type=Path, default=None,
                   help="Directorio de salida (default: gdrive premio_Roche/figuras si existe, si no figures/out)")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Raíz de CRC-digital-twin")
    p.add_argument("--respaldo", type=Path, default=None, help="Directorio respaldo_v0.2.0 (Drive)")
    p.add_argument("--mra-dir", type=Path, default=None, help="Directorio crc_mra_results")
    p.add_argument("--predictive-panel", type=Path, default=None,
                   help="network_analysis/results/crc_net_577/predictive_panel")
    p.add_argument("--svg", action="store_true", help="Escribir también SVG")
    return p


def resolve_paths(args) -> dict:
    out = {}
    out["outdir"] = args.outdir or (OUTDIR_CANDIDATES[0] if OUTDIR_CANDIDATES[0].parent.exists() else OUTDIR_CANDIDATES[1])
    out["repo"] = Path(args.repo_root)
    out["respaldo"] = args.respaldo or first_existing_optional(RESPALDO_CANDIDATES)
    out["mra"] = args.mra_dir or first_existing_optional(MRA_CANDIDATES)
    out["predictive_panel"] = args.predictive_panel or first_existing_optional(PREDICTIVE_PANEL_CANDIDATES)
    return out


def require_path(path, what: str, candidates) -> Path:
    """Falla con el mensaje útil justo donde el insumo hace falta de verdad."""
    if path is None:
        raise FileNotFoundError(
            f"No encuentro {what}; probé: " + "; ".join(str(c) for c in candidates)
            + f". Pásalo explícitamente si está en otro sitio.")
    return Path(path)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return f"p={p:.1e}".replace("e-0", "e-")
    if p < 0.01:
        return f"p={p:.4f}".rstrip("0")
    return f"p={p:.3f}".rstrip("0")


def forest_plot(ax, rows, x_label="HR (IC95%)", ref_label=None, xlim=(0.4, 6.0), text_x=None, show_p=False):
    """Forest plot genérico en escala log.

    rows: lista de dicts con keys: label, hr, lo, hi, p, color (opcional), ref (bool, fila de referencia).
    Dibuja de arriba hacia abajo en el orden dado. Escribe HR (IC) a la derecha.
    """
    n = len(rows)
    ys = list(range(n))[::-1]
    ax.axvline(1.0, color=GRAY, lw=0.6, ls="--", zorder=1)
    for y, r in zip(ys, rows):
        col = r.get("color", GRAY_DARK)
        if r.get("ref"):
            ax.plot([1.0], [y], marker="D", ms=3.5, mfc="white", mec=col, mew=0.8, zorder=3)
            txt = "1 (referencia)"
        else:
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([r["hr"]], [y], marker="s", ms=3.8, color=col, zorder=3)
            txt = f"{r['hr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})"
            if show_p and r.get("p") is not None:
                txt += "  " + fmt_p(r["p"])
        ax.text(text_x if text_x is not None else xlim[1] * 1.08, y, txt, va="center", ha="left",
                fontsize=FONT_MIN, color=INK, clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r.get("color") and r.get("bold_label", True):
            tick.set_fontweight("bold")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ticks = [0.5, 1, 2, 4]
    ax.set_xticks([t for t in ticks if xlim[0] <= t <= xlim[1]])
    ax.set_xticklabels([str(t) for t in ticks if xlim[0] <= t <= xlim[1]])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(x_label)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
