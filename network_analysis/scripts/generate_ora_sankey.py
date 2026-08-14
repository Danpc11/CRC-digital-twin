"""
Static, non-interactive publication figure: Sankey diagram of
confidence tier -> panel gene (colored by CMS axis) -> significant
ORA route (GO BP, GO MF, KEGG, Reactome, Hallmark), padj<0.05.

Fixed dark/black theme (no light-mode variant, no JS, no hover) --
meant to be dropped directly into a manuscript/poster as an SVG. For
the interactive, theme-aware exploration version see
ora_sankey_interactive.html (same underlying data, built with the
dataviz skill's HTML/SVG + hover-tooltip method).

Reads directly from the ORA outputs so the figure can't silently drift
from the actual numbers:
  - results/crc_net/ora/ora_summary.tsv          (per gene x database)
  - results/crc_net/neighborhoods_final/panel_neighborhoods_final.tsv (tier/confidence)

CMS-axis assignment is fixed (not stored in any data file) -- see
CLAUDE.md "Objetivo": CMS1 MLH1/GNLY/USP18, CMS2 MYC/AXIN2,
CMS3 FABP1/CPS1/SI, CMS4 VIM/TGFB1.
"""

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ora-summary", required=True, type=Path,
                    help="ora_summary.tsv (from ora_panel_neighborhoods.R)")
    p.add_argument("--neighborhoods-final", required=True, type=Path,
                    help="panel_neighborhoods_final.tsv (from panel_neighborhoods.py)")
    p.add_argument("--out-svg", required=True, type=Path,
                    help="output static publication SVG (dark/black theme)")
    p.add_argument("--out-png", required=True, type=Path,
                    help="output rasterized PNG (via headless Chrome)")
    p.add_argument("--dpi", type=int, default=300,
                    help="PNG raster resolution (default 300)")
    return p.parse_args()

# Headless Chrome is used purely as an SVG-to-PNG rasterizer (no page
# scripting) -- avoids adding a Python image-rendering dependency
# (cairosvg/etc.) just for this one export step.
CHROME_CANDIDATES = [
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if Path(c).is_file():
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


PRINT_DPI = 300  # final raster resolution
CHROME_DEFAULT_DPI = 96  # Chrome's CSS-px-to-physical-inch assumption


def export_png(svg_path, png_path, width_in, height_in, dpi=PRINT_DPI):
    chrome = find_chrome()
    if not chrome:
        print("WARNING: no headless Chrome/Chromium found -- skipping PNG export. "
              f"SVG is still at {svg_path}; rasterize it manually if needed.")
        return
    css_w = round(width_in * CHROME_DEFAULT_DPI)
    css_h = round(height_in * CHROME_DEFAULT_DPI)
    scale = dpi / CHROME_DEFAULT_DPI
    subprocess.run([
        chrome, "--headless", "--disable-gpu",
        f"--force-device-scale-factor={scale}",
        f"--window-size={css_w},{css_h}",
        f"--screenshot={png_path}",
        "--default-background-color=00000000",
        svg_path.resolve().as_uri(),
    ], check=True, capture_output=True)
    print(f"Wrote bitmap export: {png_path} "
          f"({round(css_w * scale)}x{round(css_h * scale)}px @ {dpi}dpi, {width_in:.2f}x{height_in:.2f}in)")

CMS_AXIS = {
    "MLH1": 1, "GNLY": 1, "USP18": 1,
    "MYC": 2, "AXIN2": 2,
    "FABP1": 3, "CPS1": 3, "SI": 3,
    "VIM": 4, "TGFB1": 4,
}
CMS_LABEL = {
    1: "CMS1 · immune / MSI",
    2: "CMS2 · canonical / Wnt",
    3: "CMS3 · metabolic / differentiation",
    4: "CMS4 · mesenchymal / EMT",
}
CMS_COLOR = {1: "#3987e5", 2: "#d95926", 3: "#199e70", 4: "#c98500"}  # dark-mode categorical slots 1-4

DB_LABEL = {"GO_BP": "GO BP", "GO_MF": "GO MF", "KEGG": "KEGG", "Reactome": "Reactome", "Hallmark": "Hallmark"}
DB_ORDER = ["GO_BP", "GO_MF", "KEGG", "Reactome", "Hallmark"]

TIER_ORDER = ["alta", "baja", "baja_nm"]
TIER_LABEL = {"alta": "High", "baja": "Low", "baja_nm": "Low, no module"}
TIER_SUB = {
    "alta": "Infomap ∩ Leiden consensus",
    "baja": "Infomap ∪ Leiden union",
    "baja_nm": "direct neighbors, min-count 2",
}
NO_SIGNIFICANT_TERM = "No significant term"
CONFIDENCE_TO_TIER = {
    "high": "alta",
    "low (consensus too small/empty)": "baja",
    "low (not module-based)": "baja_nm",
}
NULL_ROUTE_ID = "__null__"

# ---- Colors (fixed dark/black theme) --------------------------------------
BG = "#000000"
SURFACE = "#0d0d0d"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#8f8d86"
TIER_FILL = "#1c1c1a"
ROUTE_FILL = "#232322"
ROUTE_FILL_NULL = "#33322e"
HAIRLINE = "#2c2c2a"
FONT = "Helvetica Neue, Arial, sans-serif"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt_padj(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def main():
    args = parse_args()
    summary = pd.read_csv(args.ora_summary, sep="\t")
    neigh = pd.read_csv(args.neighborhoods_final, sep="\t")

    genes = []
    for _, row in neigh.iterrows():
        gene_id = row["gene"]
        tier = CONFIDENCE_TO_TIER[row["confidence"]]
        routes = []
        gsum = summary[summary["gene"] == gene_id]
        for db in DB_ORDER:
            r = gsum[gsum["database"] == db]
            if r.empty:
                continue
            term = r.iloc[0]["top_term"]
            padj = r.iloc[0]["top_padj"]
            if pd.isna(term) or pd.isna(padj):
                continue
            routes.append({"db": db, "term": term, "padj": float(padj)})
        genes.append({
            "id": gene_id, "cms": CMS_AXIS[gene_id], "tier": tier,
            "n": int(row["n"]), "routes": routes,
        })
    genes.sort(key=lambda g: TIER_ORDER.index(g["tier"]))

    # ---- route nodes (with shared null-result sink) ----
    route_nodes = {}
    route_order = []
    for g in genes:
        if not g["routes"]:
            continue
        for r in g["routes"]:
            rid = f"{g['id']}__{r['db']}"
            route_nodes[rid] = {"id": rid, "term": r["term"], "db": r["db"], "padj": r["padj"], "value": 0.0, "is_null": False}
            route_order.append(rid)
    has_null = any(not g["routes"] for g in genes)
    if has_null:
        route_nodes[NULL_ROUTE_ID] = {"id": NULL_ROUTE_ID, "term": NO_SIGNIFICANT_TERM, "db": None, "padj": None, "value": 0.0, "is_null": True}

    links_tg = [{"source": g["tier"], "target": g["id"], "value": 1.0, "gene": g} for g in genes]

    links_gr = []
    null_genes = []
    for g in genes:
        if not g["routes"]:
            links_gr.append({"source": g["id"], "target": NULL_ROUTE_ID, "value": 1.0, "gene": g, "route": route_nodes[NULL_ROUTE_ID]})
            route_nodes[NULL_ROUTE_ID]["value"] += 1.0
            null_genes.append(g["id"])
        else:
            share = 1.0 / len(g["routes"])
            for r in g["routes"]:
                rid = f"{g['id']}__{r['db']}"
                links_gr.append({"source": g["id"], "target": rid, "value": share, "gene": g, "route": route_nodes[rid], "term_data": r})
                route_nodes[rid]["value"] += share
    if has_null:
        route_order.append(NULL_ROUTE_ID)

    # ---- layout ----
    # Real physical units: 1 user-unit = 1pt, viewBox width fixed to
    # EXACTLY 11in (792pt, landscape letter) per the chosen tradeoff --
    # letter-exact width, and only the 2-3 route labels too long to fit
    # on one line at a legible font wrap to a second line (rather than
    # shrinking every route to fit the single longest one). Height is
    # NOT capped to 8.5in: with 28 distinct route rows there is no font
    # size that is both legible and fits 28 rows in 8.5in, so height is
    # left to be whatever the content needs (a taller print/supplementary
    # page) -- reported to the console (and see CLAUDE.md) rather than
    # silently shrunk.
    PT_PER_IN = 72
    TARGET_WIDTH_IN = 11.0
    W = round(TARGET_WIDTH_IN * PT_PER_IN)  # 792pt

    RIGHT_MARGIN = 16
    X0, COL0_W = 16, 195
    GAP_COL0 = 45
    X1 = X0 + COL0_W + GAP_COL0
    COL1_W = 130
    GAP_COL1 = 45
    X2 = X1 + COL1_W + GAP_COL1
    COL2_W = W - X2 - RIGHT_MARGIN

    ROUTE_FONT, ROUTE_SUB_FONT = 9, 8
    ROUTE_PAD = 12
    CHAR_W = 0.56  # avg glyph width as a fraction of font-size, Helvetica-ish
    CHAR_BUDGET = (COL2_W - 2 * ROUTE_PAD) / (CHAR_W * ROUTE_FONT)

    TIER_HEAD_FONT, TIER_SUB_FONT = 12, 8.5
    GENE_NAME_FONT, GENE_CMS_FONT, GENE_N_FONT = 13, 8.5, 8.5
    HEADER_FONT = 9.5
    LEGEND_FONT = 8.5

    GAP0, GAP1, GAP2 = 10, 6, 6
    PX_PER_UNIT = 88
    NORMAL_MIN_H = 17
    WRAP_MIN_H = 30
    TOP = 24
    BOTTOM_MARGIN = 78

    # Decide, per route, whether it needs a second line (term on line 1,
    # "DB · padj=X" on line 2) -- purely length-driven against the
    # column's actual character budget at ROUTE_FONT, not hardcoded.
    for rid, rn in route_nodes.items():
        if rn["is_null"]:
            rn["line1"] = f'{rn["term"]} — {", ".join(null_genes)}'
            rn["line2"] = None
            rn["wrap"] = False
            continue
        suffix = f'{DB_LABEL[rn["db"]]} · padj={fmt_padj(rn["padj"])}'
        oneline = f'{rn["term"]} — {suffix}'
        if len(oneline) <= CHAR_BUDGET:
            rn["line1"] = oneline
            rn["line2"] = None
            rn["wrap"] = False
        else:
            rn["line1"] = rn["term"]
            rn["line2"] = suffix
            rn["wrap"] = True

    def stack(items, gap, top, min_h_fn=None):
        y = top
        for it in items:
            it["y"] = y
            h = it["value"] * PX_PER_UNIT
            if min_h_fn:
                h = max(h, min_h_fn(it))
            it["h"] = max(h, 4)
            y += it["h"] + gap
        return y - gap

    tier_items = {t: {"id": t, "value": sum(1 for g in genes if g["tier"] == t), "y": 0, "h": 0} for t in TIER_ORDER}
    tier_list = [tier_items[t] for t in TIER_ORDER if tier_items[t]["value"] > 0]
    gene_items = {g["id"]: {"id": g["id"], "value": 1.0, "data": g, "y": 0, "h": 0} for g in genes}
    gene_list = [gene_items[g["id"]] for g in genes]
    route_items = {rid: {"id": rid, "value": route_nodes[rid]["value"], "data": route_nodes[rid], "y": 0, "h": 0} for rid in route_order}
    route_list = [route_items[rid] for rid in route_order]

    bottom0 = stack(tier_list, GAP0, TOP)
    bottom1 = stack(gene_list, GAP1, TOP)
    bottom2 = stack(route_list, GAP2, TOP,
                     min_h_fn=lambda it: WRAP_MIN_H if it["data"]["wrap"] else NORMAL_MIN_H)
    content_bottom = max(bottom0, bottom1, bottom2)
    H = math.ceil(content_bottom + BOTTOM_MARGIN)

    def assign_offsets(links, source_map, target_map):
        src_cursor, tgt_cursor = {}, {}
        for l in links:
            s, t = source_map[l["source"]], target_map[l["target"]]
            so, to = src_cursor.get(l["source"], 0.0), tgt_cursor.get(l["target"], 0.0)
            l["sy0"] = s["y"] + so * PX_PER_UNIT
            l["sy1"] = l["sy0"] + l["value"] * PX_PER_UNIT
            l["ty0"] = t["y"] + to * PX_PER_UNIT
            l["ty1"] = l["ty0"] + l["value"] * PX_PER_UNIT
            src_cursor[l["source"]] = so + l["value"]
            tgt_cursor[l["target"]] = to + l["value"]

    assign_offsets(links_tg, tier_items, gene_items)
    assign_offsets(links_gr, gene_items, route_items)

    def link_path(x0, y0a, y0b, x1, y1a, y1b):
        mx = (x0 + x1) / 2
        return (f"M{x0},{y0a} C{mx},{y0a} {mx},{y1a} {x1},{y1a} "
                f"L{x1},{y1b} C{mx},{y1b} {mx},{y0b} {x0},{y0b} Z")

    # ---- SVG assembly ----
    width_in = W / PT_PER_IN
    height_in = H / PT_PER_IN
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
                f'width="{width_in:.2f}in" height="{height_in:.2f}in" font-family="{FONT}">')
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{BG}"/>')

    # Column headers
    svg.append(f'<text x="{X0}" y="{TOP - 8}" font-size="{HEADER_FONT}" font-weight="700" fill="{INK_MUTED}" '
                f'letter-spacing="0.03em">NEIGHBORHOOD CONFIDENCE</text>')
    svg.append(f'<text x="{X1}" y="{TOP - 8}" font-size="{HEADER_FONT}" font-weight="700" fill="{INK_MUTED}" '
                f'letter-spacing="0.03em">PANEL GENE (CMS AXIS)</text>')
    svg.append(f'<text x="{X2}" y="{TOP - 8}" font-size="{HEADER_FONT}" font-weight="700" fill="{INK_MUTED}" '
                f'letter-spacing="0.03em">RECOVERED BIOLOGICAL PATHWAY</text>')

    # Links (draw before nodes so nodes sit on top)
    for l in links_tg:
        color = CMS_COLOR[l["gene"]["cms"]]
        d = link_path(X0 + COL0_W, l["sy0"], l["sy1"], X1, l["ty0"], l["ty1"])
        svg.append(f'<path d="{d}" fill="{color}" opacity="0.18"/>')
    for l in links_gr:
        color = CMS_COLOR[l["gene"]["cms"]]
        op = 0.24 if l["route"]["is_null"] else 0.34
        d = link_path(X1 + COL1_W, l["sy0"], l["sy1"], X2, l["ty0"], l["ty1"])
        svg.append(f'<path d="{d}" fill="{color}" opacity="{op}"/>')

    # Tier nodes
    for it in tier_list:
        tid = it["id"]
        svg.append(f'<rect x="{X0}" y="{it["y"]:.1f}" width="{COL0_W}" height="{it["h"]:.1f}" rx="5" fill="{TIER_FILL}"/>')
        mid = it["y"] + it["h"] / 2
        svg.append(f'<text x="{X0 + 12}" y="{mid - 4:.1f}" font-size="{TIER_HEAD_FONT}" font-weight="700" fill="{INK_PRIMARY}">'
                    f'{esc(TIER_LABEL[tid])} · n={it["value"]:.0f}</text>')
        svg.append(f'<text x="{X0 + 12}" y="{mid + 12:.1f}" font-size="{TIER_SUB_FONT}" fill="{INK_SECONDARY}">'
                    f'{esc(TIER_SUB[tid])}</text>')

    # Gene nodes
    for it in gene_list:
        gdata = it["data"]
        color = CMS_COLOR[gdata["cms"]]
        svg.append(f'<rect x="{X1}" y="{it["y"]:.1f}" width="{COL1_W}" height="{it["h"]:.1f}" rx="5" fill="{color}"/>')
        mid = it["y"] + it["h"] / 2
        svg.append(f'<text x="{X1 + 10}" y="{mid - 4:.1f}" font-size="{GENE_NAME_FONT}" font-weight="700" fill="#ffffff">'
                    f'{esc(gdata["id"])}</text>')
        svg.append(f'<text x="{X1 + COL1_W - 9}" y="{mid - 4:.1f}" font-size="{GENE_CMS_FONT}" font-weight="600" '
                    f'fill="#ffffff" opacity="0.85" text-anchor="end">CMS{gdata["cms"]}</text>')
        svg.append(f'<text x="{X1 + 10}" y="{mid + 12:.1f}" font-size="{GENE_N_FONT}" fill="#ffffff" opacity="0.75">'
                    f'n={gdata["n"]}</text>')

    # Route nodes -- single line normally; the 2-3 whose term+stats text
    # doesn't fit CHAR_BUDGET at ROUTE_FONT wrap to a second (muted,
    # smaller) line instead of being shrunk to fit the longest one.
    for it in route_list:
        rdata = it["data"]
        fill = ROUTE_FILL_NULL if rdata["is_null"] else ROUTE_FILL
        svg.append(f'<rect x="{X2}" y="{it["y"]:.1f}" width="{COL2_W}" height="{it["h"]:.1f}" rx="5" fill="{fill}"/>')
        mid = it["y"] + it["h"] / 2
        if rdata["wrap"]:
            svg.append(f'<text x="{X2 + ROUTE_PAD}" y="{mid - 3.5:.1f}" font-size="{ROUTE_FONT}" fill="{INK_PRIMARY}">'
                        f'{esc(rdata["line1"])}</text>')
            svg.append(f'<text x="{X2 + ROUTE_PAD}" y="{mid + 10.5:.1f}" font-size="{ROUTE_SUB_FONT}" fill="{INK_SECONDARY}">'
                        f'{esc(rdata["line2"])}</text>')
        else:
            svg.append(f'<text x="{X2 + ROUTE_PAD}" y="{mid + 3:.1f}" font-size="{ROUTE_FONT}" fill="{INK_PRIMARY}">'
                        f'{esc(rdata["line1"])}</text>')

    # Legend -- two rows (CMS axes, then node-type key).
    CHAR_W_LEGEND = 0.56 * LEGEND_FONT
    SW = 10  # legend swatch side

    def legend_row(y, items):
        lx = X0
        for fill, text, stroke in items:
            stroke_attr = f' stroke="{stroke}"' if stroke else ""
            svg.append(f'<rect x="{lx}" y="{y - SW + 2:.1f}" width="{SW}" height="{SW}" rx="2" fill="{fill}"{stroke_attr}/>')
            svg.append(f'<text x="{lx + SW + 6}" y="{y:.1f}" font-size="{LEGEND_FONT}" fill="{INK_SECONDARY}">{esc(text)}</text>')
            lx += SW + 6 + CHAR_W_LEGEND * len(text) + 24

    ly1 = content_bottom + 34
    ly2 = ly1 + 22
    svg.append(f'<line x1="{X0}" y1="{ly1 - SW - 6:.1f}" x2="{W - RIGHT_MARGIN}" y2="{ly1 - SW - 6:.1f}" stroke="{HAIRLINE}" stroke-width="1"/>')
    legend_row(ly1, [(CMS_COLOR[c], CMS_LABEL[c], None) for c in (1, 2, 3, 4)])
    legend_row(ly2, [
        (TIER_FILL, "neighborhood confidence", HAIRLINE),
        (ROUTE_FILL, "biological pathway (padj<0.05)", HAIRLINE),
        (ROUTE_FILL_NULL, NO_SIGNIFICANT_TERM.lower(), HAIRLINE),
    ])

    svg.append("</svg>")

    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.write_text("\n".join(svg), encoding="utf-8")
    n_wrapped = sum(1 for rn in route_nodes.values() if rn["wrap"])
    print(f"Wrote static publication figure: {args.out_svg} "
          f"({width_in:.2f}x{height_in:.2f}in, {W}x{H}pt, {n_wrapped} wrapped route label(s), "
          f"route font {ROUTE_FONT}pt)")

    export_png(args.out_svg, args.out_png, width_in, height_in, dpi=args.dpi)


if __name__ == "__main__":
    main()
