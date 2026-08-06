"""
build_gse17537_dataset.py

Fusion para GSE17537 (cohorte de validacion EXTERNA, Smith et al.) --
mismo flujo que build_gse39582_dataset.py, misma plataforma GPL570
(reutiliza la anotacion ya descargada), pero columnas de supervivencia
distintas: esta cohorte trae 'dfs_time'/'dfs_event' (disease-free
survival) en vez de 'rfs.delay'/'rfs.event'. Mismo endpoint conceptual
(recidiva), nombre de campo distinto en la fuente -- confirmado via
inspeccion real de gse17537_phenotype.tsv, no asumido.

REQUIERE:
    1. curl del series_matrix de GSE17537 + parse_series_matrix()
       (trae gse17537_phenotype.tsv y gse17537_expression_probes.tsv)
    2. data/raw_geo/GPL570.txt (ya descargado para GSE39582, misma plataforma)
    3. data/raw_synapse/tcga_cms_labels/cms_labels_public_all.txt (ya descargado)

IMPORTANTE: este archivo se usa con external_validation.py, que aplica
los patrones YA CALIBRADOS con GSE39582 -- este script NO debe usarse
con run_pipeline.py para recalibrar, eso invalidaria la validacion
externa.
"""

from pathlib import Path

import pandas as pd

RAW_GEO = Path(__file__).resolve().parents[1] / "data" / "raw_geo"
RAW_SYNAPSE = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gse17537_cms_labeled.tsv"

TARGET_SYMBOLS = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}

CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"

# Nombres reales confirmados en gse17537_phenotype.tsv (distintos a
# GSE39582 -- no asumir que son iguales entre cohortes)
DFS_TIME_COL = "characteristics__dfs_time"
DFS_EVENT_COL = "characteristics__dfs_event (disease free survival; cancer recurrence)"


def parse_platform_annotation(path) -> pd.DataFrame:
    table_lines = []
    header = None
    in_table = False
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                in_table = False
                continue
            if in_table:
                if header is None:
                    header = line.split("\t")
                else:
                    table_lines.append(line)
    if header is None:
        raise ValueError(f"No se encontro tabla de plataforma en {path}")
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", names=header)


def main():
    expr_path = RAW_GEO / "gse17537_expression_probes.tsv"
    pheno_path = RAW_GEO / "gse17537_phenotype.tsv"
    annot_path = RAW_GEO / "GPL570.txt"
    labels_path = RAW_SYNAPSE / "tcga_cms_labels" / "cms_labels_public_all.txt"

    for p in (expr_path, pheno_path, annot_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(f"No se encontro {p}. Revisa los pasos previos en el docstring.")

    print("Cargando anotacion de plataforma GPL570 (reutilizada de GSE39582)...")
    annot = parse_platform_annotation(annot_path)
    symbol_col = [c for c in annot.columns if "symbol" in c.lower()][0]
    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    probe_to_symbol = annot.set_index(id_col)[symbol_col]

    print("Cargando expresion (probes)...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    print(f"Mapeando probes a simbolos: {TARGET_SYMBOLS}")
    gene_data = {}
    for symbol in TARGET_SYMBOLS:
        matching_probes = probe_to_symbol[probe_to_symbol == symbol].index
        matching_probes = [p for p in matching_probes if p in expr.index]
        if not matching_probes:
            print(f"  AVISO: {symbol} no disponible en esta plataforma/cohorte -- se omite")
            continue
        gene_data[symbol] = expr.loc[matching_probes].mean(axis=0)
        if len(matching_probes) > 1:
            print(f"  {symbol}: promediado sobre {len(matching_probes)} probes")

    if not gene_data:
        raise ValueError("Ningun gen del panel se pudo mapear.")

    gene_expr = pd.DataFrame(gene_data)

    print("Cargando fenotipo (DFS) y etiquetas CMS...")
    pheno = pd.read_csv(pheno_path, sep="\t", index_col=0)
    for col in (DFS_TIME_COL, DFS_EVENT_COL):
        if col not in pheno.columns:
            raise ValueError(f"Columna '{col}' no encontrada. Columnas disponibles: {list(pheno.columns)}")

    labels = pd.read_csv(labels_path, sep="\t")
    gse_labels = labels[labels["dataset"] == "gse17537"].set_index("sample")

    if len(gse_labels) == 0:
        print(
            "\nAVISO: el consorcio CMS no incluyo 'gse17537' en cms_labels_public_all.txt "
            "(GSE17536 si esta, GSE17537 no -- son dos subseries relacionadas del mismo "
            "estudio de Smith et al., pero el consorcio solo etiqueto una). Se continua SIN "
            "etiqueta CMS oficial -- la validacion contra supervivencia real sigue siendo "
            "valida, solo no habra linea base de comparacion para esta cohorte especifica."
        )
        print("Fusionando (sin etiqueta CMS oficial)...")
        merged = gene_expr.join(pheno[[DFS_TIME_COL, DFS_EVENT_COL]], how="left")
        n_with_cms = len(merged)
        # 'none' explicito (no ausencia de columna) -- calibration.py
        # exige la columna cms_label para cargar el TSV, y "none" es
        # honesto: no es que sepamos que no tiene subtipo, es que el
        # consorcio nunca lo etiqueto.
        merged["cms_label"] = "none"
    else:
        print("Fusionando...")
        merged = gene_expr.join(gse_labels[[CMS_LABEL_COLUMN]], how="inner")
        n_with_cms = len(merged)
        merged = merged.join(pheno[[DFS_TIME_COL, DFS_EVENT_COL]], how="left")
        merged = merged.rename(columns={CMS_LABEL_COLUMN: "cms_label"})
        merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    merged = merged.rename(columns={
        DFS_TIME_COL: "relapse_free_months",
        DFS_EVENT_COL: "relapse_event",
    })
    merged["relapse_free_months"] = pd.to_numeric(merged["relapse_free_months"], errors="coerce")

    # relapse_event viene como texto ("recurrence"/"no recurrence"), NO
    # numerico -- confirmado inspeccionando los valores crudos. Mapear
    # explicitamente en vez de pd.to_numeric(), que forzaria todo a NaN
    # silenciosamente (como paso la primera vez: 177/177 NaN sin ningun
    # error ni aviso -- pd.to_numeric con errors='coerce' NO avisa
    # cuando falla, solo devuelve NaN).
    event_map = {"recurrence": 1, "no recurrence": 0}
    unmapped = set(merged["relapse_event"].dropna().unique()) - set(event_map.keys())
    if unmapped:
        raise ValueError(
            f"Valores inesperados en relapse_event no cubiertos por el mapeo: {unmapped}. "
            f"Mapeo actual: {event_map}. Verifica los valores crudos y actualiza event_map."
        )
    merged["relapse_event"] = merged["relapse_event"].map(event_map)

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    n_missing_duration = merged["relapse_free_months"].isna().sum()
    n_missing_event = merged["relapse_event"].isna().sum()
    print(f"\n{n_with_cms} muestras con expresion + CMS. "
          f"{n_missing_duration} sin dato de duracion (DFS time), "
          f"{n_missing_event} sin dato de evento (DFS event).")
    print("\nDistribucion de subtipo CMS:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"\nGuardado: {OUTPUT_PATH} ({len(merged)} muestras, {len(gene_data)} genes)")
    print(
        "\nSiguiente paso -- validacion externa con patrones YA CALIBRADOS de GSE39582 "
        "(NO usar run_pipeline.py con este archivo, eso recalibrarias sobre la cohorte "
        "externa e invalidaria la validacion):\n\n"
        "  python3 src/external_validation.py \\\n"
        "    --patterns results_gse39582_final/calibrated_patterns.tsv \\\n"
        "    --input data/gse17537_cms_labeled.tsv \\\n"
        "    --output results_external_gse17537/"
    )


if __name__ == "__main__":
    main()
