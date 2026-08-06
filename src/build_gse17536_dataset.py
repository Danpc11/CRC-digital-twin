"""
build_gse17536_dataset.py

Fusion para GSE17536 (cohorte de validacion EXTERNA, Smith et al.) --
mismo flujo que build_gse39582_dataset.py, misma plataforma GPL570
(reutiliza la anotacion ya descargada), pero columnas de supervivencia
distintas: esta cohorte trae 'dfs_time'/'dfs_event' (disease-free
survival) en vez de 'rfs.delay'/'rfs.event'. Mismo endpoint conceptual
(recidiva), nombre de campo distinto en la fuente -- confirmado via
inspeccion real de gse17536_phenotype.tsv, no asumido.

REQUIERE:
    1. curl del series_matrix de GSE17536 + parse_series_matrix()
       (trae gse17536_phenotype.tsv y gse17536_expression_probes.tsv)
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
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gse17536_cms_labeled.tsv"

TARGET_SYMBOLS = ["MLH1", "GZMB", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}

CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"

# Nombres reales confirmados en gse17536_phenotype.tsv (distintos a
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
    expr_path = RAW_GEO / "gse17536_expression_probes.tsv"
    pheno_path = RAW_GEO / "gse17536_phenotype.tsv"
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
    gse_labels = labels[labels["dataset"] == "gse17536"].set_index("sample")
    if len(gse_labels) == 0:
        raise ValueError(
            "No se encontraron muestras con dataset=='gse17536' en cms_labels_public_all.txt -- "
            "verifica el nombre exacto del dataset (podria ser 'gse17536' con otra capitalizacion "
            "o no estar incluido en el dump del consorcio)."
        )

    print("Fusionando...")
    merged = gene_expr.join(gse_labels[[CMS_LABEL_COLUMN]], how="inner")
    n_with_cms = len(merged)
    merged = merged.join(pheno[[DFS_TIME_COL, DFS_EVENT_COL]], how="left")

    merged = merged.rename(columns={
        CMS_LABEL_COLUMN: "cms_label",
        DFS_TIME_COL: "relapse_free_months",
        DFS_EVENT_COL: "relapse_event",
    })
    merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)
    merged["relapse_free_months"] = pd.to_numeric(merged["relapse_free_months"], errors="coerce")
    merged["relapse_event"] = pd.to_numeric(merged["relapse_event"], errors="coerce")

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    n_missing = merged["relapse_free_months"].isna().sum()
    print(f"\n{n_with_cms} muestras con expresion + CMS. {n_missing} sin dato de DFS.")
    print("\nDistribucion de subtipo CMS:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"\nGuardado: {OUTPUT_PATH} ({len(merged)} muestras, {len(gene_data)} genes)")
    print(
        "\nSiguiente paso -- validacion externa con patrones YA CALIBRADOS de GSE39582 "
        "(NO usar run_pipeline.py con este archivo, eso recalibrarias sobre la cohorte "
        "externa e invalidaria la validacion):\n\n"
        "  python3 src/external_validation.py \\\n"
        "    --patterns results_gse39582_v2/calibrated_patterns.tsv \\\n"
        "    --input data/gse17536_cms_labeled.tsv \\\n"
        "    --output results_external_gse17536/"
    )


if __name__ == "__main__":
    main()
