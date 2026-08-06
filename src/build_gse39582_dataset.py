"""
build_gse39582_dataset.py

Fusion final: expresion de GSE39582 (probes Affymetrix) + anotacion de
plataforma GPL570 (probe -> simbolo de gen) + etiquetas CMS del
consorcio (Synapse, ya descargadas) + supervivencia libre de recidiva
real (rfs.delay/rfs.event, ya confirmado presente en el phenotype de
GEO) -- el TSV final en el esquema que espera calibration.py.

REQUIERE (en este orden):
    1. src/download_synapse_data.py  (trae cms_labels_public_all.txt)
    2. curl del series_matrix de GSE39582 + src/parse_geo_series_matrix.py
       (trae gse39582_phenotype.tsv y gse39582_expression_probes.tsv)
    3. curl de GPL570.txt (registro propio de la plataforma, texto plano):
       curl -L -o data/raw_geo/GPL570.txt \\
         "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=self&acc=GPL570&form=text&view=full"
       (NO usar GPL570_family.soft.gz -- ese trae TODAS las series
       historicas de la plataforma, decenas de GB para GPL570)
"""

from pathlib import Path

import pandas as pd

RAW_GEO = Path(__file__).resolve().parents[1] / "data" / "raw_geo"
RAW_SYNAPSE = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gse39582_cms_labeled.tsv"

TARGET_SYMBOLS = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}

CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"


def parse_platform_annotation(path) -> pd.DataFrame:
    """
    Parsea el registro SOFT propio de la plataforma (texto plano, NO
    gzip -- descargado via acc.cgi?targ=self&form=text&view=full, no
    via el .annot.gz deprecado ni el _family.soft.gz que trae TODAS
    las series historicas de la plataforma -- ese ultimo puede pesar
    decenas de GB para plataformas populares como GPL570, no es lo que
    queremos). Misma estructura de tabla que el series_matrix:
    '!platform_table_begin' / '!platform_table_end'.
    """
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
    annot = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", names=header)
    return annot


def main():
    expr_path = RAW_GEO / "gse39582_expression_probes.tsv"
    pheno_path = RAW_GEO / "gse39582_phenotype.tsv"
    annot_path = RAW_GEO / "GPL570.txt"
    labels_path = RAW_SYNAPSE / "tcga_cms_labels" / "cms_labels_public_all.txt"

    for p in (expr_path, pheno_path, annot_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(f"No se encontro {p}. Revisa los pasos previos en el docstring.")

    print("Cargando anotacion de plataforma GPL570...")
    annot = parse_platform_annotation(annot_path)
    symbol_col_candidates = [c for c in annot.columns if "symbol" in c.lower()]
    if not symbol_col_candidates:
        raise ValueError(
            f"No se encontro columna de simbolo de gen en la anotacion. "
            f"Columnas disponibles: {list(annot.columns)}"
        )
    symbol_col = symbol_col_candidates[0]
    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    probe_to_symbol = annot.set_index(id_col)[symbol_col]

    print("Cargando expresion (probes)...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    print(f"Mapeando probes a simbolos para el panel objetivo: {TARGET_SYMBOLS}")
    gene_data = {}
    for symbol in TARGET_SYMBOLS:
        matching_probes = probe_to_symbol[probe_to_symbol == symbol].index
        matching_probes = [p for p in matching_probes if p in expr.index]
        if not matching_probes:
            print(f"  AVISO: {symbol} no tiene ningun probe mapeado en GPL570 -- se omite")
            continue
        # si varios probes mapean al mismo gen (comun en Affymetrix),
        # promediar -- practica estandar cuando no hay razon a priori
        # para preferir un probe sobre otro
        gene_data[symbol] = expr.loc[matching_probes].mean(axis=0)
        if len(matching_probes) > 1:
            print(f"  {symbol}: promediado sobre {len(matching_probes)} probes ({matching_probes})")

    if not gene_data:
        raise ValueError("Ningun gen del panel objetivo se pudo mapear. Revisa la anotacion.")

    gene_expr = pd.DataFrame(gene_data)  # index = sample GSM IDs, columns = genes

    print("Cargando fenotipo (RFS) y etiquetas CMS...")
    pheno = pd.read_csv(pheno_path, sep="\t", index_col=0)
    rfs_cols = [c for c in pheno.columns if "rfs" in c.lower()]
    print(f"  Columnas RFS encontradas: {rfs_cols}")

    labels = pd.read_csv(labels_path, sep="\t")
    gse_labels = labels[labels["dataset"] == "gse39582"].set_index("sample")

    if CMS_LABEL_COLUMN not in gse_labels.columns:
        raise ValueError(f"Columna '{CMS_LABEL_COLUMN}' no encontrada en las etiquetas CMS")

    print("Fusionando...")
    merged = gene_expr.join(gse_labels[[CMS_LABEL_COLUMN]], how="inner")
    n_with_cms = len(merged)

    merged = merged.join(
        pheno[["characteristics__rfs.delay", "characteristics__rfs.event"]], how="left"
    )
    merged = merged.rename(columns={
        CMS_LABEL_COLUMN: "cms_label",
        "characteristics__rfs.delay": "relapse_free_months",
        "characteristics__rfs.event": "relapse_event",
    })
    merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    # rfs.delay/rfs.event vienen como texto ('NA' para faltantes) del
    # parser de series_matrix -- convertir a numerico
    merged["relapse_free_months"] = pd.to_numeric(merged["relapse_free_months"], errors="coerce")
    merged["relapse_event"] = pd.to_numeric(merged["relapse_event"], errors="coerce")

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    n_missing_survival = merged["relapse_free_months"].isna().sum()
    print(
        f"\n{n_with_cms} muestras con expresion + CMS. "
        f"{n_missing_survival} sin dato de RFS (se excluiran automaticamente "
        "en survival_validation.py via dropna)."
    )
    print("\nDistribucion de subtipo CMS:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"\nGuardado: {OUTPUT_PATH} ({len(merged)} muestras, {len(gene_data)} genes)")
    print(
        "\nEste archivo usa 'relapse_free_months'/'relapse_event' (RFS real, "
        "no OS) -- es el endpoint correcto segun la literatura original de "
        "CMS. Correr: python3 run_pipeline.py --input data/gse39582_cms_labeled.tsv --output results_gse39582/"
    )


if __name__ == "__main__":
    main()
