"""
build_external_cohort_generic.py

Version generalizada para agregar cohortes CRCSC adicionales (mas alla
de GSE39582/GSE17536/GSE17537) al analisis de validacion externa
combinada. Ya tienen etiqueta CMS oficial en cms_labels_public_all.txt:
    gse2109, gse14333, gse13294, gse37892, gse33113, gse20916,
    gse13067, gse35896, gse23878
(petacc3 y kfsyscc tambien aparecen en el dump del consorcio, pero no
son accesiones GSE literales -- PETACC-3 en particular es un ensayo
clinico europeo que probablemente requiere acceso controlado, no esta
en GEO publico sin mas. Se dejan fuera de este script; investigar
acceso por separado si se quieren incluir.)

A diferencia de los scripts especificos por cohorte (build_gse17536_
dataset.py, etc.), este NO asume plataforma ni nombres de columna de
supervivencia -- las distintas cohortes usan cosas distintas. Flujo de
dos pasos obligatorio:

PASO 1 -- diagnostico (no descarga nada nuevo si ya tienes el series_matrix):
    curl -L -o data/raw_geo/GSE13294_series_matrix.txt.gz \\
      "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/GSE13294_series_matrix.txt.gz"

    python3 src/build_external_cohort_generic.py --gse GSE13294 --diagnose

PASO 2 -- construccion, con los nombres reales de columna que el diagnostico revelo:
    python3 src/build_external_cohort_generic.py --gse GSE13294 \\
        --dataset gse13294 \\
        --duration-col "characteristics__xxx" \\
        --event-col "characteristics__yyy" \\
        --event-map "recurrence=1,no recurrence=0"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW_GEO = Path(__file__).resolve().parents[1] / "data" / "raw_geo"
RAW_SYNAPSE = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"

TARGET_SYMBOLS = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}
CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"


def get_platform_id(path):
    """Escanea el series_matrix por la linea '!Series_platform_id'."""
    import gzip
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if line.startswith("!Series_platform_id"):
                return line.split("\t")[1].strip().strip('"')
    return None


def parse_platform_annotation(path):
    table_lines, header, in_table = [], None, False
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gse", required=True, help="Ej. GSE13294")
    parser.add_argument("--diagnose", action="store_true",
                         help="Solo inspeccionar plataforma y columnas disponibles, sin construir nada")
    parser.add_argument("--dataset", default=None,
                         help="Valor de la columna 'dataset' en cms_labels_public_all.txt (default: gse en minusculas)")
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--event-col", default=None)
    parser.add_argument("--event-map", default=None,
                         help="Mapeo texto->numero si event-col no es ya 0/1, formato 'valorA=1,valorB=0'")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    matrix_path = RAW_GEO / f"{args.gse}_series_matrix.txt.gz"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"No se encontro {matrix_path}. Descargalo primero con curl "
            f"(ver docstring de este script para el patron de URL)."
        )

    from parse_geo_series_matrix import parse_series_matrix
    pheno, expr = parse_series_matrix(matrix_path)

    platform_id = get_platform_id(matrix_path)
    print(f"Plataforma: {platform_id}")
    annot_path = RAW_GEO / f"{platform_id}.txt"
    has_annot = annot_path.exists()
    print(f"Anotacion local ({annot_path.name}): {'encontrada' if has_annot else 'NO encontrada'}")
    if not has_annot:
        print(
            f"  Descargala con:\n"
            f"  curl -L -o {annot_path} \\\n"
            f'    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=self&acc={platform_id}&form=text&view=full"'
        )

    if args.diagnose:
        print(f"\nMuestras: {len(pheno)}")
        print(f"Columnas de fenotipo disponibles:\n{list(pheno.columns)}")
        print(
            "\nBusca columnas relacionadas a supervivencia (rfs/dfs/relapse/recur/event/delay/time/survival) "
            "e identifica cuales usar como --duration-col y --event-col para el paso de construccion."
        )
        return

    if not (args.duration_col and args.event_col):
        raise ValueError("Faltan --duration-col y/o --event-col. Corre primero con --diagnose.")
    if not has_annot:
        raise FileNotFoundError(f"Falta la anotacion de plataforma en {annot_path}. Descargala primero.")

    dataset_name = args.dataset or args.gse.lower()
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}_cms_labeled.tsv"
    )

    print("\nCargando anotacion de plataforma...")
    annot = parse_platform_annotation(annot_path)
    symbol_col_candidates = [c for c in annot.columns if "symbol" in c.lower()]
    if not symbol_col_candidates:
        raise ValueError(f"No se encontro columna de simbolo de gen. Columnas: {list(annot.columns)}")
    symbol_col = symbol_col_candidates[0]
    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    probe_to_symbol = annot.set_index(id_col)[symbol_col]

    print(f"Mapeando probes a simbolos: {TARGET_SYMBOLS}")
    gene_data = {}
    for symbol in TARGET_SYMBOLS:
        matching_probes = probe_to_symbol[probe_to_symbol == symbol].index
        matching_probes = [p for p in matching_probes if p in expr.index]
        if not matching_probes:
            print(f"  AVISO: {symbol} no disponible en esta plataforma -- se omite")
            continue
        gene_data[symbol] = expr.loc[matching_probes].mean(axis=0)
        if len(matching_probes) > 1:
            print(f"  {symbol}: promediado sobre {len(matching_probes)} probes")

    if not gene_data:
        raise ValueError("Ningun gen del panel se pudo mapear en esta plataforma.")

    gene_expr = pd.DataFrame(gene_data)

    labels_path = RAW_SYNAPSE / "tcga_cms_labels" / "cms_labels_public_all.txt"
    labels = pd.read_csv(labels_path, sep="\t")
    gse_labels = labels[labels["dataset"] == dataset_name].set_index("sample")
    print(f"\nEtiquetas CMS para dataset='{dataset_name}': {len(gse_labels)} muestras")

    if len(gse_labels) == 0:
        print("AVISO: sin etiqueta CMS oficial -- se continua con cms_label='none' para todas las muestras.")
        merged = gene_expr.join(pheno[[args.duration_col, args.event_col]], how="left")
        merged["cms_label"] = "none"
    else:
        merged = gene_expr.join(gse_labels[[CMS_LABEL_COLUMN]], how="inner")
        merged = merged.join(pheno[[args.duration_col, args.event_col]], how="left")
        merged = merged.rename(columns={CMS_LABEL_COLUMN: "cms_label"})
        merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    merged = merged.rename(columns={
        args.duration_col: "relapse_free_months",
        args.event_col: "relapse_event",
    })

    # GEO a veces codifica valores faltantes como texto literal ("NA",
    # "N/A", etc.) en vez de celda vacia -- normalizar a NaN real ANTES
    # de mapear/validar, o el validador de --event-map los marca como
    # "no cubiertos" y truena innecesariamente.
    NA_TOKENS = {"NA", "N/A", "n/a", "na", "NaN", "nan", ""}
    merged["relapse_free_months"] = merged["relapse_free_months"].replace(NA_TOKENS, pd.NA)
    merged["relapse_event"] = merged["relapse_event"].replace(NA_TOKENS, pd.NA)

    merged["relapse_free_months"] = pd.to_numeric(merged["relapse_free_months"], errors="coerce")

    if args.event_map:
        event_map = dict(pair.split("=") for pair in args.event_map.split(","))
        event_map = {k: int(v) for k, v in event_map.items()}
        unmapped = set(merged["relapse_event"].dropna().unique()) - set(event_map.keys())
        if unmapped:
            raise ValueError(f"Valores no cubiertos por --event-map: {unmapped}. Mapeo actual: {event_map}")
        merged["relapse_event"] = merged["relapse_event"].map(event_map)
    else:
        merged["relapse_event"] = pd.to_numeric(merged["relapse_event"], errors="coerce")

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    print(f"\n{len(merged)} muestras con expresion. "
          f"{merged['relapse_free_months'].isna().sum()} sin duracion, "
          f"{merged['relapse_event'].isna().sum()} sin evento.")
    print("\nDistribucion CMS:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(output_path, sep="\t", index=False)
    print(f"\nGuardado: {output_path} ({len(merged)} muestras, {len(gene_data)} genes)")
    print(
        f"\nSiguiente paso -- validacion externa con patrones congelados:\n"
        f"  python3 src/external_validation.py \\\n"
        f"    --patterns results_gse39582_final/calibrated_patterns.tsv \\\n"
        f"    --input {output_path} \\\n"
        f"    --output results_external_{dataset_name}/\n"
        f"\nY luego agregar '{output_path.name} scored' al pooled_cox_validation.py junto con las demas cohortes."
    )


if __name__ == "__main__":
    main()
