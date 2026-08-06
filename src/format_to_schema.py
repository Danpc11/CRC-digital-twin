"""
format_to_schema.py

Convierte los datos crudos descargados de Synapse al esquema TSV que
espera calibration.py, usando el subconjunto TCGA (573 muestras) --
es el unico subconjunto que trae expresion + etiqueta CMS +
supervivencia real (dfsMo/dfsStat) ya emparejados 1 a 1, confirmado
por diagnose_id_mapping.py (overlap 573/573).

GSE39582 (566 muestras) tiene expresion + CMS pero NO trajo
supervivencia en este dump de Synapse -- para incluirlo hace falta
bajar su metadata clinica por separado desde GEO y cruzarla por
sample ID (los IDs en cms_labels_public_all.txt para gse39582 son
formato GSM, hay que mapearlos a los IDs de muestra del paper original
de Marisa et al. si se quiere supervivencia de esa cohorte tambien).

Mapeo Entrez ID -> simbolo de gen para el panel actual. KRAS_sig fue
un placeholder sin gen real detras (nunca existio como identificador
biologico) -- se excluye aqui. Si quieres restaurar un octavo gen
para el "eje metabolico" (CMS3), hay que elegir uno real de la
literatura antes de mapear su Entrez ID.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "tcga_cms_labeled.tsv"

ENTREZ_TO_SYMBOL = {
    "4292": "MLH1",
    "10578": "GNLY",   # reemplaza GZMB -- AUC=0.886 vs 0.665 de GZMB, misma familia funcional (granulos citotoxicos)
    "11274": "USP18",  # eje CMS1 -- AUC=0.888, gen estimulado por interferon tipo I
    "4609": "MYC",
    "8313": "AXIN2",
    "2168": "FABP1",
    "1373": "CPS1",
    "6476": "SI",
    "7431": "VIM",
    "7040": "TGFB1",
}

# Alternativas si el marcador principal no esta en el panel de 5973 genes
# de formatted_crc_data.txt (interseccion entre plataformas -- algunos
# genes se pierden aunque el Entrez ID sea correcto). Mismo eje biologico
# que el marcador que reemplazan.
FALLBACK_ENTREZ = {
    "MLH1":  [("4436", "MSH2"), ("2956", "MSH6"), ("5395", "PMS2")],   # otros genes MMR/MSI
    "GNLY":  [("3001", "GZMA"), ("5551", "PRF1"), ("925", "CD8A")],    # otros marcadores citotoxicos/inmunes
}

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}

CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"


def main():
    expr_path = RAW / "crcsc_combined" / "formatted_crc_data.txt"
    labels_path = RAW / "tcga_cms_labels" / "cms_labels_public_all.txt"
    clinical_path = RAW / "tcga_rnaseq" / "TCGACRC_clinical-merged.tsv"

    for p in (expr_path, labels_path, clinical_path):
        if not p.exists():
            raise FileNotFoundError(f"No se encontro {p}. Corre download_synapse_data.py primero.")

    print("Cargando expresion...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr.columns = expr.columns.astype(str)

    missing_entrez = [eid for eid in ENTREZ_TO_SYMBOL if eid not in expr.columns]
    if missing_entrez:
        print(f"AVISO: Entrez IDs no encontrados en el panel de 5973 genes: {missing_entrez}")
        for eid in missing_entrez:
            symbol = ENTREZ_TO_SYMBOL[eid]
            resolved = False
            for fallback_eid, fallback_symbol in FALLBACK_ENTREZ.get(symbol, []):
                if fallback_eid in expr.columns:
                    print(f"  {symbol} (Entrez {eid}) no disponible -> usando {fallback_symbol} (Entrez {fallback_eid}) en su lugar")
                    del ENTREZ_TO_SYMBOL[eid]
                    ENTREZ_TO_SYMBOL[fallback_eid] = fallback_symbol
                    resolved = True
                    break
            if not resolved:
                print(f"  {symbol} (Entrez {eid}) no disponible y ningun fallback conocido tampoco -- se omite del panel")
                del ENTREZ_TO_SYMBOL[eid]

    if len(ENTREZ_TO_SYMBOL) == 0:
        raise ValueError("Ningun gen del panel (ni sus fallbacks) esta disponible en formatted_crc_data.txt.")

    print(f"\nPanel final ({len(ENTREZ_TO_SYMBOL)} genes): {list(ENTREZ_TO_SYMBOL.values())}")

    gene_expr = expr[list(ENTREZ_TO_SYMBOL.keys())].copy()
    gene_expr = gene_expr.rename(columns=ENTREZ_TO_SYMBOL)

    print("Cargando etiquetas CMS...")
    labels = pd.read_csv(labels_path, sep="\t")
    tcga_labels = labels[labels["dataset"] == "tcga"].set_index("sample")

    if CMS_LABEL_COLUMN not in tcga_labels.columns:
        raise ValueError(
            f"Columna '{CMS_LABEL_COLUMN}' no encontrada. Columnas disponibles: "
            f"{list(tcga_labels.columns)}"
        )

    print("Cargando supervivencia clinica (TCGA)...")
    clinical = pd.read_csv(clinical_path, sep="\t", index_col="id")

    # dfsMo (supervivencia libre de recidiva) esta vacio en TCGA-COAD/READ
    # (0/603 no-nulos, verificado) -- TCGA curo consistentemente muerte
    # (overall survival) pero no recidiva especifica. Usamos osMo/osStat
    # en su lugar, con nombres de columna que dejan explicito que esto es
    # supervivencia GLOBAL, no libre de recidiva -- son desenlaces
    # distintos (muerte por cualquier causa vs. recurrencia del cancer)
    # y no se deben tratar como intercambiables en ningun analisis
    # posterior.
    for col in ("osMo", "osStat"):
        if col not in clinical.columns:
            raise ValueError(f"Columna '{col}' no encontrada en {clinical_path.name}")

    print("Fusionando...")
    merged = gene_expr.join(tcga_labels[[CMS_LABEL_COLUMN]], how="inner")
    n_before_clinical = len(merged)
    merged = merged.join(clinical[["osMo", "osStat"]], how="left")

    merged = merged.rename(columns={
        CMS_LABEL_COLUMN: "cms_label",
        "osMo": "overall_survival_months",
        "osStat": "death_event",
    })
    merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    n_missing_survival = merged["overall_survival_months"].isna().sum()
    print(
        f"\n{n_before_clinical} muestras con expresion + CMS (subset TCGA). "
        f"{n_missing_survival} sin dato de supervivencia global -- estas se "
        "excluiran automaticamente en survival_validation.py via dropna."
    )
    print(
        "\nNOTA: esto es supervivencia GLOBAL (muerte por cualquier causa), "
        "NO supervivencia libre de recidiva -- TCGA-COAD/READ no tiene dfsMo "
        "curado (0/603 no-nulos, verificado). Si mas adelante consigues "
        "GSE39582 con supervivencia libre de recidiva real, usa nombres de "
        "columna 'relapse_free_months'/'relapse_event' para ese archivo, "
        "para no mezclar ambos endpoints bajo el mismo nombre."
    )
    print("\nDistribucion de subtipo CMS en este subset:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"\nGuardado: {OUTPUT_PATH} ({len(merged)} muestras, {len(ENTREZ_TO_SYMBOL)} genes)")
    print(f"Panel final usado: {list(ENTREZ_TO_SYMBOL.values())}")
    print(
        "\nPara correr calibration.py/run_pipeline.py con este archivo no hace "
        "falta ningun cambio -- infiere las columnas de genes automaticamente, "
        "funciona con cualquier numero de genes."
    )


if __name__ == "__main__":
    main()
