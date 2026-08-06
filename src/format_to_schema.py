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
    "3002": "GZMB",
    "4609": "MYC",
    "8313": "AXIN2",
    "2168": "FABP1",
    "7431": "VIM",
    "7040": "TGFB1",
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
        raise ValueError(
            f"Entrez IDs no encontrados en la matriz de expresion: {missing_entrez}. "
            "Verifica que formatted_crc_data.txt no haya cambiado de formato."
        )

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
    for col in ("dfsMo", "dfsStat"):
        if col not in clinical.columns:
            raise ValueError(f"Columna '{col}' no encontrada en {clinical_path.name}")

    print("Fusionando...")
    merged = gene_expr.join(tcga_labels[[CMS_LABEL_COLUMN]], how="inner")
    n_before_clinical = len(merged)
    merged = merged.join(clinical[["dfsMo", "dfsStat"]], how="left")

    merged = merged.rename(columns={
        CMS_LABEL_COLUMN: "cms_label",
        "dfsMo": "relapse_free_months",
        "dfsStat": "relapse_event",
    })
    merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    n_missing_survival = merged["relapse_free_months"].isna().sum()
    print(
        f"\n{n_before_clinical} muestras con expresion + CMS (subset TCGA). "
        f"{n_missing_survival} sin dato de supervivencia (dfsMo faltante en la fuente "
        "clinica original) -- estas se excluiran automaticamente en "
        "survival_validation.py via dropna, no hace falta filtrarlas aqui."
    )
    print("\nDistribucion de subtipo CMS en este subset:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"\nGuardado: {OUTPUT_PATH} ({len(merged)} muestras, {len(ENTREZ_TO_SYMBOL)} genes)")
    print(
        "\nNOTA: el panel quedo en 7 genes (sin KRAS_sig, que era un placeholder "
        "sin gen real detras). Para correr calibration.py/run_pipeline.py con "
        "este archivo no hace falta ningun cambio -- infiere las columnas de "
        "genes automaticamente, funciona igual con 7 que con 8."
    )


if __name__ == "__main__":
    main()
