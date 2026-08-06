"""
diagnose_id_mapping.py

Diagnostico ANTES de fusionar formatted_crc_data.txt con
cms_labels_public_all.txt -- verifica si los sample IDs se corresponden
directamente o si hace falta un mapeo.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"

print("=== formatted_crc_data.txt: IDs de muestra ===")
expr = pd.read_csv(RAW / "crcsc_combined" / "formatted_crc_data.txt", sep="\t", index_col=0, nrows=50)
print(f"Shape (primeras 50 filas para inspeccion): {expr.shape}")
print(f"Primeros 10 sample IDs: {list(expr.index[:10])}")
print(f"Primeras 10 columnas (deberian ser Entrez IDs): {list(expr.columns[:10])}")

print("\n=== cms_labels_public_all.txt: datasets disponibles ===")
labels = pd.read_csv(RAW / "tcga_cms_labels" / "cms_labels_public_all.txt", sep="\t")
print(labels["dataset"].value_counts())

print("\n=== cms_labels_public_all.txt: muestra de IDs por dataset ===")
for ds in labels["dataset"].unique():
    sample_ids = labels.loc[labels["dataset"] == ds, "sample"].head(3).tolist()
    print(f"  {ds}: {sample_ids}")

print("\n=== Verificando overlap directo de IDs ===")
expr_full_index = pd.read_csv(
    RAW / "crcsc_combined" / "formatted_crc_data.txt", sep="\t", index_col=0, usecols=[0]
).index
expr_ids = set(expr_full_index.astype(str))
label_ids = set(labels["sample"].astype(str))

exact_overlap = expr_ids & label_ids
print(f"IDs en formatted_crc_data.txt: {len(expr_ids)}")
print(f"IDs en cms_labels_public_all.txt: {len(label_ids)}")
print(f"Overlap exacto: {len(exact_overlap)}")

# Overlap case-insensitive, por si difieren solo en mayusculas/minusculas
expr_ids_lower = {i.lower() for i in expr_ids}
label_ids_lower = {i.lower() for i in label_ids}
overlap_ci = expr_ids_lower & label_ids_lower
print(f"Overlap case-insensitive: {len(overlap_ci)}")

print("\n=== TCGACRC_clinical-merged.tsv: overlap con cms_labels (dataset=tcga?) ===")
clinical = pd.read_csv(RAW / "tcga_rnaseq" / "TCGACRC_clinical-merged.tsv", sep="\t")
clinical_ids = set(clinical["id"].astype(str))
print(f"IDs clinicos TCGA (formato barcode): {list(clinical_ids)[:5]}")
tcga_like_datasets = [d for d in labels["dataset"].unique() if "tcga" in d.lower()]
print(f"Datasets con 'tcga' en el nombre dentro de cms_labels: {tcga_like_datasets}")
if tcga_like_datasets:
    tcga_label_ids = set(labels.loc[labels["dataset"].isin(tcga_like_datasets), "sample"].astype(str))
    print(f"Overlap clinical TCGA vs cms_labels (subset tcga): {len(clinical_ids & tcga_label_ids)}")
