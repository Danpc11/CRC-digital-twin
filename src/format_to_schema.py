"""
format_to_schema.py

ESQUELETO -- no correr tal cual. Convierte los datos crudos
descargados de Synapse (download_synapse_data.py) al esquema TSV que
espera calibration.py:

    sample_id   cms_label   GENE1   GENE2   ...   relapse_free_months   relapse_event

Por que es un esqueleto y no un script terminado: no pude inspeccionar
la estructura real de los archivos de Synapse en este entorno (sin
acceso de red a synapse.org). La estructura tipica de estos datasets es:
    - Matriz de expresion: genes en filas, muestras en columnas (hay
      que transponerla)
    - Metadata clinica en un archivo separado, indexada por sample_id
    - Las etiquetas CMS pueden venir como columna separada o como
      archivo aparte

Ajusta las secciones marcadas con TODO despues de inspeccionar los
archivos reales descargados en data/raw_synapse/.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gse39582_cms_labeled.tsv"

# El panel de genes que usa attractor_model.py -- ajusta si calibras
# con un panel distinto (calibration.py acepta cualquier subconjunto).
TARGET_GENES = ["MLH1", "GZMB", "MYC", "AXIN2", "FABP1", "KRAS_sig", "VIM", "TGFB1"]


def main():
    # TODO: reemplazar con el nombre real del archivo descargado
    expr_path = RAW_DIR / "crcsc_combined" / "expression_matrix.tsv"
    clinical_path = RAW_DIR / "crcsc_combined" / "clinical_metadata.tsv"

    if not expr_path.exists():
        raise FileNotFoundError(
            f"No se encontro {expr_path}. Corre primero "
            "download_synapse_data.py e inspecciona la estructura real "
            "de los archivos descargados para ajustar este script."
        )

    # TODO: verificar separador, orientacion (genes en filas o columnas),
    # y si el indice de genes es simbolo (MYC) o Entrez ID (necesitaria
    # mapeo, ver org.Hs.eg.db si vienen como Entrez).
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    # Si genes estan en filas y muestras en columnas, transponer:
    # expr = expr.T

    missing_genes = set(TARGET_GENES) - set(expr.columns)
    if missing_genes:
        print(
            f"AVISO: {missing_genes} no encontrados en la matriz de expresion. "
            "Revisa si el identificador es simbolo de gen vs Entrez ID, o si "
            "el gen simplemente no esta en la plataforma usada."
        )

    clinical = pd.read_csv(clinical_path, sep="\t", index_col=0)

    # TODO: verificar nombres reales de columnas en el archivo de metadata.
    # Los nombres tipicos en estudios CRCSC son similares a:
    #   'cms_label' o 'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples'
    #   'rfs_delay' / 'relapse_free_survival'
    #   'rfs_event' / 'relapse'
    rename_map = {
        # "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples": "cms_label",
        # "rfs_delay": "relapse_free_months",
        # "rfs_event": "relapse_event",
    }
    clinical = clinical.rename(columns=rename_map)

    merged = expr[TARGET_GENES].join(clinical, how="inner")
    merged.index.name = "sample_id"
    merged = merged.reset_index()

    # Normalizar etiquetas CMS al formato esperado por calibration.py
    cms_rename = {
        "CMS1": "CMS1_MSI_immune",
        "CMS2": "CMS2_canonical_WNT",
        "CMS3": "CMS3_metabolic",
        "CMS4": "CMS4_mesenchymal",
        "NOLBL": "none",
    }
    if "cms_label" in merged.columns:
        merged["cms_label"] = merged["cms_label"].replace(cms_rename)

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Guardado: {OUTPUT_PATH} ({len(merged)} muestras)")


if __name__ == "__main__":
    main()
