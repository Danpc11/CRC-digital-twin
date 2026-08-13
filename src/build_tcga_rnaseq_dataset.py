"""
build_tcga_rnaseq_dataset.py

Construye la cohorte TCGA-COAD/READ (RNA-seq) con los 10 genes del panel
completo -- el archivo anterior (data/tcga_cms_labeled.tsv) solo tenia
5 de 10 genes (MYC, AXIN2, FABP1, VIM, TGFB1), de una version anterior
del panel antes de que quedara congelado en los 10 actuales. Este
script reconstruye desde los datos crudos, que SI tienen los 10 genes.

Proposito especifico: prueba de VALIDEZ ENTRE PLATAFORMAS -- todas las
demas cohortes del proyecto son microarreglos Affymetrix (mayoria
U133 Plus 2.0, una U133A). TCGA es RNA-seq, una tecnologia de medicion
fundamentalmente distinta. La pregunta no es supervivencia (RFS/DFS
no existe de forma curada en TCGA -- dfsStat esta 100% vacio en las
603 muestras, verificado) -- es si la clasificacion por correlacion
concuerda con la etiqueta CMS oficial del consorcio en datos de una
plataforma que el modelo nunca vio durante calibracion.

Fuentes (ya presentes en el repo, no requiere descarga nueva):
    data/raw_synapse/tcga_rnaseq/TCGACRC_expression-merged.tsv
        (genes en filas, muestras en columnas, valores ya log-transformados)
    data/raw_synapse/tcga_cms_labels/cms_labels_public_all.txt
        (mismo archivo central de etiquetas que usa todo el proyecto,
        dataset='tcga', columna CMS_final_network_plus_RFclassifier_in_nonconsensus_samples)
    data/raw_synapse/tcga_rnaseq/TCGACRC_clinical-merged.tsv
        (osMo/osStat -- se incluyen por completitud, aunque la prueba
        principal aqui es concordancia de clasificacion, no supervivencia)

USO:
    python3 src/build_tcga_rnaseq_dataset.py --output data/tcga_rnaseq_cms_labeled.tsv
"""

import argparse
from pathlib import Path

import pandas as pd

from build_external_cohort_generic import CMS_RENAME

GENES = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_synapse" / "tcga_rnaseq"
LABELS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_synapse" / "tcga_cms_labels" / "cms_labels_public_all.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", default=str(RAW_DIR / "TCGACRC_expression-merged.tsv"))
    parser.add_argument("--clinical", default=str(RAW_DIR / "TCGACRC_clinical-merged.tsv"))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Cargando expresion: {args.expression}")
    expr = pd.read_csv(args.expression, sep="\t", index_col=0)
    print(f"  {expr.shape[0]} features x {expr.shape[1]} muestras")

    genes_presentes = [g for g in GENES if g in expr.index]
    genes_faltantes = [g for g in GENES if g not in expr.index]
    if genes_faltantes:
        print(f"AVISO: genes del panel NO encontrados en la expresion: {genes_faltantes}")
    print(f"Genes del panel encontrados: {len(genes_presentes)}/10")

    # transponer: genes en filas -> muestras en filas (mismo formato que
    # el resto de las cohortes del proyecto)
    expr_panel = expr.loc[genes_presentes].T
    expr_panel.index.name = "sample_id"
    expr_panel = expr_panel.reset_index()

    print(f"\nCargando etiquetas CMS: {args.labels}")
    labels = pd.read_csv(args.labels, sep="\t")
    labels_tcga = labels[labels["dataset"] == "tcga"][
        ["sample", "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"]
    ].rename(columns={
        "sample": "sample_id",
        "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples": "cms_label",
    })
    print(f"  {len(labels_tcga)} muestras TCGA con etiqueta en el archivo central")

    print(f"\nCargando clinico (OS, referencia): {args.clinical}")
    clinical = pd.read_csv(args.clinical, sep="\t")
    clinical = clinical.rename(columns={
        "id": "sample_id", "osMo": "overall_survival_months", "osStat": "os_event",
    })[["sample_id", "overall_survival_months", "os_event"]]

    merged = expr_panel.merge(labels_tcga, on="sample_id", how="left")
    merged = merged.merge(clinical, on="sample_id", how="left")

    # el archivo central da forma corta (CMS1, CMS4, NOLBL, ...) --
    # el resto del pipeline (load_labeled_dataset) exige la forma larga
    # (CMS1_MSI_immune, ...) -- mismo mapeo canonico que usa
    # build_external_cohort_generic.py, no uno propio inventado.
    merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    n_sin_etiqueta = merged["cms_label"].isna().sum()
    if n_sin_etiqueta:
        print(f"AVISO: {n_sin_etiqueta} muestras de expresion sin etiqueta CMS en el archivo "
              "central -- se marcan 'none' (no se excluyen, igual que en las demas cohortes "
              "sin verdad de referencia oficial para todas sus muestras).")
    merged["cms_label"] = merged["cms_label"].fillna("none")

    print(f"\n{len(merged)} muestras con expresion de los 10 genes.")
    print("Distribucion CMS:")
    print(merged["cms_label"].value_counts())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
