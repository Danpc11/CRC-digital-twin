"""
download_geo_gse39582.py

Descarga GSE39582 (Marisa et al. 2013) directamente de GEO -- la
cohorte con supervivencia libre de recidiva (RFS) curada que TCGA no
tiene. Contiene expresion (microarray Affymetrix HG-U133 Plus 2,
plataforma GPL570) + metadata clinica en el mismo archivo (series
matrix).

REQUIERE:
    pip install GEOparse

USO:
    python3 src/download_geo_gse39582.py

NOTA: no se pudo probar en este entorno (sin acceso de red a NCBI/GEO
desde el sandbox de desarrollo). Corre esto en tu maquina local.
"""

from pathlib import Path

import GEOparse

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_geo"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Descargando GSE39582 (esto puede tardar varios minutos, "
          "el archivo de expresion es grande)...")
    gse = GEOparse.get_GEO(geo="GSE39582", destdir=str(OUTPUT_DIR))

    # Metadata clinica -- viene de los campos "characteristics_ch1" de
    # cada muestra, GEOparse los expone como columnas del phenotype_data
    pheno = gse.phenotype_data
    pheno_path = OUTPUT_DIR / "gse39582_phenotype.tsv"
    pheno.to_csv(pheno_path, sep="\t")
    print(f"Metadata clinica guardada en: {pheno_path}")
    print(f"Columnas disponibles: {list(pheno.columns)}")

    # Matriz de expresion -- probes de Affymetrix en filas, muestras en
    # columnas. Requiere mapeo probe->gen via la plataforma GPL570.
    expr = gse.pivot_samples("VALUE")
    expr_path = OUTPUT_DIR / "gse39582_expression_probes.tsv"
    expr.to_csv(expr_path, sep="\t")
    print(f"Expresion (por probe, sin mapear a gen) guardada en: {expr_path}")

    # Anotacion de plataforma para mapear probe ID -> simbolo de gen
    gpl = gse.gpls.get("GPL570")
    if gpl is not None:
        annot_path = OUTPUT_DIR / "gpl570_annotation.tsv"
        gpl.table.to_csv(annot_path, sep="\t", index=False)
        print(f"Anotacion de plataforma (probe -> gen) guardada en: {annot_path}")
    else:
        print("AVISO: no se encontro la plataforma GPL570 en el objeto GSE -- "
              "revisa gse.gpls.keys() para ver que plataformas trajo.")

    print(
        "\nDescarga completa. Antes de escribir el script de fusion, inspecciona "
        "los nombres reales de columna en gse39582_phenotype.tsv (deberian "
        "incluir algo relacionado a rfs.delay/rfs.event/tnm.stage segun el "
        "paper de Marisa et al., pero GEOparse a veces las expone con "
        "nombres ligeramente distintos -- no asumas el nombre exacto sin "
        "verlo)."
    )


if __name__ == "__main__":
    main()
