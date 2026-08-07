"""
parse_geo_series_matrix.py

Parsea GSE39582_series_matrix.txt.gz manualmente (sin GEOparse, cuyo
downloader fallo por un corte de conexion FTP en un archivo grande).
El formato series_matrix es texto plano con:
    - Lineas "!Series_..." -- metadata de la serie completa
    - Lineas "!Sample_..." -- una fila por atributo, una columna por
      muestra. "!Sample_characteristics_ch1" se repite multiples veces
      por muestra (una por cada atributo clinico: edad, sexo,
      rfs.delay, rfs.event, etc.), cada valor con formato "atributo: valor"
    - Entre "!series_matrix_table_begin" y "!series_matrix_table_end":
      la matriz de expresion (probes en filas, muestras en columnas)

USO:
    python3 src/parse_geo_series_matrix.py
"""

import gzip
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_geo"
MATRIX_PATH = RAW / "GSE39582_series_matrix.txt.gz"


def parse_series_matrix(path):
    """
    Parsea el series_matrix por CELDA, no por posicion de fila.

    BUG CORREGIDO: la version anterior asumia que todas las muestras
    tienen sus '!Sample_characteristics_ch1' en el mismo orden, y
    usaba el nombre de atributo de la PRIMERA muestra para etiquetar
    toda la fila. En GSE17537 (serie de 2010) esto desalineo datos --
    la columna 'overall_event' termino mezclando grados de
    diferenciacion tumoral con death/no death porque distintas
    muestras tienen sus caracteristicas en distinto orden.

    Este parser en cambio extrae cada celda como su propio par
    'atributo: valor' y la asigna al atributo correcto para esa
    muestra especifica, sin asumir alineacion posicional entre
    muestras.
    """
    metadata_per_sample = {}
    sample_ids = None
    table_lines = []
    in_table = False

    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue

            if in_table:
                table_lines.append(line)
                continue

            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip('"') for x in line.split("\t")[1:]]
                for sid in sample_ids:
                    metadata_per_sample[sid] = {}
                continue

            if line.startswith("!Sample_characteristics_ch1"):
                if sample_ids is None:
                    raise ValueError(
                        "'!Sample_characteristics_ch1' aparecio antes de "
                        "'!Sample_geo_accession' -- el archivo no tiene el "
                        "formato esperado."
                    )
                values = [x.strip('"') for x in line.split("\t")[1:]]
                for sid, v in zip(sample_ids, values):
                    # Una celda puede traer UN solo "atributo: valor"
                    # (formato de GSE39582/GSE17536/GSE17537) o VARIOS
                    # empacados con ';' en una sola celda (formato de
                    # GSE14333: "Location: Right; DukesStage: A;
                    # DFS_Time: 3.64; DFS_Cens: 1; ..."). Separar por
                    # ';' primero maneja ambos casos: con un solo
                    # segmento, el resultado es identico al caso simple.
                    for segment in v.split(";"):
                        segment = segment.strip()
                        if ":" in segment:
                            attr_name, attr_value = segment.split(":", 1)
                            metadata_per_sample[sid][attr_name.strip()] = attr_value.strip()
                        # segmento sin ':' no se puede asociar a un
                        # atributo con seguridad -- se omite en vez de
                        # adivinar
                continue

    if sample_ids is None:
        raise ValueError("No se encontro '!Sample_geo_accession' -- verifica el archivo.")

    phenotype = pd.DataFrame.from_dict(metadata_per_sample, orient="index")
    phenotype.columns = [f"characteristics__{c}" for c in phenotype.columns]
    phenotype = phenotype.reindex(sample_ids)  # preservar orden original de muestras

    from io import StringIO
    table_str = "\n".join(table_lines)
    expr = pd.read_csv(StringIO(table_str), sep="\t", index_col=0, quotechar='"')

    return phenotype, expr


def main():
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {MATRIX_PATH}. Descargalo primero con curl (ver README)."
        )

    print("Parseando series matrix (puede tardar un momento)...")
    phenotype, expr = parse_series_matrix(MATRIX_PATH)

    pheno_path = RAW / "gse39582_phenotype.tsv"
    expr_path = RAW / "gse39582_expression_probes.tsv"
    phenotype.to_csv(pheno_path, sep="\t")
    expr.to_csv(expr_path, sep="\t")

    print(f"\nMetadata clinica: {pheno_path} ({phenotype.shape[0]} muestras, {phenotype.shape[1]} atributos)")
    print(f"Columnas de metadata encontradas:\n{list(phenotype.columns)}")
    print(f"\nExpresion: {expr_path} ({expr.shape[0]} probes, {expr.shape[1]} muestras)")
    print(f"Primeros probe IDs: {list(expr.index[:5])}")


if __name__ == "__main__":
    main()
