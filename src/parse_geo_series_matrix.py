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
    metadata_rows = {}
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
                continue

            if line.startswith("!Sample_characteristics_ch1"):
                values = [x.strip('"') for x in line.split("\t")[1:]]
                # cada valor viene como "atributo: valor" -- usar el
                # nombre del atributo (antes de ":") como clave de fila
                if values and ":" in values[0]:
                    attr_name = values[0].split(":")[0].strip()
                    parsed_values = [
                        v.split(":", 1)[1].strip() if ":" in v else v
                        for v in values
                    ]
                    key = f"characteristics__{attr_name}"
                    metadata_rows[key] = parsed_values

    if sample_ids is None:
        raise ValueError("No se encontro '!Sample_geo_accession' -- verifica el archivo.")

    phenotype = pd.DataFrame(metadata_rows, index=sample_ids)

    # Parsear tabla de expresion
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
