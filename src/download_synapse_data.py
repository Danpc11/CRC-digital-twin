"""
download_synapse_data.py

Descarga los datos del consorcio de subtipificacion de cancer
colorrectal (CRCSC) desde Synapse.

REQUIERE:
    pip install synapseclient
    Una cuenta gratuita en https://www.synapse.org
    Un Personal Access Token (Settings -> Personal Access Tokens en tu perfil)

USO:
    export SYNAPSE_AUTH_TOKEN="tu_token_aqui"
    python3 scripts/download_synapse_data.py

NOTA: este script NO se ha podido probar en este entorno porque el
sandbox de desarrollo no tiene acceso de red a synapse.org (solo
PyPI/GitHub/npm estan permitidos). Corre esto en tu maquina local.

IDs verificados en literatura (Guinney et al. 2015, CMScaller paper):
    syn4961785  -- dataset combinado CRCSC: 3232 muestras, multiples
                   cohortes, batch-corrected, con etiquetas CMS y
                   datos clinicos/patologicos. PUNTO DE PARTIDA
                   RECOMENDADO.
    syn4978511  -- etiquetas CMS especificas para TCGA-COADREAD
    syn2023932  -- RNA-seq procesado de TCGA-COADREAD (n=577)
"""

import os
from pathlib import Path

import synapseclient

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"

SYNAPSE_IDS = {
    "crcsc_combined": "syn4961785",   # dataset combinado, punto de partida recomendado
    "tcga_cms_labels": "syn4978511",  # solo si quieres TCGA por separado
    "tcga_rnaseq": "syn2023932",      # solo si quieres TCGA por separado
}


def main():
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise RuntimeError(
            "Define la variable de entorno SYNAPSE_AUTH_TOKEN con tu Personal "
            "Access Token de Synapse antes de correr este script."
        )

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, syn_id in SYNAPSE_IDS.items():
        print(f"Descargando {name} ({syn_id})...")
        entity = syn.get(syn_id, downloadLocation=str(OUTPUT_DIR / name))
        print(f"  -> guardado en: {entity.path}")

    print(
        "\nDescarga completa. Los archivos crudos de Synapse rara vez vienen "
        "en el esquema exacto que espera calibration.py -- revisa la "
        "estructura descargada (probablemente un .tsv/.RData con genes en "
        "filas, muestras en columnas, y un archivo de metadata clinica "
        "aparte) y usa scripts/format_to_schema.py como punto de partida "
        "para convertirlo al TSV requerido."
    )


if __name__ == "__main__":
    main()
