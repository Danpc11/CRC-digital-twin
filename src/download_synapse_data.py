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
    python3 src/download_synapse_data.py

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
import synapseutils

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
        dest = OUTPUT_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {name} ({syn_id})...")

        # syncFromSynapse maneja tanto archivos individuales como
        # Folders/Projects contenedores con multiples archivos adentro
        # (que es lo que causaba el AttributeError: 'path' con syn.get()
        # directo -- syn4961785 es un contenedor, no un archivo suelto).
        downloaded = synapseutils.syncFromSynapse(syn, syn_id, path=str(dest))

        if not downloaded:
            print(f"  AVISO: no se descargo ningun archivo para {syn_id}. "
                  "Verifica el ID y tus permisos de acceso en Synapse.")
        else:
            for entity in downloaded:
                print(f"  -> {entity.path}")

    print(
        "\nDescarga completa. Revisa la estructura real en data/raw_synapse/ "
        "y ajusta format_to_schema.py con los nombres de archivo/columnas "
        "reales que encuentres."
    )


if __name__ == "__main__":
    main()
