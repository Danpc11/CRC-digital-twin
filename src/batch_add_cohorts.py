"""
batch_add_cohorts.py

Descarga y diagnostica en lote las cohortes CRCSC que todavia no estan
en el analisis combinado. Automatiza los pasos repetitivos (armar la
URL de GEO, descargar, parsear, detectar plataforma) y se detiene donde
SI hace falta criterio humano: elegir que columnas son duracion y
evento de supervivencia.

POR QUE MAS COHORTES
--------------------
El analisis de poder (power_analysis.py) dio el numero concreto: para
detectar el HR observado de CMS4 (1.88) con 80% de poder harian falta
~84 eventos, y solo hay ~50 disponibles. Mas cohortes = mas eventos =
la unica solucion real, mucho mejor que seguir ajustando el panel.

FLUJO
    1) python3 src/batch_add_cohorts.py --download        descarga todo
    2) python3 src/batch_add_cohorts.py --diagnose        inspecciona columnas
    3) revisar la salida, elegir --duration-col/--event-col por cohorte
    4) construir cada una con build_external_cohort_generic.py

NOTA SOBRE LAS URLs DE GEO
--------------------------
El bucket agrupa por los ultimos 3 digitos reemplazados por 'nnn':
GSE2109 -> GSE2nnn, GSE13294 -> GSE13nnn, GSE33113 -> GSE33nnn.
Para IDs de 3 digitos o menos el bucket es literalmente 'GSEnnn'.
Esa regla esta implementada en geo_matrix_url().
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_GEO = ROOT / "data" / "raw_geo"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Cohortes del consorcio CMS que aun no estan en el analisis combinado.
# El 'n' es el numero de muestras CON etiqueta CMS oficial segun
# cms_labels_public_all.txt -- no el total de la serie en GEO.
PENDING_COHORTS = {
    "GSE2109":  {"n_cms": 293, "dataset": "gse2109",
                  "nota": "expO / IGC -- puede NO traer supervivencia curada"},
    "GSE13294": {"n_cms": 155, "dataset": "gse13294",
                  "nota": "Jorissen -- verificar si trae seguimiento"},
    "GSE37892": {"n_cms": 130, "dataset": "gse37892",
                  "nota": "Laibe/Marisa -- estadios II-III, probable RFS"},
    "GSE33113": {"n_cms": 90,  "dataset": "gse33113",
                  "nota": "AMC-AJCCII -- estadio II, probable RFS"},
    "GSE20916": {"n_cms": 90,  "dataset": "gse20916",
                  "nota": "Skrzypczak -- serie de progresion, puede no traer desenlace"},
    "GSE13067": {"n_cms": 74,  "dataset": "gse13067", "nota": "Jorissen"},
    "GSE35896": {"n_cms": 62,  "dataset": "gse35896", "nota": "Schlicker"},
    "GSE23878": {"n_cms": 35,  "dataset": "gse23878", "nota": "muestra pequena"},
}

SURVIVAL_HINTS = ("rfs", "dfs", "relapse", "recur", "surv", "event", "delay",
                   "time", "follow", "cens", "status", "os_")
STAGE_HINTS = ("stage", "dukes", "tnm", "ajcc")


def geo_matrix_url(gse: str) -> str:
    """URL del series_matrix de GEO, con la regla de bucket correcta."""
    digits = gse.replace("GSE", "")
    bucket = f"GSE{digits[:-3]}nnn" if len(digits) > 3 else "GSEnnn"
    return (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{gse}/matrix/"
            f"{gse}_series_matrix.txt.gz")


def download(gse: str, force: bool = False) -> bool:
    RAW_GEO.mkdir(parents=True, exist_ok=True)
    dest = RAW_GEO / f"{gse}_series_matrix.txt.gz"
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"  {gse}: ya existe ({size_mb:.1f} MB), se omite (--force para rebajar)")
        return True

    url = geo_matrix_url(gse)
    print(f"  {gse}: descargando desde {url}")
    # sin -C - : algunos espejos de NCBI no soportan reanudacion por rangos
    r = subprocess.run(["curl", "-L", "-s", "-f", "-o", str(dest), url])
    if r.returncode != 0:
        print(f"  {gse}: FALLO la descarga (curl codigo {r.returncode}). "
              "Puede ser el ID, la ruta, o falta de conexion.")
        dest.unlink(missing_ok=True)
        return False

    size = dest.stat().st_size
    head = dest.read_bytes()[:4]

    # Un .gz valido empieza con los magic bytes 1f 8b. Cualquier otra cosa
    # (HTML de error, respuesta vacia, pagina de 404) se descarta -- dejar
    # un archivo basura en disco hace fallar el diagnostico despues con un
    # error confuso, muy lejos de la causa real.
    if head[:2] != b"\x1f\x8b":
        preview = dest.read_bytes()[:120]
        print(f"  {gse}: el archivo NO es gzip valido ({size} bytes). "
              f"Inicio: {preview[:60]!r}")
        print("         Probable 404 o error del servidor. Verifica el ID en GEO.")
        dest.unlink(missing_ok=True)
        return False

    if size < 50_000:
        print(f"  {gse}: AVISO -- gzip valido pero sospechosamente pequeno "
              f"({size/1e3:.1f} KB). Revisalo antes de confiar en el.")

    print(f"  {gse}: OK ({size/1e6:.1f} MB)")
    return True


def diagnose(gse: str) -> dict:
    from parse_geo_series_matrix import parse_series_matrix

    path = RAW_GEO / f"{gse}_series_matrix.txt.gz"
    if not path.exists():
        print(f"\n{gse}: NO descargado -- corre primero con --download")
        return {}

    try:
        pheno, expr = parse_series_matrix(path)
    except Exception as err:
        print(f"\n{gse}: no se pudo parsear -- {err}")
        return {}

    cols = list(pheno.columns)
    surv = [c for c in cols if any(h in c.lower() for h in SURVIVAL_HINTS)]
    stage = [c for c in cols if any(h in c.lower() for h in STAGE_HINTS)]

    info = PENDING_COHORTS.get(gse, {})
    print(f"\n{'=' * 72}")
    print(f"{gse}  ({pheno.shape[0]} muestras en GEO, {info.get('n_cms', '?')} con etiqueta CMS)")
    if info.get("nota"):
        print(f"  nota: {info['nota']}")
    print(f"{'=' * 72}")

    if not surv:
        print("  SIN columnas candidatas de supervivencia -- esta cohorte probablemente")
        print("  NO sirve para el analisis de supervivencia (aunque si para clasificacion).")
    else:
        print("  Candidatas de SUPERVIVENCIA:")
        for c in surv:
            vals = pheno[c].dropna().unique()[:4]
            print(f"    - {c}")
            print(f"        ejemplos: {list(vals)}")

    if stage:
        print("  Candidatas de ESTADIO:")
        for c in stage:
            vals = pheno[c].dropna().unique()[:6]
            print(f"    - {c}  -> {list(vals)}")
    else:
        print("  Sin columna de estadio (no se podra incluir en el modelo ajustado).")

    if not surv and not stage:
        print(f"  Todas las columnas: {cols}")

    return {"gse": gse, "n": pheno.shape[0], "survival_cols": surv, "stage_cols": stage}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="Descargar los series_matrix")
    parser.add_argument("--diagnose", action="store_true", help="Inspeccionar columnas disponibles")
    parser.add_argument("--force", action="store_true", help="Redescargar aunque ya exista")
    parser.add_argument("--only", nargs="+", default=None,
                         help="Limitar a ciertas cohortes, ej: --only GSE2109 GSE13294")
    args = parser.parse_args()

    if not (args.download or args.diagnose):
        parser.error("Elige al menos --download o --diagnose")

    cohorts = args.only or list(PENDING_COHORTS.keys())
    unknown = [c for c in cohorts if c not in PENDING_COHORTS]
    if unknown:
        print(f"AVISO: cohortes no listadas en PENDING_COHORTS: {unknown} (se intentaran igual)")

    if args.download:
        print(f"DESCARGANDO {len(cohorts)} cohortes en {RAW_GEO}\n")
        ok = sum(download(g, args.force) for g in cohorts)
        print(f"\n{ok}/{len(cohorts)} descargadas correctamente.")

    if args.diagnose:
        results = [diagnose(g) for g in cohorts]
        usable = [r for r in results if r.get("survival_cols")]
        print(f"\n{'=' * 72}\nRESUMEN\n{'=' * 72}")
        print(f"Cohortes con columnas de supervivencia candidatas: "
              f"{len(usable)}/{len([r for r in results if r])}")
        for r in usable:
            n_cms = PENDING_COHORTS.get(r["gse"], {}).get("n_cms", "?")
            print(f"  {r['gse']}: {n_cms} muestras con CMS, "
                  f"{len(r['survival_cols'])} columna(s) de supervivencia")
        print(
            "\nSIGUIENTE PASO -- por cada cohorte utilizable, elige las columnas y construye:\n\n"
            "  python3 src/build_external_cohort_generic.py --gse GSEXXXXX \\\n"
            "    --dataset gsexxxxx \\\n"
            '    --duration-col "characteristics__..." \\\n'
            '    --event-col "characteristics__..." \\\n'
            '    --stage-col "characteristics__..." \\\n'
            '    --event-map "valorA=1,valorB=0"      # solo si el evento no es 0/1\n\n'
            "Recuerda verificar la DIRECCION del evento antes de mapear: en GSE14333, "
            "'DFS_Cens=1' significaba CENSURADO, no evento. Cruzar contra estadio "
            "(la proporcion de eventos debe SUBIR con el estadio) es una buena comprobacion."
        )


if __name__ == "__main__":
    main()
