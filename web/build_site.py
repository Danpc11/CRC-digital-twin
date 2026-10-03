#!/usr/bin/env python3
"""
web/build_site.py -- construye la version web de ColoQ (GitHub Pages).

La app de Streamlit corre COMPLETA en el navegador del usuario mediante
stlite (Streamlit sobre Pyodide/WebAssembly): no hay servidor, ningun dato
que el usuario suba sale de su computadora.

Que hace:
  1. Copia app.py y src/*.py a <out>/app/.
  2. Incluye un calibrated_patterns.tsv para que la app arranque con patrones
     ya cargados:
       - si existe web/calibrated_patterns.tsv (patrones reales, p. ej. los de
         GSE39582), se monta como results_gse39582/calibrated_patterns.tsv;
       - si no, genera patrones DEMO con datos sinteticos (`cli.py demo`) y
         los monta como results_demo/calibrated_patterns.tsv.
     app.py ya busca results*/calibrated_patterns.tsv por defecto, asi que no
     hace falta tocar la app.
  3. Escribe <out>/index.html con el montaje de stlite y <out>/.nojekyll.

Uso local:
    python3 web/build_site.py --out _site
    python3 -m http.server -d _site 8000     # abrir http://localhost:8000

En CI lo corre .github/workflows/pages.yml en cada push a main.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Version de stlite fijada (CDN jsDelivr). Para actualizar: cambiar aqui y
# comprobar que la pagina carga y que las cinco pestanas funcionan.
STLITE_VERSION = "1.9.2"

# Paquetes que micropip instala dentro de Pyodide. numpy/scipy/pandas/
# matplotlib vienen precompilados en Pyodide; lifelines y reportlab son
# Python puro y se bajan de PyPI. Sin versiones fijadas a proposito: en el
# navegador mandan las versiones que trae la distribucion de Pyodide.
# (synapseclient, GEOparse, scikit-learn y pytest no los usa la app.)
REQUIREMENTS = ["numpy", "scipy", "pandas", "matplotlib", "lifelines", "reportlab"]

REPO_URL = "https://github.com/Danpc11/CRC-digital-twin"

INDEX_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <title>ColoQ · gemelo digital de cáncer colorrectal</title>
  <meta name="description" content="ColoQ: subtipificación molecular CMS1–4 de cáncer colorrectal con dinámica de atractores tipo Hopfield sobre un panel RT-qPCR de 10 genes. Corre completo en el navegador." />
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>◧</text></svg>" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__STLITE__/build/stlite.css" />
  <style>
    #notice {
      font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
      background: #1e2327; color: #e6e6e6; padding: 6px 14px;
      display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
    }
    #notice a { color: #8ecbff; }
  </style>
</head>
<body>
  <div id="notice">
    <span>◧ <strong>ColoQ</strong> — herramienta de investigación, no es un dispositivo médico.</span>
    <span>Corre completa en tu navegador: los archivos que subas no salen de tu computadora.
      La primera carga tarda ~1 min.</span>
    <span>__PATTERNS_NOTE__</span>
    <span><a href="__REPO__">Código</a> ·
      <a href="__REPO__/blob/main/LICENSE">Licencia (PolyForm Noncommercial 1.0.0)</a></span>
  </div>
  <div id="root"></div>
  <noscript>Esta aplicación requiere JavaScript.</noscript>
  <script type="module">
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__STLITE__/build/stlite.js";
    mount(
      {
        requirements: __REQUIREMENTS__,
        entrypoint: "app.py",
        files: __FILES__,
        streamlitConfig: {
          "client.toolbarMode": "viewer",
          "client.showErrorDetails": true,
        },
      },
      document.getElementById("root"),
    );
  </script>
</body>
</html>
"""


def resolve_patterns(out_app: Path) -> tuple[Path, str]:
    """Devuelve (ruta relativa dentro de la app, nota para el banner)."""
    real = ROOT / "web" / "calibrated_patterns.tsv"
    if real.exists():
        rel = Path("results_gse39582") / "calibrated_patterns.tsv"
        (out_app / rel.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(real, out_app / rel)
        return rel, "Patrones precargados: calibrados en GSE39582."

    demo = ROOT / "results_demo" / "calibrated_patterns.tsv"
    if not demo.exists():
        print("No hay web/calibrated_patterns.tsv: generando patrones DEMO "
              "(datos sinteticos) con `cli.py demo`...", flush=True)
        subprocess.run([sys.executable, "cli.py", "demo", "--output", "results_demo"],
                       cwd=ROOT, check=True)
    rel = Path("results_demo") / "calibrated_patterns.tsv"
    (out_app / rel.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(demo, out_app / rel)
    return rel, ("Patrones precargados: DEMO con datos sintéticos "
                 "(puedes subir tu propio calibrated_patterns.tsv en la barra lateral).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="_site", help="carpeta de salida (default: _site)")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out_app = out / "app"
    (out_app / "src").mkdir(parents=True)

    shutil.copy2(ROOT / "app.py", out_app / "app.py")
    rel_files = [Path("app.py")]
    for py in sorted((ROOT / "src").glob("*.py")):
        shutil.copy2(py, out_app / "src" / py.name)
        rel_files.append(Path("src") / py.name)

    patterns_rel, patterns_note = resolve_patterns(out_app)
    rel_files.append(patterns_rel)

    # Cada archivo se monta en el sistema de archivos virtual de Pyodide con
    # la misma ruta relativa que tiene en el repo, asi ROOT/"src" y
    # ROOT.glob("results*/...") de app.py funcionan sin cambios.
    files = {p.as_posix(): {"url": f"./app/{p.as_posix()}"} for p in rel_files}

    html = (INDEX_TEMPLATE
            .replace("__STLITE__", STLITE_VERSION)
            .replace("__REPO__", REPO_URL)
            .replace("__PATTERNS_NOTE__", patterns_note)
            .replace("__REQUIREMENTS__", json.dumps(REQUIREMENTS))
            .replace("__FILES__", json.dumps(files, indent=2)))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Sitio listo en {out}  ({len(files)} archivos montados en la app)")


if __name__ == "__main__":
    main()
