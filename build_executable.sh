#!/usr/bin/env bash
# build_executable.sh -- construye ColoQ como ejecutable de un solo
# archivo (Linux/Mac). En Windows, correr los mismos dos comandos
# directamente en PowerShell (pip install / pyinstaller), este script
# de shell no aplica ahi.
#
# Requiere: Python 3.11+ y las dependencias de requirements.txt ya
# instaladas (o correr esto dentro del mismo entorno conda/venv del
# proyecto).
#
# Salida: dist/ColoQ (Linux/Mac) -- un solo archivo, ~170 MB.
# En Windows el resultado seria dist/ColoQ.exe.
#
# IMPORTANTE: el ejecutable generado es especifico de la plataforma y
# arquitectura donde se construye -- un ColoQ compilado en Linux NO
# corre en Windows ni Mac. Para distribuir en las tres plataformas hay
# que correr este build en cada una por separado (o usar CI multi-OS,
# ver nota al final).

set -euo pipefail

echo "=== Instalando PyInstaller ==="
# En algunos sistemas (Debian/Ubuntu con Python del sistema) pip
# bloquea la instalacion directa por PEP 668 ("externally-managed-
# environment"). Si el usuario ya esta en un venv/conda, esto no pasa
# y el primer intento basta.
pip install --quiet pyinstaller==6.21.0 2>/tmp/pip_err.log || {
    if grep -q "externally-managed-environment" /tmp/pip_err.log; then
        echo "  (entorno Python gestionado por el sistema -- reintentando con --break-system-packages)"
        pip install --quiet --break-system-packages pyinstaller==6.21.0
    else
        cat /tmp/pip_err.log
        exit 1
    fi
}

echo "=== Construyendo ejecutable (puede tardar varios minutos) ==="
pyinstaller ColoQ.spec

echo ""
echo "=== Build completo ==="
ls -lh dist/

echo ""
echo "Para probarlo:"
echo "  ./dist/ColoQ"
echo ""
echo "Se abre el navegador automaticamente en http://localhost:8501"
echo "(o el siguiente puerto libre si ese esta ocupado)."
echo ""
echo "NOTA -- distribucion multiplataforma real: este script solo construye"
echo "para el sistema operativo donde se corre. Para generar los tres"
echo "ejecutables (Windows/Mac/Linux) automaticamente sin tener las tres"
echo "maquinas a mano, se puede extender el workflow de GitHub Actions ya"
echo "existente (.github/workflows/tests.yml) con una matriz de SO que"
echo "corra este mismo build y suba los binarios como artifacts o release."
