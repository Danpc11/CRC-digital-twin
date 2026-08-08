"""
launcher.py -- punto de entrada para el ejecutable empacado con PyInstaller.

Arranca Streamlit programaticamente (via su propio CLI interno) y abre
el navegador -- el usuario solo hace doble clic, sin terminal, sin
`pip install`, sin Python instalado por separado.

No confundir con `cli.py app` (ese asume Python + dependencias ya
instaladas en el sistema; este asume NADA, todo va empacado dentro del
ejecutable).
"""

import os
import sys
import socket
import threading
import time
import webbrowser
from pathlib import Path


def resource_path(relative: str) -> str:
    """
    Ruta a un recurso empacado -- funciona tanto corriendo el script
    normal (desarrollo) como dentro del ejecutable de PyInstaller
    (donde los archivos se extraen a una carpeta temporal referenciada
    por sys._MEIPASS).
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, relative)


def find_free_port(start: int = 8501, tries: int = 20) -> int:
    """Busca un puerto libre a partir de 8501 -- evita chocar si el usuario
    ya tiene otra instancia corriendo o el puerto por defecto esta ocupado."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start  # si todos ocupados, dejar que streamlit reporte el error


def open_browser_when_ready(url: str, timeout: float = 20.0):
    """Espera a que el servidor responda antes de abrir el navegador --
    si se abre antes de tiempo, el usuario ve 'conexion rechazada'."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.3)
    # si no respondio a tiempo, abrir de todas formas -- el usuario vera
    # el navegador cargando, mejor que no abrir nada
    webbrowser.open(url)


PORT = find_free_port()

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    app_path = resource_path("app.py")
    url = f"http://localhost:{PORT}"

    print(f"ColoQ arrancando en {url} ...")
    print("(cerrar esta ventana detiene la aplicacion)")

    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]
    sys.exit(stcli.main())
