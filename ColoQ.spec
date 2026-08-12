# ColoQ.spec -- build reproducible del ejecutable de un solo archivo.
#
# Construir con:
#   pyinstaller ColoQ.spec
#
# Probado end-to-end EN LINUX: el ejecutable resultante sirve la app
# real en http://127.0.0.1:8501 (HTTP 200, /_stcore/health responde
# "ok"). NO verificado en Windows/Mac por falta de esas maquinas --
# ver build-executables.yml para el CI que construye en las 4
# plataformas.
#
# BUG REAL ENCONTRADO EN MAC (no reproducido en Linux): 'lifelines' no
# se incluyo en el build de Mac -- "ModuleNotFoundError: No module
# named 'lifelines'" al arrancar. La deteccion automatica de
# dependencias de PyInstaller (seguir los imports del codigo) no es
# 100% consistente entre plataformas/versiones, especialmente con
# paquetes de cadena de dependencias compleja como lifelines (que a su
# vez depende de autograd, autograd-gamma, formulaic). Correccion:
# declarar EXPLICITAMENTE cada paquete cientifico critico via
# collect_all, en vez de confiar en la deteccion automatica.
#
# Los flags --collect-all y --copy-metadata para streamlit tambien son
# necesarios -- sin ellos el build aparenta completar bien pero el
# ejecutable falla al arrancar: le faltan los assets estaticos del
# frontend de Streamlit (collect_all) y revienta con
# "PackageNotFoundError" al hacer streamlit introspeccion de su propia
# version via importlib.metadata en tiempo de ejecucion (copy_metadata).
#
# Salida: dist/ColoQ (Linux/Mac) o dist/ColoQ.exe (Windows).
# Tamano esperado: ~180-200 MB -- incluye numpy/scipy/pandas/
# matplotlib/lifelines/streamlit completos, no hay forma de evitarlo
# sin quitar dependencias reales del proyecto.

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('app.py', '.'), ('src', 'src')]
binaries = []
hiddenimports = ['streamlit.web.cli']

# streamlit necesita copy_metadata ademas de collect_all -- hace
# introspeccion de su propia version en tiempo de ejecucion
datas += copy_metadata('streamlit')

# Cada paquete cientifico usado en tiempo de ejecucion por app.py y su
# cadena de imports (src/*.py) se declara explicitamente, sin confiar
# en la deteccion automatica -- es lo que fallo con lifelines en Mac.
for pkg in ['streamlit', 'lifelines', 'numpy', 'scipy', 'pandas', 'matplotlib', 'reportlab']:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ColoQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX comprime el binario pero dispara falsos
    upx_exclude=[],          # positivos de antivirus en Windows con frecuencia --
    runtime_tmpdir=None,     # no vale la pena el ahorro de tamano
    console=False,            # sin consola visible -- app.py ya tiene su
                               # propio boton "Salir" que mata el proceso
                               # (os._exit(0), verificado que funciona desde
                               # el hilo de Streamlit). Con consola=True, en
                               # Mac via clic derecho->Abrir, Terminal.app
                               # no cierra la ventana solo al terminar el
                               # proceso (configuracion propia del emulador,
                               # no controlable desde Python) -- quitarla
                               # evita el problema de raiz.
                               # RIESGO ACEPTADO: si algo falla ANTES de que
                               # el navegador abra, no hay ningun mensaje de
                               # error visible en ninguna plataforma. Si
                               # aparece un problema de arranque nuevo (como
                               # el de 'lifelines' faltante en Mac), va a ser
                               # mas dificil de diagnosticar sin consola --
                               # considerar volver a console=True temporalmente
                               # para depurar si eso pasa.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,   # agregar identidad de firma aqui para distribuir
    entitlements_file=None,   # en macOS sin advertencia de Gatekeeper
)
