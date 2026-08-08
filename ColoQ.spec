# ColoQ.spec -- build reproducible del ejecutable de un solo archivo.
#
# Construir con:
#   pyinstaller ColoQ.spec
#
# Probado end-to-end: el ejecutable resultante sirve la app real en
# http://127.0.0.1:8501 (HTTP 200, /_stcore/health responde "ok").
#
# Los flags --collect-all y --copy-metadata para streamlit son
# necesarios -- sin ellos el build aparenta completar bien pero el
# ejecutable falla al arrancar: le faltan los assets estaticos del
# frontend de Streamlit (collect_all) y revienta con
# "PackageNotFoundError" al hacer streamlit introspeccion de su propia
# version via importlib.metadata en tiempo de ejecucion (copy_metadata).
#
# Salida: dist/ColoQ (Linux/Mac) o dist/ColoQ.exe (Windows).
# Tamano esperado: ~170-180 MB -- incluye numpy/scipy/pandas/
# matplotlib/scikit-learn/streamlit completos, no hay forma de
# evitarlo sin quitar dependencias reales del proyecto.

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('app.py', '.'), ('src', 'src')]
binaries = []
hiddenimports = ['streamlit.web.cli']
datas += copy_metadata('streamlit')
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


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
    console=True,             # deja la consola visible -- el usuario ve el
                               # estado de arranque y como cerrar la app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,   # agregar identidad de firma aqui para distribuir
    entitlements_file=None,   # en macOS sin advertencia de Gatekeeper
)
