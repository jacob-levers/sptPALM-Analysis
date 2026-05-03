# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for sptPALM Analysis Pipeline.

Build (from the project root):
  macOS:   pyinstaller sptpalm.spec
  Windows: pyinstaller sptpalm.spec

Outputs:
  macOS:   dist/sptPALM.app   (then wrap in DMG via CI)
  Windows: dist/sptPALM/      (zip via CI)
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import sys

# ── Hidden imports ─────────────────────────────────────────────────────────────
# PyInstaller's static analysis misses many scientific-Python sub-packages.

hidden = []
hidden += collect_submodules("trackpy")
hidden += collect_submodules("scipy")
hidden += collect_submodules("skimage")
hidden += collect_submodules("matplotlib")
hidden += collect_submodules("joblib")
hidden += [
    # CZI readers
    "czifile", "aicspylibczi",
    # TIFF
    "tifffile",
    # Tkinter backends
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends._backend_tk",
    # PIL
    "PIL._tkinter_finder", "PIL.Image", "PIL.ImageTk",
    # pandas / numpy internals
    "pandas._libs.tslibs.np_datetime",
    "pandas._libs.tslibs.nattype",
    "pandas._libs.tslibs.timedeltas",
    "pandas._libs.tslibs.timestamps",
    # multiprocessing (spawn support)
    "multiprocessing.pool",
    "multiprocessing.managers",
    # memory management (conditionally imported inside function)
    "psutil",
]

# ── Datas ─────────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("skimage")
datas += collect_data_files("matplotlib")
# Include the science engine alongside the GUI
datas += [("sptpalm_analysis.py", ".")]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["app_tk.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit", "tornado", "altair", "bokeh", "IPython",
              "notebook", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sptPALM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can corrupt scientific binaries; leave off
    console=False,      # no terminal window
    argv_emulation=False,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sptPALM",
)

# macOS .app bundle (ignored on Windows)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="sptPALM.app",
        icon=None,                          # swap in an .icns file here if you have one
        bundle_identifier="com.jacoblevers.sptpalm",
        info_plist={
            "CFBundleName": "sptPALM",
            "CFBundleDisplayName": "sptPALM Analysis Pipeline",
            "CFBundleVersion": "1.0.0",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            # Allow spawning child processes without the macOS sandbox blocking them
            "NSAppleEventsUsageDescription": "Required for analysis.",
        },
    )
