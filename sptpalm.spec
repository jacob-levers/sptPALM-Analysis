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
hidden = []
hidden += collect_submodules("trackpy")
hidden += collect_submodules("scipy")
hidden += collect_submodules("skimage")
hidden += collect_submodules("matplotlib")
hidden += collect_submodules("joblib")
hidden += collect_submodules("aicspylibczi")
hidden += collect_submodules("imagecodecs")
hidden += [
    "czifile", "aicspylibczi", "imagecodecs",
    "tifffile",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends._backend_tk",
    "PIL._tkinter_finder", "PIL.Image", "PIL.ImageTk",
    "pandas._libs.tslibs.np_datetime",
    "pandas._libs.tslibs.nattype",
    "pandas._libs.tslibs.timedeltas",
    "pandas._libs.tslibs.timestamps",
    "multiprocessing.pool",
    "multiprocessing.managers",
    "psutil",
]

# ── Datas ─────────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("skimage")
datas += collect_data_files("matplotlib")
datas += collect_data_files("aicspylibczi")
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
    upx=False,
    console=False,
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
        icon=None,
        bundle_identifier="com.jacoblevers.sptpalm",
        info_plist={
            "CFBundleName": "sptPALM",
            "CFBundleDisplayName": "sptPALM Analysis Pipeline",
            "CFBundleVersion": "1.0.0",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSAppleEventsUsageDescription": "Required for analysis.",
        },
    )
