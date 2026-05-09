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
import glob
import os
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
    # PIL image format plugins (loaded lazily — missed by static analysis)
    "PIL.ImageFile",
    "PIL.PngImagePlugin",
    "PIL.JpegImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.BmpImagePlugin",
    "PIL.GifImagePlugin",
    "PIL.WebPImagePlugin",
    "pandas._libs.tslibs.np_datetime",
    "pandas._libs.tslibs.nattype",
    "pandas._libs.tslibs.timedeltas",
    "pandas._libs.tslibs.timestamps",
    "multiprocessing.pool",
    "multiprocessing.managers",
    # concurrent.futures — used for ThreadPoolExecutor in analysis pipeline
    "concurrent.futures",
    "concurrent.futures.thread",
    "psutil",
    # encoding tables sometimes missed in frozen builds
    "encodings.utf_8",
    "encodings.ascii",
    "encodings.latin_1",
]

# ── Datas ─────────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("skimage")
datas += collect_data_files("matplotlib")
datas += collect_data_files("aicspylibczi")
datas += collect_data_files("PIL")           # Pillow data (ICC profiles etc.)
datas += [("sptpalm_analysis.py", ".")]

# ── Tcl/Tk data (Windows only) ─────────────────────────────────────────────────
# PyInstaller's pyi_rthkinter runtime hook looks for _tcl_data / _tk_data
# inside sys._MEIPASS.  On Windows these directories must be bundled
# explicitly — they are NOT auto-collected by PyInstaller's tkinter hook.
#
# Strategy: ask Tcl itself via a Tk root (most reliable), then fall back to
# globbing sys.prefix/tcl/ for tcl8*/tk8* or tcl9*/tk9* directories.
if sys.platform == "win32":
    _tcl_src = None
    _tk_src  = None

    # Method 1: instantiate Tk and query $tcl_library / $tk_library
    try:
        import tkinter as _tkmod
        _r = _tkmod.Tk()
        _r.withdraw()
        _tcl_src = _r.tk.exprstring("$tcl_library")
        _tk_src  = _r.tk.exprstring("$tk_library")
        _r.destroy()
        del _tkmod, _r
    except Exception as _e:
        print(f"[spec] tkinter query failed ({_e}), falling back to glob")

    # Method 2: glob sys.prefix/tcl/ for version directories
    if not (_tcl_src and os.path.isdir(_tcl_src)):
        for _pat in ("tcl8*", "tcl9*"):
            _hits = sorted(
                d for d in glob.glob(os.path.join(sys.prefix, "tcl", _pat))
                if os.path.isdir(d)
            )
            if _hits:
                _tcl_src = _hits[-1]
                break

    if not (_tk_src and os.path.isdir(_tk_src)):
        for _pat in ("tk8*", "tk9*"):
            _hits = sorted(
                d for d in glob.glob(os.path.join(sys.prefix, "tcl", _pat))
                if os.path.isdir(d)
            )
            if _hits:
                _tk_src = _hits[-1]
                break

    if _tcl_src and os.path.isdir(_tcl_src):
        datas.append((_tcl_src, "_tcl_data"))
        print(f"[spec] Bundling Tcl data : {_tcl_src}")
    else:
        print("[spec] WARNING: Could not locate Tcl data directory — "
              "_tcl_data will be missing from the bundle!")

    if _tk_src and os.path.isdir(_tk_src):
        datas.append((_tk_src, "_tk_data"))
        print(f"[spec] Bundling Tk  data : {_tk_src}")
    else:
        print("[spec] WARNING: Could not locate Tk data directory — "
              "_tk_data will be missing from the bundle!")

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
