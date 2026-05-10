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
    "threadpoolctl",   # used by sptpalm_analysis to expand BLAS during localise
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
# inside sys._MEIPASS.  On Windows these are NOT auto-collected; bundle them
# here so the hook always finds them regardless of Python install layout.
#
# Uses _tkinter.TCL_VERSION / TK_VERSION (C constants — no Tk window needed)
# to build exact paths, with a broad glob fallback.
if sys.platform == "win32":
    _tcl_src = None
    _tk_src  = None

    # Method 1: _tkinter constants → exact versioned directory (most reliable)
    try:
        import _tkinter as _tki
        _tcl_ver = _tki.TCL_VERSION   # e.g. "8.6"
        _tk_ver  = _tki.TK_VERSION    # e.g. "8.6"
        print(f"[spec] _tkinter reports TCL={_tcl_ver}  TK={_tk_ver}")
        for _base in (sys.prefix, os.path.join(sys.prefix, "tcl")):
            _c = os.path.join(_base, f"tcl{_tcl_ver}")
            if os.path.isdir(_c):
                _tcl_src = _c
                break
            _c = os.path.join(_base, "tcl", f"tcl{_tcl_ver}")
            if os.path.isdir(_c):
                _tcl_src = _c
                break
        for _base in (sys.prefix, os.path.join(sys.prefix, "tcl")):
            _c = os.path.join(_base, f"tk{_tk_ver}")
            if os.path.isdir(_c):
                _tk_src = _c
                break
            _c = os.path.join(_base, "tcl", f"tk{_tk_ver}")
            if os.path.isdir(_c):
                _tk_src = _c
                break
    except Exception as _e:
        print(f"[spec] _tkinter version query failed: {_e}")

    # Method 2: broad glob across common install layouts
    if not (_tcl_src and os.path.isdir(_tcl_src)):
        for _search_root in (os.path.join(sys.prefix, "tcl"), sys.prefix):
            for _pat in ("tcl8*", "tcl9*"):
                _hits = sorted(
                    d for d in glob.glob(os.path.join(_search_root, _pat))
                    if os.path.isdir(d)
                )
                if _hits:
                    _tcl_src = _hits[-1]
                    break
            if _tcl_src:
                break

    if not (_tk_src and os.path.isdir(_tk_src)):
        for _search_root in (os.path.join(sys.prefix, "tcl"), sys.prefix):
            for _pat in ("tk8*", "tk9*"):
                _hits = sorted(
                    d for d in glob.glob(os.path.join(_search_root, _pat))
                    if os.path.isdir(d)
                )
                if _hits:
                    _tk_src = _hits[-1]
                    break
            if _tk_src:
                break

    # Method 3: check TCL_LIBRARY / TK_LIBRARY environment variables
    if not (_tcl_src and os.path.isdir(_tcl_src)):
        _env = os.environ.get("TCL_LIBRARY", "")
        if _env and os.path.isdir(_env):
            _tcl_src = _env
    if not (_tk_src and os.path.isdir(_tk_src)):
        _env = os.environ.get("TK_LIBRARY", "")
        if _env and os.path.isdir(_env):
            _tk_src = _env

    if _tcl_src and os.path.isdir(_tcl_src):
        datas.append((_tcl_src, "_tcl_data"))
        print(f"[spec] Bundling Tcl data : {_tcl_src}")
    else:
        print("[spec] WARNING: Could not locate Tcl data — CI fallback step must provide it")

    if _tk_src and os.path.isdir(_tk_src):
        datas.append((_tk_src, "_tk_data"))
        print(f"[spec] Bundling Tk  data : {_tk_src}")
    else:
        print("[spec] WARNING: Could not locate Tk data — CI fallback step must provide it")

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["app_tk.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=["hooks/rthook_tcl_tk.py"],
    excludes=["streamlit", "tornado", "altair", "bokeh", "IPython",
              "notebook", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ── Windows: ONEFILE mode (single self-contained .exe) ────────────────────────
# Extracting a zip on the user's Windows machine is fundamentally unreliable —
# Windows Explorer's built-in extractor silently drops files (long paths,
# special chars), and Defender quarantines .tcl scripts.  Onefile sidesteps
# this entirely: the bundle is a single .exe that extracts itself to %TEMP%
# at runtime where no user/AV can accidentally damage it.
if sys.platform == "win32":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="sptPALM",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,           # default %TEMP%\_MEIxxxxx
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    # macOS/Linux: ONEDIR mode (then wrap in .app/.dmg on macOS)
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
