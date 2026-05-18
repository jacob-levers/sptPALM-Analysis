# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for FIREFLY (PySide6 + napari frontend, v2.0).

Build (from the project root):
  macOS:   pyinstaller sptpalm.spec
  Windows: pyinstaller sptpalm.spec

Outputs:
  macOS:   dist/FIREFLY.app  (then wrap in a DMG via CI)
  Windows: dist/FIREFLY.exe  (onefile)
"""

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, copy_metadata)
import os
import sys

# ── Hidden imports ───────────────────────────────────────────────────────────
hidden = []

# Scientific Python — collect every submodule because lazy imports under
# numpy._core / pandas._libs / scipy.* are missed by static analysis.
hidden += collect_submodules("numpy")
hidden += collect_submodules("pandas")
hidden += collect_submodules("trackpy")
hidden += collect_submodules("scipy")
hidden += collect_submodules("skimage")
hidden += collect_submodules("sklearn")
hidden += collect_submodules("matplotlib")
hidden += collect_submodules("joblib")
hidden += collect_submodules("aicspylibczi")
hidden += collect_submodules("imagecodecs")

# Qt / PySide6 — the Qt6 stack.  collect_submodules pulls in plugin loaders.
hidden += collect_submodules("PySide6")
hidden += collect_submodules("shiboken6")

# Napari + its plugin discovery + the vispy backend.  napari is loaded
# lazily by the Workspace tab; if any of these are missed at freeze time
# the tab quietly degrades to the "not installed" placeholder, which is
# acceptable — but we'd rather have it work.
hidden += collect_submodules("napari")
hidden += collect_submodules("vispy")
hidden += collect_submodules("magicgui")
hidden += collect_submodules("npe2")
hidden += collect_submodules("qtpy")
hidden += collect_submodules("pydantic")
hidden += collect_submodules("superqt")

# PyTorch — GPU localiser backend.  Only collect if the dep is installed;
# otherwise we'd inflate the bundle with non-existent stubs.
try:
    import torch  # noqa: F401
    hidden += collect_submodules("torch")
except ImportError:
    pass

hidden += [
    "czifile", "aicspylibczi", "imagecodecs",
    "tifffile",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_agg",
    "PIL._tkinter_finder", "PIL.Image", "PIL.ImageTk",
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
    "concurrent.futures",
    "concurrent.futures.thread",
    "psutil",
    "threadpoolctl",
    # FIREFLY's own helper modules — explicitly named so the spawned
    # subprocess can find them even if static analysis misses the
    # firefly_worker.run_analysis cross-module reference.
    "firefly_worker",
    "crash_reporter",
    "sptpalm_analysis",
    "cuda_installer",
    # Encoding tables sometimes missed in frozen builds
    "encodings.utf_8",
    "encodings.ascii",
    "encodings.latin_1",
]

# ── Datas ─────────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("skimage")
datas += collect_data_files("matplotlib")
datas += collect_data_files("aicspylibczi")
datas += collect_data_files("PIL")

# Napari ships configuration JSON + theme files + plugin manifests as
# package-data.  Collect them so the embedded viewer starts correctly.
try:
    datas += collect_data_files("napari")
    datas += collect_data_files("vispy")
    datas += collect_data_files("magicgui")
    datas += collect_data_files("npe2")
except Exception:
    pass

# `.dist-info` metadata for scientific packages.  Without this, pandas's
# `import_optional_dependency("numpy")` (and similar checks in napari,
# scikit-image, scipy, etc.) raises the misleading
#   "Missing optional dependency 'numpy'"
# error at runtime in the frozen build — even though the package itself
# is bundled and importable.  `importlib.metadata.version(pkg)` resolves
# against the .dist-info directory, which PyInstaller doesn't pick up
# automatically for collect_submodules.  copy_metadata fixes that.
for _pkg in ("numpy", "pandas", "scipy", "scikit-image", "scikit-learn",
             "matplotlib", "napari", "vispy", "magicgui", "npe2",
             "tifffile", "trackpy", "joblib", "Pillow", "psutil",
             "dask", "torch",
             # Qt bindings — napari probes for these via
             # `importlib.metadata.version("PySide6")` to decide which
             # Qt backend to use.  Without the .dist-info the probe
             # raises PackageNotFoundError and napari falls back to
             # "Cannot show napari window".
             "PySide6", "shiboken6", "qtpy", "superqt", "pydantic",
             "imageio", "scikit-image", "cachey", "lazy_loader",
             "pint", "app-model", "in-n-out", "psygnal"):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        # Package not installed at freeze time — fine, just skip it.
        pass

# Dask is napari's preferred lazy-load backend.  Pull its submodules
# in too so napari's image-loading code path doesn't bail with the
# "Dask array requirements are not installed" prompt.
try:
    hidden += collect_submodules("dask")
    datas   += collect_data_files("dask")
except Exception:
    pass

datas += [("sptpalm_analysis.py", ".")]
datas += [("firefly_worker.py",   ".")]
datas += [("crash_reporter.py",   ".")]
datas += [("cuda_installer.py",   ".")]

# Bundle the app icon PNG so the Qt window/dock icon can be loaded
# at runtime from sys._MEIPASS/assets/icon.png in frozen mode.
if os.path.isfile(os.path.join(SPECPATH, "assets", "icon.png")):
    datas += [(os.path.join(SPECPATH, "assets", "icon.png"), "assets")]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["app_qt.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit", "tornado", "altair", "bokeh", "IPython",
              "notebook", "jupyter",
              # Tkinter no longer used after v2.0 — exclude to keep
              # the bundle from carrying the Tcl/Tk runtime
              "tkinter", "_tkinter", "tkinterdnd2"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

_ICON_WIN = os.path.join(SPECPATH, "assets", "icon.ico") if (
    'SPECPATH' in dir() and os.path.isfile(
        os.path.join(SPECPATH, "assets", "icon.ico"))) else None
_ICON_MAC = os.path.join(SPECPATH, "assets", "icon.icns") if (
    'SPECPATH' in dir() and os.path.isfile(
        os.path.join(SPECPATH, "assets", "icon.icns"))) else None

# ── Windows: ONEFILE mode (single self-contained .exe) ───────────────────────
if sys.platform == "win32":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="FIREFLY",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=_ICON_WIN,
    )
else:
    # macOS / Linux: ONEDIR mode (wrapped in .app/.dmg on macOS)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="FIREFLY",
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
        name="FIREFLY",
    )

    if sys.platform == "darwin":
        app = BUNDLE(
            coll,
            name="FIREFLY.app",
            icon=_ICON_MAC,
            bundle_identifier="com.jacoblevers.firefly",
            info_plist={
                "CFBundleName": "FIREFLY",
                "CFBundleDisplayName": "FIREFLY — Fluorescence Inference & Reconstruction Engine",
                "CFBundleVersion": "2.0.0",
                "CFBundleShortVersionString": "2.0.0",
                "NSHighResolutionCapable": True,
                "LSMinimumSystemVersion": "11.0",
                "NSAppleEventsUsageDescription": "Required for analysis.",
                # napari's vispy backend uses Metal — disable the App Sandbox
                # so the GPU access works without prompting.  The app
                # doesn't need network or filesystem entitlements beyond
                # what macOS grants normally.
            },
        )
