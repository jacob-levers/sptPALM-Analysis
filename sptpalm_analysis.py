#!/usr/bin/env python3
import multiprocessing
import sys
import os

# Fix macOS multiprocessing crashes — must be set before any other imports
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

"""
sptPALM Analysis Pipeline for Zeiss Elyra Microscope Data  (OPTIMISED)
=======================================================================
Supports .czi (native Zeiss) and .tif/.tiff files.
Pixel size and frame interval are read automatically from CZI metadata.

Speed optimisations vs the original version:
  - Background subtraction:  rolling_ball (~4800 ms/frame)
                           -> uniform_filter (~3 ms/frame)  [~1700x faster]
  - Preprocessing:           serial -> parallel across all CPU cores
  - Localisation:            single core -> all CPU cores
  - Memory:                  entire stack in RAM -> chunked processing
  - Progress:                silent -> live progress bars

Usage
-----
  # Typical usage — everything auto-detected from CZI:
  python sptpalm_analysis.py my_experiment.czi

  # With output folder:
  python sptpalm_analysis.py my_experiment.czi --output-dir C:\\results

  # Override metadata if needed:
  python sptpalm_analysis.py my_experiment.czi --pixel-size 0.104 --frame-interval 0.05

  # Limit CPU cores (default: all available):
  python sptpalm_analysis.py my_experiment.czi --workers 4

  # Use legacy rolling-ball background (slower but more accurate for uneven illumination):
  python sptpalm_analysis.py my_experiment.czi --bg-method rolling_ball

All options:
  --pixel-size       um per pixel (auto from CZI metadata)
  --frame-interval   seconds per frame (auto from CZI metadata)
  --diameter         PSF diameter in pixels, must be odd (default: 7)
  --minmass          Min integrated brightness (auto if omitted)
  --search-range     Max displacement between frames in px (default: 5)
  --memory           Frames a particle may vanish and reappear (default: 3)
  --min-track-length Discard tracks shorter than this (default: 5)
  --max-lagtime      MSD lag time points (default: 20)
  --bg-method        Background method: uniform_filter (fast) or rolling_ball (default: uniform_filter)
  --bg-radius        Background radius in pixels (default: 50)
  --workers          CPU cores to use (default: all)
  --chunk-size       Frames per processing chunk, reduce if RAM is low (default: 500)
  --channel          Channel index for multi-channel CZI (default: 0)
  --output-dir       Where to save results (default: same folder as input)
"""

import argparse
import multiprocessing
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

# ── Prevent BLAS/OpenBLAS/MKL thread oversubscription ─────────────────────────
# Scientific libraries (numpy, scipy, skimage) each spin up their own internal
# threads (via OpenBLAS or MKL).  When joblib ALSO spawns N_CPUS Python threads
# each calling numpy, you end up with N_CPUS² threads fighting for N_CPUS cores
# — performance collapses, especially on Windows.  Setting these to "1" caps
# internal BLAS parallelism so joblib's threads are the ones that scale linearly.
# This must be done *before* numpy is imported to take full effect.
for _blas_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_blas_env, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import trackpy as tp
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import correlate as _correlate2d
from scipy.stats import gaussian_kde
from skimage import filters, exposure
from tqdm import tqdm

# On Windows with console=False (PyInstaller GUI build), sys.stderr is None.
# tqdm writes to sys.stderr by default and crashes with AttributeError.
# Point it at a safe no-op stream instead.
import io as _io
_TQDM_FILE = sys.stderr if (sys.stderr is not None) else _io.StringIO()

def _tqdm(*args, **kwargs):
    """tqdm wrapper that always uses a valid output stream."""
    kwargs.setdefault("file", _TQDM_FILE)
    return tqdm(*args, **kwargs)

# Optional readers
try:
    import aicspylibczi
    HAS_AICS = True
except (ImportError, OSError):
    # OSError covers the case where the package is installed but its
    # bundled C++ shared library cannot be found (common in PyInstaller bundles)
    HAS_AICS = False

try:
    import czifile
    HAS_CZIFILE = True
except ImportError:
    HAS_CZIFILE = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

tp.quiet()

N_CPUS = multiprocessing.cpu_count()


# ══════════════════════════════════════════════════════════════════════════════
#  CZI LOADING + METADATA
# ══════════════════════════════════════════════════════════════════════════════

def _parse_czi_metadata(xml_str):
    meta = {"pixel_size_um": None, "frame_interval_s": None}
    if not xml_str:
        return meta
    try:
        root = ET.fromstring(xml_str)
        for dist in root.iter("Distance"):
            if dist.get("Id", "") in ("X", "Y"):
                el = dist.find("Value")
                if el is not None:
                    try:
                        val = float(el.text)
                        if 1e-9 < val < 1e-3:
                            meta["pixel_size_um"] = round(val * 1e6, 6)
                            break
                    except (TypeError, ValueError):
                        pass
        for tag in ("TimeIncrement", "Interval"):
            el = root.find(f".//{tag}")
            if el is not None:
                text = el.text or (
                    el.find("Value").text
                    if el.find("Value") is not None else None)
                if text:
                    try:
                        val = float(text)
                        if 1e-6 < val < 3600:
                            meta["frame_interval_s"] = round(val, 6)
                            break
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass
    return meta


def _dim_size(v, default=1):
    """aicspylibczi returns dims as int or (start, size) tuple."""
    if isinstance(v, tuple):
        return int(v[1])
    return int(v) if v is not None else default


def load_projection_fast(path, channel=0, max_frames=100):
    """
    Return a normalised [0,1] float32 mean-projection image using at most
    *max_frames* evenly-spaced frames.  Much faster than load_file() for
    large datasets because frames are read individually (no full stack load).
    Used by the ROI Editor so it doesn't have to load all 16K frames.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".czi":
        if HAS_AICS:
            czi  = aicspylibczi.CziFile(path)
            dims = dict(czi.get_dims_shape()[0])
            n_t  = _dim_size(dims.get("T"), 1)
            n_c  = _dim_size(dims.get("C"), 1)
            ch   = min(channel, n_c - 1)
            indices = np.linspace(0, n_t - 1, min(max_frames, n_t),
                                  dtype=int)
            frames = []
            for t in indices:
                img, _ = czi.read_image(T=int(t), C=ch)
                frame  = img.squeeze()
                if frame.ndim > 2:
                    frame = frame[0]
                frames.append(frame.astype(np.float32))
            proj = np.stack(frames).mean(axis=0)
        elif HAS_CZIFILE:
            # Use czifile subblock directory for per-frame access (avoids
            # loading the entire stack into RAM).
            # NOTE: No full-load fallback — czi.asarray() on a 16K-frame file
            # takes ~30 minutes and effectively hangs the app.  If subblock
            # reading fails we surface a clear error immediately.
            with czifile.CziFile(path) as czi:
                entries = list(czi.subblock_directory)
                if not entries:
                    raise RuntimeError(
                        "No subblocks found in CZI file. "
                        "Install aicspylibczi for full CZI support: "
                        "pip install aicspylibczi")
                n      = len(entries)
                step   = max(1, n // max_frames)
                frames = []
                for entry in entries[::step]:
                    try:
                        seg = entry.data_segment()
                        arr = np.asarray(seg.data(raw=False),
                                         dtype=np.float32).squeeze()
                        if arr.size == 0:
                            continue
                        # Peel leading size-1 / channel dims until we have 2D
                        while arr.ndim > 2:
                            arr = arr[0]
                        if arr.ndim == 2:
                            frames.append(arr)
                    except Exception:
                        continue   # skip unreadable subblocks; keep going
                if not frames:
                    raise RuntimeError(
                        "czifile could not decode any preview frames.\n"
                        "Fix: pip install imagecodecs\n"
                        "Or:  pip install aicspylibczi")
                proj = np.stack(frames).mean(axis=0)
        else:
            raise RuntimeError("Cannot read CZI: install aicspylibczi or czifile.")

    elif ext in (".tif", ".tiff"):
        if not HAS_TIFFFILE:
            raise RuntimeError("Run: pip install tifffile")
        with tifffile.TiffFile(path) as tif:
            n     = len(tif.pages)
            step  = max(1, n // max_frames)
            pages = [tif.pages[i].asarray().astype(np.float32)
                     for i in range(0, n, step)]
        proj = np.stack(pages).mean(axis=0)
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")

    lo, hi = proj.min(), proj.max()
    return (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)


def _find_czi_series(path):
    """
    Zeiss splits long acquisitions into companion files named:
        experiment.czi
        experiment(1).czi
        experiment(2).czi  …

    Given the primary path, return an ordered list of all files in that
    series (including the primary itself).  If no companions are found the
    list contains only the original path.
    """
    import glob, re
    directory = os.path.dirname(path) or "."
    basename  = os.path.splitext(os.path.basename(path))[0]

    # Strip any trailing "(N)" so we get the root name
    root = re.sub(r"\(\d+\)$", "", basename).rstrip()

    # Collect all matching files
    pattern  = os.path.join(directory, glob.escape(root) + "*.czi")
    candidates = sorted(glob.glob(pattern))

    # Keep only: root.czi  and  root(N).czi  (not unrelated names)
    series_re = re.compile(
        r"^" + re.escape(root) + r"(\(\d+\))?\.czi$", re.IGNORECASE)
    series = [f for f in candidates
              if series_re.match(os.path.basename(f))]

    # Natural sort so (1) < (2) < (10)
    def _nat_key(s):
        m = re.search(r"\((\d+)\)\.czi$", s, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    series.sort(key=_nat_key)

    if len(series) > 1:
        print(f"  Multi-file CZI series detected ({len(series)} files):")
        for f in series:
            print(f"    {os.path.basename(f)}")
    return series if series else [path]


class _Cancelled(Exception):
    """Raised inside loaders when a stop_event fires mid-load."""
    pass


def _load_single_czi(path, channel=0, stop_event=None):
    """Load a single CZI file and return (stack, pixel_size_um, frame_interval_s).

    stop_event : threading.Event or None
        If set, loading is aborted and _Cancelled is raised.
        The check runs every 500 frames so the UI stays responsive.
    """
    def _chk():
        if stop_event is not None and stop_event.is_set():
            raise _Cancelled()

    if HAS_AICS:
        czi  = aicspylibczi.CziFile(path)
        xml  = czi.meta if hasattr(czi, "meta") else None
        meta = _parse_czi_metadata(xml)
        dims = dict(czi.get_dims_shape()[0])
        n_t  = _dim_size(dims.get("T"), 1)
        n_c  = _dim_size(dims.get("C"), 1)
        ch   = min(channel, n_c - 1)
        print(f"  Frames: {n_t}  |  Channels: {n_c}  |  Using channel: {ch}", flush=True)
        # Read first frame to discover H×W, then pre-allocate the full array.
        # This avoids building a Python list + np.stack which doubles peak RAM.
        img0, _ = czi.read_image(T=0, C=ch)
        f0 = img0.squeeze()
        if f0.ndim > 2:
            f0 = f0[0]
        H, W  = f0.shape
        stack = np.empty((n_t, H, W), dtype=np.float32)
        stack[0] = f0.astype(np.float32)
        for t in range(1, n_t):
            img, _ = czi.read_image(T=t, C=ch)
            frame  = img.squeeze()
            if frame.ndim > 2:
                frame = frame[0]
            stack[t] = frame.astype(np.float32)
            if t % 500 == 0:
                print(f"  Loading: {t}/{n_t} frames...", flush=True)
                _chk()
        return stack, meta["pixel_size_um"], meta["frame_interval_s"]

    if HAS_CZIFILE:
        # Use subblock-by-subblock reading — czi.asarray() loads the whole
        # file at once and hangs for large (16K-frame) datasets.
        # Note: czifile needs imagecodecs to decompress JPEG XR frames
        # (the default compression for Zeiss Elyra).
        # Install with: pip install imagecodecs
        with czifile.CziFile(path) as czi:
            xml  = czi.metadata()
            meta = _parse_czi_metadata(xml)
            entries = list(czi.subblock_directory)
            if not entries:
                raise RuntimeError(
                    "No subblocks found in CZI file.\n"
                    "Try: pip install aicspylibczi imagecodecs")
            n      = len(entries)
            print(f"  Subblocks: {n}  |  Using channel: {channel}", flush=True)
            frames = []
            _first_err = None   # log first decode error for diagnosis
            for i, entry in enumerate(entries):
                try:
                    seg = entry.data_segment()
                    arr = np.asarray(seg.data(raw=False),
                                     dtype=np.float32).squeeze()
                    if arr.size == 0:
                        continue
                    while arr.ndim > 2:
                        arr = arr[0]
                    if arr.ndim == 2:
                        frames.append(arr)
                except Exception as exc:
                    if _first_err is None:
                        _first_err = exc
                    continue
                if i % 500 == 0 and i > 0:
                    print(f"  Loading: {i}/{n} subblocks...", flush=True)
                    _chk()
            if not frames:
                hint = (f"\nFirst decode error: {_first_err}" if _first_err else "")
                raise RuntimeError(
                    "czifile could not decode any frames from this CZI.\n"
                    "This usually means the JPEG XR codec is missing.\n"
                    "Fix: pip install imagecodecs\n"
                    "Or:  pip install aicspylibczi"
                    + hint)
            data = np.stack(frames)
        return data, meta["pixel_size_um"], meta["frame_interval_s"]

    raise RuntimeError(
        "Cannot read CZI: install aicspylibczi or czifile.\n"
        "Run:  pip install aicspylibczi imagecodecs")


def load_czi(path, channel=0, stop_event=None):
    # Detect multi-file series (Zeiss splits large datasets into companion files)
    series = _find_czi_series(path)

    if len(series) == 1:
        # Single file — straightforward load
        print(f"  Loading CZI: {path}")
        stack, px_um, fi_s = _load_single_czi(path, channel, stop_event)
        print(f"  Shape: {stack.shape}  (T x Y x X)", flush=True)
        return stack, px_um, fi_s

    # Multi-file series — load each file and concatenate along time axis.
    # Metadata (pixel size, frame interval) is taken from the first file.
    print(f"  Loading CZI series: {len(series)} files", flush=True)
    stacks   = []
    px_um_out  = None
    fi_s_out   = None
    for i, fpath in enumerate(series):
        print(f"  [{i+1}/{len(series)}] {os.path.basename(fpath)}", flush=True)
        st, px, fi = _load_single_czi(fpath, channel, stop_event)
        stacks.append(st)
        if i == 0:
            px_um_out = px
            fi_s_out  = fi

    combined = np.concatenate(stacks, axis=0)
    print(f"  Combined shape: {combined.shape}  (T x Y x X)", flush=True)
    return combined, px_um_out, fi_s_out


def _parse_ome_metadata(tif):
    """
    Extract pixel size (µm) and frame interval (s) from a tifffile.TiffFile.

    Checks in priority order:
      1. OME-XML embedded in the first page (OME-TIFF standard)
      2. ImageJ metadata dict (files saved by Fiji/ImageJ)
      3. XResolution TIFF tag (gives pixels per unit; combined with ResolutionUnit)

    Returns (pixel_size_um, frame_interval_s) — either value may be None if
    the corresponding metadata is absent.
    """
    px_um = None
    fi_s  = None

    # ── 1. OME-XML ────────────────────────────────────────────────────────────
    try:
        ome = tif.ome_metadata          # returns XML string or None
        if ome:
            root = ET.fromstring(ome)
            # Strip namespace: '{http://www.openmicroscopy.org/Schemas/OME/...}Pixels'
            ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
            prefix = f"{{{ns}}}" if ns else ""

            def _find_el(tag):
                # Try with and without namespace
                el = root.find(f".//{prefix}{tag}")
                if el is None:
                    el = root.find(f".//{tag}")
                return el

            pixels = _find_el("Pixels")
            if pixels is not None:
                # PhysicalSizeX is stored in µm by OME convention
                psx = pixels.get("PhysicalSizeX")
                psx_unit = pixels.get("PhysicalSizeXUnit", "µm")
                if psx:
                    try:
                        v = float(psx)
                        # Convert to µm if necessary
                        unit_lc = psx_unit.lower().replace("μ", "u").replace("µ", "u")
                        if unit_lc in ("nm", "nanometer", "nanometre"):
                            v /= 1000.0
                        elif unit_lc in ("mm", "millimeter", "millimetre"):
                            v *= 1000.0
                        if 0.001 < v < 100:
                            px_um = round(v, 6)
                    except (TypeError, ValueError):
                        pass

                ti = pixels.get("TimeIncrement")
                ti_unit = pixels.get("TimeIncrementUnit", "s")
                if ti:
                    try:
                        v = float(ti)
                        unit_lc = ti_unit.lower()
                        if unit_lc in ("ms", "millisecond", "milliseconds"):
                            v /= 1000.0
                        elif unit_lc in ("min", "minute", "minutes"):
                            v *= 60.0
                        if 1e-6 < v < 3600:
                            fi_s = round(v, 6)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    # ── 2. ImageJ metadata ────────────────────────────────────────────────────
    try:
        ij = tif.imagej_metadata        # dict or None
        if ij:
            if px_um is None:
                # ImageJ stores resolution as pixels/unit in TIFF XResolution tag.
                # We read it from the tag below; here we can get unit from ij dict.
                pass  # handled in section 3

            if fi_s is None:
                finterval = ij.get("finterval")  # seconds
                if finterval is not None:
                    try:
                        v = float(finterval)
                        if 1e-6 < v < 3600:
                            fi_s = round(v, 6)
                    except (TypeError, ValueError):
                        pass

                # Some ImageJ files store frame rate instead
                if fi_s is None:
                    fps = ij.get("fps")
                    if fps is not None:
                        try:
                            v = float(fps)
                            if v > 0:
                                fi_s = round(1.0 / v, 6)
                        except (TypeError, ValueError):
                            pass
    except Exception:
        pass

    # ── 3. XResolution TIFF tag (works for ImageJ TIFFs) ─────────────────────
    try:
        if px_um is None and tif.pages:
            page = tif.pages[0]
            xres = page.tags.get("XResolution")
            runit = page.tags.get("ResolutionUnit")
            if xres is not None:
                val = xres.value
                # Value is a rational (numerator, denominator) or plain float
                if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                    pixels_per_unit = val[0] / val[1]
                else:
                    pixels_per_unit = float(val)
                if pixels_per_unit > 0:
                    # ResolutionUnit: 1=no units, 2=inch, 3=cm
                    unit_code = runit.value if runit is not None else 2
                    if unit_code == 3:          # centimetres
                        um_per_pixel = 1e4 / pixels_per_unit
                    elif unit_code == 2:        # inches
                        um_per_pixel = 25400.0 / pixels_per_unit
                    else:
                        um_per_pixel = None

                    # ImageJ often uses µm as "unit" and encodes pixels/µm
                    # by hacking the ResolutionUnit.  Check ij metadata unit.
                    try:
                        ij = tif.imagej_metadata or {}
                        ij_unit = ij.get("unit", "")
                        if ij_unit.lower() in ("um", "µm", "μm", "micron"):
                            um_per_pixel = 1.0 / pixels_per_unit
                    except Exception:
                        pass

                    if um_per_pixel and 0.001 < um_per_pixel < 100:
                        px_um = round(um_per_pixel, 6)
    except Exception:
        pass

    return px_um, fi_s


def load_tif(path, stop_event=None):
    if not HAS_TIFFFILE:
        raise RuntimeError("Run: pip install tifffile")
    print(f"  Loading TIF: {path}")
    with tifffile.TiffFile(path) as tif:
        px_um, fi_s = _parse_ome_metadata(tif)
        n_pages = len(tif.pages)
        if stop_event is not None and n_pages > 500:
            # Large file: load page-by-page so stop can interrupt mid-load
            frames = []
            for i, page in enumerate(tif.pages):
                frames.append(page.asarray().astype(np.float32))
                if i % 500 == 0 and i > 0:
                    print(f"  Loading: {i}/{n_pages} frames...", flush=True)
                    if stop_event.is_set():
                        raise _Cancelled()
            stack = np.stack(frames)
        else:
            stack = tif.asarray().astype(np.float32)

    if   stack.ndim == 2: stack = stack[np.newaxis]
    elif stack.ndim == 4:
        stack = stack[:, 0] if stack.shape[1] == 1 else stack.mean(axis=1)
    print(f"  Shape: {stack.shape}  (T x Y x X)")

    if px_um is not None:
        print(f"  Pixel size  : {px_um} µm  (from file metadata)")
    if fi_s is not None:
        print(f"  Frame interval: {fi_s} s  (from file metadata)")

    return stack, px_um, fi_s


def load_file(path, channel=0, stop_event=None):
    ext = os.path.splitext(path)[1].lower()
    if   ext == ".czi":            return load_czi(path, channel, stop_event)
    elif ext in (".tif", ".tiff"): return load_tif(path, stop_event)
    else: sys.exit(f"ERROR: Unsupported file '{ext}'. Use .czi or .tif")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (fast path + parallel)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_fast(frame, bg_radius=50, sigma=1.0):
    """
    Fast background subtraction using uniform_filter.
    ~1700x faster than rolling_ball with comparable results for PALM data.
    """
    bg        = uniform_filter(frame, size=int(bg_radius * 2 + 1))
    corrected = np.clip(frame - bg, 0, None)
    smoothed  = filters.gaussian(corrected, sigma=sigma, preserve_range=True)
    mn, mx    = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)
    return smoothed.astype(np.float32)


def _preprocess_rolling(frame, bg_radius=50, sigma=1.0):
    """Legacy rolling-ball background subtraction (slow but thorough)."""
    from skimage.restoration import rolling_ball
    bg        = rolling_ball(frame, radius=bg_radius)
    corrected = np.clip(frame - bg, 0, None)
    smoothed  = filters.gaussian(corrected, sigma=sigma, preserve_range=True)
    mn, mx    = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)
    return smoothed.astype(np.float32)


def preprocess_stack(stack, bg_radius=50, bg_method="uniform_filter",
                     workers=N_CPUS):
    n = len(stack)
    fn = _preprocess_fast if bg_method == "uniform_filter" else _preprocess_rolling

    print(f"  Background method : {bg_method}")
    print(f"  Workers           : {workers} / {N_CPUS} CPU cores")
    t0 = time.perf_counter()

    if workers == 1:
        processed = [fn(f, bg_radius) for f in
                     _tqdm(stack, desc="  Preprocessing", unit="fr", ncols=70)]
    else:
        processed = Parallel(n_jobs=workers, prefer="threads")(
            delayed(fn)(f, bg_radius)
            for f in _tqdm(stack, desc="  Preprocessing", unit="fr", ncols=70))

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/n*1000:.1f} ms/frame)")
    return np.stack(processed)




# ══════════════════════════════════════════════════════════════════════════════
#  ROI  —  simple intensity threshold
# ══════════════════════════════════════════════════════════════════════════════

def auto_threshold(image_norm, method="auto"):
    """
    Determine an intensity threshold automatically from the normalised
    mean projection using scikit-image thresholding algorithms.

    Parameters
    ----------
    image_norm : float array (Y, X), values in [0, 1]
    method     : "auto"     — tries otsu, li, triangle; picks best for sptPALM
                 "otsu"     — maximises inter-class variance
                 "li"       — minimises cross-entropy (good for sparse cells)
                 "triangle" — geometric method, good for large dark backgrounds

    Returns
    -------
    threshold : float in [0, 1]
    chosen    : str — which method was used
    all_vals  : dict — thresholds from all three methods for reference
    """
    from skimage.filters import threshold_otsu, threshold_li, threshold_triangle

    results = {}
    try:    results["otsu"]     = float(threshold_otsu(image_norm))
    except: results["otsu"]     = None
    try:    results["li"]       = float(threshold_li(image_norm))
    except: results["li"]       = None
    try:    results["triangle"] = float(threshold_triangle(image_norm))
    except: results["triangle"] = None

    print(f"  Auto-threshold candidates:")
    for name, val in results.items():
        print(f"    {name:<10} : {val:.4f}" if val is not None
              else f"    {name:<10} : failed")

    if method != "auto":
        chosen = method
        val    = results.get(method)
        if val is None:
            print(f"  WARNING: {method} failed, falling back to otsu")
            chosen = "otsu"
            val    = results["otsu"]
    else:
        # For sptPALM the cell typically occupies < 30% of the frame.
        # Triangle handles large dark backgrounds best in this scenario.
        # Fall back to Li then Otsu if triangle is unavailable.
        for preferred in ("triangle", "li", "otsu"):
            if results[preferred] is not None:
                chosen = preferred
                val    = results[preferred]
                break

    print(f"  Selected method : {chosen}  ->  threshold = {val:.4f}")
    return val, chosen, results


def build_roi_mask_mean(stack, threshold=0.15, smooth_sigma=5):
    """
    Mean-projection ROI: one mask derived from the average intensity across
    ALL frames.  Stable and recommended for most experiments.

    Works by averaging all T frames into a single image, smoothing it, then
    thresholding.  Because fluorophores blink on and off, individual frames
    are mostly dark even inside the cell — the mean projection reveals the
    underlying cell structure by accumulating signal over time.

    The mean projection is normalised to [0,1] before thresholding so that
    the threshold is always relative to the brightest region in the image,
    regardless of fluorophore density or acquisition settings.

    Returns
    -------
    mask : bool array (Y, X)
    """
    from skimage.morphology import binary_closing, disk
    mean_proj = stack.mean(axis=0)
    smoothed  = filters.gaussian(mean_proj, sigma=smooth_sigma,
                                 preserve_range=True)
    # Normalise to [0,1] so threshold is relative to brightest region
    mn, mx = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed_norm = (smoothed - mn) / (mx - mn)
    else:
        smoothed_norm = smoothed
    mask = binary_closing(smoothed_norm > threshold, disk(5))
    return mask, mean_proj


def build_roi_mask_perframe(stack, threshold=0.15, smooth_sigma=5):
    """
    Per-frame ROI: a separate mask computed independently for each frame.

    Each frame is smoothed and thresholded individually, so the ROI can
    change shape frame-to-frame.  Useful when illumination drifts during
    acquisition or when imaging moving/growing cells.

    Note: because individual sptPALM frames are very sparse (only a handful
    of fluorophores are visible at once), per-frame masks are inherently
    noisier than the mean-projection mask.  Use with caution and always
    check the preview.

    Returns
    -------
    masks : bool array (T, Y, X) — one mask per frame
    mean_proj : float array (Y, X) — mean projection for display purposes
    """
    from skimage.morphology import binary_closing, disk
    T = len(stack)
    masks = np.zeros(stack.shape, dtype=bool)
    for t in _tqdm(range(T), desc="  Building per-frame masks",
                   unit="fr", ncols=70):
        smoothed = filters.gaussian(stack[t], sigma=smooth_sigma,
                                    preserve_range=True)
        # Normalise each frame to [0,1] before thresholding
        mn, mx = smoothed.min(), smoothed.max()
        if mx > mn:
            smoothed = (smoothed - mn) / (mx - mn)
        masks[t] = binary_closing(smoothed > threshold, disk(5))
    mean_proj = stack.mean(axis=0)
    return masks, mean_proj


def build_roi_mask(stack=None, threshold=None, smooth_sigma=5,
                   mode="mean", threshold_method="auto", save_path=None,
                   precomputed_mean_proj=None):
    """
    Build ROI mask(s) from a stack or a pre-computed mean projection.

    Parameters
    ----------
    stack                 : preprocessed float32 stack (T x Y x X).
                            Not required when precomputed_mean_proj is supplied.
    threshold             : manual intensity cutoff on [0,1].
                            If None, determined automatically.
    smooth_sigma          : Gaussian sigma (px) before thresholding (default: 5)
    mode                  : "mean"     — one mask from mean projection (default)
                            "perframe" — separate mask per frame (needs stack)
    threshold_method      : "auto" | "otsu" | "li" | "triangle"
    save_path             : if given, saves a preview PNG for inspection
    precomputed_mean_proj : float32 (Y, X) normalised [0,1] mean projection.
                            When supplied, stack is not needed for mean-mode ROI,
                            saving the memory cost of holding the full stack.
                            perframe mode silently falls back to mean mode here.

    Returns
    -------
    mask : bool array (Y, X) for mode="mean"
           bool array (T, Y, X) for mode="perframe"
    """
    # ── Mean projection ────────────────────────────────────────────────────────
    if precomputed_mean_proj is not None:
        mean_proj_norm = precomputed_mean_proj.astype(np.float32)
        mn, mx = mean_proj_norm.min(), mean_proj_norm.max()
        if mx > mn:
            mean_proj_norm = (mean_proj_norm - mn) / (mx - mn)
        if mode == "perframe":
            print("  NOTE: perframe ROI mode needs the full stack; "
                  "falling back to mean mode.")
            mode = "mean"
    elif stack is not None:
        mean_proj_norm = stack.mean(axis=0)
        mn, mx = mean_proj_norm.min(), mean_proj_norm.max()
        if mx > mn:
            mean_proj_norm = (mean_proj_norm - mn) / (mx - mn)
    else:
        raise ValueError(
            "build_roi_mask: supply either 'stack' or 'precomputed_mean_proj'.")

    auto_method_used = None
    if threshold is None:
        threshold, auto_method_used, all_thresh = auto_threshold(
            mean_proj_norm, method=threshold_method)
    else:
        print(f"  Threshold : {threshold:.4f}  [manual]")
        all_thresh = None

    # ── Build mask ─────────────────────────────────────────────────────────────
    if mode == "perframe" and stack is not None:
        mask, mean_proj = build_roi_mask_perframe(stack, threshold, smooth_sigma)
        display_mask = mask.mean(axis=0) > 0.5
        mode_label   = "Per-frame"
    else:
        if stack is not None:
            mask, mean_proj = build_roi_mask_mean(stack, threshold, smooth_sigma)
        else:
            # Streaming mode: build mask directly from precomputed mean projection
            from skimage.morphology import binary_closing, disk
            smoothed = filters.gaussian(mean_proj_norm, sigma=smooth_sigma,
                                        preserve_range=True)
            smn, smx = smoothed.min(), smoothed.max()
            snorm = (smoothed - smn) / (smx - smn) if smx > smn else smoothed
            mask = binary_closing(snorm > threshold, disk(5))
            mean_proj = mean_proj_norm  # use normalised version for display
        display_mask = mask
        mode_label   = "Mean projection"

    # ── Stats ──────────────────────────────────────────────────────────────────
    n_px  = display_mask.sum()
    total = display_mask.size
    print(f"  ROI mode  : {mode_label}")
    print(f"  ROI area  : {n_px:,} / {total:,} pixels  "
          f"({100*n_px/total:.1f}% of frame)")

    # ── Preview ────────────────────────────────────────────────────────────────
    if save_path:
        import matplotlib.pyplot as plt
        # 4 panels when auto-thresholding so we can show all three candidates
        if all_thresh is not None:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d1117")
            # Panel 4: comparison of auto-threshold candidates on histogram
            ax_h = axes[3]
            ax_h.set_facecolor("#0d1117")
            ax_h.hist(mean_proj_norm.ravel(), bins=200,
                      color="#58a6ff", alpha=0.7, log=True)
            colors_thresh = {"otsu":"#f78166", "li":"#7ed321", "triangle":"#f5a623"}
            for name, val in all_thresh.items():
                if val is not None:
                    lw  = 2.5 if name == auto_method_used else 1.2
                    ls  = "-"  if name == auto_method_used else "--"
                    ax_h.axvline(val, color=colors_thresh[name], lw=lw, ls=ls,
                                 label=f"{name}={val:.3f}"
                                       + (" *" if name == auto_method_used else ""))
            ax_h.legend(fontsize=8, facecolor="#0d1117",
                        edgecolor="#30363d", labelcolor="#e6edf3")
            ax_h.set_xlabel("Normalised intensity", color="#e6edf3", fontsize=9)
            ax_h.set_ylabel("Pixel count (log)", color="#e6edf3", fontsize=9)
            ax_h.set_title("Threshold comparison  (* selected)",
                           color="white", fontsize=9)
            ax_h.tick_params(colors="#e6edf3")
            for sp in ax_h.spines.values(): sp.set_edgecolor("#30363d")
            panel_axes = axes[:3]
        else:
            fig, panel_axes = plt.subplots(1, 3, figsize=(15, 5),
                                           facecolor="#0d1117")

        thresh_label = (f"auto:{auto_method_used}={threshold:.3f}"
                        if auto_method_used else f"manual={threshold:.3f}")
        titles = ["Mean projection",
                  f"ROI mask  ({mode_label})",
                  f"Overlay  ({thresh_label})"]
        imgs  = [mean_proj, display_mask.astype(float), mean_proj]
        cmaps = ["inferno", "Greens", "inferno"]
        for ax, img, ttl, cm in zip(panel_axes, imgs, titles, cmaps):
            ax.set_facecolor("#0d1117")
            ax.imshow(img, cmap=cm, origin="lower")
            ax.set_title(ttl, color="white", fontsize=10)
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        panel_axes[2].contour(display_mask.astype(float), levels=[0.5],
                              colors=["#58a6ff"], linewidths=[1.5])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ROI preview saved -> {save_path}")

    return mask


def apply_roi_mask(locs, mask):
    """
    Filter a localisations DataFrame to keep only points inside the ROI mask.

    Parameters
    ----------
    locs : DataFrame with columns 'x', 'y', 'frame', in pixels
    mask : bool array (Y, X)      — mean-projection mode: same mask every frame
           bool array (T, Y, X)   — per-frame mode: each frame gets its own mask

    Returns
    -------
    Filtered DataFrame
    """
    xi = np.clip(locs["x"].values.astype(int), 0, mask.shape[-1] - 1)
    yi = np.clip(locs["y"].values.astype(int), 0, mask.shape[-2] - 1)

    if mask.ndim == 2:
        # Mean-projection mode — same mask for every localisation
        inside = mask[yi, xi]
    else:
        # Per-frame mode — look up the mask for each localisation's frame
        fi = np.clip(locs["frame"].values.astype(int), 0, mask.shape[0] - 1)
        inside = mask[fi, yi, xi]

    filtered  = locs[inside].reset_index(drop=True)
    n_removed = len(locs) - len(filtered)
    mode_str  = "per-frame" if mask.ndim == 3 else "mean-projection"
    print(f"  ROI filter ({mode_str}): kept {len(filtered):,} / {len(locs):,} "
          f"localisations  ({n_removed:,} outside ROI removed)")
    return filtered

# ══════════════════════════════════════════════════════════════════════════════
#  DRIFT CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def correct_drift(locs, n_seg_frames=200, upsampling=4, smooth_sigma=1.5):
    """
    Reference-free drift correction via cross-correlation of localization
    density maps (simplified RCC approach; Wang et al. 2014, Nat Methods).

    The acquisition is divided into time segments.  A 2-D localization density
    histogram is built for each segment at ``upsampling``× the raw pixel
    resolution.  Consecutive histograms are cross-correlated (FFT) to measure
    the inter-segment drift.  The cumulative, Gaussian-smoothed drift trajectory
    is interpolated to per-frame resolution and subtracted from every
    localization.

    Applied *before* linking so that drift-corrected positions produce better
    trajectories.

    Parameters
    ----------
    locs          : DataFrame with 'x', 'y', 'frame' columns (in pixels)
    n_seg_frames  : target number of frames per time segment (default 200).
                    Smaller → finer time resolution but fewer localisations
                    per segment (noisier cross-correlation).
    upsampling    : density-map super-resolution factor.  upsampling=4 gives
                    ~25 nm accuracy at 0.1 µm/px (default 4).
    smooth_sigma  : Gaussian smoothing sigma in units of *segments* applied to
                    the raw drift trajectory before interpolation (default 1.5).

    Returns
    -------
    locs_corrected : DataFrame with corrected 'x' and 'y'
    drift_df       : DataFrame with columns ['frame', 'dx', 'dy'] (pixels)
    """
    if len(locs) == 0:
        return locs.copy(), pd.DataFrame({"frame": [0], "dx": [0.0], "dy": [0.0]})

    x = locs["x"].values.astype(np.float64)
    y = locs["y"].values.astype(np.float64)
    f = locs["frame"].values.astype(int)

    n_frames   = int(f.max()) + 1
    n_segments = max(4, int(np.ceil(n_frames / n_seg_frames)))
    n_segments = min(n_segments, max(2, len(locs) // 10))  # need ≥10 locs/seg

    print(f"  Drift correction : {n_segments} segments "
          f"(~{n_frames // n_segments} frames each, upsampling={upsampling})")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    W = max(int((x_max - x_min) * upsampling) + 1, 16)
    H = max(int((y_max - y_min) * upsampling) + 1, 16)

    seg_bounds  = np.linspace(0, n_frames, n_segments + 1).astype(int)
    seg_centers = (seg_bounds[:-1] + seg_bounds[1:]) / 2.0

    # ── Build upsampled density maps ──────────────────────────────────────────
    density_maps = []
    seg_counts   = []
    for i in range(n_segments):
        sel = (f >= seg_bounds[i]) & (f < seg_bounds[i + 1])
        seg_counts.append(int(sel.sum()))
        dm  = np.zeros((H, W), dtype=np.float32)
        if sel.sum() > 0:
            xi = np.clip(((x[sel] - x_min) * upsampling).astype(int), 0, W - 1)
            yi = np.clip(((y[sel] - y_min) * upsampling).astype(int), 0, H - 1)
            np.add.at(dm, (yi, xi), 1.0)
            dm = gaussian_filter(dm, sigma=upsampling * 0.7)   # spread spots
        density_maps.append(dm)

    print(f"  Localisations/segment: min {min(seg_counts):,}, "
          f"max {max(seg_counts):,}")

    # ── Cross-correlate consecutive density maps ───────────────────────────────
    dx_steps = [0.0]
    dy_steps = [0.0]
    for i in range(1, n_segments):
        if seg_counts[i - 1] < 5 or seg_counts[i] < 5:
            dx_steps.append(0.0); dy_steps.append(0.0)
            continue
        corr = _correlate2d(density_maps[i - 1], density_maps[i],
                            mode="full", method="fft")
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        # peak position relative to centre gives the shift of segment i vs i-1
        dy_steps.append(float(peak[0] - (H - 1)))
        dx_steps.append(float(peak[1] - (W - 1)))

    # ── Integrate → cumulative drift in density-map pixels ────────────────────
    dx_cum = np.cumsum(dx_steps)
    dy_cum = np.cumsum(dy_steps)

    # Smooth then convert to localization pixels
    dx_sm = gaussian_filter1d(dx_cum, sigma=smooth_sigma) / upsampling
    dy_sm = gaussian_filter1d(dy_cum, sigma=smooth_sigma) / upsampling

    # Zero-centre so overall position is preserved
    dx_sm -= dx_sm.mean()
    dy_sm -= dy_sm.mean()

    rng_x, rng_y = float(np.ptp(dx_sm)), float(np.ptp(dy_sm))
    print(f"  Drift range  x={rng_x:.3f} px  y={rng_y:.3f} px")

    # ── Interpolate to every frame ────────────────────────────────────────────
    frame_arr = np.arange(n_frames, dtype=float)
    ix = interp1d(seg_centers, dx_sm, kind="linear",
                  bounds_error=False, fill_value=(dx_sm[0], dx_sm[-1]))
    iy = interp1d(seg_centers, dy_sm, kind="linear",
                  bounds_error=False, fill_value=(dy_sm[0], dy_sm[-1]))
    drift_x = ix(frame_arr)
    drift_y = iy(frame_arr)

    # ── Subtract from localisations ────────────────────────────────────────────
    locs_out = locs.copy()
    fi       = np.clip(f, 0, n_frames - 1)
    locs_out["x"] = x - drift_x[fi]
    locs_out["y"] = y - drift_y[fi]

    drift_df = pd.DataFrame({"frame": frame_arr.astype(int),
                             "dx": drift_x, "dy": drift_y})
    return locs_out, drift_df


# ══════════════════════════════════════════════════════════════════════════════
#  LOCALISATION  (parallel + chunked)
# ══════════════════════════════════════════════════════════════════════════════

def _ram_strategy(stack, headroom: float = 0.75) -> tuple[bool, float, float]:
    """
    Decide whether the full preprocessed stack fits in free RAM.

    Returns (use_fast, free_gb, needed_gb).
    Falls back to streaming if psutil is not installed.
    """
    needed_gb = stack.nbytes / 1e9   # preprocessed copy ≈ same dtype/shape
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
        return needed_gb < free_gb * headroom, free_gb, needed_gb
    except ImportError:
        return False, 0.0, needed_gb


def _fast_preprocess_and_localise(stack, diameter=7, minmass=None, percentile=64,
                                   bg_radius=50, bg_method="uniform_filter",
                                   workers=N_CPUS, chunk_size=500,
                                   preview_cb=None):
    """
    Fast path (ample RAM): preprocess the full stack in parallel, then localise
    in parallel chunks.  Faster than streaming because all preprocessing jobs
    run simultaneously rather than serially.

    Returns (locs, mean_proj_norm, minmass_used)  — same contract as the stream path.
    """
    import gc
    if diameter % 2 == 0:
        diameter += 1

    stack_pp = preprocess_stack(stack, bg_radius=bg_radius,
                                bg_method=bg_method, workers=workers)

    if minmass is None:
        minmass = float(np.percentile(stack_pp[min(5, len(stack_pp) - 1)], 99) * 0.4)
        print(f"  Auto minmass: {minmass:.4f}")

    mean_proj = stack_pp.mean(axis=0).astype(np.float32)
    mn, mx    = mean_proj.min(), mean_proj.max()
    if mx > mn:
        mean_proj = (mean_proj - mn) / (mx - mn)

    locs = localise_particles(stack_pp, diameter=diameter, minmass=minmass,
                              percentile=percentile, workers=workers,
                              chunk_size=chunk_size, preview_cb=preview_cb)
    del stack_pp
    gc.collect()
    return locs, mean_proj, minmass


def preprocess_and_localise_adaptive(stack, diameter=7, minmass=None, percentile=64,
                                     bg_radius=50, bg_method="uniform_filter",
                                     workers=N_CPUS, chunk_size=500,
                                     ram_headroom: float = 0.75,
                                     preview_cb=None, stop_event=None):
    """
    Adaptive dispatcher — automatically selects the fastest strategy that fits
    in available RAM.

    Fast path   (plenty of RAM): full parallel preprocessing → parallel localisation.
                                 Scales with both CPU count and RAM size.
    Stream path (tight RAM):     one chunk preprocessed + localised + discarded at
                                 a time.  Peak extra RAM = one chunk only.

    The decision is made at runtime using psutil to query free memory.
    ``ram_headroom`` (default 0.75) means the preprocessed copy must fit in
    75 % of currently free RAM so the OS and other processes retain a buffer.

    Returns (locs, mean_proj_norm, minmass_used)
    """
    use_fast, free_gb, needed_gb = _ram_strategy(stack, headroom=ram_headroom)

    if use_fast:
        print(f"  RAM strategy : FAST (parallel)   — "
              f"{free_gb:.1f} GB free, {needed_gb:.1f} GB needed")
        return _fast_preprocess_and_localise(
            stack, diameter, minmass, percentile,
            bg_radius, bg_method, workers, chunk_size,
            preview_cb=preview_cb)
    else:
        print(f"  RAM strategy : STREAM (low-mem)  — "
              f"{free_gb:.1f} GB free, {needed_gb:.1f} GB needed")
        return preprocess_and_localise_stream(
            stack, diameter, minmass, percentile,
            bg_radius, bg_method, workers, chunk_size,
            preview_cb=preview_cb, stop_event=stop_event)


def preprocess_and_localise_stream(stack, diameter=7, minmass=None, percentile=64,
                                   bg_radius=50, bg_method="uniform_filter",
                                   workers=N_CPUS, chunk_size=500,
                                   preview_cb=None, stop_event=None):
    """
    Memory-efficient single streaming pass: preprocess + localise without ever
    materialising the full preprocessed stack in RAM.

    Each chunk is preprocessed, localised, and immediately discarded, so peak
    extra memory above the raw stack is one chunk (~chunk_size frames).
    For a 10 000-frame 512×512 stack this cuts peak RAM from ~2× to ~1× stack size.

    Parameters
    ----------
    stack    : raw float32 stack (T x Y x X)
    minmass  : if None, auto-detected from the first preprocessed chunk

    Returns
    -------
    locs             : DataFrame of all localised particles
    mean_proj_norm   : float32 (Y, X) normalised [0,1] mean of preprocessed frames
                       — suitable for ROI thresholding
    minmass          : the minmass value actually used
    """
    import gc
    if diameter % 2 == 0:
        diameter += 1

    fn       = _preprocess_fast if bg_method == "uniform_filter" else _preprocess_rolling
    n_frames = len(stack)
    n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))
    workers_ = max(1, min(workers, N_CPUS))

    print(f"  Mode      : streaming preprocess + localise  (low memory)")
    print(f"  Diameter  : {diameter}px  |  bg_method: {bg_method}")
    print(f"  Chunks    : {n_chunks} × ~{chunk_size} frames  |  workers: {workers_}")
    t0 = time.perf_counter()

    # ── First chunk: preprocess now so we can auto-detect minmass ─────────────
    first_end  = min(chunk_size, n_frames)
    first_pp   = np.stack(Parallel(n_jobs=workers_, prefer="threads")(
        delayed(fn)(f, bg_radius) for f in stack[:first_end]))

    if minmass is None:
        minmass = float(np.percentile(first_pp[min(5, first_end - 1)], 99) * 0.4)
        print(f"  Auto minmass: {minmass:.4f}")
    else:
        print(f"  Minmass   : {minmass:.4f}")

    # ── Stream all chunks ──────────────────────────────────────────────────────
    all_locs  = []
    mean_acc  = first_pp.sum(axis=0).astype(np.float64)
    frame_count = len(first_pp)

    # Localise first chunk (already preprocessed)
    locs0 = tp.batch(first_pp, diameter=diameter, minmass=minmass,
                     percentile=percentile, processes=1)
    if len(locs0) > 0:
        all_locs.append(locs0)

    # Emit preview for the first chunk (middle frame + its localisations)
    if preview_cb is not None:
        try:
            mid = len(first_pp) // 2
            preview_frame = first_pp[mid]
            mid_locs = locs0[locs0["frame"] == mid] if len(locs0) > 0 else None
            xs = mid_locs["x"].values if mid_locs is not None and len(mid_locs) > 0 else []
            ys = mid_locs["y"].values if mid_locs is not None and len(mid_locs) > 0 else []
            preview_cb(mid, preview_frame, xs, ys, n_frames)
        except Exception:
            pass

    del first_pp
    gc.collect()

    # Remaining chunks
    for i in _tqdm(range(1, n_chunks), desc="  Streaming", unit="chunk", ncols=70):
        # Honour a stop request between chunks
        if stop_event is not None and stop_event.is_set():
            print("  Streaming stopped by user.")
            break

        start     = i * chunk_size
        end       = min(start + chunk_size, n_frames)
        chunk_pp  = np.stack(Parallel(n_jobs=workers_, prefer="threads")(
            delayed(fn)(f, bg_radius) for f in stack[start:end]))

        mean_acc   += chunk_pp.sum(axis=0)
        frame_count += len(chunk_pp)

        locs_i = tp.batch(chunk_pp, diameter=diameter, minmass=minmass,
                          percentile=percentile, processes=1)

        if len(locs_i) > 0:
            locs_i = locs_i.copy()
            locs_i["frame"] += start
            all_locs.append(locs_i)

        # Live preview: middle frame of this chunk with detected particles
        if preview_cb is not None:
            try:
                mid_local = len(chunk_pp) // 2
                preview_frame = chunk_pp[mid_local]
                mid_global    = start + mid_local
                if len(locs_i) > 0:
                    sel = locs_i[locs_i["frame"] == mid_global]
                    xs, ys = sel["x"].values, sel["y"].values
                else:
                    xs, ys = [], []
                preview_cb(mid_global, preview_frame, xs, ys, n_frames)
            except Exception:
                pass

        del chunk_pp
        gc.collect()

    # ── Mean projection (normalised) ──────────────────────────────────────────
    mean_proj = (mean_acc / frame_count).astype(np.float32)
    mn, mx    = mean_proj.min(), mean_proj.max()
    if mx > mn:
        mean_proj = (mean_proj - mn) / (mx - mn)

    result  = pd.concat(all_locs, ignore_index=True) if all_locs else pd.DataFrame()
    elapsed = time.perf_counter() - t0
    print(f"  Found {len(result):,} localisations in {elapsed:.1f}s  "
          f"({n_frames / elapsed:.0f} frames/s)")
    return result, mean_proj, minmass


def _localise_chunk(chunk, diameter, minmass, percentile, frame_offset):
    """Localise one chunk and apply global frame offset. Always uses processes=1
    so that parallelism is handled at the chunk level via joblib threads,
    which avoids the macOS multiprocessing-fork crash."""
    locs = tp.batch(chunk, diameter=diameter, minmass=minmass,
                    percentile=percentile, processes=1)
    if len(locs) > 0:
        locs = locs.copy()
        locs["frame"] += frame_offset
    return locs


def localise_particles(stack, diameter=7, minmass=0.1, percentile=64,
                       workers=N_CPUS, chunk_size=500, preview_cb=None):
    if diameter % 2 == 0:
        diameter += 1

    n_frames = len(stack)
    n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))
    workers  = max(1, min(workers, N_CPUS))

    print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}")
    print(f"  Chunks    : {n_chunks} x ~{chunk_size} frames  |  Workers: {workers}")

    t0      = time.perf_counter()
    chunks  = np.array_split(stack, n_chunks)
    offsets = [i * chunk_size for i in range(len(chunks))]

    # Parallelise across chunks using threads (safe on macOS — no fork).
    # tp.batch internally uses NumPy/SciPy which release the GIL, so
    # thread-level parallelism gives real speedup.
    if preview_cb is None:
        chunk_results = Parallel(n_jobs=workers, prefer="threads")(
            delayed(_localise_chunk)(chunk, diameter, minmass, percentile, offset)
            for chunk, offset in _tqdm(
                zip(chunks, offsets), total=n_chunks,
                desc="  Localising", unit="chunk", ncols=70))
    else:
        # Sequential collection so we can emit preview frames as we go.
        # We still parallelise inside each chunk via tp.batch (processes=1
        # but trackpy releases the GIL during heavy NumPy work).
        chunk_results = []
        for chunk, offset in _tqdm(zip(chunks, offsets), total=n_chunks,
                                   desc="  Localising", unit="chunk", ncols=70):
            df = _localise_chunk(chunk, diameter, minmass, percentile, offset)
            chunk_results.append(df)
            try:
                mid_local  = len(chunk) // 2
                mid_global = offset + mid_local
                preview_frame = chunk[mid_local]
                if df is not None and len(df) > 0:
                    sel = df[df["frame"] == mid_global]
                    xs, ys = sel["x"].values, sel["y"].values
                else:
                    xs, ys = [], []
                preview_cb(mid_global, preview_frame, xs, ys, n_frames)
            except Exception:
                pass

    valid = [df for df in chunk_results if df is not None and len(df) > 0]
    result = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

    elapsed = time.perf_counter() - t0
    print(f"  Found {len(result):,} localisations in {elapsed:.1f}s  "
          f"({n_frames / elapsed:.0f} frames/s)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  LINKING
# ══════════════════════════════════════════════════════════════════════════════

def link_trajectories(locs, search_range=5, memory=3, min_len=5, max_len=None):
    print(f"  Linking (search_range={search_range}px, memory={memory}) ...")
    t0 = time.perf_counter()
    try:
        linked = tp.link(locs, search_range=search_range, memory=memory)
    except Exception as exc:
        if "SubnetOversizeException" in type(exc).__name__ or "Subnetwork" in str(exc):
            # Particle density is too high for the recursive solver at this
            # search_range.  The nonrecursive strategy handles arbitrarily
            # large subnetworks and is recommended for dense sptPALM data.
            print(f"  WARNING: SubnetOversizeException — switching to "
                  f"nonrecursive linker (consider reducing Search range)")
            linked = tp.link(locs, search_range=search_range, memory=memory,
                             link_strategy="nonrecursive")
        else:
            raise
    filtered = tp.filter_stubs(linked, min_len)
    if max_len is not None and max_len > 0:
        lengths  = filtered.groupby("particle")["frame"].count()
        keep     = lengths[lengths <= max_len].index
        filtered = filtered[filtered["particle"].isin(keep)]
        print(f"  Max-length filter (<={max_len}): {filtered['particle'].nunique():,} remain")
    elapsed = time.perf_counter() - t0
    n       = filtered["particle"].nunique()
    max_str = str(max_len) if max_len else "inf"
    print(f"  {n:,} trajectories (len {min_len}-{max_str}) in {elapsed:.1f}s")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  MSD + DIFFUSION  (custom parallel — replaces slow tp.imsd)
# ══════════════════════════════════════════════════════════════════════════════

def msd_linear(t, D, offset):
    return 4 * D * t + offset


def classify_motion(alpha):
    if   alpha < 0.5: return "Immobile"
    elif alpha < 0.9: return "Confined"
    elif alpha < 1.1: return "Brownian"
    else:             return "Directed"


def _msd_and_fit_one(xy_um, frames, pid, lag_times, max_lagtime, n_fit):
    """
    Compute per-track MSD array AND fit D + alpha in a single pass.

    Uses actual frame numbers (not row indices) so that gaps in a trajectory
    caused by memory-linking do not inflate the MSD.  Only pairs of positions
    whose frame difference exactly equals the requested lag are included.
    """
    msd_vals = np.full(max_lagtime, np.nan)
    for lag_idx, lag in enumerate(range(1, max_lagtime + 1)):
        if lag >= len(xy_um):
            break
        # Only use pairs where the actual frame separation equals lag
        frame_diff = frames[lag:] - frames[:-lag]
        valid      = frame_diff == lag
        if valid.sum() > 0:
            d = xy_um[lag:][valid] - xy_um[:-lag][valid]
            msd_vals[lag_idx] = np.mean(d[:, 0] ** 2 + d[:, 1] ** 2)

    # Fit using first n_fit lag times
    t   = lag_times[:n_fit]
    m   = msd_vals[:n_fit]
    ok  = np.isfinite(m) & (m > 0)
    D = alpha = np.nan
    if ok.sum() >= 3:
        try:    alpha = np.polyfit(np.log(t[ok]), np.log(m[ok]), 1)[0]
        except: pass
        try:
            popt, _ = curve_fit(msd_linear, t[ok], m[ok], p0=[0.01, 0],
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                                maxfev=2000)
            D = popt[0]
        except: pass

    motion = classify_motion(alpha) if np.isfinite(alpha) else "Unknown"

    # Confinement radius: mean distance of all positions from the track centroid
    centroid       = xy_um.mean(axis=0)
    conf_radius_um = float(np.mean(np.sqrt(np.sum((xy_um - centroid) ** 2, axis=1))))

    return pid, msd_vals, dict(particle=pid, D=D, alpha=alpha, motion=motion,
                               confinement_radius_um=conf_radius_um)


def compute_msd_and_fit(tracks, pixel_size, frame_interval,
                        max_lagtime=20, n_fit=5, workers=N_CPUS):
    """
    Single parallel pass that computes both MSD and diffusion fits.
    Replaces tp.imsd + tp.emsd + separate fit loop — all in one go.
    """
    lag_times  = np.arange(1, max_lagtime + 1) * frame_interval
    grouped    = tracks.groupby("particle")
    pid_list   = list(grouped.groups.keys())
    n_tracks   = len(pid_list)

    print(f"  Tracks to process : {n_tracks:,}")
    print(f"  Workers           : {workers} / {N_CPUS} CPU cores")
    t0 = time.perf_counter()

    results = Parallel(n_jobs=workers, prefer="threads")(
        delayed(_msd_and_fit_one)(
            grouped.get_group(pid)[["x", "y"]].values * pixel_size,
            grouped.get_group(pid)["frame"].values,
            pid, lag_times, max_lagtime, n_fit)
        for pid in _tqdm(pid_list, desc="  MSD + fitting", unit="track", ncols=70))

    elapsed = time.perf_counter() - t0
    rate    = n_tracks / elapsed
    print(f"  Done in {elapsed:.1f}s  ({rate:.0f} tracks/s)")

    # Assemble imsd DataFrame  (rows = lag index, cols = particle id)
    msd_matrix = np.array([r[1] for r in results]).T   # shape: (max_lagtime, n_tracks)
    imsd_df    = pd.DataFrame(msd_matrix,
                              index=np.arange(1, max_lagtime + 1),
                              columns=[r[0] for r in results])

    # Ensemble MSD = nanmean across tracks at each lag
    emsd_series = pd.Series(np.nanmean(msd_matrix, axis=1),
                            index=np.arange(1, max_lagtime + 1))

    diff_df = pd.DataFrame([r[2] for r in results])

    # Merge per-track mean localisation precision (pixels → nm)
    if "ep" in tracks.columns:
        ep_nm = (tracks.groupby("particle")["ep"].mean() * pixel_size * 1000
                 ).rename("loc_precision_nm").reset_index()
        diff_df = diff_df.merge(ep_nm, on="particle", how="left")

    return imsd_df, emsd_series, diff_df


# ══════════════════════════════════════════════════════════════════════════════
#  JUMP DISTANCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_jdd(tracks, pixel_size_um, frame_interval_s, n_components=2):
    """
    Jump Distance Distribution (JDD) analysis.

    Extracts single-frame displacements from all tracks, then fits the
    empirical CDF to a mixture of 2D Brownian populations:

        CDF(r) = 1 - Σᵢ fᵢ · exp(–r² / 4Dᵢ Δt)

    Fitting the CDF (rather than histogram) avoids binning artefacts and
    gives robust estimates even with short tracks — ideal for sptPALM where
    many tracks have only 2–5 frames.

    Parameters
    ----------
    n_components : 1, 2, or 3

    Returns
    -------
    dict or None (if too few jumps to fit)
    """
    dt = frame_interval_s
    jumps = []

    for pid, grp in tracks.groupby("particle"):
        grp    = grp.reset_index(drop=True).sort_values("frame")
        frames = grp["frame"].values
        x      = grp["x"].values * pixel_size_um
        y      = grp["y"].values * pixel_size_um
        for i in range(len(frames) - 1):
            if frames[i + 1] - frames[i] == 1:   # consecutive frames only
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                jumps.append(np.sqrt(dx * dx + dy * dy))

    jumps = np.asarray(jumps, dtype=np.float64)
    if len(jumps) < 30:
        return None

    r_sorted = np.sort(jumps)
    cdf_emp  = np.arange(1, len(r_sorted) + 1) / len(r_sorted)

    # ── CDF model definitions ─────────────────────────────────────────────────
    def _cdf1(r, D1):
        return 1.0 - np.exp(-r ** 2 / (4 * D1 * dt))

    def _cdf2(r, D1, D2, f1):
        f2 = 1.0 - f1
        return 1.0 - f1 * np.exp(-r**2 / (4*D1*dt)) \
                   - f2 * np.exp(-r**2 / (4*D2*dt))

    def _cdf3(r, D1, D2, D3, f1, f2):
        f3 = 1.0 - f1 - f2
        return (1.0 - f1 * np.exp(-r**2 / (4*D1*dt))
                    - f2 * np.exp(-r**2 / (4*D2*dt))
                    - f3 * np.exp(-r**2 / (4*D3*dt)))

    configs = {
        1: (_cdf1, [0.05],                   ([1e-6],        [100.0])),
        2: (_cdf2, [0.005, 0.3, 0.4],        ([1e-6, 1e-5, 0.01], [10.0, 100.0, 0.99])),
        3: (_cdf3, [0.003, 0.05, 0.5, 0.3, 0.35],
                                              ([1e-6, 1e-5, 1e-4, 0.01, 0.01],
                                               [1.0, 10.0, 100.0, 0.97, 0.97])),
    }

    model, p0, (lb, ub) = configs[n_components]
    try:
        popt, _ = curve_fit(model, r_sorted, cdf_emp,
                            p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception:
        return None

    # ── Extract sorted (D, fraction) pairs ───────────────────────────────────
    if n_components == 1:
        pairs = [(popt[0], 1.0)]
    elif n_components == 2:
        pairs = sorted([(popt[0], popt[2]), (popt[1], 1.0 - popt[2])])
    else:
        f3    = 1.0 - popt[3] - popt[4]
        pairs = sorted([(popt[0], popt[3]), (popt[1], popt[4]), (popt[2], f3)])

    D_values  = [p[0] for p in pairs]
    fractions = [p[1] for p in pairs]

    # ── PDF for plotting ──────────────────────────────────────────────────────
    # Rayleigh-like: f_i(r) = r/(2DᵢΔt) · exp(–r²/4DᵢΔt)
    r_range = np.linspace(0, np.percentile(jumps, 99.5), 500)

    def _pdf_component(r, D):
        return (r / (2 * D * dt)) * np.exp(-r**2 / (4 * D * dt))

    pdfs = [frac * _pdf_component(r_range, D)
            for D, frac in zip(D_values, fractions)]
    pdf_total = np.sum(pdfs, axis=0)

    return {
        "jumps":         jumps,
        "D_values":      D_values,
        "fractions":     fractions,
        "n_components":  n_components,
        "n_jumps":       len(jumps),
        "r_range":       r_range,
        "pdfs":          pdfs,           # per-component PDF arrays
        "pdf_total":     pdf_total,
        "cdf_r":         r_sorted,
        "cdf_empirical": cdf_emp,
        "cdf_fit":       model(r_sorted, *popt),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TURNING ANGLES
# ══════════════════════════════════════════════════════════════════════════════

def compute_turning_angles(tracks):
    """
    For each track with ≥3 points, compute step-to-step turning angles in degrees.

    Returns a flat np.array of all angles across all tracks.
    """
    all_angles = []
    for pid, grp in tracks.groupby("particle"):
        grp = grp.reset_index(drop=True).sort_values("frame")
        xy  = grp[["x", "y"]].values
        if len(xy) < 3:
            continue
        v1 = np.diff(xy, axis=0)[:-1]   # shape (n-2, 2)
        v2 = np.diff(xy, axis=0)[1:]    # shape (n-2, 2)
        dot   = np.sum(v1 * v2, axis=1)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        cos_a = dot / (norm1 * norm2 + 1e-12)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_a))
        all_angles.append(angles)
    if all_angles:
        return np.concatenate(all_angles)
    return np.array([])


# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE FRACTION OVER TIME
# ══════════════════════════════════════════════════════════════════════════════

def compute_mobile_fraction_over_time(tracks, diff_df, frame_interval,
                                       window_frames=100):
    """
    Compute mobile fraction in sliding windows of `window_frames` frames.

    Returns DataFrame with columns: time_s, mobile_fraction, n_tracks.
    Only windows with ≥5 tracks are included.
    """
    if len(tracks) == 0 or len(diff_df) == 0:
        return pd.DataFrame(columns=["time_s", "mobile_fraction", "n_tracks"])

    track_times = tracks.groupby("particle")["frame"].mean().reset_index()
    track_times.columns = ["particle", "mean_frame"]
    merged = track_times.merge(diff_df[["particle", "motion"]], on="particle", how="inner")

    max_frame = int(tracks["frame"].max())
    windows   = range(0, max_frame, window_frames)
    rows = []
    for w in windows:
        sel = merged[(merged["mean_frame"] >= w) &
                     (merged["mean_frame"] < w + window_frames)]
        total = len(sel)
        if total < 5:
            continue
        mobile = sel["motion"].isin(["Free diffusion", "Directed", "Brownian"]).sum()
        rows.append({
            "time_s":          (w + window_frames / 2) * frame_interval,
            "mobile_fraction": mobile / total,
            "n_tracks":        total,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER ANALYSIS  (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════

def compute_clusters(locs, pixel_size_um, eps_um=0.05, min_samples=5,
                     max_locs=250_000):
    from sklearn.cluster import DBSCAN
    from scipy.spatial import ConvexHull
    xy = locs[["x", "y"]].values * pixel_size_um
    subsampled = len(xy) > max_locs
    if subsampled:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xy), max_locs, replace=False)
        xy = xy[idx]
    labels = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(xy)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    rows = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        pts = xy[labels == c]
        n = len(pts)
        try:
            area = ConvexHull(pts).volume if n >= 3 else np.nan
        except Exception:
            area = np.nan
        density = n / area if (area and area > 0) else np.nan
        rows.append({"cluster_id": int(c), "n_locs": int(n),
                     "area_um2": area, "density_locs_per_um2": density,
                     "centroid_x_um": pts[:,0].mean(),
                     "centroid_y_um": pts[:,1].mean()})
    return labels, pd.DataFrame(rows), int(n_clusters), xy


# ══════════════════════════════════════════════════════════════════════════════
#  DWELL TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_dwell_times(tracks, diff_df, frame_interval):
    confined_pids = diff_df[diff_df["motion"].isin(["Confined", "Immobile"])]["particle"]
    rows = []
    for pid in confined_pids:
        n = len(tracks[tracks["particle"] == pid])
        if n > 0:
            rows.append({"particle": int(pid), "dwell_time_s": n * frame_interval})
    dwell_df = pd.DataFrame(rows)
    tau = np.nan
    if len(dwell_df) >= 10:
        try:
            dt = np.sort(dwell_df["dwell_time_s"].values)
            cdf = np.arange(1, len(dt) + 1) / len(dt)
            popt, _ = curve_fit(lambda t, tau: 1 - np.exp(-t / tau),
                                dt, cdf, p0=[dt.mean()], bounds=(1e-6, np.inf),
                                maxfev=2000)
            tau = float(popt[0])
        except Exception:
            pass
    return dwell_df, tau


# ══════════════════════════════════════════════════════════════════════════════
#  MOMENT SCALING SPECTRUM  (MSS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mss(tracks, pixel_size_um, frame_interval, max_lagtime=10):
    q_values = [1, 2, 3, 4]
    results = []
    for pid, grp in (tracks.reset_index(drop=True)
                          .sort_values("frame").groupby("particle")):
        xy = grp[["x", "y"]].values * pixel_size_um
        n = len(xy)
        if n < max(max_lagtime + 2, 6):
            continue
        gammas = []
        lag_arr = list(range(1, min(max_lagtime + 1, n // 2)))
        if len(lag_arr) < 3:
            continue
        for q in q_values:
            moments = []
            for lag in lag_arr:
                r = np.sqrt(np.sum((xy[lag:] - xy[:-lag]) ** 2, axis=1))
                moments.append(np.mean(r ** q))
            log_t = np.log(np.array(lag_arr, dtype=float) * frame_interval)
            log_m = np.log(np.array(moments) + 1e-15)
            gammas.append(np.polyfit(log_t, log_m, 1)[0])
        mss_slope = np.polyfit(q_values, gammas, 1)[0]
        results.append({"particle": int(pid), "mss_slope": float(mss_slope)})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

MC   = {"Immobile":"#e05252","Confined":"#f5a623","Brownian":"#4a90d9",
        "Directed":"#7ed321","Unknown":"#aaaaaa"}
MORD = ["Immobile","Confined","Brownian","Directed"]


def _draw_track(grp, color, ax, lw=0.8, alpha=0.6):
    xy = grp[["x","y"]].values
    if len(xy) < 2: return
    for i in range(len(xy)-1):
        a = 0.2 + (alpha-0.2)*i/max(len(xy)-2, 1)
        ax.plot(xy[i:i+2,0], xy[i:i+2,1], "-",
                color=color, lw=lw, alpha=a, solid_capstyle="round")


def make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, output_path, roi_mask=None,
                fig_theme="Dark", proj_cmap="Inferno", jdd=None,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None):
    print("  Rendering figure ...")

    # ── Theme palettes ─────────────────────────────────────────────────────────
    if fig_theme == "Light":
        BG, PNL   = "#ffffff", "#f6f8fa"
        TXT, GRD  = "#24292f", "#d0d7de"
        ACC       = "#0969da"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _pie_text = "#ffffff"
        _font     = "sans-serif"
    elif fig_theme == "Publication":
        BG, PNL   = "#ffffff", "#ffffff"
        TXT, GRD  = "#000000", "#cccccc"
        ACC       = "#333333"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _pie_text = "#ffffff"
        _font     = "serif"
    else:                                    # Dark (default)
        BG, PNL   = "#0d1117", "#161b22"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _pie_text = "#0d1117"
        _font     = "monospace"

    # ── Projection colourmap ───────────────────────────────────────────────────
    _cmap_map = {
        "Inferno": "inferno",
        "Hot":     "hot",
        "Viridis": "viridis",
        "Plasma":  "plasma",
        "Greys":   "Greys" if fig_theme in ("Light", "Publication") else "Greys_r",
    }
    _pcmap = _cmap_map.get(proj_cmap, "inferno")

    plt.rcParams.update({
        "text.color":       TXT, "axes.labelcolor": TXT,
        "xtick.color":      TXT, "ytick.color":     TXT,
        "axes.edgecolor":   GRD, "axes.facecolor":  PNL,
        "grid.color":       GRD, "grid.alpha":      0.4,
        "font.family":      _font})

    _has_jdd = jdd is not None
    fig = plt.figure(figsize=(20, 32), facecolor=BG)
    gs  = GridSpec(5, 3, figure=fig, hspace=0.42, wspace=0.32,
                   left=0.06, right=0.97, top=0.94, bottom=0.04)

    _panels = []   # (letter, axes) collected for per-panel export

    def sax(ax, ltr, ttl):
        ax.set_facecolor(PNL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        ax.set_title(f"  {ttl}", loc="left", fontsize=11,
                     color=TXT, pad=8, fontweight="bold")
        ax.text(-0.04,1.06,ltr,transform=ax.transAxes,fontsize=14,
                color=ACC,fontweight="bold",va="top",ha="right")
        _panels.append((ltr, ax))

    # Use up to 200 evenly-spaced frames for the max projection to save memory
    idx  = np.linspace(0, len(stack)-1, min(200, len(stack)), dtype=int)
    proj = stack[idx].max(axis=0)
    from skimage import exposure as _exp
    proj_eq = _exp.equalize_adapthist(
        (proj / proj.max()).astype(np.float32), clip_limit=0.03)
    mcol = diff_df.set_index("particle")["motion"].to_dict()

    # A — max projection
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(proj_eq, cmap=_pcmap, origin="lower", aspect="equal")
    bp = 5/pixel_size; y0,x0 = proj.shape[0]*.05, proj.shape[1]*.05
    ax.plot([x0,x0+bp],[y0,y0],"-",color="white",lw=3)
    ax.text(x0+bp/2,y0+proj.shape[0]*.025,"5 um",
            ha="center",va="bottom",color="white",fontsize=8)
    ax.set_xlabel(f"X  ({pixel_size} um/px)",fontsize=9)
    ax.set_ylabel("Y (px)",fontsize=9)
    if roi_mask is not None:
        ax.contour(roi_mask.astype(float), levels=[0.5],
                   colors=["#58a6ff"], linewidths=[1.2], alpha=0.8)
        ax.text(0.02, 0.02, f"ROI", transform=ax.transAxes,
                color="#58a6ff", fontsize=8, va="bottom")
    sax(ax,"A","Max Projection")

    # B — trajectory map coloured by motion type (subsample if very many tracks)
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(proj_eq,cmap=_traj_bg,origin="lower",aspect="equal",alpha=0.35)
    all_pids  = list(tracks["particle"].unique())
    draw_pids = set(np.random.default_rng(42).choice(
        all_pids, min(2000, len(all_pids)), replace=False))
    n_drawn = 0
    for pid, grp in (tracks[tracks["particle"].isin(draw_pids)]
                     .reset_index(drop=True).sort_values("frame")
                     .groupby("particle")):
        _draw_track(grp, MC.get(mcol.get(pid,"Unknown"),"#aaa"), ax)
        n_drawn += 1
    els = [Line2D([0],[0],color=MC[m],lw=2,label=m)
           for m in MORD if m in mcol.values()]
    ax.legend(handles=els,fontsize=8,loc="upper right",
              framealpha=0.7,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.set_xlim(0,proj.shape[1]); ax.set_ylim(0,proj.shape[0])
    ax.set_xlabel("X (px)",fontsize=9); ax.set_ylabel("Y (px)",fontsize=9)
    shown = f"{n_drawn:,}" + (f" of {len(all_pids):,}" if n_drawn < len(all_pids) else "")
    sax(ax,"B",f"Trajectories  (n={shown})")

    # C — trajectories coloured by D value
    ax = fig.add_subplot(gs[0,2])
    ax.imshow(proj_eq, cmap=_traj_bg, origin="lower", aspect="equal", alpha=0.35)
    d_map = diff_df.set_index("particle")["D"].to_dict()
    d_vals_valid = [v for v in d_map.values() if v is not None and np.isfinite(v) and v > 0]
    if d_vals_valid:
        log_d_vals = np.log10(d_vals_valid)
        _p5  = np.percentile(log_d_vals, 5)
        _p95 = np.percentile(log_d_vals, 95)
        _cmap_d = plt.cm.plasma
        _norm_d = plt.Normalize(vmin=_p5, vmax=_p95)
        _sm_d   = plt.cm.ScalarMappable(cmap=_cmap_d, norm=_norm_d)
        _sm_d.set_array([])
        draw_pids_c = set(np.random.default_rng(43).choice(
            all_pids, min(2000, len(all_pids)), replace=False))
        for pid, grp in (tracks[tracks["particle"].isin(draw_pids_c)]
                         .reset_index(drop=True).sort_values("frame")
                         .groupby("particle")):
            D_val = d_map.get(pid)
            if D_val is not None and np.isfinite(D_val) and D_val > 0:
                col = _cmap_d(_norm_d(np.log10(D_val)))
            else:
                col = "#555555"
            _draw_track(grp, col, ax)
        cb = plt.colorbar(_sm_d, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(D)  [µm²/s]", fontsize=8, color=TXT)
        cb.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT, fontsize=7)
    ax.set_xlim(0, proj.shape[1]); ax.set_ylim(0, proj.shape[0])
    ax.set_xlabel("X (px)", fontsize=9); ax.set_ylabel("Y (px)", fontsize=9)
    sax(ax, "C", "Trajectories by D value")

    # D — MSD curves
    ax = fig.add_subplot(gs[1,0])
    lt  = emsd_df.index.values * frame_interval
    rng = np.random.default_rng(42)
    for pid in rng.choice(list(imsd_df.columns), min(200,len(imsd_df.columns)), replace=False):
        v  = imsd_df[pid].values
        t  = imsd_df.index.values * frame_interval
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() >= 2:
            ax.plot(t[ok],v[ok],"-",color="#8b949e",lw=0.4,alpha=0.3)
    ax.plot(lt,emsd_df.values,"-o",color=ACC,lw=2.5,ms=4,zorder=5,
            label="Ensemble MSD")
    try:
        t6,m6 = lt[:6], emsd_df.values[:6].ravel()
        ok6   = np.isfinite(m6) & (m6>0)
        po,_  = curve_fit(msd_linear,t6[ok6],m6[ok6],p0=[0.01,0],maxfev=2000)
        te    = np.linspace(t6[0],lt[-1],200)
        ax.plot(te,msd_linear(te,*po),"--",color="#f78166",lw=2,
                label=f"Fit D={po[0]:.4f} um2/s")
    except: pass
    ax.set_xlabel("Lag time (s)",fontsize=9)
    ax.set_ylabel("MSD (um2)",fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True,which="both",ls=":",alpha=0.3)
    ax.legend(fontsize=8,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    sax(ax,"D","MSD Curves")

    # E — D distribution
    ax = fig.add_subplot(gs[1,1])
    dv = diff_df["D"].dropna()
    dv = dv[(dv>0) & (dv<dv.quantile(0.995))]
    if len(dv) > 5:
        ld   = np.log10(dv)
        bins = np.linspace(ld.min(), ld.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & (diff_df["D"]>0)]
            if len(sub):
                ax.hist(np.log10(sub["D"].clip(1e-6)),bins=bins,
                        color=MC[m],alpha=0.7,label=m,edgecolor="none")
        if len(ld) > 10:
            kde = gaussian_kde(ld)
            xk  = np.linspace(ld.min(), ld.max(), 300)
            ax.plot(xk, kde(xk)*len(dv)*(bins[1]-bins[0]),
                    "-",color=_kde_col,lw=2)
        ax.axvline(np.log10(dv.median()),color=ACC,ls="--",lw=1.5,
                   label=f"Median={dv.median():.4f}")
        ax.set_xlabel("log10(D)  [um2/s]",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=8,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls=":",alpha=0.3)
    sax(ax,"E","Diffusion Coefficient Distribution")

    # F — pie chart
    ax = fig.add_subplot(gs[1,2])
    mc_ = diff_df["motion"].value_counts()
    lbl = [m for m in MORD if m in mc_]
    sz  = [mc_[m] for m in lbl]
    co  = [MC[m] for m in lbl]
    _,_,ats = ax.pie(sz,labels=lbl,colors=co,autopct="%1.1f%%",startangle=140,
                      textprops={"color":TXT,"fontsize":9},
                      wedgeprops={"edgecolor":PNL,"linewidth":2})
    for at in ats: at.set_fontsize(8); at.set_color(_pie_text)
    sax(ax,"F","Motion Classification")

    # G — alpha distribution
    ax = fig.add_subplot(gs[2,0])
    av = diff_df["alpha"].dropna()
    av = av[(av>-1) & (av<4)]
    if len(av) > 5:
        ba = np.linspace(av.min(), av.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & diff_df["alpha"].notna()]
            if len(sub):
                ax.hist(sub["alpha"].clip(-1,4),bins=ba,
                        color=MC[m],alpha=0.7,label=m,edgecolor="none")
        for xv,lb,ls in [(0.5,"a=0.5",":"),(1.0,"a=1 Brownian","--"),(2.0,"a=2 directed",":")]:
            ax.axvline(xv,color=GRD,ls=ls,lw=1.2,label=lb)
        ax.set_xlabel("Anomalous exponent alpha",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=7,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls=":",alpha=0.3)
    sax(ax,"G","Anomalous Exponent Alpha Distribution")

    # H — Position Density Heatmap
    ax = fig.add_subplot(gs[2, 1])
    try:
        x_um = tracks["x"].values * pixel_size
        y_um = tracks["y"].values * pixel_size
        h, xe, ye = np.histogram2d(x_um, y_um, bins=120)
        from scipy.ndimage import gaussian_filter as _gf
        h_sm = _gf(h, sigma=1.5)
        ax.imshow(h_sm.T, origin="lower", cmap="hot",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  aspect="equal", interpolation="bilinear")
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        if roi_mask is not None:
            H_px, W_px = roi_mask.shape
            ax.contour(
                np.linspace(0, W_px * pixel_size, W_px),
                np.linspace(0, H_px * pixel_size, H_px),
                roi_mask.astype(float), levels=[0.5],
                colors=["#58a6ff"], linewidths=[1.0], alpha=0.7)
    except Exception:
        pass
    sax(ax, "H", "Position Density Map")

    # I — Turning Angle Distribution
    ax = fig.add_subplot(gs[2, 2])
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        _ta_bins = np.linspace(0, 180, 37)
        ax.hist(turning_angles, bins=_ta_bins, color=ACC, alpha=0.8, edgecolor="none")
        ax.axvline(90, color=GRD, lw=1.5, ls="--", label="90°")
        ax.text(45,  ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                "← Confined", ha="center", color="#f78166", fontsize=9)
        ax.text(135, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                "Directed →", ha="center", color="#3fb950", fontsize=9)
        ax.set_xlabel("Turning angle (°)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.6, facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
        ax.grid(True, ls=":", alpha=0.3)
    sax(ax, "I", "Turning Angle Distribution")

    # J — Mobile Fraction Over Time
    ax = fig.add_subplot(gs[3, 0])
    if mobile_frac_df is None or len(mobile_frac_df) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ts  = mobile_frac_df["time_s"].values
        mf  = mobile_frac_df["mobile_fraction"].values * 100
        ax.plot(ts, mf, "o-", color=ACC, lw=2, ms=5)
        ax.fill_between(ts, 0, mf, alpha=0.2, color=ACC)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Mobile fraction (%)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
    sax(ax, "J", "Mobile Fraction Over Time")

    # K — Jump Distance Distribution (spans cols 1–2)
    ax = fig.add_subplot(gs[3, 1:])
    if _has_jdd:
        _jdd_colors = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff"]

        r_max_plot = np.percentile(jdd["jumps"], 99.5)
        bins = np.linspace(0, r_max_plot, 60)
        ax.hist(jdd["jumps"], bins=bins, density=True,
                color="#8b949e", alpha=0.45, edgecolor="none",
                label=f"Observed  (n={jdd['n_jumps']:,})")

        _comp_labels = ["Slow", "Medium", "Fast"]
        for k, (pdf_k, D_k, f_k) in enumerate(
                zip(jdd["pdfs"], jdd["D_values"], jdd["fractions"])):
            lbl = (f"{_comp_labels[k]}  D={D_k:.4f} µm²/s  "
                   f"({f_k*100:.1f}%)")
            ax.plot(jdd["r_range"], pdf_k,
                    color=_jdd_colors[k], lw=2, label=lbl)

        ax.plot(jdd["r_range"], jdd["pdf_total"],
                color=TXT, lw=2.5, ls="--", label="Total fit")
        ax.set_xlabel("Jump distance  (µm)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.set_xlim(0, r_max_plot)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.6,
                  facecolor=PNL, edgecolor=GRD, labelcolor=TXT,
                  loc="upper right")
        sax(ax, "K",
            f"Jump Distance Distribution  "
            f"({jdd['n_components']}-population fit  |  "
            f"{jdd['n_jumps']:,} jumps)")
    else:
        ax.text(0.5, 0.5, "JDD not computed", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
        sax(ax, "K", "Jump Distance Distribution")

    # L — Cluster Map
    ax = fig.add_subplot(gs[4, 0])
    if cluster_labels is not None and cluster_locs is not None and len(cluster_locs) > 0:
        xy_um = cluster_locs  # already in µm, subsampled to match labels
        noise = cluster_labels == -1
        if noise.any():
            ax.scatter(xy_um[noise, 0], xy_um[noise, 1],
                       s=0.5, c="#444", alpha=0.3, linewidths=0, rasterized=True)
        clustered = ~noise
        if clustered.any():
            n_c = max(cluster_labels.max() + 1, 1)
            cmap_c = plt.cm.get_cmap("tab20", n_c)
            ax.scatter(xy_um[clustered, 0], xy_um[clustered, 1],
                       s=1.5, c=cluster_labels[clustered], cmap=cmap_c,
                       alpha=0.7, linewidths=0, rasterized=True,
                       vmin=0, vmax=n_c - 1)
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        n_shown = int(cluster_labels.max()) + 1 if cluster_labels.max() >= 0 else 0
        ax.text(0.02, 0.98, f"n={n_shown} clusters",
                transform=ax.transAxes, fontsize=8, color=TXT, va="top")
    else:
        ax.text(0.5, 0.5, "Cluster analysis\nnot computed",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10)
    sax(ax, "L", "Cluster Map  (DBSCAN)")

    # M — Dwell Time Distribution
    ax = fig.add_subplot(gs[4, 1])
    if dwell_df is not None and len(dwell_df) >= 5:
        dt_vals = dwell_df["dwell_time_s"].values
        ax.hist(dt_vals, bins=30, color=ACC, alpha=0.75, edgecolor="none", density=True)
        if np.isfinite(dwell_tau):
            t_fit = np.linspace(0, dt_vals.max(), 200)
            ax.plot(t_fit, (1/dwell_tau) * np.exp(-t_fit / dwell_tau),
                    "--", color="#f78166", lw=2,
                    label=f"τ = {dwell_tau:.2f} s")
            ax.legend(fontsize=8, framealpha=0.6, facecolor=PNL,
                      edgecolor=GRD, labelcolor=TXT)
        ax.set_xlabel("Dwell time  (s)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data\n(need confined/immobile tracks)",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10)
    sax(ax, "M", "Dwell Time Distribution")

    # N — MSS Slope Distribution
    ax = fig.add_subplot(gs[4, 2])
    if "mss_slope" in diff_df.columns and diff_df["mss_slope"].notna().sum() >= 5:
        ms = diff_df["mss_slope"].dropna()
        ms = ms[ms.between(-0.5, 1.5)]
        bins = np.linspace(ms.min(), ms.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"] == m) & diff_df["mss_slope"].notna()]
            sub = sub[sub["mss_slope"].between(-0.5, 1.5)]
            if len(sub):
                ax.hist(sub["mss_slope"], bins=bins, color=MC[m],
                        alpha=0.7, label=m, edgecolor="none")
        for xv, lb, ls_ in [(0.25, "Confined", ":"), (0.5, "Brownian", "--"), (0.75, "Directed", ":")]:
            ax.axvline(xv, color=GRD, ls=ls_, lw=1.2, label=lb)
        ax.set_xlabel("MSS slope  (ν)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.6, facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
        ax.grid(True, ls=":", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "MSS not computed\n(tracks too short)",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10)
    sax(ax, "N", "Moment Scaling Spectrum  (MSS slope)")

    md = diff_df["D"].dropna().median()
    ma = diff_df["alpha"].dropna().median()
    fig.suptitle(
        f"sptPALM Analysis  |  {diff_df.shape[0]:,} trajectories  |  "
        f"Median D = {md:.4f} um2/s  |  Median alpha = {ma:.2f}",
        fontsize=13,color=TXT,y=0.97,fontweight="bold")

    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure -> {output_path}")

    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure (PDF) -> {pdf_path}")

    # Per-panel export — each labelled panel saved individually
    try:
        from matplotlib.transforms import Bbox as _Bbox
        panel_dir = os.path.join(os.path.dirname(output_path), "panels")
        os.makedirs(panel_dir, exist_ok=True)
        fig.canvas.draw()
        renderer  = fig.canvas.get_renderer()
        stem_base = os.path.splitext(os.path.basename(output_path))[0]
        pad_px    = fig.dpi * 0.10   # 0.1 inch padding in display units
        for ltr, pax in _panels:
            bbox = pax.get_tightbbox(renderer)
            if bbox is None:
                continue
            bbox_padded = _Bbox([[bbox.x0 - pad_px, bbox.y0 - pad_px],
                                  [bbox.x1 + pad_px, bbox.y1 + pad_px]])
            bbox_in = bbox_padded.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(panel_dir, f"{stem_base}_panel_{ltr}.png"),
                        bbox_inches=bbox_in, dpi=180,
                        facecolor=fig.get_facecolor())
            fig.savefig(os.path.join(panel_dir, f"{stem_base}_panel_{ltr}.pdf"),
                        bbox_inches=bbox_in, facecolor=fig.get_facecolor())
        print(f"  Panels -> {panel_dir}/")
    except Exception as _pe:
        print(f"  Per-panel export skipped: {_pe}")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="sptPALM for Zeiss Elyra CZI/TIF (optimised)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--pixel-size",       type=float, default=None)
    p.add_argument("--frame-interval",   type=float, default=None)
    p.add_argument("--diameter",         type=int,   default=7)
    p.add_argument("--minmass",          type=float, default=None)
    p.add_argument("--search-range",     type=float, default=5)
    p.add_argument("--memory",           type=int,   default=3)
    p.add_argument("--min-track-length", type=int,   default=5)
    p.add_argument("--max-lagtime",      type=int,   default=20)
    p.add_argument("--bg-method",        default="uniform_filter",
                   choices=["uniform_filter","rolling_ball"])
    p.add_argument("--bg-radius",        type=float, default=50)
    p.add_argument("--workers",          type=int,   default=N_CPUS)
    p.add_argument("--chunk-size",       type=int,   default=500)
    p.add_argument("--channel",          type=int,   default=0)
    p.add_argument("--output-dir",       default=None)
    p.add_argument("--roi-threshold",      type=float, default=None,
                   help="Manual intensity threshold for ROI mask on [0,1]. "
                        "If omitted with --roi-auto, threshold is determined "
                        "automatically. Omit both to process the full frame.")
    p.add_argument("--roi-auto",           action="store_true", default=False,
                   help="Automatically determine ROI threshold from the data. "
                        "Uses --roi-auto-method to select the algorithm.")
    p.add_argument("--roi-auto-method",    default="auto",
                   choices=["auto", "otsu", "li", "triangle"],
                   help="Algorithm for automatic ROI thresholding. "
                        "auto     = picks best method for sptPALM (default). "
                        "otsu     = maximises inter-class variance. "
                        "li       = minimises cross-entropy (sparse cells). "
                        "triangle = best for large dark backgrounds.")
    p.add_argument("--roi-mode",           default="mean",
                   choices=["mean", "perframe"],
                   help="ROI masking mode. "
                        "mean     = one mask from mean projection (default). "
                        "perframe = separate mask computed per frame.")
    return p.parse_args()


def main():
    args    = parse_args()
    t_start = time.perf_counter()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: File not found: {args.input}")

    stem    = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*67)
    print("  sptPALM Analysis Pipeline  --  Zeiss Elyra  --  By Jacob Levers")
    print("="*67)
    print(f"  CPU cores available : {N_CPUS}  |  Using: {args.workers}")

    # 1 — Load
    print("\n[1/6] Loading file")
    stack, meta_px, meta_fi = load_file(args.input, channel=args.channel)
    n_frames = len(stack)

    pixel_size = args.pixel_size or meta_px
    if pixel_size is None:
        print("  WARNING: Pixel size not in metadata. Using 0.104 um/px.")
        print("  (Override with --pixel-size)")
        pixel_size = 0.104
    else:
        src = "command line" if args.pixel_size else "CZI metadata"
        print(f"  Pixel size     : {pixel_size} um/px  [{src}]")

    frame_interval = args.frame_interval or meta_fi
    if frame_interval is None:
        print("  WARNING: Frame interval not in metadata. Using 0.05 s.")
        print("  (Override with --frame-interval)")
        frame_interval = 0.05
    else:
        src = "command line" if args.frame_interval else "CZI metadata"
        print(f"  Frame interval : {frame_interval} s/frame  [{src}]")

    print(f"  Total frames   : {n_frames:,}")
    print(f"  Output dir     : {out_dir}")

    # 2 — Preprocess
    print("\n[2/6] Preprocessing")
    stack_pp = preprocess_stack(stack, bg_radius=args.bg_radius,
                                bg_method=args.bg_method,
                                workers=args.workers)

    if args.minmass is None:
        sample = stack_pp[min(5, n_frames-1)]
        args.minmass = float(np.percentile(sample, 99) * 0.4)
        print(f"  Auto minmass: {args.minmass:.4f}")

    # 2b — ROI mask (optional)
    roi_mask = None
    use_roi  = (args.roi_threshold is not None) or args.roi_auto
    if use_roi:
        manual_thresh = args.roi_threshold  # None = auto
        auto_method   = args.roi_auto_method if args.roi_auto else None
        if manual_thresh is not None:
            mode_str = f"threshold={manual_thresh}, mode={args.roi_mode}"
        else:
            mode_str = f"auto-threshold ({args.roi_auto_method}), mode={args.roi_mode}"
        print(f"\n[2b/6] Building ROI mask  ({mode_str})")
        roi_preview = os.path.join(out_dir, f"{stem}_roi_mask.png")
        roi_mask = build_roi_mask(
            stack_pp,
            threshold=manual_thresh,
            mode=args.roi_mode,
            threshold_method=args.roi_auto_method if args.roi_auto else "auto",
            save_path=roi_preview)
    else:
        print("  ROI: disabled  "
              "(use --roi-auto for automatic, or --roi-threshold 0.15 for manual)")

    # 3 — Localise
    print("\n[3/6] Localisation")
    locs = localise_particles(stack_pp, diameter=args.diameter,
                              minmass=args.minmass,
                              workers=args.workers,
                              chunk_size=args.chunk_size)
    if len(locs) == 0:
        sys.exit("ERROR: No particles found. Try adding --minmass 0.05")

    if roi_mask is not None:
        locs = apply_roi_mask(locs, roi_mask)
        if len(locs) == 0:
            sys.exit("ERROR: No localisations inside ROI. "
                     "Lower --roi-threshold or remove it.")

    # 4 — Link
    print("\n[4/6] Linking trajectories")
    tracks = link_trajectories(locs, search_range=args.search_range,
                               memory=args.memory,
                               min_len=args.min_track_length)
    if tracks["particle"].nunique() == 0:
        sys.exit("ERROR: No trajectories found. Lower --min-track-length.")

    # 5 — MSD + diffusion (single parallel pass — no tp.imsd)
    print("\n[5/6] MSD & diffusion fitting")
    imsd_df, emsd_df, diff_df = compute_msd_and_fit(
        tracks, pixel_size, frame_interval,
        max_lagtime=args.max_lagtime, workers=args.workers)

    # 5b — JDD
    print("\n[5b/6] Jump Distance Distribution")
    jdd = compute_jdd(tracks, pixel_size, frame_interval, n_components=2)
    if jdd:
        print(f"  Jumps: {jdd['n_jumps']:,}")
        for k, (D, f) in enumerate(zip(jdd["D_values"], jdd["fractions"])):
            print(f"  Population {k+1}: D={D:.4f} um2/s  fraction={f*100:.1f}%")
    else:
        print("  Too few jumps to fit JDD.")

    # 6 — Save
    print("\n[6/6] Saving outputs")
    for df, suffix in [(locs,"localisations"), (tracks,"trajectories"),
                       (diff_df,"diffusion_summary")]:
        path = os.path.join(out_dir, f"{stem}_{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"  {suffix:<25} -> {path}")

    emsd_out  = emsd_df.to_frame("msd_um2").reset_index(names="lag_frame")
    emsd_path = os.path.join(out_dir, f"{stem}_ensemble_msd.csv")
    emsd_out.to_csv(emsd_path, index=False)
    print(f"  ensemble_msd              -> {emsd_path}")

    fig_path = os.path.join(out_dir, f"{stem}_sptpalm_figure.png")
    make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, fig_path,
                roi_mask=roi_mask, jdd=jdd,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None)

    # Summary
    total = time.perf_counter() - t_start
    print("\n" + "="*67)
    print("  RESULTS SUMMARY")
    print("="*67)
    print(f"  Raw localisations : {len(locs):>8,}")
    print(f"  Final trajectories: {tracks['particle'].nunique():>8,}")
    mc_ = diff_df["motion"].value_counts()
    for m in MORD:
        cnt = mc_.get(m, 0)
        print(f"    {m:<12}  {cnt:>6,}  ({100*cnt/max(len(diff_df),1):.1f}%)")
    print(f"\n  Median D  : {diff_df['D'].median():.5f} um2/s")
    print(f"  Mean D    : {diff_df['D'].mean():.5f} um2/s")
    print(f"  Median a  : {diff_df['alpha'].median():.3f}")
    print(f"\n  Total time: {total:.1f}s  ({total/60:.1f} min)")
    print("="*67)
    print(f"\n  Done! Results in: {out_dir}\n")


if __name__ == "__main__":
    main()
