#!/usr/bin/env python3
import multiprocessing
import sys
import os

__version__ = "2.3.1"

# Fix macOS multiprocessing crashes — must be set before any other imports
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

"""
FIREFLY — Fluorescence Inference & Reconstruction Engine  (OPTIMISED)
=======================================================================
Framework for Localization Yields.  Supports .czi (Zeiss native) and
.tif / .tiff files.
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

# ── BLAS / OpenBLAS / MKL threading policy ─────────────────────────────────────
# Cap internal BLAS threads to 1.  We use ThreadPoolExecutor for preprocessing
# (one Python thread per frame, all calling scipy.ndimage which uses BLAS).
# Without this cap, we get N² threads (Python pool × BLAS pool) on N cores,
# which deadlocks Windows frozen apps before the first preview frame is sent.
#
# Per-frame numpy/scipy operations on small (256×256) images are too fast to
# benefit from BLAS threading anyway — chunk-level Python threading wins.
# This MUST be set before numpy is imported to take effect.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    # Fallback no-op context manager if threadpoolctl unavailable
    from contextlib import contextmanager as _cm
    @_cm
    def _threadpool_limits(limits=None, user_api=None):
        yield
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import correlate as _correlate2d
from scipy.stats import gaussian_kde
from skimage import filters, exposure
from tqdm import tqdm

# On Windows with console=False (PyInstaller GUI build), sys.stderr is None.
# tqdm writes to sys.stderr by default and crashes with AttributeError.
# Use sys.stdout instead — the GUI redirects stdout to its log panel, so
# tqdm progress lines will appear there in real time.
import io as _io

def _tqdm(*args, **kwargs):
    """tqdm wrapper that writes to stdout (captured by the GUI log panel).
    Falls back to a no-op StringIO if stdout is somehow invalid."""
    out = sys.stdout if (sys.stdout is not None) else _io.StringIO()
    kwargs.setdefault("file", out)
    # Disable ANSI colour codes — the log panel is plain text.
    kwargs.setdefault("colour", None)
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


def load_czi(path, channel=0, stop_event=None, files=None):
    """Load a CZI (or multi-file CZI series) into one stack.

    `files`, when provided, overrides the auto-discovery of sibling
    files — used by the GUI to honour per-file checkbox selections.
    """
    if files:
        seen = set()
        series = []
        for f in sorted(files, key=lambda p: os.path.basename(p)):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f); series.append(f)
        if not series:
            series = [path]
        print(f"  CZI series override: {len(series)} files",
              flush=True)
    else:
        # Detect multi-file series (Zeiss splits large datasets into companion files)
        series = _find_czi_series(path)

    if len(series) == 1:
        # Single file — straightforward load.  Use series[0] so a
        # per-file override that selects a non-primary sister still
        # loads the right one.
        only = series[0]
        print(f"  Loading CZI: {only}")
        stack, px_um, fi_s = _load_single_czi(only, channel, stop_event)
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


def _find_tif_series(path):
    """Find split TIFF files like name.tif, name(1).tif, name(2).tif"""
    import glob, re
    directory = os.path.dirname(path) or "."
    basename  = os.path.splitext(os.path.basename(path))[0]
    ext       = os.path.splitext(path)[1].lower()  # .tif or .tiff

    # Strip any trailing "(N)" so we get the root name
    root = re.sub(r"\(\d+\)$", "", basename).rstrip()

    # Collect all matching files
    pattern  = os.path.join(directory, glob.escape(root) + "*" + ext)
    candidates = sorted(glob.glob(pattern))

    # Keep only: root.tif and root(N).tif
    series_re = re.compile(
        r"^" + re.escape(root) + r"(\(\d+\))?" + re.escape(ext) + r"$", re.IGNORECASE)
    series = [f for f in candidates
              if series_re.match(os.path.basename(f))]

    # Natural sort so (1) < (2) < (10)
    def _nat_key(s):
        m = re.search(r"\((\d+)\)" + re.escape(ext) + r"$", s, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    series.sort(key=_nat_key)

    if len(series) > 1:
        print(f"  Multi-file TIF series detected ({len(series)} files):")
        for f in series:
            print(f"    {os.path.basename(f)}")
    return series if series else [path]

def _load_single_tif(path, stop_event=None):
    """Load a single TIF file and return its stack, pixel size, and frame interval.

    Strategy: tifffile's `asarray()` reads + decompresses pages with internal
    multithreading via `maxworkers`, which is far faster than looping
    page.asarray() (each call re-opens its own thread pool).  For very
    large files we'd still like a cancel-poll, so we read in chunks of
    `BATCH` pages — each batch goes through the fast path, and we check
    stop_event + log progress between batches.
    """
    if not HAS_TIFFFILE:
        raise RuntimeError("Run: pip install tifffile")
    BATCH = 2000
    with tifffile.TiffFile(path) as tif:
        px_um, fi_s = _parse_ome_metadata(tif)
        n_pages = len(tif.pages)
        if n_pages > BATCH:
            # Inspect first page for output shape + dtype so we can pre-allocate
            sample = tif.pages[0].asarray()
            shape = (n_pages,) + tuple(sample.shape)
            stack = np.empty(shape, dtype=np.float32)
            t0 = time.perf_counter()
            for start in range(0, n_pages, BATCH):
                if stop_event is not None and stop_event.is_set():
                    raise _Cancelled()
                end = min(start + BATCH, n_pages)
                # tifffile asarray(key=range(...)) uses multithreaded decode
                try:
                    chunk = tif.asarray(key=range(start, end),
                                         maxworkers=N_CPUS)
                except TypeError:
                    chunk = tif.asarray(key=range(start, end))
                # asarray may return (1, H, W) for a single page, normalise
                if chunk.ndim == 2:
                    chunk = chunk[np.newaxis]
                stack[start:end] = chunk.astype(np.float32, copy=False)
                # Free intermediate so peak memory stays at one batch above
                # the pre-allocated stack
                del chunk
                if start > 0:
                    rate = (start) / max(time.perf_counter() - t0, 1e-3)
                    print(f"  Loading: {end}/{n_pages} frames "
                          f"({rate:.0f} fr/s)...", flush=True)
        else:
            # Small files — fastest path is a single asarray()
            try:
                stack = tif.asarray(maxworkers=N_CPUS).astype(
                    np.float32, copy=False)
            except TypeError:
                stack = tif.asarray().astype(np.float32, copy=False)

    if   stack.ndim == 2: stack = stack[np.newaxis]
    elif stack.ndim == 4:
        stack = stack[:, 0] if stack.shape[1] == 1 else stack.mean(axis=1)

    return stack, px_um, fi_s


# ── Memmap cleanup ────────────────────────────────────────────────────────────
# When the multi-file loader falls back to a disk-backed memmap (because the
# combined stack won't fit in RAM), we leave the file on disk for the duration
# of the run.  Register an atexit hook to remove these temp files so they
# don't accumulate.
_firefly_temp_stack_paths: list = []

def _register_temp_stack_path(p: str) -> None:
    import atexit
    if not _firefly_temp_stack_paths:
        atexit.register(_cleanup_temp_stack_paths)
    _firefly_temp_stack_paths.append(p)

def _cleanup_temp_stack_paths() -> None:
    for p in list(_firefly_temp_stack_paths):
        try:    os.remove(p)
        except Exception: pass


#  How much physical RAM to leave for the OS + the user's other apps.
#  Without this reserve, FIREFLY's memory checks would happily consume
#  every free byte; the moment the user opens a Safari tab the system
#  starts swapping or OOM-killing.  We hold back the LARGER of:
#     • a fixed floor (4 GB)                          — covers macOS itself
#     • 20% of total RAM                              — scales with system size
#  Tweaked via env var FIREFLY_USER_RAM_RESERVE_GB if you really need to.
def _user_ram_reserve_gb() -> float:
    """RAM (in GB) we deliberately keep available for non-FIREFLY uses."""
    try:
        env = os.environ.get("FIREFLY_USER_RAM_RESERVE_GB")
        if env:
            return max(0.5, float(env))
    except Exception:
        pass
    try:
        import psutil as _ps
        total_gb = _ps.virtual_memory().total / 1e9
    except Exception:
        total_gb = 8.0   # conservative fallback if psutil is missing
    return max(4.0, 0.20 * total_gb)


def _probe_tif_shape_and_count(path: str):
    """Read just enough of a TIF to return (n_pages, (H, W))."""
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        sample = tif.pages[0].asarray()
        H, W = sample.shape[-2:]
    return n, (int(H), int(W))


def load_tif(path, stop_event=None, files=None):
    """Load `path` and (when present) its sibling files into one stack.

    If `files` is a non-empty list, it overrides auto-discovery — the
    GUI uses this to honour per-file checkbox selections within a series.
    The override is sorted to match _find_tif_series ordering so frame
    indices line up with the user's expectation.
    """
    if files:
        # De-dup and sort by the same key the auto-discovery uses so the
        # frame order doesn't depend on how the GUI sent the list.
        seen = set()
        series = []
        for f in sorted(files, key=lambda p: os.path.basename(p)):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f); series.append(f)
        if not series:
            series = [path]
        print(f"  TIF series override: {len(series)} files",
              flush=True)
    else:
        series = _find_tif_series(path)

    if len(series) == 1:
        # Single file — straightforward load.  Use series[0] (not the
        # original `path`) so a per-file override that selects a
        # non-primary sister file still loads the right file.
        only = series[0]
        print(f"  Loading TIF: {only}")
        stack, px_um, fi_s = _load_single_tif(only, stop_event)
        print(f"  Shape: {stack.shape}  (T x Y x X)")
        if px_um is not None: print(f"  Pixel size  : {px_um} µm  (from file metadata)")
        if fi_s is not None:  print(f"  Frame interval: {fi_s} s  (from file metadata)")
        return stack, px_um, fi_s

    # ── Multi-file series ────────────────────────────────────────────────
    # The old path loaded every file into a `stacks` list and called
    # `np.concatenate(stacks)`, which allocates a brand-new combined array
    # while the source list is still alive — peak memory = 2× the combined
    # size.  On a 16.8 GB series that's a 33.6 GB working set on a 16 GB
    # machine.  System swap takes minutes and pegs the disk.
    #
    # The new path:
    #   1. Probes each file's frame count via TiffFile headers (no data load)
    #   2. Pre-allocates the destination — in RAM if it fits, on disk via
    #      np.memmap if not
    #   3. Loads each source, copies into the destination slice, frees the
    #      source.  Peak = combined + one source ≈ 1.25× total.
    print(f"  Loading TIF series: {len(series)} files", flush=True)

    n_per_file: list[int] = []
    H = W = 0
    for fpath in series:
        n, (h, w) = _probe_tif_shape_and_count(fpath)
        n_per_file.append(n)
        H, W = h, w
    n_total = sum(n_per_file)
    bytes_per_frame = 4 * H * W      # float32
    total_size = n_total * bytes_per_frame
    total_gb = total_size / 1e9

    # Decide RAM vs memmap.  We need the combined stack + headroom for
    # one source file at a time + downstream allocations + (critically!)
    # a reserve so the user's OS / browser / etc. don't get squeezed
    # into swap.
    use_memmap = False
    free_gb    = None
    reserve_gb = _user_ram_reserve_gb()
    try:
        import psutil as _psutil
        free_gb = _psutil.virtual_memory().available / 1e9
        # Usable budget = free RAM minus the bytes we promised to leave
        # for everything else on the machine.
        usable_gb = free_gb - reserve_gb
        # And we need at least 1.2 × the combined stack to cover the
        # intermediate per-file source array + downstream copies.
        if usable_gb < total_gb * 1.2:
            use_memmap = True
    except Exception:
        pass

    if use_memmap:
        import tempfile
        tmp_fh = tempfile.NamedTemporaryFile(
            prefix="firefly_stack_", suffix=".raw", delete=False)
        tmp_path = tmp_fh.name
        tmp_fh.close()
        _register_temp_stack_path(tmp_path)
        free_disp = f"{free_gb:.1f}" if free_gb is not None else "?"
        print(f"  Combined stack would need {total_gb:.1f} GB and we "
              f"reserve {reserve_gb:.1f} GB for the OS / other apps; "
              f"only {free_disp} GB free — backing it with a memmap on "
              f"disk at {tmp_path}.", flush=True)
        combined = np.memmap(tmp_path, dtype=np.float32, mode="w+",
                             shape=(n_total, H, W))
    else:
        print(f"  Allocating combined stack ({n_total:,} frames, "
              f"{total_gb:.1f} GB) in RAM…", flush=True)
        combined = np.empty((n_total, H, W), dtype=np.float32)

    # Load each file, copy into the destination slice, free immediately.
    px_um_out = None
    fi_s_out  = None
    offset = 0
    import gc as _gc
    for i, fpath in enumerate(series):
        print(f"  [{i+1}/{len(series)}] {os.path.basename(fpath)}",
              flush=True)
        st, px, fi = _load_single_tif(fpath, stop_event)
        if i == 0:
            px_um_out = px
            fi_s_out  = fi
        combined[offset:offset + st.shape[0]] = st
        offset += st.shape[0]
        del st
        _gc.collect()
    if use_memmap:
        combined.flush()
    print(f"  Combined shape: {combined.shape}  (T x Y x X)", flush=True)
    if px_um_out is not None: print(f"  Pixel size  : {px_um_out} µm  (from file metadata)")
    if fi_s_out is not None:  print(f"  Frame interval: {fi_s_out} s  (from file metadata)")
    return combined, px_um_out, fi_s_out


def load_file(path, channel=0, stop_event=None, files=None):
    """Load `path` (or, if `files` is provided, the explicit list of
    files) as a single stack.

    `files` lets a caller override the auto-discovery of sister files
    that `_find_tif_series` / `_find_czi_series` does — useful when the
    GUI wants to load only a user-selected subset of a multi-file series.
    """
    ext = os.path.splitext(path)[1].lower()
    if   ext == ".czi":            return load_czi(path, channel, stop_event,
                                                   files=files)
    elif ext in (".tif", ".tiff"): return load_tif(path, stop_event,
                                                   files=files)
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
        with ThreadPoolExecutor(max_workers=workers) as _exe:
            _futs = [_exe.submit(fn, f, bg_radius) for f in stack]
            processed = [_f.result() for _f in
                         _tqdm(_futs, desc="  Preprocessing", unit="fr", ncols=70)]

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

    # ── Cross-correlate ALL pairs (i, j) → solve cumulative drift ─────────────
    # This is the redundant cross-correlation (RCC) algorithm of Wang et al.
    # 2014 (Nat. Methods).  Instead of relying only on consecutive pairs, we
    # measure the inter-segment shift Δ_{ij} for every pair (i, j) with i<j
    # and then solve the over-determined linear system
    #
    #     drift[j] − drift[i] = Δ_{ij}      for all valid pairs
    #
    # by least-squares.  Drift[0] is fixed at zero (gauge fixing).  The
    # redundancy averages out cross-correlation noise far better than the
    # consecutive-only chain, and is robust to any single bad pair (e.g. a
    # segment with too few localisations).
    #
    # Performance note:  scipy.signal.correlate(method="fft") re-FFTs both
    # density maps on every pair call, so an N-segment run does ~N(N-1)
    # FFTs.  We precompute rfft2 of each (zero-padded) map ONCE and just
    # run an IFFT per pair — quadratic-cost FFT work collapses to linear,
    # plus the IFFT loop parallelises trivially via threads.
    from scipy.fft import rfft2 as _rfft2, irfft2 as _irfft2, \
                          next_fast_len as _next_fast_len
    pad_H = _next_fast_len(2 * H - 1)
    pad_W = _next_fast_len(2 * W - 1)
    fft_maps = [_rfft2(dm, s=(pad_H, pad_W)) for dm in density_maps]

    pair_indices = [(i, j) for i in range(n_segments)
                    for j in range(i + 1, n_segments)
                    if seg_counts[i] >= 5 and seg_counts[j] >= 5]

    def _pair_shift(i, j):
        # Cross-correlation r[τ] = Σ a[k+τ] b[k]  via  IFFT(F_a · conj(F_b))
        cross = _irfft2(fft_maps[i] * np.conj(fft_maps[j]),
                        s=(pad_H, pad_W))
        # Zero-lag at index 0; positive shifts up to (H-1, W-1) sit at low
        # indices, negative shifts wrap to the end.  Re-centre by treating
        # any index beyond half-extent as negative.
        peak = int(np.argmax(cross))
        py, px = divmod(peak, pad_W)
        if py >= pad_H // 2: py -= pad_H
        if px >= pad_W // 2: px -= pad_W
        return i, j, float(px), float(py)

    A_rows_x, A_rows_y = [], []
    b_x, b_y = [], []
    if pair_indices:
        with ThreadPoolExecutor(max_workers=N_CPUS) as _exe:
            for i, j, dx_pair, dy_pair in _exe.map(
                    lambda ij: _pair_shift(*ij), pair_indices):
                row = np.zeros(n_segments)
                row[i], row[j] = -1.0, 1.0
                A_rows_x.append(row); b_x.append(dx_pair)
                A_rows_y.append(row); b_y.append(dy_pair)

    if not A_rows_x:
        # Fallback: zero drift
        dx_cum = np.zeros(n_segments)
        dy_cum = np.zeros(n_segments)
    else:
        # Add gauge-fixing row: drift[0] = 0 (heavy weight)
        gauge = np.zeros(n_segments); gauge[0] = 1.0
        A = np.vstack(A_rows_x + [gauge * 1e3])
        bx = np.append(np.array(b_x), 0.0)
        by = np.append(np.array(b_y), 0.0)
        dx_cum, *_ = np.linalg.lstsq(A, bx, rcond=None)
        dy_cum, *_ = np.linalg.lstsq(A, by, rcond=None)

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

    Holds back `_user_ram_reserve_gb()` for the OS + the user's other
    apps so a parallel Safari tab doesn't push the machine into swap.
    """
    needed_gb = stack.nbytes / 1e9   # preprocessed copy ≈ same dtype/shape
    try:
        import psutil
        free_gb    = psutil.virtual_memory().available / 1e9
        reserve_gb = _user_ram_reserve_gb()
        usable_gb  = max(0.0, free_gb - reserve_gb)
        return needed_gb < usable_gb * headroom, free_gb, needed_gb
    except ImportError:
        return False, 0.0, needed_gb


def _fast_preprocess_and_localise(stack, diameter=7, minmass=None, percentile=64,
                                   bg_radius=50, bg_method="uniform_filter",
                                   workers=N_CPUS, chunk_size=500,
                                   preview_cb=None, backend="auto"):
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
        # Auto-detect minmass.  The "mass" trackpy returns is *integrated*
        # intensity over the spot (≈π(d/2)² ≈ d²/π px ≈ d²/4 effective px,
        # depending on PSF shape).  The old formula used `peak × 0.4` which
        # is the *per-pixel* threshold — that under-shoots the integrated
        # threshold by ~10× and produces 100k+ false-positive "spots" on
        # PALM-density data.  Corrected to account for the spot's pixel
        # support: `peak × diameter² / 8` (0.5 × effective area).
        # This is still a heuristic and may need manual tuning; users with
        # known data should set minmass explicitly via the GUI spinbox.
        _peak = float(np.percentile(stack_pp[min(5, len(stack_pp) - 1)], 99))
        minmass = float(_peak * (diameter ** 2) / 8.0)
        print(f"  Auto minmass: {minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")

    mean_proj = stack_pp.mean(axis=0).astype(np.float32)
    mn, mx    = mean_proj.min(), mean_proj.max()
    if mx > mn:
        mean_proj = (mean_proj - mn) / (mx - mn)

    locs = localise_particles(stack_pp, diameter=diameter, minmass=minmass,
                              percentile=percentile, workers=workers,
                              chunk_size=chunk_size, preview_cb=preview_cb,
                              backend=backend)
    del stack_pp
    gc.collect()
    return locs, mean_proj, minmass


def preprocess_and_localise_adaptive(stack, diameter=7, minmass=None, percentile=64,
                                     bg_radius=50, bg_method="uniform_filter",
                                     workers=N_CPUS, chunk_size=500,
                                     ram_headroom: float = 0.75,
                                     preview_cb=None, stop_event=None,
                                     mass_cb=None, backend="auto"):
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
    # Resolve and announce the backend once, up front — visible in the log
    # regardless of which RAM strategy we end up taking (the FAST path goes
    # through localise_particles which re-prints; the STREAM path bypasses it
    # entirely, so we need this line here too).
    try:
        _impl = _resolve_backend(backend)
        print(f"  Backend   : {_impl.name}  (requested: {backend})")
    except Exception as _e:
        print(f"  Backend   : (resolution failed: {_e})")

    use_fast, free_gb, needed_gb = _ram_strategy(stack, headroom=ram_headroom)
    reserve_gb = _user_ram_reserve_gb()

    if use_fast:
        print(f"  RAM strategy : FAST (parallel)   — "
              f"{free_gb:.1f} GB free, {needed_gb:.1f} GB needed, "
              f"{reserve_gb:.1f} GB reserved for OS/apps")
        return _fast_preprocess_and_localise(
            stack, diameter, minmass, percentile,
            bg_radius, bg_method, workers, chunk_size,
            preview_cb=preview_cb, backend=backend)
    else:
        print(f"  RAM strategy : STREAM (low-mem)  — "
              f"{free_gb:.1f} GB free, {needed_gb:.1f} GB needed, "
              f"{reserve_gb:.1f} GB reserved for OS/apps")
        return preprocess_and_localise_stream(
            stack, diameter, minmass, percentile,
            bg_radius, bg_method, workers, chunk_size,
            preview_cb=preview_cb, stop_event=stop_event,
            mass_cb=mass_cb, backend=backend)


def preprocess_and_localise_stream(stack, diameter=7, minmass=None, percentile=64,
                                   bg_radius=50, bg_method="uniform_filter",
                                   workers=N_CPUS, chunk_size=500,
                                   preview_cb=None, stop_event=None,
                                   mass_cb=None, backend="auto"):
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

    # Resolve the backend up front so each chunk goes through the same
    # implementation.  Trackpy is special-cased below to skip the per-chunk
    # process-pool spawn cost; everything else delegates to .localise().
    #
    # NOTE: an earlier version of this code bumped chunk_size to 1500 on
    # MPS/CUDA hoping to amortize dispatch overhead.  Empirically that made
    # things *slower* on Apple Silicon — the GPU is bandwidth-limited at
    # these convolution sizes, and 500-frame chunks fit better in cache
    # than 1500-frame chunks.  Per-frame throughput dropped ~3× when we
    # tried the bigger chunks.  Sticking with the caller's chunk_size now.
    _impl = _resolve_backend(backend)
    print(f"  Mode      : streaming preprocess + localise  (low memory)")
    print(f"  Backend   : {_impl.name}")
    print(f"  Diameter  : {diameter}px  |  bg_method: {bg_method}")
    print(f"  Chunks    : {n_chunks} × ~{chunk_size} frames  |  workers: {workers_}")
    t0 = time.perf_counter()

    def _localise_chunk_via_backend(chunk_pp):
        """Run the active backend on a single preprocessed chunk and return
        a DataFrame with at least columns x, y, frame, mass.

        Trackpy: call `tp.batch` directly with processes=1 to skip the
                 multiprocessing-pool spawn overhead (per-chunk, the pool
                 startup cost would dominate the actual work).
        Other:   delegate to the backend's `.localise()` (single iteration
                 because the chunk is already smaller than chunk_size).
        """
        if _impl.name == "trackpy":
            with _threadpool_limits(limits=N_CPUS):
                return tp.batch(chunk_pp, diameter=diameter, minmass=minmass,
                                percentile=percentile, processes=1)
        return _impl.localise(chunk_pp, diameter=diameter, minmass=minmass,
                              percentile=percentile, workers=workers_,
                              chunk_size=len(chunk_pp))

    # ── First chunk: preprocess now so we can auto-detect minmass ─────────────
    first_end  = min(chunk_size, n_frames)
    with ThreadPoolExecutor(max_workers=workers_) as _exe:
        first_pp = np.stack([_f.result() for _f in
                             [_exe.submit(fn, f, bg_radius) for f in stack[:first_end]]])

    if minmass is None:
        # Auto-detect minmass.  trackpy's "mass" is *integrated* intensity
        # over a (diameter × diameter) spot patch, not a single-pixel value.
        # The old formula `peak × 0.4` was a per-pixel threshold and under-
        # shoots integrated mass by ~10×, producing 100k+ false-positive
        # spots on dense PALM data.  Corrected to `peak × d²/8` — accounts
        # for the spot's pixel support area at the standard 50% acceptance.
        # Still a heuristic; users with known data should set minmass
        # explicitly via the GUI spinbox.
        _peak = float(np.percentile(first_pp[min(5, first_end - 1)], 99))
        minmass = float(_peak * (diameter ** 2) / 8.0)
        print(f"  Auto minmass: {minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")
    else:
        print(f"  Minmass   : {minmass:.4f}")

    # ── Stream all chunks ──────────────────────────────────────────────────────
    all_locs  = []
    mean_acc  = first_pp.sum(axis=0).astype(np.float64)
    frame_count = len(first_pp)

    # Localise first chunk (already preprocessed) — through the active backend
    locs0 = _localise_chunk_via_backend(first_pp)
    if len(locs0) > 0:
        all_locs.append(locs0)
    if mass_cb is not None and len(locs0) > 0 and "mass" in locs0.columns:
        try:    mass_cb(np.asarray(locs0["mass"].values, dtype=np.float32))
        except Exception: pass

    # ── Live preview: emit EVERY frame of each chunk after localisation
    # so the GUI's live view scrolls through the actual movie at 60 Hz
    # rather than ticking once per chunk.  The GUI's repaint timer
    # naturally drops in-between frames it can't paint in time, so we
    # just fire-and-forget every frame — the message queue + per-frame
    # cost is tiny next to localisation itself.
    def _emit_chunk_previews(chunk_pp, locs_chunk, frame_offset):
        if preview_cb is None or len(chunk_pp) == 0:
            return
        # Pre-index spots by frame for cheap per-frame lookups
        spots_by_frame = {}
        if len(locs_chunk) > 0 and "frame" in locs_chunk.columns:
            for f, sub in locs_chunk.groupby("frame"):
                spots_by_frame[int(f)] = (sub["x"].values, sub["y"].values)
        for local_i in range(len(chunk_pp)):
            global_i = frame_offset + local_i
            sxy = spots_by_frame.get(global_i, ([], []))
            try:
                preview_cb(global_i, chunk_pp[local_i],
                           sxy[0], sxy[1], n_frames)
            except Exception:
                pass

    _emit_chunk_previews(first_pp, locs0, frame_offset=0)

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
        with ThreadPoolExecutor(max_workers=workers_) as _exe:
            chunk_pp = np.stack([_f.result() for _f in
                                 [_exe.submit(fn, f, bg_radius) for f in stack[start:end]]])

        mean_acc   += chunk_pp.sum(axis=0)
        frame_count += len(chunk_pp)

        locs_i = _localise_chunk_via_backend(chunk_pp)

        if len(locs_i) > 0:
            locs_i = locs_i.copy()
            locs_i["frame"] += start
            all_locs.append(locs_i)
        if mass_cb is not None and len(locs_i) > 0 and "mass" in locs_i.columns:
            try:    mass_cb(np.asarray(locs_i["mass"].values, dtype=np.float32))
            except Exception: pass

        # Live previews — multiple evenly-spaced frames within this chunk
        _emit_chunk_previews(chunk_pp, locs_i, frame_offset=start)

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
    """Localise one chunk and apply global frame offset."""
    locs = tp.batch(chunk, diameter=diameter, minmass=minmass,
                    percentile=percentile, processes=1)
    if len(locs) > 0:
        locs = locs.copy()
        locs["frame"] += frame_offset
    return locs


def _localise_chunk_mp(args):
    """Picklable wrapper for multiprocessing.Pool.imap_unordered.
    Returns (index, dataframe) so we can preserve order despite unordered iteration."""
    idx, chunk, diameter, minmass, percentile, frame_offset = args
    result = _localise_chunk(chunk, diameter, minmass, percentile, frame_offset)
    return idx, result


# ══════════════════════════════════════════════════════════════════════════════
#  LOCALISER BACKENDS
# ══════════════════════════════════════════════════════════════════════════════
#
# A backend takes a *preprocessed* stack (T × Y × X, float32) and returns a
# DataFrame with at least the columns: x, y, frame, mass.  Preprocessing
# (background subtraction, bandpass) is handled separately so the fast / stream
# RAM strategies in this file stay backend-agnostic.
#
# Registration model: subclass LocaliserBackend, set `.name`, implement
# `.is_available()` (classmethod) and `.localise(stack, **params)`, then append
# to _BACKEND_REGISTRY in the preference order used by `backend="auto"`.
#
# Phase A1: only TrackpyBackend exists (refactor — no behaviour change).
# Phase A2: TorchBackend (CPU) lands here.
# Phase A3: device selection (MPS / CUDA) inside TorchBackend.

class LocaliserBackend:
    """Abstract base for particle-localisation backends.

    Subclasses must set `name` and implement `is_available()` + `localise()`.
    """
    name: str = "abstract"

    @classmethod
    def is_available(cls) -> bool:
        return False

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None, **kwargs):
        raise NotImplementedError


class TrackpyBackend(LocaliserBackend):
    """CPU localiser using trackpy's Crocker-Grier centroid detection.

    Parallelised via multiprocessing.Pool (spawn) for true multi-core scaling;
    falls back to a single-process BLAS-threaded path if Pool creation fails
    (rare, but happens on locked-down Windows boxes and inside some sandboxes).

    Accepted params:
        diameter     — odd integer, spot diameter in px (auto-bumped if even)
        minmass      — minimum integrated intensity for a spot
        percentile   — local-noise threshold (passed straight to tp.batch)
        workers      — process pool size (defaults to N_CPUS)
        chunk_size   — frames per chunk (memory / parallelism tradeoff)
    """
    name = "trackpy"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import trackpy  # noqa: F401
            return True
        except ImportError:
            return False

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None, **_):
        if diameter % 2 == 0:
            diameter += 1

        n_frames = len(stack)
        n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))
        workers  = max(1, min(workers if workers is not None else N_CPUS, N_CPUS))

        print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}")
        print(f"  Chunks    : {n_chunks} x ~{chunk_size} frames")

        t0       = time.perf_counter()
        chunks   = np.array_split(stack, n_chunks)
        offsets  = [i * chunk_size for i in range(len(chunks))]
        chunk_pairs = list(zip(chunks, offsets))

        # ── True multi-core via multiprocessing.Pool ──────────────────────
        # Each worker is a separate Python process with its own GIL — N workers
        # genuinely use N CPU cores.  Spawn context is required for Windows +
        # macOS frozen apps; PyInstaller's freeze_support (called in app_qt.py
        # main) makes spawn workers reuse the parent's _MEIPASS extraction, so
        # workers start in seconds rather than minutes.  Falls back to a
        # BLAS-pool serial path if Pool creation fails for any reason.
        n_workers = min(workers, n_chunks, N_CPUS)
        chunk_results = [None] * n_chunks
        use_mp_ok = False
        try:
            print(f"  Parallelism : multiprocessing.Pool × {n_workers} (spawn — true multi-core)")
            print(f"  Spawning workers (one-time ~10-30s; chunks then process truly in parallel)...")
            ctx = multiprocessing.get_context("spawn")
            mp_args = [(i, c, diameter, minmass, percentile, o)
                       for i, (c, o) in enumerate(chunk_pairs)]
            with ctx.Pool(processes=n_workers) as pool:
                for idx, result in _tqdm(
                        pool.imap_unordered(_localise_chunk_mp, mp_args),
                        total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
                    chunk_results[idx] = result
            use_mp_ok = True
        except Exception as exc:
            print(f"  multiprocessing failed ({type(exc).__name__}: {exc})")
            print(f"  Falling back to BLAS-pool parallelism (slower, single-process)")

        if not use_mp_ok:
            with _threadpool_limits(limits=N_CPUS):
                chunk_results = [_localise_chunk(chunk, diameter, minmass, percentile, offset)
                                 for chunk, offset in _tqdm(chunk_pairs, total=n_chunks,
                                                            desc="  Localising", unit="chunk",
                                                            ncols=70)]

        valid = [df for df in chunk_results if df is not None and len(df) > 0]
        result = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

        elapsed = time.perf_counter() - t0
        print(f"  Found {len(result):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return result


class TorchBackend(LocaliserBackend):
    """PyTorch-based localiser — CPU for now (MPS / CUDA arrive in A3).

    Algorithm (matches trackpy's default centroid-of-mass semantics so the
    sub-pixel positions stay close to within a few nm):

      1.  Bandpass = signal − local-average-background, then small-σ Gaussian
          smoothing.  Implemented as batched F.avg_pool2d + separable conv2d.
      2.  Threshold = `percentile`-th percentile of the bandpassed image
          (trackpy's `percentile` argument has the same meaning).
      3.  Local maxima = pixels where signal equals its diameter-window
          max-pool output AND exceeds the threshold (F.max_pool2d trick).
      4.  Patch extraction: gather a (diameter × diameter) tile around every
          candidate via fancy indexing — fully vectorised.
      5.  Mass = sum over patch; filter spots by `mass >= minmass`.
      6.  Sub-pixel refinement = centroid of mass on the patch.

    Returns a DataFrame with the standard columns `x, y, frame, mass`.

    Frames are processed in chunks of `chunk_size` to bound peak GPU memory.
    Step 1 (bandpass) is the bandwidth bottleneck on CPU; expect roughly the
    same wall-clock as trackpy on a fast laptop.  The point of this backend
    is the GPU path landing in A3 — CPU is here for correctness validation.
    """
    name = "torch"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def list_devices(cls) -> list[str]:
        """Return all torch devices we could plausibly run on, fastest first.
        Used by the GUI to populate a device-override picker and by the
        crash reporter to record what was actually visible.
        """
        try:
            import torch
        except ImportError:
            return []
        devs: list[str] = []
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devs.append("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devs.append(f"cuda:{i}" if torch.cuda.device_count() > 1 else "cuda")
        devs.append("cpu")
        return devs

    @classmethod
    def _device_sanity_check(cls, dev: str) -> bool:
        """Run the exact ops used in the hot path on `dev` to confirm full
        kernel coverage AND correctness.  Some PyTorch builds advertise MPS
        or CUDA support but either lack kernels for specific ops, or have
        kernels that silently return garbage (no Python exception raised).
        We test both.

        Metal native errors fire on C-level stderr — Python try/except
        can't catch those.  On MPS we run the probe inside an OS-level
        stderr redirect so a broken Metal context produces a clean
        "sanity check failed" message instead of flooding the terminal.
        """
        # OS-level stderr redirect (catches C / Metal native prints too).
        # Only used for MPS probing where we expect this class of noise.
        import contextlib as _cl

        @_cl.contextmanager
        def _quiet_native_stderr():
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved   = os.dup(2)
            try:
                os.dup2(devnull, 2)
                yield
            finally:
                os.dup2(saved, 2)
                os.close(devnull)
                os.close(saved)

        ctx = _quiet_native_stderr() if dev == "mps" else _cl.nullcontext()
        try:
            with ctx:
                import torch
                import torch.nn.functional as F
                t = torch.device(dev)
                # 4×4 linear solve (same kernel as the Gaussian fit).  Use
                # an identity matrix and verify the result matches the
                # input — broken MPS can return garbage with no exception.
                A = torch.eye(4, device=t, dtype=torch.float32).unsqueeze(0)
                v = torch.ones(4, device=t, dtype=torch.float32).view(1, 4, 1)
                sol = torch.linalg.solve(A, v)
                if not torch.allclose(sol, v, rtol=1e-2, atol=1e-3):
                    return False
                # avg_pool2d (bandpass) and max_pool2d (local maxima)
                x = torch.zeros(1, 1, 8, 8, device=t, dtype=torch.float32)
                _ = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                _ = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
                # einsum (used in normal-equations assembly)
                _ = torch.einsum('ni,ij,ik->njk',
                                 torch.ones(2, 4, device=t),
                                 torch.ones(4, 4, device=t),
                                 torch.ones(4, 4, device=t))
            return True
        except Exception as exc:
            print(f"  Device sanity check failed on {dev}: "
                  f"{type(exc).__name__}: {exc}")
            return False

    # Cached result of the device-selection sanity walk — recomputing it on
    # every chunk in the streaming path would add a few-ms penalty per call
    # for no information gain (hardware doesn't change mid-run).
    _cached_device: "str | None" = None

    @classmethod
    def select_device(cls) -> str:
        """Auto-pick the best device that actually works on this machine.

        Preference order: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU.
        Each candidate goes through a self-test before we commit.  This
        prevents the analysis from picking MPS, running the bandpass + max-
        pool fine, then dying on `torch.linalg.solve` halfway through a
        16 000-frame stack.  Result is cached for the process lifetime.
        """
        if cls._cached_device is not None:
            return cls._cached_device
        for cand in cls.list_devices():
            if cls._device_sanity_check(cand):
                cls._cached_device = cand
                return cand
        cls._cached_device = "cpu"
        return "cpu"

    @staticmethod
    def _gaussian_blur(x, sigma, device):
        """Separable 1-D Gaussian blur via two conv1d-flavoured conv2d calls."""
        import torch
        import torch.nn.functional as F
        radius = max(1, int(round(3 * sigma)))
        kx = torch.arange(-radius, radius + 1, device=device, dtype=x.dtype)
        kernel_1d = torch.exp(-(kx ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # (1, 1, 1, k) — horizontal
        kh = kernel_1d.view(1, 1, 1, -1)
        # (1, 1, k, 1) — vertical
        kv = kernel_1d.view(1, 1, -1, 1)
        x = F.conv2d(x, kh, padding=(0, radius))
        x = F.conv2d(x, kv, padding=(radius, 0))
        return x

    @staticmethod
    def _build_gaussian_design_matrix(dy_grid, dx_grid):
        """Precompute the (k², 4) design matrix and its pseudo-inverse for the
        log-Gaussian linear least-squares fit.

        Model:   log(I) = a + b·x + c·y + p·(x² + y²)
                 where  p = -1/(2σ²),  b = -2·x₀·p,  c = -2·y₀·p
                 ⇒    x₀ = -b/(2p),   y₀ = -c/(2p)

        M is identical for every spot (only depends on the patch geometry),
        so we precompute its pseudo-inverse once and reuse it as a batched
        matrix-multiply per chunk.  Cost: a single (N, k²) @ (k², 4) gemm.
        """
        import torch
        x_flat = dx_grid.reshape(-1)
        y_flat = dy_grid.reshape(-1)
        ones   = torch.ones_like(x_flat)
        M = torch.stack([ones, x_flat, y_flat, x_flat**2 + y_flat**2], dim=1)
        # Pseudoinverse: M_pinv = (MᵀM)⁻¹Mᵀ  — shape (4, k²)
        M_pinv = torch.linalg.pinv(M)
        return M, M_pinv

    @staticmethod
    def _gaussian_lstsq_refine(patches, dy_grid, dx_grid, M):
        """Batched analytical 2D-Gaussian fit on patches via the *normal
        equations* of a weighted log-linearisation.

        Why normal equations and not `torch.linalg.lstsq`?
        --------------------------------------------------
        `torch.linalg.lstsq` is NOT implemented on the MPS device in current
        PyTorch builds (it raises NotImplementedError for `aten::linalg_lstsq.out`).
        `torch.linalg.solve` is — and for full-rank weighted least-squares,
        solving the 4×4 normal equations `(MᵀWᵀWM) b = MᵀWᵀW y` gives the
        identical answer.  The reformulation buys us cross-device support
        (CPU, CUDA, MPS) at the cost of a slightly higher condition number,
        which is irrelevant for the well-posed 4-parameter Gaussian fit.

        Why weighted?
        -------------
        Unweighted log-space LSQ gives every pixel — including dim, noisy
        edge pixels — equal influence on the centroid.  This inflates per-
        spot variance, which manifests as a depressed MSD α (because
        MSD = MSD_true + 4σ²_loc; higher σ_loc flattens the apparent log-log
        slope at short lags).  Weighting each pixel by √I (Poisson-likelihood
        weighting in log-space) means bright spot-centre pixels dominate the
        fit, restoring centroid-of-mass-like noise behaviour while preserving
        the unbiased mean-position accuracy of the Gaussian fit.

        Math
        ----
        Model:    log(I) = a + b·x + c·y + p·(x² + y²)            (linear in params)
        Weights:  w² = I       ⇒  weighted residual = √I · (a + b·x + c·y + p·(x²+y²) − log(I))
        Normal eq: A b = v,   A = MᵀWᵀWM = Σᵢ Iᵢ·MᵢMᵢᵀ,   v = MᵀWᵀWy = Σᵢ Iᵢ·log(Iᵢ)·Mᵢ
        Recover:  x₀ = −b/(2p),   y₀ = −c/(2p),   σ² = −1/(2p)

        Inputs
        ------
        patches : (N, k, k) float tensor — non-negative pixel intensities
        dy_grid : (k, k)    float tensor — y offsets relative to patch centre
        dx_grid : (k, k)    float tensor — x offsets relative to patch centre
        M       : (k², 4)   float tensor — design matrix [1, x, y, x²+y²]

        Returns (dy_sub, dx_sub, ok) where:
          dy_sub, dx_sub : (N,) sub-pixel offsets relative to the patch centre
          ok             : (N,) bool mask — True for spots whose fit is valid
        """
        import torch
        N, k, _ = patches.shape
        eps = 1e-6
        I_flat = patches.clamp(min=eps).reshape(N, k * k)          # (N, k²)
        Y_log  = torch.log(I_flat)                                  # (N, k²)

        # Normal equations: per-spot A is (4, 4); per-spot v is (4,)
        # A[n, j, k] = Σᵢ I[n, i] · M[i, j] · M[i, k]
        # v[n, j]    = Σᵢ I[n, i] · log(I[n, i]) · M[i, j]
        A = torch.einsum('ni,ij,ik->njk', I_flat, M, M)             # (N, 4, 4)
        v = torch.einsum('ni,ij->nj', I_flat * Y_log, M)            # (N, 4)

        # Tikhonov-style ridge for numerical conditioning on near-flat patches.
        # 1e-6 * trace(A) per spot is small enough not to bias real spots but
        # keeps degenerate ones from blowing up the solver.
        ridge = 1e-6 * torch.diagonal(A, dim1=1, dim2=2).mean(dim=1)
        eye   = torch.eye(4, device=A.device, dtype=A.dtype)
        A = A + ridge.view(-1, 1, 1) * eye.unsqueeze(0)

        # Solve N independent 4×4 systems.  `torch.linalg.solve` is supported
        # on CPU / CUDA / MPS — unlike `lstsq` which lacks MPS coverage.
        try:
            sol = torch.linalg.solve(A, v.unsqueeze(-1)).squeeze(-1)   # (N, 4)
        except (NotImplementedError, RuntimeError) as exc:
            # Final belt-and-braces fallback: shuttle to CPU.  Should never
            # trigger in normal operation, but it means a single missing
            # kernel won't kill the run.
            print(f"  [TorchBackend] linalg.solve fallback to CPU: {exc}")
            sol = torch.linalg.solve(A.cpu(),
                                     v.unsqueeze(-1).cpu()).squeeze(-1).to(A.device)

        a, b, c, p = sol.unbind(dim=1)
        # Guard against degenerate fits: p must be negative (peak, not pit)
        safe_p = torch.where(p < -1e-8, p, torch.full_like(p, -1e-8))
        dx_sub = -b / (2.0 * safe_p)
        dy_sub = -c / (2.0 * safe_p)
        # Reject fits whose centroid lies well outside the patch — clamping to
        # ≤ 1.5 px keeps spurious "edge wins" from leaking through.  A real
        # spot's Gaussian fit lands within ±0.5 px of the integer maximum.
        ok = (p < -1e-8) & (dx_sub.abs() <= 1.5) & (dy_sub.abs() <= 1.5)
        return dy_sub, dx_sub, ok

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None,
                 device=None, **_):
        import torch
        import torch.nn.functional as F

        if diameter % 2 == 0:
            diameter += 1
        radius = diameter // 2
        k = diameter

        # Resolve device: explicit `device=` arg > `_forced_device` set by
        # the 'torch-mps'/'torch-cuda'/'torch-cpu' GUI pins > auto-select.
        dev_str = (device
                   or getattr(self, "_forced_device", None)
                   or self.select_device())
        dev     = torch.device(dev_str)
        # Float32 is plenty for centroid math; saves memory on GPUs and
        # avoids dtype gotchas with MPS (which dislikes float64).
        dtype = torch.float32

        # See note in preprocess_and_localise_stream re: why we don't bump
        # chunk_size on GPU — Apple Silicon is bandwidth-limited, not
        # dispatch-limited, so the caller's chunk_size (typically 500) is
        # actually optimal.  Honour it as passed.
        n_frames = len(stack)
        n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))

        print(f"  Device    : {dev_str}")
        print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}  "
              f"|  percentile: {percentile}")
        print(f"  Chunks    : {n_chunks} × ~{chunk_size} frames")

        t0 = time.perf_counter()
        all_locs: list[dict] = []

        # Index grid used for sub-pixel refinement (cached on device).  Same
        # tensor is shared by the centroid-of-mass and Gaussian-LSQ paths.
        dy_grid, dx_grid = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=dev, dtype=dtype),
            torch.arange(-radius, radius + 1, device=dev, dtype=dtype),
            indexing="ij")

        # Precompute the Gaussian-LSQ design matrix once per call — it
        # depends only on the patch geometry.  (The unweighted pseudo-inverse
        # is computed too, kept for reference but no longer used since we
        # switched to weighted batched LSQ for better noise behaviour.)
        _M, _M_pinv = self._build_gaussian_design_matrix(dy_grid, dx_grid)

        for chunk_idx, chunk_start in enumerate(range(0, n_frames, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_np  = np.asarray(stack[chunk_start:chunk_end], dtype=np.float32)

            # (T, 1, Y, X)
            x = torch.from_numpy(chunk_np).to(dev, dtype=dtype).unsqueeze(1)
            T, _, Y, X = x.shape

            # ── 1. Bandpass: subtract local background, then small smooth ───
            bg = F.avg_pool2d(x, kernel_size=2 * radius + 1,
                              stride=1, padding=radius)
            smooth_sigma = max(1.0, diameter / 4.0)
            signal = self._gaussian_blur(x - bg, sigma=smooth_sigma, device=dev)
            signal = torch.clamp(signal, min=0.0)

            # ── 2. Percentile threshold per chunk ───────────────────────────
            # torch.quantile is exact for small inputs; for big tensors use
            # sample-based estimate to bound memory.
            flat = signal.reshape(-1)
            if flat.numel() > 5_000_000:
                idx = torch.randint(0, flat.numel(),
                                    (5_000_000,), device=dev)
                sample = flat[idx]
                threshold = torch.quantile(sample, percentile / 100.0)
            else:
                threshold = torch.quantile(flat, percentile / 100.0)

            # ── 3. Local maxima via max-pool == self ────────────────────────
            maxp   = F.max_pool2d(signal, kernel_size=k, stride=1, padding=radius)
            is_max = (signal == maxp) & (signal > threshold)
            # nonzero → (N, 4) columns: (t, c, y, x)
            coords = is_max.nonzero(as_tuple=False)
            if coords.numel() == 0:
                continue

            # Drop maxima too close to the edge to extract a full patch
            edge_ok = (
                (coords[:, 2] >= radius) & (coords[:, 2] < Y - radius) &
                (coords[:, 3] >= radius) & (coords[:, 3] < X - radius)
            )
            coords = coords[edge_ok]
            if coords.numel() == 0:
                continue

            t_ix = coords[:, 0]
            y_ix = coords[:, 2]
            x_ix = coords[:, 3]

            # ── 4. Patch extraction via batched advanced indexing ───────────
            # ys: (N, k, k), xs: (N, k, k), ts: (N, k, k)
            ys = y_ix[:, None, None] + dy_grid.long()[None]
            xs = x_ix[:, None, None] + dx_grid.long()[None]
            ts = t_ix[:, None, None].expand_as(ys)
            patches = signal[ts, 0, ys, xs]   # (N, k, k)

            # ── 5. Mass + filter ────────────────────────────────────────────
            mass = patches.sum(dim=(1, 2))
            keep = mass >= minmass
            if keep.sum() == 0:
                continue
            patches = patches[keep]
            t_ix    = t_ix[keep]
            y_ix    = y_ix[keep]
            x_ix    = x_ix[keep]
            mass    = mass[keep]

            # ── 6. Sub-pixel refinement ─────────────────────────────────────
            # Primary path: analytical 2D-Gaussian fit on log-intensities,
            #   one batched solve per sub-batch.  Matches trackpy's iterative
            #   refinement to within ≈10 nm and tightens trajectory recovery
            #   vs centroid-of-mass alone.
            # Fallback path: centroid of mass — used only for the small set
            #   of spots whose Gaussian fit was rejected.
            #
            # Sub-batching: when N is large (low-minmass / noisy data can
            # easily produce 10s of thousands of "spots" per chunk), feeding
            # all of them into `torch.linalg.solve` in a single call has
            # been observed to misbehave on MPS — typically subsequent
            # chunks then return 0 maxima as the MPS allocator state stays
            # degraded.  Splitting the fit into ≤5000-spot sub-batches
            # avoids that edge case while keeping batched-LSQ efficient.
            MAX_FIT_BATCH = 5_000
            N_spots = patches.shape[0]
            if N_spots > MAX_FIT_BATCH:
                dy_g_parts, dx_g_parts, ok_parts = [], [], []
                for _start in range(0, N_spots, MAX_FIT_BATCH):
                    _end  = min(_start + MAX_FIT_BATCH, N_spots)
                    _dyg, _dxg, _okg = self._gaussian_lstsq_refine(
                        patches[_start:_end], dy_grid, dx_grid, _M)
                    dy_g_parts.append(_dyg)
                    dx_g_parts.append(_dxg)
                    ok_parts.append(_okg)
                dy_g = torch.cat(dy_g_parts)
                dx_g = torch.cat(dx_g_parts)
                ok   = torch.cat(ok_parts)
            else:
                dy_g, dx_g, ok = self._gaussian_lstsq_refine(
                    patches, dy_grid, dx_grid, _M)

            patch_sum = patches.sum(dim=(1, 2)).clamp(min=1e-6)
            dy_cm = (patches * dy_grid[None]).sum(dim=(1, 2)) / patch_sum
            dx_cm = (patches * dx_grid[None]).sum(dim=(1, 2)) / patch_sum

            # Combine: use Gaussian where OK, fall back to centroid otherwise
            dy_off = torch.where(ok, dy_g, dy_cm)
            dx_off = torch.where(ok, dx_g, dx_cm)

            x_sub = x_ix.to(dtype) + dx_off
            y_sub = y_ix.to(dtype) + dy_off
            frame_abs = (t_ix + chunk_start).to(torch.int64)

            all_locs.append({
                "x":     x_sub.detach().cpu().numpy(),
                "y":     y_sub.detach().cpu().numpy(),
                "frame": frame_abs.detach().cpu().numpy(),
                "mass":  mass.detach().cpu().numpy(),
            })

            # Free chunk allocations promptly.  PyTorch's reference-counting
            # releases the Python handles, but on MPS the underlying device
            # memory isn't actually returned until queued command buffers
            # complete.  Sequence here:
            #   1. del Python handles
            #   2. synchronize: wait for the device's command queue to drain
            #   3. empty_cache: release the pool back to the system
            # Without the synchronize, mps.empty_cache() returns immediately
            # and the memory stays committed — which on a 16 GB unified
            # M-series machine can starve downstream stages (matplotlib
            # rendering, Qt repaint) of GPU memory and produce confusing
            # OOM errors that look unrelated to the localisation step.
            del x, bg, signal, maxp, is_max, coords, patches
            if dev_str == "mps":
                try:
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except Exception:
                    pass
            elif dev_str.startswith("cuda"):
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # Drop the cached on-device tensors (design matrix, index grids) and
        # force a full GPU drain before returning.  Otherwise the next
        # CPU-only stage (linking) inherits a degraded MPS context — its
        # finalizers run when Python GC kicks in during link_trajectories
        # and produce "command buffer exited with error" OOM messages that
        # have nothing to do with the actual cause.
        del dy_grid, dx_grid, _M, _M_pinv
        if dev_str == "mps":
            try:
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        elif dev_str.startswith("cuda"):
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

        if not all_locs:
            print("  Found 0 localisations")
            return pd.DataFrame(columns=["x", "y", "frame", "mass"])

        df = pd.DataFrame({
            col: np.concatenate([d[col] for d in all_locs])
            for col in ("x", "y", "frame", "mass")
        })

        elapsed = time.perf_counter() - t0
        print(f"  Found {len(df):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return df


# Order matters: `backend="auto"` resolves to the first available entry.
# TorchBackend stays AFTER TrackpyBackend in A2 so "auto" still picks trackpy
# while users validate the new path explicitly by selecting "torch" in the GUI.
# A3 will swap the order once we've confirmed numerical agreement on real data.
_BACKEND_REGISTRY: list[type[LocaliserBackend]] = [TrackpyBackend, TorchBackend]


def list_available_backends() -> list[str]:
    """Return the names of all backends usable on this machine.

    For TorchBackend this expands to one entry per visible device
    (`torch` = auto-select fastest; `torch-mps` / `torch-cuda` / `torch-cpu`
    = explicit device pin, useful for benchmarking or reproducibility).
    """
    out: list[str] = []
    for b in _BACKEND_REGISTRY:
        if not b.is_available():
            continue
        out.append(b.name)
        if b is TorchBackend:
            for dev in TorchBackend.list_devices():
                out.append(f"torch-{dev.replace(':', '')}")
    return out


def _resolve_backend(name: str | None):
    """Look up a backend by name; resolve 'auto' to the FASTEST available
    backend that's actually healthy on this machine.

    Auto-selection logic:
      1. Prefer TorchBackend if a GPU device (MPS / CUDA) passes the sanity
         check — that's the only configuration where torch beats trackpy.
      2. Otherwise pick TrackpyBackend.  Torch-on-CPU is comparable to
         trackpy in speed but less battle-tested, so trackpy wins ties.

    This keeps users on M-series Macs out of the MPS-OOM trap when their
    Metal context is degraded (e.g. after an aborted prior process): the
    sanity check fails, select_device() returns "cpu", and auto picks
    trackpy.  After a reboot when MPS works again, auto picks torch
    automatically — the user never has to touch the dropdown.

    Accepts torch-device pins (`torch-mps`, `torch-cuda`, `torch-cpu`) that
    pre-set the device on the returned instance — used for benchmarking
    and to let users force a specific device path.
    """
    if name in (None, "", "auto"):
        # Smart-auto: GPU-first.  Order is CUDA → MPS → trackpy → torch-CPU.
        #
        # Earlier versions skipped MPS in auto-resolution because of
        # reliability issues observed on macOS 26 + M4 + PyTorch 2.12 (the
        # MPS allocator producing Metal command-buffer OOMs at extreme
        # spot density).  Most of those have been mitigated since:
        #   • PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 set at process start
        #   • per-chunk + end-of-localise mps.synchronize + empty_cache
        #   • Gaussian fit sub-batched at 5k spots/call to avoid the
        #     batched linalg.solve issue
        #   • subprocess isolation so Qt's Metal claim doesn't compete
        #     with PyTorch's MPS for unified memory on Apple Silicon
        # With those in place, MPS is the right default on Apple Silicon
        # (~6× faster than CPU on typical SPT stacks).  If a specific
        # machine still has trouble, users can manually pick Trackpy or
        # Torch — CPU from the dropdown.
        if TorchBackend.is_available():
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    inst = TorchBackend()
                    inst._forced_device = "cuda"
                    return inst
                if (hasattr(_torch.backends, "mps")
                        and _torch.backends.mps.is_available()):
                    inst = TorchBackend()
                    inst._forced_device = "mps"
                    return inst
            except Exception:
                pass
        # No GPU available → reference CPU implementation (trackpy).
        for cls in _BACKEND_REGISTRY:
            if cls is TorchBackend:
                continue
            if cls.is_available():
                return cls()
        # Last resort: torch on CPU, if even trackpy is missing.
        if TorchBackend.is_available():
            inst = TorchBackend()
            inst._forced_device = "cpu"
            return inst
        raise RuntimeError(
            "No localiser backend available — install trackpy or torch.")

    # Torch-device pins (e.g. 'torch-mps', 'torch-cuda:0', 'torch-cpu')
    if name.startswith("torch-"):
        if not TorchBackend.is_available():
            raise RuntimeError(
                "Torch device pin requested but PyTorch isn't installed.")
        forced = name[len("torch-"):]
        inst = TorchBackend()
        inst._forced_device = forced
        return inst

    for cls in _BACKEND_REGISTRY:
        if cls.name == name:
            if not cls.is_available():
                raise RuntimeError(
                    f"Localiser backend '{name}' is registered but its "
                    f"dependencies aren't installed on this machine.")
            return cls()
    raise ValueError(
        f"Unknown localiser backend '{name}'. "
        f"Registered: {[c.name for c in _BACKEND_REGISTRY]}; "
        f"available here: {list_available_backends()}.")


def localise_particles(stack, diameter=7, minmass=0.1, percentile=64,
                       workers=N_CPUS, chunk_size=500, preview_cb=None,
                       backend="auto"):
    """Localise spots in every frame of a preprocessed stack.

    `backend` selects the implementation:
        "auto"     — first available entry in _BACKEND_REGISTRY
        "trackpy"  — Crocker-Grier centroid (CPU, multi-process)
        (future)   — "torch" for GPU acceleration

    Returns a DataFrame with columns: x, y, frame, mass.
    """
    impl = _resolve_backend(backend)
    print(f"  Backend   : {impl.name}")
    return impl.localise(stack, diameter=diameter, minmass=minmass,
                         percentile=percentile, workers=workers,
                         chunk_size=chunk_size, preview_cb=preview_cb)


# ══════════════════════════════════════════════════════════════════════════════
#  LINKING
# ══════════════════════════════════════════════════════════════════════════════

def link_trajectories(locs, search_range=5, memory=3, min_len=5, max_len=None,
                       progress_cb=None, stop_event=None):
    """Link localisations into trajectories.

    progress_cb : callable(fraction) → None
        Optional.  Called periodically with a [0, 1] float so the host
        can update a progress bar.  Updates are throttled to roughly
        once every 32 frames + once on completion.
    stop_event  : threading.Event-like
        Optional.  Polled between frames; if `.is_set()` the linker
        raises `_Cancelled` and aborts cleanly.

    Uses `tp.link_iter` when available so the user can see progress
    and cancel mid-link.  Falls back to atomic `tp.link` on older
    trackpy versions or if the iterator path errors out (some
    edge-case densities switch trackpy to a non-iter strategy).
    """
    print(f"  Linking {len(locs):,} localisations  "
          f"(search_range={search_range}px, memory={memory}) ...")
    t0 = time.perf_counter()

    iter_ok = hasattr(tp, "link_iter") and len(locs) > 0
    linked = None
    if iter_ok:
        # Per-frame coordinate iterator + index map so we can re-attach
        # particle IDs to the original locs DataFrame.
        try:
            frame_nums = sorted(int(f) for f in locs["frame"].unique())
            grouped = locs.groupby("frame")
            coords_per_frame: list = []
            indices_per_frame: list = []
            for f in frame_nums:
                sub = grouped.get_group(f)
                coords_per_frame.append(sub[["y", "x"]].to_numpy())
                indices_per_frame.append(sub.index.to_numpy())
            n_frames = len(frame_nums)

            particle_ids = np.full(len(locs), -1, dtype=np.int64)
            iterator = tp.link_iter(
                iter(coords_per_frame),
                search_range=search_range, memory=memory)
            for f_idx, p_ids in enumerate(iterator):
                row_idx = indices_per_frame[f_idx]
                arr = np.asarray(p_ids, dtype=np.int64)
                if arr.shape[0] != row_idx.shape[0]:
                    # Mismatch — trackpy's iter may have emitted in a
                    # different shape than we expected.  Bail to atomic.
                    raise RuntimeError("link_iter shape mismatch")
                particle_ids[row_idx] = arr
                # Progress + cancel — only every 32 frames to keep cost
                # well under linking cost itself
                if (f_idx & 31) == 0:
                    if progress_cb is not None:
                        try:    progress_cb((f_idx + 1) / max(1, n_frames))
                        except Exception: pass
                    if stop_event is not None and stop_event.is_set():
                        raise _Cancelled()
            if progress_cb is not None:
                try:    progress_cb(1.0)
                except Exception: pass

            linked = locs.copy()
            linked["particle"] = particle_ids
            linked = linked[linked["particle"] >= 0].reset_index(drop=True)
            print(f"  tp.link_iter done — filtering stubs "
                  f"(min_len={min_len}) ...")
        except _Cancelled:
            raise
        except Exception as exc:
            # Iter path didn't work — fall through to atomic tp.link
            print(f"  link_iter failed ({type(exc).__name__}: {exc}); "
                  f"falling back to atomic tp.link")
            linked = None

    if linked is None:
        # Atomic path — uninterruptible but works on any trackpy version
        if stop_event is not None and stop_event.is_set():
            raise _Cancelled()
        try:
            linked = tp.link(locs, search_range=search_range, memory=memory)
            print(f"  tp.link done — filtering stubs (min_len={min_len}) ...")
        except Exception as exc:
            if ("SubnetOversizeException" in type(exc).__name__
                    or "Subnetwork" in str(exc)):
                print(f"  WARNING: SubnetOversizeException — switching to "
                      f"nonrecursive linker (consider reducing Search range)")
                linked = tp.link(locs, search_range=search_range,
                                  memory=memory,
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


# Default alpha-exponent thresholds for the four-class motion classifier.
# Conventional sptPALM values: 0.5 / 0.9 / 1.1.  These are now the *defaults*
# but every public function that classifies motion accepts a thresholds=
# triple so users can tune the boundaries to their lab's convention.
ALPHA_THRESHOLDS_DEFAULT = (0.5, 0.9, 1.1)

# Default D cutoff for splitting Mobile / Immobile populations (µm²/s).
# 0.05 is the conventional membrane-protein threshold used throughout the
# sptPALM literature; tracks with D ≥ this value are considered Mobile.
# Defined here at the top so functions defined later in the file can use
# it as a default argument (Python evaluates defaults at definition time).
MOBILE_D_THRESHOLD_DEFAULT = 0.05


def classify_motion(alpha, thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """Classify a track by its anomalous exponent α.

    thresholds = (t_immobile, t_confined, t_directed):
        α  <  t_immobile   → "Immobile"
        t_immobile  ≤ α  <  t_confined → "Confined"
        t_confined  ≤ α  <  t_directed → "Brownian"
        α  ≥  t_directed   → "Directed"
    """
    t_imm, t_conf, t_dir = thresholds
    if   alpha < t_imm:  return "Immobile"
    elif alpha < t_conf: return "Confined"
    elif alpha < t_dir:  return "Brownian"
    else:                return "Directed"


def _msd_and_fit_one(xy_um, frames, pid, lag_times, max_lagtime, n_fit,
                     alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT):
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
    msd0 = np.nan        # linear-fit intercept (PALM-Tracer "MSD(0)")
    mse  = np.nan        # mean squared residual of the linear fit
    if ok.sum() >= 3:
        try:    alpha = np.polyfit(np.log(t[ok]), np.log(m[ok]), 1)[0]
        except: pass
        try:
            popt, _ = curve_fit(msd_linear, t[ok], m[ok], p0=[0.01, 0],
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                                maxfev=2000)
            D = popt[0]
            msd0 = float(popt[1])
            _resid = m[ok] - msd_linear(t[ok], *popt)
            mse = float(np.mean(_resid ** 2))
        except: pass

    motion = classify_motion(alpha, alpha_thresholds) if np.isfinite(alpha) else "Unknown"

    # Two distinct radial-spread metrics, both useful and named explicitly:
    #   mean_radial_displacement_um  = ⟨|r − r̄|⟩       (1st moment)
    #   radius_of_gyration_um        = √⟨|r − r̄|²⟩    (RMS, the standard Rg)
    centroid    = xy_um.mean(axis=0)
    sq_dists    = np.sum((xy_um - centroid) ** 2, axis=1)
    mean_radial = float(np.mean(np.sqrt(sq_dists)))
    rg          = float(np.sqrt(np.mean(sq_dists)))

    return pid, msd_vals, dict(particle=pid, D=D, alpha=alpha, motion=motion,
                               MSD0=msd0, MSE=mse,
                               mean_radial_displacement_um=mean_radial,
                               radius_of_gyration_um=rg)


def compute_msd_and_fit(tracks, pixel_size, frame_interval,
                        max_lagtime=20, n_fit=5, workers=N_CPUS,
                        alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT):
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

    # Defensive: if linking produced zero trajectories (e.g. localiser
    # returned no spots, or every spot is an isolated singleton), return
    # empty results instead of crashing pandas with "Empty data passed
    # with indices specified".  The caller still sees the empty result
    # and can produce a sensible "no tracks found" log message.
    if n_tracks == 0:
        print("  No trajectories — skipping MSD/fit (returning empty result).")
        imsd_empty = pd.DataFrame(
            np.full((max_lagtime, 0), np.nan, dtype=float),
            index=np.arange(1, max_lagtime + 1))
        emsd_empty = pd.Series(
            np.full(max_lagtime, np.nan, dtype=float),
            index=np.arange(1, max_lagtime + 1))
        diff_empty = pd.DataFrame(columns=[
            "particle", "D", "alpha", "motion", "MSD0", "MSE",
            "mean_radial_displacement_um", "radius_of_gyration_um"])
        return imsd_empty, emsd_empty, diff_empty

    with ThreadPoolExecutor(max_workers=workers) as _exe:
        _futs = [_exe.submit(
                    _msd_and_fit_one,
                    grouped.get_group(pid)[["x", "y"]].values * pixel_size,
                    grouped.get_group(pid)["frame"].values,
                    pid, lag_times, max_lagtime, n_fit, alpha_thresholds)
                 for pid in pid_list]
        results = [_f.result() for _f in
                   _tqdm(_futs, desc="  MSD + fitting", unit="track", ncols=70)]

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
    print(f"  JDD analysis      : {n_components} component(s)  "
          f"|  {tracks['particle'].nunique():,} tracks")
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
    """For each track with ≥3 points, compute step-to-step **signed** turning
    angles in degrees, in the range (-180°, +180°].

    Sign convention (standard 2D right-handed):
        +90°  =  90° left turn (counter-clockwise rotation from v1 to v2)
        -90°  =  90° right turn (clockwise rotation)
          0°  =  continued straight
        ±180° =  full reversal

    Computation: for consecutive step vectors v1 = r(t_{i+1}) - r(t_i)
    and v2 = r(t_{i+2}) - r(t_{i+1}),

        θ = atan2( v1.x · v2.y - v1.y · v2.x,    v1 · v2 )

    where the first argument is the z-component of the 3-D cross product
    v1 × v2 (positive for counter-clockwise rotation). Returns a flat
    array of all angles across all tracks, in degrees.
    """
    print(f"  Turning angles    : {tracks['particle'].nunique():,} tracks")
    all_angles = []
    for pid, grp in tracks.groupby("particle"):
        grp = grp.reset_index(drop=True).sort_values("frame")
        xy  = grp[["x", "y"]].values
        if len(xy) < 3:
            continue
        v1 = np.diff(xy, axis=0)[:-1]   # shape (n-2, 2)
        v2 = np.diff(xy, axis=0)[1:]    # shape (n-2, 2)
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        dot   = np.sum(v1 * v2, axis=1)
        # Skip steps where either vector is zero-length (atan2(0,0) is 0
        # but isn't meaningful for a non-existent rotation).
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        valid = (norm1 > 0) & (norm2 > 0)
        if not valid.any():
            continue
        angles = np.degrees(np.arctan2(cross[valid], dot[valid]))
        all_angles.append(angles)
    if all_angles:
        result = np.concatenate(all_angles)
        # Distribution sanity check — Brownian motion should produce a
        # roughly symmetric distribution around 0°.  Strong asymmetry can
        # indicate uncorrected drift, an asymmetric cellular geometry, or
        # a real biological turn bias.  Printed for diagnostic verification.
        if len(result) > 0:
            pos = int((result > 0).sum())
            neg = int((result < 0).sum())
            zer = int((result == 0).sum())
            print(f"    signed turning angles: "
                  f"{pos:,} positive  /  {neg:,} negative  /  {zer:,} zero  "
                  f"|  min={result.min():.1f}°  max={result.max():.1f}°  "
                  f"mean={result.mean():.2f}°  median={np.median(result):.2f}°")
        return result
    return np.array([])


# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE FRACTION OVER TIME
# ══════════════════════════════════════════════════════════════════════════════

def compute_mobile_fraction_over_time(tracks, diff_df, frame_interval,
                                       window_frames=100,
                                       d_threshold=MOBILE_D_THRESHOLD_DEFAULT):
    """Compute mobile fraction in sliding windows of `window_frames` frames.

    Mobile = tracks with D ≥ d_threshold (consistent with _mob_immob_ratio
    and the LogD-distribution panel's threshold line).  Tracks with
    non-finite D are excluded from the window denominator.

    Returns DataFrame with columns: time_s, mobile_fraction, n_tracks.
    Only windows with ≥5 valid tracks are included.
    """
    if len(tracks) == 0 or len(diff_df) == 0:
        return pd.DataFrame(columns=["time_s", "mobile_fraction", "n_tracks"])

    track_times = tracks.groupby("particle")["frame"].mean().reset_index()
    track_times.columns = ["particle", "mean_frame"]
    merged = track_times.merge(diff_df[["particle", "D"]], on="particle", how="inner")
    # Drop tracks where D could not be fit
    merged = merged[np.isfinite(merged["D"]) & (merged["D"] > 0)]

    max_frame = int(tracks["frame"].max())
    windows   = range(0, max_frame, window_frames)
    rows = []
    for w in windows:
        sel = merged[(merged["mean_frame"] >= w) &
                     (merged["mean_frame"] < w + window_frames)]
        total = len(sel)
        if total < 5:
            continue
        mobile = int((sel["D"] >= d_threshold).sum())
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
        print(f"  Cluster analysis  : sub-sampled to {max_locs:,} localisations")
    else:
        print(f"  Cluster analysis  : {len(xy):,} localisations  "
              f"(eps={eps_um*1000:.0f} nm, min_samples={min_samples})")
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
    """Per-track dwell times for confined / immobile tracks.

    Returns a DataFrame with three durations per track:

      dwell_time_total_s     (last_frame − first_frame + 1) × Δt   ← canonical
      dwell_time_observed_s  n_observations × Δt                   ← fewer if gaps
      dwell_time_s           alias for dwell_time_total_s          ← back-compat

    The exponential τ is fit to dwell_time_total_s (residence-time semantics).
    """
    confined_pids = diff_df[diff_df["motion"].isin(["Confined", "Immobile"])]["particle"]
    print(f"  Dwell times       : {len(confined_pids):,} confined/immobile tracks")
    rows = []
    # Group once by particle for speed
    grouped = tracks.groupby("particle")["frame"]
    for pid in confined_pids:
        if pid not in grouped.groups:
            continue
        frames = grouped.get_group(pid).values
        n_obs = len(frames)
        if n_obs == 0:
            continue
        f_min = int(frames.min())
        f_max = int(frames.max())
        dur_total = (f_max - f_min + 1) * frame_interval
        dur_obs   = n_obs * frame_interval
        rows.append({
            "particle":              int(pid),
            "dwell_time_s":          dur_total,   # back-compat alias
            "dwell_time_total_s":    dur_total,   # full duration including gaps
            "dwell_time_observed_s": dur_obs,     # observed frames × Δt
            "n_observations":        int(n_obs),
        })
    dwell_df = pd.DataFrame(rows)
    tau = np.nan
    if len(dwell_df) >= 10:
        try:
            dt = np.sort(dwell_df["dwell_time_total_s"].values)
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
    n_tracks = tracks["particle"].nunique()
    print(f"  MSS analysis      : {n_tracks:,} tracks")
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
                pixel_size, frame_interval, output_path=None, roi_mask=None,
                fig_theme="Dark", proj_cmap="Inferno", jdd=None,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None, return_pdf_bytes=False):
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
    # Grid expanded from 5 to 6 rows in v1.0.64 to fit the new Radial
    # Distribution polar panel.
    fig = plt.figure(figsize=(20, 38), facecolor=BG)
    gs  = GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.32,
                   left=0.06, right=0.97, top=0.95, bottom=0.035)

    _panels          = []   # (letter, axes) collected for per-panel export
    _letter_artists  = []   # text objects for letter labels (hidden for panel renders)

    def sax(ax, ltr, ttl):
        ax.set_facecolor(PNL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        ax.set_title(f"  {ttl}", loc="left", fontsize=11,
                     color=TXT, pad=8, fontweight="bold")
        txt = ax.text(-0.04, 1.06, ltr, transform=ax.transAxes, fontsize=14,
                      color=ACC, fontweight="bold", va="top", ha="right")
        _panels.append((ltr, ax))
        _letter_artists.append(txt)

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
    # Plotted as a single LINE following the count of each |angle| bin,
    # using UNSIGNED magnitudes (|θ|) so the x-axis runs 0°–180°.
    # 0° = continued straight; 180° = full reversal; 90° = right-angle
    # deflection; the radial-distribution panel (O) shows the rotational
    # direction (sign) separately.
    ax = fig.add_subplot(gs[2, 2])
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ta_unsigned = np.abs(np.asarray(turning_angles, dtype=float))
        _ta_bins = np.linspace(0, 180, 37)            # 5° bins
        _ta_centres = 0.5 * (_ta_bins[:-1] + _ta_bins[1:])
        _ta_counts, _ = np.histogram(ta_unsigned, bins=_ta_bins)
        # Normalise to relative frequency so the shape is comparable across
        # runs (and consistent with the Compare-mode panel).  Total track
        # count is already reported in the suptitle / Summary tab.
        _ta_freq = (_ta_counts / _ta_counts.sum()
                    if _ta_counts.sum() else _ta_counts)
        ax.plot(_ta_centres, _ta_freq, "-o",
                color=ACC, lw=2, ms=3, alpha=0.95)
        # Uniform-distribution reference line (1/N_bins)
        ax.axhline(1.0 / len(_ta_centres),
                   color=GRD, lw=0.6, ls=":", label="uniform")
        # Reference verticals: 90° (right-angle), 180° (full reversal)
        ax.axvline(90,  color=GRD, lw=0.8, ls="--")
        ax.axvline(180, color=GRD, lw=0.6, ls=":")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("|Turning angle|  (°)", fontsize=9)
        ax.set_ylabel("Relative frequency", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="best")
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
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
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
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
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
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "N", "Moment Scaling Spectrum  (MSS slope)")

    # O — Radial Distribution of turning angles (polar)
    # A polar histogram of signed turning angles, oriented so 0° (straight
    # ahead) is at the top and positive angles sweep CLOCKWISE around to the
    # right (i.e. right hemisphere = positive turns, left hemisphere =
    # negative turns).  The bars radiate outward; their angular position is
    # the turning direction, their height the relative frequency.  Uniform
    # circle = Brownian motion; lobe at 0° = directional persistence; lobe
    # at ±180° = back-tracking / confinement.
    # Placed at the centre column of row 5 so it sits visually balanced
    # rather than pinned to a corner.
    ax = fig.add_subplot(gs[5, 1], projection="polar")
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ta_arr = np.asarray(turning_angles, dtype=float)
        is_signed = bool(np.any(ta_arr < -1e-3))
        print(f"  Radial-dist input: n={len(ta_arr):,}  "
              f"signed={is_signed}  "
              f"pos={int((ta_arr>0).sum()):,}  neg={int((ta_arr<0).sum()):,}  "
              f"min={ta_arr.min():.1f}°  max={ta_arr.max():.1f}°")
        if not is_signed:
            ta_arr = np.concatenate([ta_arr, -ta_arr])
        # CRITICAL: matplotlib polar's ax.bar() does NOT render correctly
        # when theta values are in (-π, +π].  Half the bars (the side with
        # negative theta after applying set_theta_direction) silently fail
        # to draw, producing only a half-circle of bars.
        # Empirical fix: shift the angles to [0, 2π) before histogramming.
        # The xticks are then placed at positive-only angles too, but
        # *labelled* with the signed values the user expects.
        angles_rad = np.mod(np.deg2rad(ta_arr), 2 * np.pi)
        n_bins = 36
        bins   = np.linspace(0, 2 * np.pi, n_bins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins, density=True)
        theta = 0.5 * (edges[:-1] + edges[1:])
        width = bins[1] - bins[0]
        ax.bar(theta, counts, width=width * 0.95, bottom=0.0,
               color=ACC, alpha=0.75, edgecolor=GRD, linewidth=0.5)
        ax.set_theta_zero_location("N")     # 0° at the top
        ax.set_theta_direction(-1)          # clockwise positive (right = +)
        # xticks at 0°, 45°, ..., 315° (positive only); labels show signed
        # equivalents so the reader still sees "-45°" on the left, etc.
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                            "−135°", "−90°", "−45°"], fontsize=8)
        # Hide the radial-axis numeric labels.
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False)
        ax.grid(True, ls=":", alpha=0.4)
    sax(ax, "O", "Radial Distribution  (signed turning angles)")

    md = diff_df["D"].dropna().median()
    ma = diff_df["alpha"].dropna().median()
    fig.suptitle(
        f"FIREFLY Analysis  |  {diff_df.shape[0]:,} trajectories  |  "
        f"Median D = {md:.4f} um2/s  |  Median alpha = {ma:.2f}",
        fontsize=13,color=TXT,y=0.97,fontweight="bold")

    import io as _io
    from matplotlib.transforms import Bbox as _Bbox

    from PIL import Image as _PILImage

    # Render individual panels WITHOUT letter labels
    for _txt in _letter_artists:
        _txt.set_visible(False)
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _pad_px   = fig.dpi * 0.12
    panel_images = {}
    for _ltr, _pax in _panels:
        _bbox = _pax.get_tightbbox(_renderer)
        if _bbox is None:
            continue
        _bbox_pad = _Bbox([[_bbox.x0 - _pad_px, _bbox.y0 - _pad_px],
                            [_bbox.x1 + _pad_px, _bbox.y1 + _pad_px]])
        _bbox_in  = _bbox_pad.transformed(fig.dpi_scale_trans.inverted())
        _pbuf = _io.BytesIO()
        fig.savefig(_pbuf, format="png", dpi=150, bbox_inches=_bbox_in,
                    facecolor=fig.get_facecolor())
        _pbuf.seek(0)
        panel_images[_ltr] = _PILImage.open(_pbuf).copy()
        _pbuf.close()

    # Restore letter labels then render combined figure
    for _txt in _letter_artists:
        _txt.set_visible(True)
    fig.canvas.draw()
    _buf = _io.BytesIO()
    fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    _buf.seek(0)
    combined_pil = _PILImage.open(_buf).copy()
    _buf.close()

    # Save to disk only if output_path explicitly provided (CLI / legacy callers)
    if output_path:
        combined_pil.save(output_path, dpi=(150, 150))
        print(f"  Figure -> {output_path}")
        _pdf = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Figure (PDF) -> {_pdf}")

    pdf_bytes = None
    if return_pdf_bytes:
        try:
            _pdfbuf = _io.BytesIO()
            fig.savefig(_pdfbuf, format="pdf", bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            pdf_bytes = _pdfbuf.getvalue()
            _pdfbuf.close()
        except Exception as _exc:
            print(f"  WARN: PDF render failed: {_exc}")

    plt.close(fig)
    print("  Figure rendered.")
    return {
        "combined":     combined_pil,
        "panels":       panel_images,
        "panel_titles": {ltr: ax.get_title().strip() for ltr, ax in _panels},
        "pdf_bytes":    pdf_bytes,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FIREFLY — Fluorescence Inference & Reconstruction Engine "
                    "(CZI / TIF, optimised)",
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
        _peak = float(np.percentile(sample, 99))
        # See _fast_preprocess_and_localise for the rationale on the d²/8 factor.
        args.minmass = float(_peak * (args.diameter ** 2) / 8.0)
        print(f"  Auto minmass: {args.minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")

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


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON  —  group A vs group B over multiple analysis output folders
# ══════════════════════════════════════════════════════════════════════════════
#
# A "group" is a list of analysis output folders.  For each folder we re-load
# the per-experiment summary (MSD curve, D values, motion classes, etc.) and
# compute scalar metrics (AUC, mob/immob ratio, mean track length).  We then
# render a multi-panel figure overlaying the two groups, with scatter dots
# per replicate and t-test significance stars on bar charts when n≥2 each.
# Layout matches the lab's Pre/Post style: MSD curve overlay, AUC bar chart,
# LogD frequency distribution, mobile/immobile ratio bar chart, motion class
# fractions, track length distribution, JDD, dwell time CDF, turning angles.

def _find_stem(data_dir):
    """Find the experiment stem from filenames like {stem}_params.json or
    {stem}_diffusion_summary.csv inside an analysis output folder's data/ dir."""
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_params.json"):
            return f[:-len("_params.json")]
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_diffusion_summary.csv"):
            return f[:-len("_diffusion_summary.csv")]
    raise FileNotFoundError(f"No analysis CSVs found in {data_dir}")


def _is_palmtracer_folder(folder):
    """Return True if `folder` contains raw PALM-Tracer output."""
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    # PALM-Tracer files have no stem prefix (e.g. 'locPALMTracer.txt')
    has_loc = any(n.lower() == "locpalmtracer.txt" or n.lower() == "locpalmtracer.csv"
                  for n in names)
    has_trc = any(n.lower() == "trcpalmtracer.txt" or n.lower() == "trcpalmtracer.csv"
                  for n in names)
    return has_loc and has_trc


def _read_palmtracer_table(path, header_lines):
    """Read a PALM-Tracer file (tab- or comma-separated), skipping comment /
    metadata rows.  `header_lines` is the number of non-data leading rows."""
    # PALM-Tracer's reference files are TSV; FIREFLY-emitted ones are CSV.
    # Sniff the separator from the first data line.
    with open(path, "r") as fh:
        for _ in range(header_lines):
            fh.readline()
        first = fh.readline()
    sep = "\t" if "\t" in first and first.count("\t") >= first.count(",") else ","
    return pd.read_csv(path, sep=sep, header=None, comment="#",
                       skiprows=header_lines, engine="python")


def load_summary_from_palmtracer(folder):
    """
    Read a raw PALM-Tracer output folder and return the same dict shape as
    `load_summary_from_folder` so the Compare tab can treat it identically.

    PALM-Tracer does not store FIREFLY-specific quantities (alpha, motion
    class, dwell times, turning angles, JDD, mobile fraction, Rg) — these are
    re-derived on the fly from the imported trajectories using the same
    pipeline functions FIREFLY normally runs.
    """
    # ── Locate the six PALM-Tracer files (tab or csv) ────────────────────
    def _pick(*candidates):
        for c in candidates:
            p = os.path.join(folder, c)
            if os.path.isfile(p):
                return p
        return None

    loc_path = _pick("locPALMTracer.txt", "locPALMTracer.csv")
    trc_path = _pick("trcPALMTracer.txt", "trcPALMTracer.csv")
    d_path   = _pick("trcPALMTracer-AllROI-D.txt", "trcPALMTracer-AllROI-D.csv",
                     "trcPALMTracer-1-D.txt",     "trcPALMTracer-1-D.csv")
    msd_path = _pick("trcPALMTracer-AllROI-MSD.txt", "trcPALMTracer-AllROI-MSD.csv",
                     "trcPALMTracer-1-MSD.txt",     "trcPALMTracer-1-MSD.csv")

    if not (loc_path and trc_path):
        raise FileNotFoundError(f"PALM-Tracer files not found in {folder}")

    # ── Parse loc / trc metadata header (line 2 contains values) ─────────
    pixel_size_um    = 0.106
    frame_interval_s = 0.02
    width = height = n_frames = 0
    try:
        with open(loc_path, "r") as fh:
            _hdr_names  = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
            _hdr_values = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
        meta = {k.strip(): v.strip() for k, v in zip(_hdr_names, _hdr_values)}
        pixel_size_um    = float(meta.get("Pixel_Size(um)", pixel_size_um))
        frame_interval_s = float(meta.get("Frame_Duration(s)", frame_interval_s))
        width    = int(float(meta.get("Width",  0) or 0))
        height   = int(float(meta.get("Height", 0) or 0))
        n_frames = int(float(meta.get("nb_Planes", 0) or 0))
    except Exception:
        pass

    # ── Localisations ────────────────────────────────────────────────────
    # Header rows in loc/trc files: metadata-names, metadata-values, column-names
    loc_df = _read_palmtracer_table(loc_path, header_lines=3)
    loc_df.columns = ["id", "Plane", "Index", "Channel", "Integrated_Intensity",
                      "CentroidX_px", "CentroidY_px", "SigmaX_px", "SigmaY_px",
                      "Angle_rad", "MSE_Gauss", "CentroidZ_um", "MSE_Z_um",
                      "Pair_Distance_px"][:loc_df.shape[1]]
    locs = pd.DataFrame({
        "x":     loc_df["CentroidX_px"].astype(float).values,
        "y":     loc_df["CentroidY_px"].astype(float).values,
        "frame": (loc_df["Plane"].astype(int).values - 1),   # 1-based → 0-based
        "mass":  loc_df["Integrated_Intensity"].astype(float).values,
    })

    # ── Trajectories ─────────────────────────────────────────────────────
    trc_df = _read_palmtracer_table(trc_path, header_lines=3)
    trc_df.columns = ["Track", "Plane", "CentroidX_px", "CentroidY_px",
                      "CentroidZ_um", "Integrated_Intensity", "id",
                      "Pair_Distance_px"][:trc_df.shape[1]]
    tracks = pd.DataFrame({
        "particle": trc_df["Track"].astype(int).values,
        "frame":    trc_df["Plane"].astype(int).values - 1,
        "x":        trc_df["CentroidX_px"].astype(float).values,
        "y":        trc_df["CentroidY_px"].astype(float).values,
        "mass":     trc_df["Integrated_Intensity"].astype(float).values,
    }).sort_values(["particle", "frame"]).reset_index(drop=True)

    # ── Re-derive D, alpha, motion via FIREFLY's own pipeline ────────────
    # This guarantees the Compare tab sees the same column names and
    # identical statistics it would for a native FIREFLY run.
    imsd_df, emsd_series, diff_df = compute_msd_and_fit(
        tracks, pixel_size_um, frame_interval_s, max_lagtime=20, n_fit=5)

    emsd_df = (emsd_series.to_frame("msd_um2")
                          .reset_index(names="lag_frame"))

    # FIREFLY-only metrics — re-derive on the fly
    try:
        jdd = compute_jdd(tracks, pixel_size_um, frame_interval_s)
    except Exception:
        jdd = None
    try:
        dwell_df, _ = compute_dwell_times(tracks, diff_df, frame_interval_s)
    except Exception:
        dwell_df = None
    try:
        ta_deg = compute_turning_angles(tracks)
    except Exception:
        ta_deg = None
    try:
        mobile_frac_df = compute_mobile_fraction_over_time(
            tracks, diff_df, frame_interval_s)
    except Exception:
        mobile_frac_df = None

    stem = os.path.basename(folder.rstrip(os.sep)) or "palmtracer_run"
    if stem.lower().endswith(".pt"):
        stem = stem[:-3]

    # ── Cache the recomputed FIREFLY-only metrics next to the PALM-Tracer
    # files so re-opening this folder in the Compare tab is instant.  The
    # cache lives in <folder>/firefly_extras/ and uses FIREFLY's native
    # CSV/JSON schema.
    try:
        import json as _json
        extras_dir = os.path.join(folder, "firefly_extras")
        os.makedirs(extras_dir, exist_ok=True)
        diff_df.to_csv(
            os.path.join(extras_dir, f"{stem}_diffusion_summary.csv"), index=False)
        tracks.to_csv(
            os.path.join(extras_dir, f"{stem}_trajectories.csv"), index=False)
        locs.to_csv(
            os.path.join(extras_dir, f"{stem}_localisations.csv"), index=False)
        emsd_df.to_csv(
            os.path.join(extras_dir, f"{stem}_ensemble_msd.csv"), index=False)
        with open(os.path.join(extras_dir, f"{stem}_params.json"), "w") as _fp:
            _json.dump({
                "stem":             stem,
                "pixel_size_um":    pixel_size_um,
                "frame_interval_s": frame_interval_s,
                "n_localisations":  int(len(locs)),
                "n_tracks":         int(diff_df.shape[0]),
                "n_frames":         int(n_frames),
                "width":            width,
                "height":           height,
                "source":           "palmtracer (re-derived)",
            }, _fp, indent=2)
        if jdd:
            with open(os.path.join(extras_dir, f"{stem}_jdd.json"), "w") as _fp:
                _json.dump(_to_jsonable(jdd) if "_to_jsonable" in globals() else jdd,
                           _fp, indent=2, default=str)
        if dwell_df is not None and len(dwell_df):
            dwell_df.to_csv(
                os.path.join(extras_dir, f"{stem}_dwell_times.csv"), index=False)
        if ta_deg is not None and len(ta_deg):
            pd.DataFrame({"turning_angle_deg": ta_deg}).to_csv(
                os.path.join(extras_dir, f"{stem}_turning_angles.csv"), index=False)
        if mobile_frac_df is not None and len(mobile_frac_df):
            mobile_frac_df.to_csv(
                os.path.join(extras_dir, f"{stem}_mobile_fraction.csv"), index=False)
    except Exception:
        # Caching is best-effort — never fail the load over a write error
        pass

    return {
        "folder":     folder,
        "stem":       stem,
        "data_dir":   folder,
        "source":     "palmtracer",
        "params": {
            "stem":             stem,
            "pixel_size_um":    pixel_size_um,
            "frame_interval_s": frame_interval_s,
            "n_localisations":  int(len(locs)),
            "n_tracks":         int(diff_df.shape[0]),
            "n_frames":         int(n_frames),
            "width":            width,
            "height":           height,
        },
        "ensemble_msd":          emsd_df,
        "diffusion":             diff_df,
        "tracks":                tracks,
        "jdd":                   jdd,
        "dwell_times":           dwell_df,
        "turning_angles":        ta_deg if ta_deg is not None else None,
        "turning_angles_signed": True,
    }


def load_summary_from_folder(folder):
    """Load all per-experiment summary data from one analysis output folder.

    Accepts any of:
      <run_dir>/                       (containing firefly_extras/ and data/)
      <run_dir>/firefly_extras/        (the FIREFLY-extras directory itself)
      <palm_tracer_folder>/            (auto-detected, re-derived on load)
      <run_dir>/data/                  (PALM-Tracer CSVs from a FIREFLY run)
    """
    import json

    # ── Resolve which directory holds the FIREFLY-native CSVs ────────────
    # 1) <folder>/firefly_extras  (folder is the run dir)
    if os.path.isdir(os.path.join(folder, "firefly_extras")):
        data_dir = os.path.join(folder, "firefly_extras")
    # 2) folder is itself the firefly_extras dir
    elif os.path.basename(folder.rstrip(os.sep)) == "firefly_extras":
        data_dir = folder
    # 3) folder is a PALM-Tracer folder (raw or FIREFLY-emitted CSV mirrors)
    elif _is_palmtracer_folder(folder):
        return load_summary_from_palmtracer(folder)
    # 4) folder is a run dir whose `data/` holds PALM-Tracer CSVs
    elif (os.path.isdir(os.path.join(folder, "data"))
          and _is_palmtracer_folder(os.path.join(folder, "data"))):
        return load_summary_from_palmtracer(os.path.join(folder, "data"))
    else:
        raise FileNotFoundError(
            f"No firefly_extras/ directory and no PALM-Tracer files in {folder}")

    stem = _find_stem(data_dir)
    s = {"folder": folder, "stem": stem, "data_dir": data_dir}

    # Params (frame interval, pixel size, ...)
    params_path = os.path.join(data_dir, f"{stem}_params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            s["params"] = json.load(f)
    else:
        s["params"] = {"pixel_size_um": 0.104, "frame_interval_s": 0.05}

    # Ensemble MSD
    msd_path = os.path.join(data_dir, f"{stem}_ensemble_msd.csv")
    if os.path.isfile(msd_path):
        s["ensemble_msd"] = pd.read_csv(msd_path)
    else:
        s["ensemble_msd"] = None

    # Diffusion summary (per-track D, alpha, motion_class)
    diff_path = os.path.join(data_dir, f"{stem}_diffusion_summary.csv")
    if os.path.isfile(diff_path):
        s["diffusion"] = pd.read_csv(diff_path)
    else:
        s["diffusion"] = None

    # Trajectories (for track length distribution)
    tr_path = os.path.join(data_dir, f"{stem}_trajectories.csv")
    if os.path.isfile(tr_path):
        s["tracks"] = pd.read_csv(tr_path)
    else:
        s["tracks"] = None

    # JDD
    jdd_path = os.path.join(data_dir, f"{stem}_jdd.json")
    if os.path.isfile(jdd_path):
        with open(jdd_path) as f:
            s["jdd"] = json.load(f)
    else:
        s["jdd"] = None

    # Dwell times
    dwell_path = os.path.join(data_dir, f"{stem}_dwell_times.csv")
    if os.path.isfile(dwell_path):
        s["dwell_times"] = pd.read_csv(dwell_path)
    else:
        s["dwell_times"] = None

    # Turning angles — signed degrees (-180..+180°)
    ta_path = os.path.join(data_dir, f"{stem}_turning_angles.csv")
    if os.path.isfile(ta_path):
        _ta_df = pd.read_csv(ta_path)
        s["turning_angles"]        = _ta_df["turning_angle_deg"].values
        s["turning_angles_signed"] = True
    else:
        s["turning_angles"]        = None
        s["turning_angles_signed"] = False

    return s


def save_palmtracer_csvs(out_dir, stem, locs, tracks, diff_df, imsd_df,
                         pixel_size_um, frame_interval_s,
                         width=None, height=None, n_frames=None,
                         mobile_D_threshold=None):
    """
    Emit PALM-Tracer-compatible CSV files alongside FIREFLY's native outputs.

    Files written (all comma-separated, written into `out_dir`):
        <stem>_locPALMTracer.csv              (one row per localisation)
        <stem>_trcPALMTracer.csv              (one row per trajectory plane)
        <stem>_trcPALMTracer-1-D.csv          (per-track D, MSD(0), MSE, LogD)
        <stem>_trcPALMTracer-1-MSD.csv        (per-track MSD curve, jagged)
        <stem>_trcPALMTracer-AllROI-D.csv     (per-track D summary)
        <stem>_trcPALMTracer-AllROI-MSD.csv   (per-track MSD curve, jagged)

    Column ordering, naming and unit conventions follow PALM-Tracer
    (Bordeaux Imaging Center).  ROI is hard-coded to 1 (FIREFLY does not
    sub-ROI tracks).  Fields FIREFLY does not measure (SigmaX/Y, Angle,
    MSE(Gauss), CentroidZ, MSE_Z, Pair_Distance) are filled with the
    PALM-Tracer "unused" sentinels (-1 or 0).
    """
    import csv as _csv
    import numpy as _np
    import pandas as _pd
    import os as _os

    if mobile_D_threshold is None:
        mobile_D_threshold = MOBILE_D_THRESHOLD_DEFAULT

    width    = int(width)    if width    is not None else 0
    height   = int(height)   if height   is not None else 0
    n_frames = int(n_frames) if n_frames is not None else int(
        max(locs["frame"].max() + 1, tracks["frame"].max() + 1))

    print(f"  PALM-Tracer: {len(locs):,} locs, {len(diff_df):,} tracks, "
          f"imsd_df shape {imsd_df.shape if imsd_df is not None else None}")

    # ── 1. locPALMTracer.csv ─────────────────────────────────────────────
    n_loc = len(locs)
    loc_path = _os.path.join(out_dir, f"{stem}_locPALMTracer.csv")
    with open(loc_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Points",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_loc,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["id", "Plane", "Index", "Channel",
                    "Integrated_Intensity",
                    "CentroidX(px)", "CentroidY(px)",
                    "SigmaX(px)", "SigmaY(px)", "Angle(rad)", "MSE(Gauss)",
                    "CentroidZ(um)", "MSE_Z(um)", "Pair_Distance(px)"])
        frames_l = locs["frame"].values
        xs       = locs["x"].values
        ys       = locs["y"].values
        mass     = (locs["mass"].values if "mass" in locs.columns
                    else _np.zeros(n_loc))
        for i in range(n_loc):
            w.writerow([i + 1, int(frames_l[i]) + 1, i + 1, -1,
                        float(mass[i]),
                        float(xs[i]), float(ys[i]),
                        0.0, 0.0, 0.0, 0.0,
                        -1.0, -1.0, 0.0])

    # ── 2. trcPALMTracer.csv ─────────────────────────────────────────────
    tr_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer.csv")
    # Re-number particles 1..n in PALM-Tracer style
    pid_order  = (diff_df["particle"].values if "particle" in diff_df.columns
                  else sorted(tracks["particle"].unique()))
    pid_to_new = {int(p): i + 1 for i, p in enumerate(pid_order)}
    n_tracks   = len(pid_to_new)

    with open(tr_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Tracks",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_tracks,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["Track", "Plane", "CentroidX(px)", "CentroidY(px)",
                    "CentroidZ(um)", "Integrated_Intensity", "id",
                    "Pair_Distance(px)"])
        # trackpy.link sets `frame` as the index AND keeps it as a column —
        # pandas refuses to disambiguate in sort_values, so drop the index first.
        tr_sorted = tracks.reset_index(drop=True).sort_values(["particle", "frame"])
        pids      = tr_sorted["particle"].values
        frames_t  = tr_sorted["frame"].values
        xs_t      = tr_sorted["x"].values
        ys_t      = tr_sorted["y"].values
        mass_t    = (tr_sorted["mass"].values if "mass" in tr_sorted.columns
                     else _np.zeros(len(tr_sorted)))
        for k in range(len(tr_sorted)):
            new_id = pid_to_new.get(int(pids[k]))
            if new_id is None:
                continue
            w.writerow([new_id, int(frames_t[k]) + 1,
                        float(xs_t[k]), float(ys_t[k]),
                        -1, float(mass_t[k]), k + 1, 0])

    print(f"  PALM-Tracer: wrote loc + trc; starting D files")

    # ── 3 & 5. D files ───────────────────────────────────────────────────
    D_arr     = diff_df["D"].values
    msd0_arr  = (diff_df["MSD0"].values if "MSD0" in diff_df.columns
                 else _np.zeros(len(diff_df)))
    mse_arr   = (diff_df["MSE"].values  if "MSE"  in diff_df.columns
                 else _np.zeros(len(diff_df)))
    logD_arr  = _np.where(D_arr > 0, _np.log10(_np.where(D_arr > 0, D_arr, 1)),
                          _np.nan)
    mobile_n  = int(_np.sum(D_arr > mobile_D_threshold))
    immob_n   = int(_np.sum(D_arr <= mobile_D_threshold))
    mob_ratio = (mobile_n / immob_n) if immob_n else _np.nan

    d1_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-D.csv")
    with open(d1_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE",
                    "LogD", "Mobile/Immobile", "Tracks"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            row = [1, new_id,
                   float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                   float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                   float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else "",
                   float(logD_arr[i]) if _np.isfinite(logD_arr[i]) else "",
                   "", ""]
            if i == 0:
                row[6] = mob_ratio if _np.isfinite(mob_ratio) else ""
                row[7] = n_tracks
            w.writerow(row)

    dA_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-D.csv")
    with open(dA_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            w.writerow([1, new_id,
                        float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                        float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                        float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else ""])

    print(f"  PALM-Tracer: wrote D files; starting MSD files")

    # ── 4 & 6. MSD files (jagged: one column per surviving lag) ──────────
    def _write_msd(path):
        with open(path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["#MSD(DeltaT) in um2"])
            w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                        f"{frame_interval_s}sec"])
            for pid in pid_order:
                if int(pid) not in imsd_df.columns and pid not in imsd_df.columns:
                    continue
                col = imsd_df[pid] if pid in imsd_df.columns else imsd_df[int(pid)]
                vals = col.values
                finite_idx = _np.where(_np.isfinite(vals))[0]
                if len(finite_idx) == 0:
                    continue
                last = finite_idx[-1] + 1
                row = [1, pid_to_new[int(pid)]]
                row.extend(float(v) if _np.isfinite(v) else ""
                           for v in vals[:last])
                w.writerow(row)

    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"))
    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"))
    print(f"  PALM-Tracer: all 6 files written successfully")

    return {
        "loc":           loc_path,
        "trc":           tr_path,
        "D_1":           d1_path,
        "D_AllROI":      dA_path,
        "MSD_1":         _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"),
        "MSD_AllROI":    _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"),
    }


def _msd_auc(emsd_df, frame_interval):
    """Trapezoidal AUC of the MSD curve in µm²·s units."""
    if emsd_df is None or len(emsd_df) == 0:
        return np.nan
    t = emsd_df["lag_frame"].values * frame_interval
    y = emsd_df["msd_um2"].values
    order = np.argsort(t)
    # NumPy 2.x renamed trapz → trapezoid
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(y[order], t[order]))


def _mob_immob_ratio(diff_df, d_threshold=MOBILE_D_THRESHOLD_DEFAULT):
    """Mobile / Immobile ratio defined by a diffusion-coefficient threshold.

    Tracks with D ≥ d_threshold count as Mobile; D < d_threshold count as
    Immobile.  Tracks with non-finite D (alpha fit failed) are excluded
    from BOTH numerator and denominator — they contribute neither mobility
    state, which avoids inflating either count.
    """
    if diff_df is None or "D" not in diff_df.columns:
        return np.nan
    d = diff_df["D"].values
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.nan
    d = d[valid]
    n_mob = int((d >= d_threshold).sum())
    n_imm = int((d <  d_threshold).sum())
    return float(n_mob / n_imm) if n_imm > 0 else np.nan


def _motion_fractions(diff_df):
    """Return dict of fractions per motion class."""
    if diff_df is None or "motion" not in diff_df.columns:
        return {}
    counts = diff_df["motion"].value_counts()
    total = counts.sum()
    if total == 0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def _track_lengths(tracks_df, frame_interval):
    """Return per-track lengths in seconds."""
    if tracks_df is None or "particle" not in tracks_df.columns:
        return np.array([])
    counts = tracks_df.groupby("particle").size().values
    return counts * frame_interval


def _stat_test(a, b):
    """Two-sample test on per-experiment scalars.  Welch's t by default,
    Mann-Whitney as fallback for non-normal data.  Returns (p, label)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, "")
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        normal = True
        for arr in (a, b):
            if 3 <= len(arr) <= 5000:
                try:
                    if shapiro(arr).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass
        if normal:
            p = ttest_ind(a, b, equal_var=False).pvalue
        else:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        if not np.isfinite(p):
            return (np.nan, "")
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        else: stars = "ns"
        return (float(p), stars)
    except Exception:
        return (np.nan, "")


def _theme_palette(theme):
    """Return a dict with figure colours matching the Analyse-mode themes
    ('Dark', 'Light', 'Publication')."""
    t = (theme or "Dark")
    # Accept various spellings
    t = {"dark": "Dark", "light": "Light", "publication": "Publication"}.get(t.lower(), t)
    if t == "Light":
        return dict(theme="Light",
                    BG="#ffffff", PNL="#f6f8fa",
                    TXT="#24292f", GRD="#d0d7de",
                    BAR_FILL="#ffffff", SIG="#000000",
                    FONT="sans-serif")
    if t == "Publication":
        return dict(theme="Publication",
                    BG="#ffffff", PNL="#ffffff",
                    TXT="#000000", GRD="#cccccc",
                    BAR_FILL="#ffffff", SIG="#000000",
                    FONT="serif")
    # Default: Dark
    return dict(theme="Dark",
                BG="#0d1117", PNL="#161b22",
                TXT="#e6edf3", GRD="#30363d",
                BAR_FILL="#161b22", SIG="#e6edf3",
                FONT="monospace")


def _stat_test_n(arrays, labels):
    """Statistical test across N≥2 groups.

    Returns
    -------
    omnibus : dict with keys {"test", "p", "stars"} or None if n<2 each
    pairwise : list of dicts with keys
        {"i", "j", "label_i", "label_j", "test", "p", "stars",
         "n_i", "n_j", "mean_i", "mean_j", "sem_i", "sem_j"}
    """
    arrs = [np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
            for a in arrays]
    valid_idx = [i for i, a in enumerate(arrs) if len(a) >= 2]

    omnibus = None
    pairwise = []

    def _star(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    if len(valid_idx) < 2:
        # Still record per-pair "ns" rows for stats CSV completeness
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": "n<2", "p": np.nan, "stars": "",
                    "n_i": int(len(arrs[i])), "n_j": int(len(arrs[j])),
                    "mean_i": float(arrs[i].mean()) if len(arrs[i]) else np.nan,
                    "mean_j": float(arrs[j].mean()) if len(arrs[j]) else np.nan,
                    "sem_i": (float(arrs[i].std(ddof=1) / np.sqrt(len(arrs[i])))
                              if len(arrs[i]) > 1 else np.nan),
                    "sem_j": (float(arrs[j].std(ddof=1) / np.sqrt(len(arrs[j])))
                              if len(arrs[j]) > 1 else np.nan),
                })
        return omnibus, pairwise

    # Omnibus test
    try:
        from scipy.stats import f_oneway, kruskal, shapiro
        valid_arrs = [arrs[i] for i in valid_idx]

        normal = True
        for a in valid_arrs:
            if 3 <= len(a) <= 5000:
                try:
                    if shapiro(a).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass

        if len(valid_arrs) == 2:
            from scipy.stats import ttest_ind, mannwhitneyu
            if normal:
                p = ttest_ind(*valid_arrs, equal_var=False).pvalue
                test_name = "Welch's t-test"
            else:
                p = mannwhitneyu(*valid_arrs, alternative="two-sided").pvalue
                test_name = "Mann-Whitney U"
        else:
            if normal:
                p = f_oneway(*valid_arrs).pvalue
                test_name = "One-way ANOVA"
            else:
                p = kruskal(*valid_arrs).pvalue
                test_name = "Kruskal-Wallis"
        if np.isfinite(p):
            omnibus = {"test": test_name, "p": float(p), "stars": _star(p)}
    except Exception:
        pass

    # Pairwise comparisons
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                a, b = arrs[i], arrs[j]
                if len(a) < 2 or len(b) < 2:
                    p = np.nan
                    test_name = "n<2"
                else:
                    is_normal = True
                    for arr in (a, b):
                        if 3 <= len(arr) <= 5000:
                            try:
                                if shapiro(arr).pvalue < 0.05:
                                    is_normal = False
                                    break
                            except Exception:
                                pass
                    if is_normal:
                        p = ttest_ind(a, b, equal_var=False).pvalue
                        test_name = "Welch's t-test"
                    else:
                        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                        test_name = "Mann-Whitney U"
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": test_name,
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "stars": _star(p) if np.isfinite(p) else "",
                    "n_i": int(len(a)), "n_j": int(len(b)),
                    "mean_i": float(a.mean()) if len(a) else np.nan,
                    "mean_j": float(b.mean()) if len(b) else np.nan,
                    "sem_i": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan,
                    "sem_j": float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else np.nan,
                })
    except Exception:
        pass

    return omnibus, pairwise


def _bar_with_dots_n(ax, data_per_group, labels, colors, palette,
                     ylabel="", record_stats=None, metric_name=""):
    """Bar chart with mean ± SEM and individual replicate dots, generalised
    to N groups.

    For 2 groups: shows pairwise stars on a bracket (matches lab style).
    For 3+ groups: shows omnibus ANOVA / Kruskal p-value as a panel
    annotation; full pairwise comparisons go to record_stats[metric_name]."""
    fill = palette["BAR_FILL"]
    sig_col = palette["SIG"]

    arrs = [np.asarray(d, dtype=float) for d in data_per_group]
    arrs = [a[np.isfinite(a)] for a in arrs]
    n = len(arrs)
    means = [float(a.mean()) if len(a) else 0.0 for a in arrs]
    sems  = [float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
             for a in arrs]
    x = np.arange(n)
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[fill] * n,
           edgecolor=colors, linewidth=1.5,
           ecolor=sig_col)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arrs):
        if len(a):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a,
                       color=colors[i], s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15 if n > 3 else 0)
    ax.set_ylabel(ylabel)

    # Stats
    omnibus, pairwise = _stat_test_n(arrs, labels)
    if record_stats is not None and metric_name:
        record_stats[metric_name] = {"omnibus": omnibus, "pairwise": pairwise}

    # Annotation
    top_data = max([a.max() if len(a) else 0 for a in arrs] + [max(means) * 1.2 if max(means) > 0 else 1])
    if n == 2 and pairwise:
        pair = pairwise[0]
        if pair["stars"] and np.isfinite(pair["p"]):
            top = top_data * 1.05
            ax.plot([0, 0, 1, 1], [top, top * 1.03, top * 1.03, top],
                    color=sig_col, lw=0.8)
            # Numeric p plus stars, e.g. "p = 0.003  **"
            p_str = (f"p = {pair['p']:.2e}" if pair['p'] < 0.001
                     else f"p = {pair['p']:.3f}")
            label = f"{p_str}  {pair['stars']}"
            ax.text(0.5, top * 1.05, label, ha="center", va="bottom",
                    fontsize=9, color=sig_col)
            # Make room above the bracket for the longer label
            ax.set_ylim(0, top * 1.30)
    elif n > 2 and omnibus:
        # Show test name + omnibus p + stars in the upper-left corner.
        # Numeric format adapts to magnitude: scientific < 0.001, fixed otherwise.
        p_val = omnibus['p']
        p_str = (f"p = {p_val:.2e}" if p_val < 0.001
                 else f"p = {p_val:.3f}")
        text = f"{omnibus['test']}\n{p_str}   {omnibus['stars']}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=sig_col,
                bbox=dict(facecolor=palette["PNL"], edgecolor="none",
                          alpha=0.7, pad=3))


def compare_groups(groups,
                   output_dir=None, output_stem="comparison",
                   panels=None, theme="Dark",
                   pdf_report=True,
                   mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   progress_cb=None):
    """Compare N≥2 groups of analysis output folders and render a multi-panel
    figure, summary CSV, statistics CSV and combined PDF report.

    Parameters
    ----------
    groups : list[dict]
        [{"folders": [path, ...], "label": "Pre", "color": "#000000"}, ...]
    output_dir : str or None
        Where to save the figure / CSVs / PDF report.  If None, nothing is
        saved to disk and only the figure is returned.
    panels : set[str] or None
        Subset of panels to render.  Default: all of {"msd", "auc",
        "logd_dist", "mob_immob", "motion_classes", "track_length",
        "jdd", "dwell_cdf", "turning_angles"}.
    theme : str
        Figure theme — "Dark" (default), "Light" or "Publication".
    pdf_report : bool
        If True (default) and output_dir is given, also write a multi-page
        PDF report bundling the figure, parameters, folder lists and stats.
    progress_cb : callable or None
        Optional callback(done:int, total:int, msg:str) for UI progress.

    Returns
    -------
    fig         : matplotlib.figure.Figure
    summary_df  : pandas.DataFrame  — per-replicate scalar metrics
    stats       : dict[str, dict]   — per-metric omnibus + pairwise tests
    """
    import matplotlib.pyplot as plt

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups; got {len(groups)}")

    if panels is None:
        panels = {"msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                  "track_length", "jdd", "dwell_cdf", "turning_angles",
                  "radial_dist"}

    n_groups = len(groups)
    labels   = [g.get("label", f"Group {i+1}") for i, g in enumerate(groups)]
    colors   = [g.get("color", "#3b6ed8")     for g in groups]
    folder_lists = [list(g["folders"]) for g in groups]

    # ── Load summaries for all groups ─────────────────────────────────────────
    all_summaries = [[] for _ in groups]
    total = sum(len(f) for f in folder_lists)
    done = 0
    for gi, folders in enumerate(folder_lists):
        for f in folders:
            if progress_cb:
                progress_cb(done, total, f"Loading: {os.path.basename(f)}")
            try:
                all_summaries[gi].append(load_summary_from_folder(f))
            except Exception as e:
                print(f"  Skipping {f}: {e}")
            done += 1

    empty_groups = [labels[i] for i, ss in enumerate(all_summaries) if len(ss) == 0]
    if empty_groups:
        raise RuntimeError(
            "Need at least one valid folder per group; these are empty: "
            + ", ".join(empty_groups))

    if progress_cb:
        progress_cb(total, total, "Computing scalars and rendering...")

    # ── Compute per-folder scalars (one row per replicate) ────────────────────
    summary_rows = []
    def _row(group_label, summary):
        p = summary["params"]
        fi = float(p.get("frame_interval_s", 0.05))
        d = summary["diffusion"]
        return {
            "group":            group_label,
            "folder":           summary["folder"],
            "stem":             summary["stem"],
            "n_tracks":         len(d) if d is not None else 0,
            "auc_msd":          _msd_auc(summary["ensemble_msd"], fi),
            "mob_immob_ratio":  _mob_immob_ratio(d, mobile_d_threshold),
            "median_D":         float(d["D"].median()) if d is not None and "D" in d.columns else np.nan,
            "median_alpha":     float(d["alpha"].median()) if d is not None and "alpha" in d.columns else np.nan,
            "mean_track_length_s": float(_track_lengths(summary["tracks"], fi).mean())
                                   if summary["tracks"] is not None else np.nan,
        }
    for gi, summaries in enumerate(all_summaries):
        for s in summaries:
            summary_rows.append(_row(labels[gi], s))
    summary_df = pd.DataFrame(summary_rows)

    # Per-metric statistics dict — populated as panels render
    stats_records = {}

    # ── Render the figure ────────────────────────────────────────────────────
    panel_order = ["msd", "auc", "logd_dist", "mob_immob",
                   "motion_classes", "track_length",
                   "jdd", "dwell_cdf", "turning_angles", "radial_dist"]
    enabled = [p for p in panel_order if p in panels]
    n_plots = len(enabled)
    if n_plots == 0:
        raise RuntimeError("No panels enabled")
    print(f"  Compare: rendering {n_plots} panel(s): {enabled}")
    if "radial_dist" not in panels:
        print(f"  Compare: 'radial_dist' NOT in requested panels — "
              f"check the 'Radial distribution (polar)' tickbox in the "
              f"Compare tab to include it.")
    ncols = 3 if n_plots > 4 else 2
    nrows = (n_plots + ncols - 1) // ncols

    pal = _theme_palette(theme)
    plt.rcParams.update({
        "text.color":      pal["TXT"], "axes.labelcolor": pal["TXT"],
        "xtick.color":     pal["TXT"], "ytick.color":     pal["TXT"],
        "axes.titlecolor": pal["TXT"],
        "axes.edgecolor":  pal["GRD"], "axes.facecolor":  pal["PNL"],
        "figure.facecolor": pal["BG"], "figure.edgecolor": pal["BG"],
        "savefig.facecolor": pal["BG"], "savefig.edgecolor": pal["BG"],
        "grid.color":      pal["GRD"], "grid.alpha": 0.4,
        "font.family":     pal["FONT"],
        "legend.facecolor": pal["PNL"], "legend.edgecolor": pal["GRD"],
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6),
                             facecolor=pal["BG"])
    axes = np.array(axes).reshape(-1)
    for ax in axes[n_plots:]:
        ax.axis("off")

    panel_idx = 0
    def _next_ax():
        nonlocal panel_idx
        ax = axes[panel_idx]; panel_idx += 1
        return ax

    def _zip_groups():
        """Iterator: (label, summaries, color) for each group."""
        for i in range(n_groups):
            yield labels[i], all_summaries[i], colors[i]

    # ── 1. MSD overlay ────────────────────────────────────────────────────────
    if "msd" in panels:
        ax = _next_ax()
        for grp_label, summaries, color in _zip_groups():
            curves = []
            tref = None
            for s in summaries:
                e = s["ensemble_msd"]
                if e is None: continue
                fi = float(s["params"].get("frame_interval_s", 0.05))
                t = e["lag_frame"].values * fi
                y = e["msd_um2"].values
                order = np.argsort(t)
                t, y = t[order], y[order]
                if tref is None:
                    tref = t
                if len(t) != len(tref) or not np.allclose(t, tref):
                    y = np.interp(tref, t, y)
                curves.append(y)
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else None
            ax.plot(tref, mean, "-o", color=color, label=grp_label, ms=4, lw=1.5)
            if sem is not None:
                ax.fill_between(tref, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.set_xlabel("Time delta (s)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title("Mean Square Displacement")
        ax.legend(frameon=False, loc="best")

    # ── 2. AUC bar chart ──────────────────────────────────────────────────────
    if "auc" in panels:
        ax = _next_ax()
        data = [summary_df.loc[summary_df["group"] == lbl, "auc_msd"].values
                for lbl in labels]
        _bar_with_dots_n(ax, data, labels, colors, pal,
                         ylabel="AUC (µm²·s)",
                         record_stats=stats_records, metric_name="auc_msd")
        ax.set_title("Area Under the Curve")

    # ── 3. LogD frequency distribution ────────────────────────────────────────
    if "logd_dist" in panels:
        ax = _next_ax()
        bins = np.linspace(-5, 1, 31)
        for grp_label, summaries, color in _zip_groups():
            all_logD = []
            for s in summaries:
                d = s["diffusion"]
                if d is None or "D" not in d.columns: continue
                vals = d["D"].values
                vals = vals[vals > 0]
                if len(vals): all_logD.append(np.log10(vals))
            if not all_logD: continue
            pooled = np.concatenate(all_logD)
            counts, edges = np.histogram(pooled, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, label=grp_label, ms=4, lw=1.2)
        ax.axvline(np.log10(mobile_d_threshold), color=pal["GRD"], ls="--", lw=0.8,
                   label=f"D = {mobile_d_threshold} µm²/s")
        ax.set_xlabel("log₁₀ D  (µm²/s)")
        ax.set_ylabel("Relative frequency")
        ax.set_title("LogD Frequency Distribution")
        ax.legend(frameon=False, loc="best")

    # ── 4. Mobile/Immobile ratio bar ──────────────────────────────────────────
    if "mob_immob" in panels:
        ax = _next_ax()
        data = [summary_df.loc[summary_df["group"] == lbl, "mob_immob_ratio"].values
                for lbl in labels]
        _bar_with_dots_n(ax, data, labels, colors, pal,
                         ylabel="Mobile/Immobile ratio",
                         record_stats=stats_records, metric_name="mob_immob_ratio")
        ax.set_title("Mobile/Immobile Ratio")

    # ── 5. Motion class fractions (grouped bars, N groups) ────────────────────
    if "motion_classes" in panels:
        ax = _next_ax()
        classes = ["Immobile", "Confined", "Brownian", "Directed"]
        def _fracs(summaries):
            rows = []
            for s in summaries:
                f = _motion_fractions(s["diffusion"])
                rows.append([f.get(c, 0.0) for c in classes])
            return np.array(rows) if rows else np.zeros((0, len(classes)))
        per_group = [_fracs(ss) for ss in all_summaries]
        x = np.arange(len(classes))
        # Group-bar width: total slot ~0.8, divided across N groups
        slot = 0.8
        w = slot / n_groups
        rng = np.random.default_rng(1)
        for gi, (grp_label, color, fracs) in enumerate(zip(labels, colors, per_group)):
            if not len(fracs): continue
            x_off = (gi - (n_groups - 1) / 2) * w
            ax.bar(x + x_off, fracs.mean(axis=0), w * 0.9,
                   yerr=fracs.std(axis=0, ddof=1)/np.sqrt(len(fracs)) if len(fracs) > 1 else None,
                   color=pal["BAR_FILL"], edgecolor=color, linewidth=1.5,
                   ecolor=pal["SIG"], capsize=3, label=grp_label)
            for ci in range(len(classes)):
                ax.scatter(np.full(len(fracs), x[ci] + x_off)
                           + rng.uniform(-w*0.25, w*0.25, len(fracs)),
                           fracs[:, ci], color=color, s=12, zorder=3)
        # Per-class stats
        for ci, cname in enumerate(classes):
            arrs = [fracs[:, ci] if len(fracs) else np.array([]) for fracs in per_group]
            omn, pw = _stat_test_n(arrs, labels)
            stats_records[f"motion_frac_{cname}"] = {"omnibus": omn, "pairwise": pw}
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
        ax.set_ylabel("Fraction of tracks")
        ax.set_title("Motion Class Fractions")
        ax.legend(frameon=False, loc="best", fontsize=8)

    # ── 6. Track length distribution (CDF, x clipped at 99th %ile) ────────────
    if "track_length" in panels:
        ax = _next_ax()
        pooled_per_group = {}
        for grp_label, summaries, _ in _zip_groups():
            arrs = []
            for s in summaries:
                fi = float(s["params"].get("frame_interval_s", 0.05))
                tl = _track_lengths(s["tracks"], fi)
                if len(tl):
                    arrs.append(tl)
            if arrs:
                pooled_per_group[grp_label] = np.concatenate(arrs)
        combined = (np.concatenate(list(pooled_per_group.values()))
                    if pooled_per_group else np.array([]))
        x_clip = float(np.percentile(combined, 99)) if len(combined) else None
        for grp_label, color in zip(labels, colors):
            p = pooled_per_group.get(grp_label)
            if p is None or len(p) == 0: continue
            x_sorted = np.sort(p)
            y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(x_sorted, y, color=color, lw=1.5, label=grp_label)
        if pooled_per_group:
            if x_clip and x_clip > 0:
                ax.set_xlim(0, x_clip)
                ax.set_title("Track Length Distribution  (x clipped at 99th %ile)")
            else:
                ax.set_title("Track Length Distribution")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Track length (s)")
            ax.set_ylabel("Cumulative fraction")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No track-length data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Track Length Distribution")
        # Stats: mean track length (per-replicate)
        arrs = [summary_df.loc[summary_df["group"] == lbl, "mean_track_length_s"].values
                for lbl in labels]
        omn, pw = _stat_test_n(arrs, labels)
        stats_records["mean_track_length_s"] = {"omnibus": omn, "pairwise": pw}

    # ── 7. JDD: per-population D + fraction (N groups) ────────────────────────
    if "jdd" in panels:
        ax = _next_ax()
        any_data = False
        max_pop_overall = 0
        # Spread groups across ±0.18 around each population index
        if n_groups > 1:
            offsets = np.linspace(-0.18, 0.18, n_groups)
        else:
            offsets = np.array([0.0])
        for gi, (grp_label, summaries, color) in enumerate(_zip_groups()):
            label_done = False
            for s in summaries:
                jd = s.get("jdd")
                if not jd or "D_values" not in jd: continue
                D = np.asarray(jd["D_values"], dtype=float)
                f = np.asarray(jd.get("fractions", np.ones_like(D)), dtype=float)
                if D.size == 0: continue
                any_data = True
                max_pop_overall = max(max_pop_overall, len(D))
                sizes = 25 + 175 * np.clip(f, 0, 1)
                xs = np.arange(len(D)) + offsets[gi]
                ax.scatter(xs, D, s=sizes, color=color,
                           alpha=0.55, edgecolor=color,
                           label=(grp_label if not label_done else None))
                label_done = True
        if any_data:
            tick_labels = ["Immobile", "Mobile", "Fast"][:max_pop_overall]
            if max_pop_overall == 1: tick_labels = ["All"]
            ax.set_xticks(np.arange(max_pop_overall))
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(-0.5, max_pop_overall - 0.5)
            ax.set_ylabel("D (µm²/s, log)")
            ax.set_yscale("log")
            ax.set_title("JDD: per-population D  (marker size ∝ population fraction)")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No JDD data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Jump Distance Distribution")

    # ── 8. Dwell time CDF (N groups) ──────────────────────────────────────────
    if "dwell_cdf" in panels:
        ax = _next_ax()
        any_data = False
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                d = s.get("dwell_times")
                if d is None or len(d) == 0: continue
                col = next((c for c in ("dwell_time_s", "dwell_s",
                                        "dwell_time", "dwell", "tau_s")
                            if c in d.columns), None)
                if col is None: continue
                pooled.extend(d[col].values)
            if not pooled: continue
            any_data = True
            arr = np.sort(np.asarray(pooled, dtype=float))
            arr = arr[arr > 0]
            if len(arr) == 0: continue
            y = 1 - np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, y, color=color, lw=1.5, label=grp_label)
        if any_data:
            ax.set_xlabel("Dwell time (s)")
            ax.set_ylabel("Survival fraction")
            ax.set_title("Dwell Time Survival")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No dwell-time data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Dwell Time Survival")

    # ── 9. Turning angle distribution (N groups, unsigned |angle|) ────────────
    # Single line per group, plotting the count of each |θ| bin on
    # the same 0°–180° x-axis.  Sign / rotational direction is handled
    # separately by the Radial Distribution panel.
    if "turning_angles" in panels:
        ax = _next_ax()
        any_data = False
        bins = np.linspace(0, 180, 37)                 # 5° bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        pooled_per_group = []
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.abs(np.asarray(ta).ravel()))
            pooled_per_group.append((grp_label, color, pooled))
        for grp_label, color, pooled in pooled_per_group:
            if not pooled: continue
            any_data = True
            counts, _ = np.histogram(pooled, bins=bins)
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, lw=1.5, ms=3, label=grp_label)
        if any_data:
            ax.set_xlabel("|Turning angle|  (°)")
            ax.set_ylabel("Relative frequency")
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_title("Turning Angle Distribution")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No turning-angle data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Turning Angle Distribution")

    # ── 10. Radial distribution (polar, signed turning angles) ────────────────
    # Polar histogram showing the angular distribution of step-to-step
    # turning angles.  Each group is plotted as a separate set of bars
    # offset around each bin centre.
    #
    # Implementation note: we replace the auto-created cartesian axis with
    # a polar one at the SAME SubplotSpec (not via fig.add_axes with raw
    # bounds), so that the polar axis remains a managed gridspec member.
    # If we used add_axes(bounds), tight_layout would later reposition the
    # other (gridspec-managed) subplots but leave the polar in its original
    # location, causing visible overlap.
    if "radial_dist" in panels:
        old_ax = axes[panel_idx]
        ss = old_ax.get_subplotspec()
        old_ax.remove()
        ax = fig.add_subplot(ss, projection="polar")
        axes[panel_idx] = ax
        panel_idx += 1

        any_data = False
        n_bins = 36
        # matplotlib polar bar() only renders correctly when theta ∈ [0, 2π);
        # shift the data accordingly.  The xticks are placed at positive-only
        # angles but labelled with their signed equivalents.
        bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_width   = (bin_edges[1] - bin_edges[0]) * 0.95

        # First pass: get raw counts per group per bin.
        counts_per_group = []     # list of (group_idx, counts_array)
        for gi in range(n_groups):
            pooled = []
            for s in all_summaries[gi]:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.asarray(ta).ravel())
            if not pooled:
                counts_per_group.append((gi, np.zeros(n_bins)))
                continue
            arr = np.asarray(pooled, dtype=float)
            if not np.any(arr < -1e-3):
                arr = np.concatenate([arr, -arr])
            angles_rad = np.mod(np.deg2rad(arr), 2 * np.pi)
            counts, _ = np.histogram(angles_rad, bins=bin_edges)
            counts_per_group.append((gi, counts.astype(float)))
            if counts.sum() > 0:
                any_data = True

        if any_data:
            # ── Normalise each group to ITS OWN total ─────────────────────
            # Otherwise a group with more total angles automatically draws
            # bigger bars everywhere — a sample-size artefact, not a real
            # shape difference.  After dividing by the per-group total, each
            # group's values sum to 1.0 across the full circle, so the bars
            # compare distribution SHAPE.
            # Bars from different groups are offset around each bin centre
            # for easy side-by-side comparison.
            per_bar_width = bar_width / max(1, n_groups) * 0.95
            for gi, counts in counts_per_group:
                total = counts.sum()
                if total <= 0:
                    continue
                normalised = counts / total
                offset = (gi - (n_groups - 1) / 2) * per_bar_width
                ax.bar(bin_centres + offset, normalised,
                       width=per_bar_width, bottom=0.0,
                       color=colors[gi], alpha=0.85,
                       edgecolor=pal["GRD"], linewidth=0.3,
                       label=labels[gi])

        if any_data:
            # Conventional orientation: 0° at top (straight ahead),
            # right hemisphere = positive turns, left hemisphere = negative.
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            # Positive-only xticks; labelled with signed equivalents.
            ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                                "−135°", "−90°", "−45°"], fontsize=7)
            # Hide the radial-axis numeric labels — bar length is
            # interpreted comparatively, not in absolute density units.
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", left=False)
            ax.set_title("Radial Distribution  (each group normalised to "
                         "its own total)", pad=14, fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
                      frameon=False, fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "No turning-angle data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Radial Distribution")

    # ── Suptitle: Group A (n=…) vs Group B (n=…) [vs Group C …] ───────────────
    parts = [f"{labels[i]}  (n={len(all_summaries[i])})" for i in range(n_groups)]
    fig.suptitle("   vs   ".join(parts),
                 fontsize=12, fontweight="bold", color=pal["TXT"])
    for ax in axes[:n_plots]:
        ax.set_facecolor(pal["PNL"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["GRD"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Build statistics dataframe (per metric × pairwise) ────────────────────
    # Bonferroni correction across pairwise comparisons WITHIN each metric:
    # multiplies the raw p-value by the number of pairs (capped at 1.0).
    # The omnibus row gets the raw p-value only — it's a single test.
    stats_rows = []
    for metric, rec in stats_records.items():
        omn = rec.get("omnibus")
        if omn:
            stars = omn["stars"]
            stars_bonf = stars  # omnibus needs no correction
            stats_rows.append({
                "metric": metric, "comparison": "omnibus",
                "test": omn["test"],
                "p_value": omn["p"], "stars": stars,
                "p_value_bonferroni": omn["p"], "stars_bonferroni": stars_bonf,
                "n_a": "", "n_b": "", "mean_a": "", "mean_b": "",
                "sem_a": "", "sem_b": "", "label_a": "all groups", "label_b": "",
            })
        pairs = rec.get("pairwise", [])
        n_pairs = max(1, len(pairs))
        for pw in pairs:
            p = pw["p"]
            if np.isfinite(p):
                p_bonf = min(1.0, p * n_pairs)
                if   p_bonf < 0.001: stars_bonf = "***"
                elif p_bonf < 0.01:  stars_bonf = "**"
                elif p_bonf < 0.05:  stars_bonf = "*"
                else:                stars_bonf = "ns"
            else:
                p_bonf = np.nan
                stars_bonf = ""
            stats_rows.append({
                "metric": metric, "comparison": f"{pw['label_i']} vs {pw['label_j']}",
                "test": pw["test"],
                "p_value": pw["p"], "stars": pw["stars"],
                "p_value_bonferroni": p_bonf, "stars_bonferroni": stars_bonf,
                "n_a": pw["n_i"], "n_b": pw["n_j"],
                "mean_a": pw["mean_i"], "mean_b": pw["mean_j"],
                "sem_a": pw["sem_i"], "sem_b": pw["sem_j"],
                "label_a": pw["label_i"], "label_b": pw["label_j"],
            })
    stats_df = pd.DataFrame(stats_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path  = os.path.join(output_dir, f"{output_stem}.png")
        pdf_path  = os.path.join(output_dir, f"{output_stem}.pdf")
        csv_path  = os.path.join(output_dir, f"{output_stem}_summary.csv")
        stats_csv = os.path.join(output_dir, f"{output_stem}_stats.csv")
        fig.savefig(png_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        summary_df.to_csv(csv_path, index=False)
        if len(stats_df):
            stats_df.to_csv(stats_csv, index=False)
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {csv_path}")
        if len(stats_df):
            print(f"  Saved: {stats_csv}")

        # ── Combined PDF report (figure + parameters + folders + stats) ──────
        if pdf_report:
            report_path = os.path.join(output_dir, f"{output_stem}_report.pdf")
            try:
                _write_pdf_report(report_path, fig, groups, all_summaries,
                                  labels, colors, summary_df, stats_df,
                                  panels=panels, theme=theme, palette=pal)
                print(f"  Saved: {report_path}")
            except Exception as exc:
                print(f"  PDF report skipped ({type(exc).__name__}: {exc})")

    return fig, summary_df, stats_records


def _write_pdf_report(path, fig, groups, all_summaries, labels, colors,
                      summary_df, stats_df, panels, theme, palette):
    """Multi-page PDF: cover + figure, parameters & folders, statistics."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pal = palette
    with PdfPages(path) as pdf:
        # ── Page 1: the comparison figure itself ──────────────────────────────
        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")

        # ── Page 2: cover / parameters ────────────────────────────────────────
        page2 = plt.figure(figsize=(8.5, 11), facecolor=pal["BG"])
        page2.text(0.5, 0.96, "sptPALM Comparison Report",
                   ha="center", fontsize=18, fontweight="bold", color=pal["TXT"])

        meta_lines = [
            f"Theme:              {theme}",
            f"Panels rendered:    {', '.join(sorted(panels))}",
            f"Number of groups:   {len(groups)}",
            "",
            "Groups:",
        ]
        for i, g in enumerate(groups):
            meta_lines.append(
                f"  • {labels[i]}   "
                f"(n={len(all_summaries[i])} folder(s), "
                f"colour {colors[i]})")
        meta_lines.append("")
        meta_lines.append("Folders:")
        for i in range(len(groups)):
            meta_lines.append(f"  [{labels[i]}]")
            for f in groups[i]["folders"]:
                meta_lines.append(f"    {f}")
            meta_lines.append("")

        page2.text(0.06, 0.92, "\n".join(meta_lines),
                   ha="left", va="top", fontsize=9, family="monospace",
                   color=pal["TXT"])
        pdf.savefig(page2, facecolor=pal["BG"], bbox_inches="tight")
        plt.close(page2)

        # ── Page 3: per-replicate scalar summary table ────────────────────────
        if len(summary_df):
            page3 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page3.text(0.5, 0.96, "Per-replicate scalar metrics",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page3.add_axes([0.04, 0.04, 0.92, 0.86])
            ax.axis("off")
            disp = summary_df.copy()
            for c in disp.select_dtypes(include="float").columns:
                disp[c] = disp[c].apply(
                    lambda x: f"{x:.4g}" if np.isfinite(x) else "")
            disp["folder"] = disp["folder"].apply(
                lambda p: "..." + p[-40:] if isinstance(p, str) and len(p) > 43 else p)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page3, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page3)

        # ── Page 4: statistical tests ─────────────────────────────────────────
        if len(stats_df):
            page4 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page4.text(0.5, 0.96, "Statistical tests",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page4.add_axes([0.03, 0.04, 0.94, 0.86])
            ax.axis("off")
            disp = stats_df.copy()
            for c in ("p_value", "mean_a", "mean_b", "sem_a", "sem_b"):
                if c in disp.columns:
                    disp[c] = disp[c].apply(
                        lambda x: f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else x)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page4, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page4)


if __name__ == "__main__":
    main()
