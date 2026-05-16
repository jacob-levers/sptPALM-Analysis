"""
FIREFLY — Fluorescence Inference & Reconstruction Engine.

PySide6 / Qt frontend.  This is the v2.0 GUI in active development under
workstream B.  Until B is feature-complete, `app_tk.py` remains the shipping
default.

NOTE on MPS environment variables (set below before any imports)
---------------------------------------------------------------
PyTorch's MPS allocator on macOS 26 / Apple M-series can leak memory across
operations even with explicit synchronize() + empty_cache() between stages.
The official mitigation is to disable the high-watermark allocator check
(PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0) and enable graceful CPU fallback for
unimplemented ops (PYTORCH_ENABLE_MPS_FALLBACK=1).  These MUST be set before
torch is imported anywhere in the process — putting them at the very top of
the entry-point module is the only reliable way.

Phase B1.0 scope (this file as of v1.x.x-dev):
    • QMainWindow with persistent left sidebar + central tab widget
    • One "Run Analysis" tab: file picker, parameters, Start/Stop, log,
      progress bar, matplotlib QtAgg canvas for the result figure
    • QThread-based worker with signals (replaces Tk's queue + after() loop)
    • Crash reporter wired in (sys.excepthook, threading.excepthook, Qt's
      qInstallMessageHandler, and a hook on the analysis worker's error path)

Later phases:
    B1.1 — Full parameters panel (everything from Tk's sidebar)
    B1.2 — Batch tab
    B1.3 — Compare tab with Qt drag-and-drop
    B2   — Napari Workspace tab (frame scrubbing + track overlay)
    B3   — Delete app_tk.py, strip Tcl/Tk from PyInstaller spec → v2.0.0
"""
from __future__ import annotations

import os
import sys

# ── MPS allocator tuning (must be set BEFORE torch import anywhere) ───────────
# Disable the high-watermark check so MPS aggressively reuses memory instead
# of holding committed blocks across ops.  Enable CPU fallback so missing
# MPS kernels don't kill the process.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import multiprocessing
import queue
import time
import traceback
from typing import Any

# macOS + multiprocessing: spawn is the only safe context for PyInstaller
# frozen apps, and it also gives the analysis subprocess a clean Python
# interpreter (no Qt, no Metal claim) — the whole point of running the
# heavy GPU work in a child process to avoid contention with Qt's window
# compositor on M-series Macs.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer

# Matplotlib Qt embedding
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

import crash_reporter


N_CPUS = multiprocessing.cpu_count()

# Worker target lives in a *Qt-free* module so the spawned analysis
# subprocess doesn't accidentally re-import PySide6 (which would defeat
# the whole point of subprocess isolation on macOS Metal — see the
# firefly_worker.py module docstring for the full rationale).
import firefly_worker
_run_analysis_in_subprocess = firefly_worker.run_analysis
_run_batch_in_subprocess    = firefly_worker.run_batch_analysis


# ══════════════════════════════════════════════════════════════════════════════
#  SUBPROCESS WORKER  (defined in firefly_worker.py — DO NOT REDEFINE HERE)
# ══════════════════════════════════════════════════════════════════════════════
# The reference is bound at the top of this module (above) to
# `firefly_worker.run_analysis`.  The worker function MUST live in a
# Qt-free module — multiprocessing.spawn re-imports the module containing
# the target function in the child process, so defining it in app_qt.py
# would pull PySide6 into the subprocess, defeating the whole point of
# subprocess isolation on Apple Silicon (see the firefly_worker.py module
# docstring for the chain of causation).
#
# Old code that used to live below this comment has been removed.  If you
# need to inspect or modify the worker, see firefly_worker.py.

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):
    """Top-level FIREFLY window: persistent left sidebar + central QTabWidget.

    The sidebar holds analysis parameters and the Start/Stop button so they
    remain visible regardless of which tab is active (per the architecture
    spec).  The central tab widget hosts the "Run Analysis" view in B1.0 and
    will gain Batch / Compare / Workspace tabs in later phases.
    """

    # Bumped manually when a stored-setting layout changes incompatibly
    SETTINGS_VERSION = 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIREFLY — Fluorescence Inference & "
                            "Reconstruction Engine")
        self.resize(1280, 820)

        # QSettings stores per-user preferences in the OS-native location
        # (~/Library/Preferences on macOS, registry on Windows).  Keyed by
        # the org/app names set in main(); no extra setup needed.
        self._settings = QtCore.QSettings("jacoblevers", "FIREFLY")

        # Subprocess + queue + cancellation event; populated when Start
        # is clicked.  See _on_run_clicked.
        self._proc:         multiprocessing.Process | None = None
        self._msg_queue:    multiprocessing.Queue   | None = None
        self._cancel_event: Any                     | None = None
        # QTimer that polls the message queue at 30 Hz when a run is
        # active.  Lives the lifetime of the window.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)   # ms
        self._poll_timer.timeout.connect(self._on_poll_queue)
        self._build_ui()
        self._install_crash_hooks()
        self._load_icon()
        self._restore_settings()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Top-level horizontal split: sidebar | tabs
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────
        # Fixed-width left panel; the scrollable parameter list lives
        # inside it so the Start/Stop button can stay pinned at the bottom
        # regardless of how far the user has scrolled.
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(340)
        sidebar.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        sb_outer = QtWidgets.QVBoxLayout(sidebar)
        sb_outer.setContentsMargins(0, 0, 0, 0)
        sb_outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_inner = QtWidgets.QWidget()
        sb_layout = QtWidgets.QVBoxLayout(scroll_inner)
        sb_layout.setContentsMargins(12, 12, 12, 12)
        sb_layout.setSpacing(8)
        scroll.setWidget(scroll_inner)
        sb_outer.addWidget(scroll, stretch=1)

        # Build all parameter sections inside the scrollable area
        self._build_sidebar(sb_layout)

        # Pinned Start/Stop button outside the scroll area
        btn_container = QtWidgets.QWidget()
        btn_layout    = QtWidgets.QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(12, 6, 12, 12)
        self.btn_run = QtWidgets.QPushButton("Start")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_layout.addWidget(self.btn_run)
        sb_outer.addWidget(btn_container)

        layout.addWidget(sidebar)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        self._build_run_tab()
        self._build_batch_tab()
        # B1.3+: Compare, Workspace tabs go here
        layout.addWidget(self.tabs, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready")

    # ── Tiny helpers for compact widget construction ──────────────────────
    @staticmethod
    def _spin_int(value: int, lo: int, hi: int, step: int = 1,
                  tip: str = "") -> QtWidgets.QSpinBox:
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setValue(value)
        if tip: s.setToolTip(tip)
        return s

    @staticmethod
    def _spin_dbl(value: float, lo: float, hi: float, step: float = 0.01,
                  decimals: int = 3, tip: str = "") -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals)
        s.setValue(value)
        if tip: s.setToolTip(tip)
        return s

    def _build_sidebar(self, layout: QtWidgets.QVBoxLayout):
        """Build the full B1.1 parameter panel — every knob from the Tk
        sidebar grouped into collapsible QGroupBox sections."""
        title = QtWidgets.QLabel("Analysis Parameters")
        f = title.font(); f.setBold(True); f.setPointSize(11); title.setFont(f)
        layout.addWidget(title)

        # ── Input & Output ────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("Input & output")
        gl  = QtWidgets.QVBoxLayout(grp)
        gl.addWidget(QtWidgets.QLabel("Input file"))
        row = QtWidgets.QHBoxLayout()
        self.e_file = QtWidgets.QLineEdit()
        self.e_file.setPlaceholderText("Browse for a .czi / .tif file…")
        b1 = QtWidgets.QPushButton("Browse"); b1.clicked.connect(self._on_browse_file)
        row.addWidget(self.e_file); row.addWidget(b1)
        gl.addLayout(row)
        gl.addWidget(QtWidgets.QLabel("Output folder (optional)"))
        row = QtWidgets.QHBoxLayout()
        self.e_outdir = QtWidgets.QLineEdit()
        self.e_outdir.setPlaceholderText("Defaults to input folder")
        b2 = QtWidgets.QPushButton("Browse"); b2.clicked.connect(self._on_browse_outdir)
        row.addWidget(self.e_outdir); row.addWidget(b2)
        gl.addLayout(row)
        layout.addWidget(grp)

        # ── Imaging metadata ──────────────────────────────────────────────
        # File-embedded values are used by default; checkbox enables manual
        # override.  Matches the Tk app's behaviour.
        grp = QtWidgets.QGroupBox("Imaging metadata")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        row = QtWidgets.QHBoxLayout()
        self.c_override_px = QtWidgets.QCheckBox("Override")
        self.c_override_px.setToolTip(
            "If unchecked, the pixel size from the file's metadata is used.\n"
            "Check this only if the metadata is missing or wrong.")
        self.s_pixel_size  = self._spin_dbl(0.106, 0.01, 1.0, 0.001, decimals=3,
            tip="Physical pixel size in µm. Used to convert px → µm for D, MSD, etc.")
        row.addWidget(self.c_override_px); row.addWidget(self.s_pixel_size, 1)
        wpx = QtWidgets.QWidget(); wpx.setLayout(row)
        gl.addRow("Pixel size (µm)", wpx)

        row = QtWidgets.QHBoxLayout()
        self.c_override_fi = QtWidgets.QCheckBox("Override")
        self.c_override_fi.setToolTip(
            "If unchecked, the frame interval from the file's metadata is used.")
        self.s_frame_interval = self._spin_dbl(0.02, 0.001, 10.0, 0.001, decimals=3,
            tip="Time between frames in seconds. Used for diffusion coefficient units.")
        row.addWidget(self.c_override_fi); row.addWidget(self.s_frame_interval, 1)
        wfi = QtWidgets.QWidget(); wfi.setLayout(row)
        gl.addRow("Frame interval (s)", wfi)

        self.s_channel = self._spin_int(0, 0, 8,
            tip="Channel index to load (CZI files only). Most single-channel data uses 0.")
        gl.addRow("Channel (CZI)", self.s_channel)
        layout.addWidget(grp)

        # ── Preprocessing ─────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("Preprocessing")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_bg_method = QtWidgets.QComboBox()
        self.c_bg_method.addItems(["Uniform Filter", "Rolling Ball"])
        self.c_bg_method.setToolTip(
            "Method for subtracting local background before detection.\n"
            "• Uniform Filter — fast box-mean subtraction. Good default.\n"
            "• Rolling Ball — slower but better on uneven illumination.")
        gl.addRow("Background method", self.c_bg_method)
        self.s_bg_radius = self._spin_int(10, 3, 200,
            tip="Radius (px) of the local-mean window for background subtraction.\n"
                "Use ~3× spot diameter for diffraction-limited spots.")
        gl.addRow("Background radius (px)", self.s_bg_radius)
        layout.addWidget(grp)

        # ── Detection ─────────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("Detection")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_diameter = self._spin_int(7, 3, 21, step=2,
            tip="Expected spot diameter in pixels. Must be ODD (the GUI enforces this).\n"
                "Use ~2× the diffraction-limited PSF FWHM. Too small misses spots; "
                "too big merges adjacent ones.")
        gl.addRow("Diameter (px, odd)", self.s_diameter)

        # Auto minmass: when checked, pipeline auto-detects from first chunk.
        # Default OFF — the auto-detect formula in sptpalm_analysis can
        # under-shoot on some data (giving e.g. 0.04 when 1.0+ is needed),
        # which on a GPU backend produces 100k+ "spots" per chunk and
        # tanks throughput.  Users with known data should set minmass
        # manually; auto-detect is for exploratory runs on new data.
        row = QtWidgets.QHBoxLayout()
        self.c_auto_minmass = QtWidgets.QCheckBox("Auto-detect")
        self.c_auto_minmass.setToolTip(
            "When checked, the pipeline picks minmass from the first chunk's\n"
            "99th-percentile pixel value × diameter²/8.  Heuristic — works on\n"
            "many datasets but may under-shoot; manual tuning is more reliable.")
        self.c_auto_minmass.setChecked(False)
        self.s_minmass = self._spin_dbl(1.0, 0.0, 20.0, 0.05, decimals=2,
            tip="Minimum integrated intensity for a spot.\n"
                "Too low → many false-positive spots, slow linking, garbage tracks.\n"
                "Too high → real spots filtered out.\n"
                "Tune by trial: start at 1.0, decrease until you see spurious tracks.")
        self.c_auto_minmass.toggled.connect(
            lambda checked: self.s_minmass.setEnabled(not checked))
        self.s_minmass.setEnabled(True)
        row.addWidget(self.c_auto_minmass); row.addWidget(self.s_minmass, 1)
        wmm = QtWidgets.QWidget(); wmm.setLayout(row)
        gl.addRow("Min mass", wmm)
        layout.addWidget(grp)

        # ── Linking ───────────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("Linking")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_search_range = self._spin_int(5, 1, 30,
            tip="Maximum pixel distance a particle can move between consecutive\n"
                "frames. Calibrate from your data: bigger search_range tolerates\n"
                "fast motion but increases linker subnetwork-explosion risk.")
        gl.addRow("Search range (px)", self.s_search_range)
        self.s_memory = self._spin_int(3, 0, 10,
            tip="Number of frames a track can disappear and still be re-linked.\n"
                "0 = strict (no gaps). 3 is typical for blinking PALM probes.")
        gl.addRow("Memory (frames)", self.s_memory)
        self.s_min_track_len = self._spin_int(8, 3, 50,
            tip="Tracks shorter than this are discarded. 8 is the de-facto minimum\n"
                "for reliable MSD fits.")
        gl.addRow("Min track length", self.s_min_track_len)
        self.s_max_track_len = self._spin_int(0, 0, 100000,
            tip="0 = disabled. If set, drops tracks longer than this. Useful for\n"
                "removing stuck/aggregated particles that masquerade as long tracks.")
        gl.addRow("Max track length (0 = off)", self.s_max_track_len)
        layout.addWidget(grp)

        # ── Diffusion fit + motion classification ─────────────────────────
        grp = QtWidgets.QGroupBox("Diffusion & motion classification")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_max_lagtime = self._spin_int(20, 5, 100,
            tip="Maximum lag-time (in frames) used in the MSD curve.")
        gl.addRow("Max lag time", self.s_max_lagtime)
        self.s_n_fit = self._spin_int(5, 2, 20,
            tip="Number of initial lag times used to fit D and α via linear LSQ.\n"
                "Fewer = more local (short-time D); more = more global.")
        gl.addRow("N fit lags", self.s_n_fit)
        self.s_alpha_immobile = self._spin_dbl(0.5, 0.0, 2.0, 0.01, decimals=2,
            tip="α below this → 'Immobile'. Default 0.5 from the SPT literature.")
        gl.addRow("α  immobile threshold", self.s_alpha_immobile)
        self.s_alpha_confined = self._spin_dbl(0.9, 0.0, 2.0, 0.01, decimals=2,
            tip="α between immobile and this → 'Confined'.")
        gl.addRow("α  confined threshold", self.s_alpha_confined)
        self.s_alpha_directed = self._spin_dbl(1.1, 0.0, 2.0, 0.01, decimals=2,
            tip="α above this → 'Directed'. Between confined and directed → 'Brownian'.")
        gl.addRow("α  directed threshold", self.s_alpha_directed)
        self.s_mobile_d_threshold = self._spin_dbl(0.05, 0.0, 10.0, 0.01, decimals=3,
            tip="Diffusion coefficient threshold separating 'mobile' from\n"
                "'immobile' tracks for the mobile-fraction-over-time panel.")
        gl.addRow("Mobile D threshold (µm²/s)", self.s_mobile_d_threshold)
        self.s_jdd_components = self._spin_int(2, 1, 4,
            tip="Number of exponential components in the Jump Distance Distribution\n"
                "fit. 2 is typical (mobile + immobile populations).")
        gl.addRow("JDD components", self.s_jdd_components)

        # Filter-by-D toggle + range
        self.c_filter_d_enabled = QtWidgets.QCheckBox("Filter tracks by D")
        self.c_filter_d_enabled.setToolTip(
            "When checked, drop tracks with D outside the [min, max] range.\n"
            "Useful for isolating a specific population for downstream analysis.")
        gl.addRow(self.c_filter_d_enabled)
        self.s_filter_d_min = self._spin_dbl(0.0, 0.0, 10.0, 0.01, decimals=3,
            tip="Minimum D (µm²/s). Tracks slower than this are excluded.")
        self.s_filter_d_max = self._spin_dbl(1.0, 0.0, 10.0, 0.01, decimals=3,
            tip="Maximum D (µm²/s). Tracks faster than this are excluded.")
        self.s_filter_d_min.setEnabled(False)
        self.s_filter_d_max.setEnabled(False)
        self.c_filter_d_enabled.toggled.connect(
            lambda checked: (self.s_filter_d_min.setEnabled(checked),
                              self.s_filter_d_max.setEnabled(checked)))
        gl.addRow("  D min (µm²/s)", self.s_filter_d_min)
        gl.addRow("  D max (µm²/s)", self.s_filter_d_max)
        layout.addWidget(grp)

        # ── ROI ───────────────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("ROI")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_roi_mode = QtWidgets.QComboBox()
        self.c_roi_mode.addItems(["None", "Auto threshold", "Manual threshold"])
        self.c_roi_mode.setCurrentText("Auto threshold")
        self.c_roi_mode.setToolTip(
            "Restrict analysis to a region of interest in the field of view.\n"
            "• None — analyse the whole image.\n"
            "• Auto threshold — pick a threshold from the mean projection.\n"
            "• Manual threshold — use the value below.")
        gl.addRow("Mode", self.c_roi_mode)
        self.c_roi_auto_method = QtWidgets.QComboBox()
        self.c_roi_auto_method.addItems(["Li", "Otsu", "Triangle", "Mean"])
        self.c_roi_auto_method.setToolTip(
            "Auto-thresholding method (from scikit-image).  Li is robust for\n"
            "low-contrast SMLM data; Otsu for bimodal histograms.")
        gl.addRow("Auto method", self.c_roi_auto_method)
        self.s_roi_threshold = self._spin_dbl(0.08, 0.0, 1.0, 0.005, decimals=3,
            tip="Manual threshold on the normalised mean projection [0, 1].")
        gl.addRow("Manual threshold", self.s_roi_threshold)
        self.c_roi_mask_mode = QtWidgets.QComboBox()
        self.c_roi_mask_mode.addItems(["Mean", "Sum"])
        self.c_roi_mask_mode.setToolTip(
            "Which projection is used to compute the ROI mask.\n"
            "Mean is appropriate when signal density is uniform; Sum\n"
            "emphasises bright sparse spots.")
        gl.addRow("Projection for ROI", self.c_roi_mask_mode)
        layout.addWidget(grp)

        # ── Drift correction ──────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("Drift correction")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_drift_correct = QtWidgets.QCheckBox("Apply RCC drift correction")
        self.c_drift_correct.setToolTip(
            "Redundant Cross-Correlation (RCC) drift correction: estimates the\n"
            "sample drift over time by all-pairs cross-correlation between\n"
            "segments, then subtracts it from every localisation.\n"
            "Strongly recommended for sptPALM movies > 1 minute long.")
        self.c_drift_correct.setChecked(True)
        gl.addRow(self.c_drift_correct)
        self.s_drift_segment = self._spin_int(500, 50, 5000, step=50,
            tip="Frames per RCC segment. Smaller = finer drift tracking but\n"
                "noisier. 500 is a reasonable default for 4000+ frame movies.")
        gl.addRow("Segment size (frames)", self.s_drift_segment)
        layout.addWidget(grp)

        # ── Clustering ────────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("Clustering (DBSCAN)")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_cluster_eps_nm = self._spin_dbl(50.0, 5.0, 1000.0, 5.0, decimals=1,
            tip="DBSCAN neighbourhood radius (nm). Two localisations are in the\n"
                "same cluster if they're within this distance.")
        gl.addRow("eps (nm)", self.s_cluster_eps_nm)
        self.s_cluster_min_samples = self._spin_int(10, 2, 100,
            tip="Minimum localisations to form a DBSCAN cluster. Lower = more\n"
                "clusters detected but noisier; higher = stricter.")
        gl.addRow("min samples", self.s_cluster_min_samples)
        layout.addWidget(grp)

        # ── Performance ───────────────────────────────────────────────────
        grp = QtWidgets.QGroupBox(f"Performance  —  {N_CPUS} cores")
        gl  = QtWidgets.QFormLayout(grp)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_backend = QtWidgets.QComboBox()
        self.c_backend.addItems(self._available_backends())
        self.c_backend.setToolTip(
            "Which implementation to use for spot localisation.\n"
            "• auto       — picks the fastest healthy backend on this machine.\n"
            "• trackpy    — reference CPU implementation (battle-tested).\n"
            "• torch      — PyTorch, device auto-selected.\n"
            "• torch-mps  — force Apple GPU.  Fast when stable; on some macOS/M-chip\n"
            "                combinations may hit memory-allocator issues at very\n"
            "                low minmass (lots of false-positive spots).\n"
            "• torch-cuda — force NVIDIA GPU.\n"
            "• torch-cpu  — force PyTorch on CPU (for benchmarking).")
        gl.addRow("Detection backend", self.c_backend)
        self.s_workers = self._spin_int(N_CPUS, 1, N_CPUS,
            tip="Parallel CPU workers for the trackpy backend's multiprocessing\n"
                "pool and the MSD fitting thread pool.  Default = all cores.")
        gl.addRow(f"CPU workers (max {N_CPUS})", self.s_workers)
        self.s_chunk_size = self._spin_int(500, 50, 5000, step=100,
            tip="Frames per processing chunk. Bigger = less per-chunk overhead\n"
                "(esp. on GPU) but more RAM. 500 is balanced; tune up if your\n"
                "stack and free RAM are large.")
        gl.addRow("Chunk size (frames)", self.s_chunk_size)
        layout.addWidget(grp)

        layout.addStretch(1)

    def _build_run_tab(self):
        """Right pane: log viewer + progress bar + matplotlib canvas."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        v.addWidget(self.progress_bar)

        # Vertical splitter: log on top, figure on bottom
        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)

        # Log viewer.  Cap the block count so the document doesn't grow
        # unbounded — a 64-file batch can dump tens of thousands of lines
        # and per-append cost climbs noticeably past ~5k blocks, which
        # starves the Qt event loop and makes the log appear "frozen"
        # while the subprocess is actually producing output in real time.
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        mono = QtGui.QFont("Menlo, Consolas, monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.log_box.setFont(mono)
        splitter.addWidget(self.log_box)

        # Matplotlib canvas wrapper
        canvas_wrap = QtWidgets.QWidget()
        cwv = QtWidgets.QVBoxLayout(canvas_wrap)
        cwv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="#222")
        self.canvas = FigureCanvas(self.fig)
        self.canvas_toolbar = NavToolbar(self.canvas, canvas_wrap)
        cwv.addWidget(self.canvas_toolbar)
        cwv.addWidget(self.canvas)
        # Placeholder
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#111")
        ax.text(0.5, 0.5, "Results figure will appear here after analysis",
                ha="center", va="center", color="#888", fontsize=12,
                transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()
        splitter.addWidget(canvas_wrap)
        splitter.setSizes([300, 500])
        v.addWidget(splitter, stretch=1)

        self.tabs.addTab(tab, "Run Analysis")

    def _build_batch_tab(self):
        """Batch tab: pick a folder, choose which files to process, run all."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # Folder picker row
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Input folder"))
        self.e_batch_folder = QtWidgets.QLineEdit()
        self.e_batch_folder.setPlaceholderText("Pick a folder containing .czi / .tif files…")
        btn_pick = QtWidgets.QPushButton("Browse")
        btn_pick.clicked.connect(self._on_batch_pick_folder)
        btn_refresh = QtWidgets.QPushButton("↻ Rescan")
        btn_refresh.setToolTip("Re-scan the folder for input files.")
        btn_refresh.clicked.connect(self._on_batch_rescan)
        row.addWidget(self.e_batch_folder, 1)
        row.addWidget(btn_pick)
        row.addWidget(btn_refresh)
        v.addLayout(row)

        # File list with checkboxes
        v.addWidget(QtWidgets.QLabel(
            "Files to process  (uncheck individual files to skip them):"))
        self.lst_batch_files = QtWidgets.QListWidget()
        self.lst_batch_files.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.lst_batch_files, stretch=1)

        # Select all / none / inverse
        sel_row = QtWidgets.QHBoxLayout()
        for label, fn in (("Select all",     self._on_batch_select_all),
                          ("Select none",    self._on_batch_select_none),
                          ("Invert selection", self._on_batch_select_inverse)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            sel_row.addWidget(b)
        sel_row.addStretch(1)
        self.lbl_batch_summary = QtWidgets.QLabel("0 files / 0 selected")
        sel_row.addWidget(self.lbl_batch_summary)
        v.addLayout(sel_row)

        # Progress bar (overall batch %) + log + per-file figure preview
        self.batch_progress = QtWidgets.QProgressBar()
        self.batch_progress.setRange(0, 100); self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setFormat("Idle")
        v.addWidget(self.batch_progress)

        self.batch_log_box = QtWidgets.QPlainTextEdit()
        self.batch_log_box.setReadOnly(True)
        # See note on log_box: cap block count for performance on long
        # batch runs that produce many thousands of log lines.
        self.batch_log_box.setMaximumBlockCount(5000)
        mono = QtGui.QFont("Menlo, Consolas, monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.batch_log_box.setFont(mono)
        v.addWidget(self.batch_log_box, stretch=1)

        self.tabs.addTab(tab, "Batch")

    # ── Batch-tab helpers ─────────────────────────────────────────────────
    @staticmethod
    def _looks_like_input_file(name: str) -> bool:
        n = name.lower()
        return n.endswith(".czi") or n.endswith(".tif") or n.endswith(".tiff")

    def _on_batch_pick_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder containing input files",
            self.e_batch_folder.text() or os.path.expanduser("~"))
        if path:
            self.e_batch_folder.setText(path)
            self._batch_rescan(path)

    def _on_batch_rescan(self):
        path = self.e_batch_folder.text().strip()
        if path:
            self._batch_rescan(path)

    def _batch_rescan(self, folder: str):
        """Populate the file list with .czi/.tif files in `folder`."""
        self.lst_batch_files.clear()
        if not os.path.isdir(folder):
            self._batch_update_summary()
            return
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            self._batch_update_summary()
            return
        for name in names:
            if not self._looks_like_input_file(name):
                continue
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, full)
            self.lst_batch_files.addItem(item)
        self.lst_batch_files.itemChanged.connect(
            lambda _: self._batch_update_summary())
        self._batch_update_summary()

    def _batch_iter_items(self):
        for i in range(self.lst_batch_files.count()):
            yield self.lst_batch_files.item(i)

    def _batch_update_summary(self):
        total = self.lst_batch_files.count()
        sel   = sum(1 for it in self._batch_iter_items()
                    if it.checkState() == Qt.CheckState.Checked)
        self.lbl_batch_summary.setText(f"{total} files / {sel} selected")

    def _on_batch_select_all(self):
        for it in self._batch_iter_items():
            it.setCheckState(Qt.CheckState.Checked)

    def _on_batch_select_none(self):
        for it in self._batch_iter_items():
            it.setCheckState(Qt.CheckState.Unchecked)

    def _on_batch_select_inverse(self):
        for it in self._batch_iter_items():
            it.setCheckState(
                Qt.CheckState.Unchecked if it.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked)

    def _batch_checked_files(self) -> list[str]:
        return [it.data(Qt.ItemDataRole.UserRole)
                for it in self._batch_iter_items()
                if it.checkState() == Qt.CheckState.Checked]

    # ── Settings persistence ──────────────────────────────────────────────
    # ── Settings layout ───────────────────────────────────────────────────
    def _setting_specs(self):
        """Single source of truth for every persisted widget.

        Each entry is (qsettings_key, widget, caster).  Used by both
        `_save_settings` and `_restore_settings` so they can't drift out
        of sync.  Widgets whose value is a string (combo / line edit) use
        a `str` caster; spinboxes use `int` or `float`; checkboxes use
        a small lambda that handles QSettings' "true"/"false" round-trip.
        """
        def _bool_cast(v):
            if isinstance(v, bool): return v
            if isinstance(v, str):  return v.lower() in ("1", "true", "yes")
            return bool(v)

        return [
            # ── Paths ─────────────────────────────────────────────────────
            ("analysis/file",            self.e_file,            "text"),
            ("analysis/outdir",          self.e_outdir,          "text"),

            # ── Imaging metadata ──────────────────────────────────────────
            ("analysis/override_px",     self.c_override_px,     "check", _bool_cast),
            ("analysis/pixel_size",      self.s_pixel_size,      "spin",  float),
            ("analysis/override_fi",     self.c_override_fi,     "check", _bool_cast),
            ("analysis/frame_interval",  self.s_frame_interval,  "spin",  float),
            ("analysis/channel",         self.s_channel,         "spin",  int),

            # ── Preprocessing ─────────────────────────────────────────────
            ("analysis/bg_method",       self.c_bg_method,       "combo"),
            ("analysis/bg_radius",       self.s_bg_radius,       "spin",  int),

            # ── Detection ─────────────────────────────────────────────────
            ("analysis/diameter",        self.s_diameter,        "spin",  int),
            ("analysis/auto_minmass",    self.c_auto_minmass,    "check", _bool_cast),
            ("analysis/minmass",         self.s_minmass,         "spin",  float),

            # ── Linking ───────────────────────────────────────────────────
            ("analysis/search_range",    self.s_search_range,    "spin",  int),
            ("analysis/memory",          self.s_memory,          "spin",  int),
            ("analysis/min_track_len",   self.s_min_track_len,   "spin",  int),
            ("analysis/max_track_len",   self.s_max_track_len,   "spin",  int),

            # ── Diffusion & motion ────────────────────────────────────────
            ("analysis/max_lagtime",     self.s_max_lagtime,     "spin",  int),
            ("analysis/n_fit",           self.s_n_fit,           "spin",  int),
            ("analysis/alpha_immobile",  self.s_alpha_immobile,  "spin",  float),
            ("analysis/alpha_confined",  self.s_alpha_confined,  "spin",  float),
            ("analysis/alpha_directed",  self.s_alpha_directed,  "spin",  float),
            ("analysis/mobile_d",        self.s_mobile_d_threshold, "spin", float),
            ("analysis/jdd_components",  self.s_jdd_components,  "spin",  int),
            ("analysis/filter_d_enable", self.c_filter_d_enabled,"check", _bool_cast),
            ("analysis/filter_d_min",    self.s_filter_d_min,    "spin",  float),
            ("analysis/filter_d_max",    self.s_filter_d_max,    "spin",  float),

            # ── ROI ───────────────────────────────────────────────────────
            ("analysis/roi_mode",        self.c_roi_mode,        "combo"),
            ("analysis/roi_auto_method", self.c_roi_auto_method, "combo"),
            ("analysis/roi_threshold",   self.s_roi_threshold,   "spin",  float),
            ("analysis/roi_mask_mode",   self.c_roi_mask_mode,   "combo"),

            # ── Drift correction ──────────────────────────────────────────
            ("analysis/drift_correct",   self.c_drift_correct,   "check", _bool_cast),
            ("analysis/drift_segment",   self.s_drift_segment,   "spin",  int),

            # ── Clustering ────────────────────────────────────────────────
            ("analysis/cluster_eps_nm",  self.s_cluster_eps_nm,  "spin",  float),
            ("analysis/cluster_min_samples", self.s_cluster_min_samples, "spin", int),

            # ── Performance ───────────────────────────────────────────────
            ("analysis/backend",         self.c_backend,         "combo"),
            ("analysis/workers",         self.s_workers,         "spin",  int),
            ("analysis/chunk_size",      self.s_chunk_size,      "spin",  int),
        ]

    def _restore_settings(self):
        """Restore the user's saved selections.  Best-effort — silently
        ignores any malformed values rather than failing the launch."""
        s = self._settings
        try:
            geom = s.value("window/geometry")
            if geom is not None:
                self.restoreGeometry(geom)
        except Exception:
            pass

        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            try:
                v = s.value(key)
                if v is None or v == "":
                    continue
                if kind == "text":
                    widget.setText(str(v))
                elif kind == "combo":
                    if str(v) in [widget.itemText(i) for i in range(widget.count())]:
                        widget.setCurrentText(str(v))
                elif kind == "spin":
                    caster = spec[3]
                    widget.setValue(caster(v))
                elif kind == "check":
                    caster = spec[3]
                    widget.setChecked(caster(v))
            except Exception:
                pass

        # Sync derived enabled-state (auto-minmass disables minmass spin,
        # filter-D toggles the D-min/max spins) AFTER restoring values.
        try:
            self.s_minmass.setEnabled(not self.c_auto_minmass.isChecked())
            on = self.c_filter_d_enabled.isChecked()
            self.s_filter_d_min.setEnabled(on)
            self.s_filter_d_max.setEnabled(on)
        except Exception:
            pass

    def _save_settings(self):
        """Write current selections to QSettings.  Called when starting a
        run and on window close."""
        s = self._settings
        s.setValue("settings/version", self.SETTINGS_VERSION)
        try:
            s.setValue("window/geometry", self.saveGeometry())
        except Exception:
            pass
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            try:
                if kind == "text":
                    s.setValue(key, widget.text())
                elif kind == "combo":
                    s.setValue(key, widget.currentText())
                elif kind == "spin":
                    s.setValue(key, widget.value())
                elif kind == "check":
                    s.setValue(key, bool(widget.isChecked()))
            except Exception:
                pass

    def closeEvent(self, event):
        """Persist state on close and tear down any running subprocess."""
        try:
            self._save_settings()
        except Exception:
            pass
        # Make sure we don't leave an orphan analysis subprocess running
        # after the GUI is closed.
        try:
            if self._proc is not None and self._proc.is_alive():
                if self._cancel_event is not None:
                    self._cancel_event.set()
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
        except Exception:
            pass
        super().closeEvent(event)

    # ── Backend availability helper ───────────────────────────────────────
    def _available_backends(self) -> list[str]:
        """Return the static list of selectable backends.

        IMPORTANT: we deliberately do NOT probe torch here.  On some macOS /
        PyTorch / Apple-Silicon combinations (e.g. macOS 26 + M4 + PyTorch
        2.12.0), just importing torch and calling
        `torch.backends.mps.is_available()` is enough to trigger noisy MPS
        command-buffer errors on stderr and, in the worst case, kill the
        process before the GUI is fully up.

        The fix is to keep the dropdown population torch-free: show every
        backend the user might want, then probe each one lazily inside the
        analysis worker only when actually selected.  Unsupported
        selections (e.g. `torch-cuda` on a Mac) raise a clean
        RuntimeError at run time with an actionable message, which the
        crash-report dialog surfaces — way better UX than a launch abort.
        """
        return ["auto", "trackpy", "torch",
                "torch-mps", "torch-cuda", "torch-cpu"]

    # ── Event handlers ────────────────────────────────────────────────────
    def _on_browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select input file", os.path.expanduser("~"),
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if path:
            self.e_file.setText(path)
            if not self.e_outdir.text():
                self.e_outdir.setText(os.path.dirname(path))

    def _on_browse_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", os.path.expanduser("~"))
        if path:
            self.e_outdir.setText(path)

    # ── Params builder (shared by single-file and batch) ──────────────────
    def _build_params_for_file(self, fpath: str, out_dir: str | None) -> dict:
        """Build the full analysis-params dict for one input file.

        Reads every spinbox / combo / checkbox in the sidebar and produces
        the kwargs dict the worker expects.  Used by both the Run-Analysis
        and Batch tabs — the only thing that differs between modes is the
        input file path and (for batch) the per-file output folder.
        """
        bg_method_map = {
            "Uniform Filter": "uniform_filter",
            "Rolling Ball":   "rolling_ball",
        }
        roi_mode_map = {
            "None":              "none",
            "Auto threshold":    "auto",
            "Manual threshold":  "manual",
        }
        max_tl = int(self.s_max_track_len.value())
        return {
            "file":              fpath,
            "out_dir":           out_dir,
            "pixel_size":        (self.s_pixel_size.value()
                                  if self.c_override_px.isChecked() else None),
            "frame_interval":    (self.s_frame_interval.value()
                                  if self.c_override_fi.isChecked() else None),
            "channel":           int(self.s_channel.value()),
            "bg_method":         bg_method_map.get(
                                    self.c_bg_method.currentText(),
                                    "uniform_filter"),
            "bg_radius":         int(self.s_bg_radius.value()),
            "diameter":          int(self.s_diameter.value()),
            "auto_minmass":      bool(self.c_auto_minmass.isChecked()),
            "minmass":           float(self.s_minmass.value()),
            "search_range":      int(self.s_search_range.value()),
            "memory":            int(self.s_memory.value()),
            "min_track_len":     int(self.s_min_track_len.value()),
            "max_track_len":     max_tl if max_tl > 0 else None,
            "max_lagtime":       int(self.s_max_lagtime.value()),
            "n_fit":             int(self.s_n_fit.value()),
            "alpha_thresholds":  (float(self.s_alpha_immobile.value()),
                                  float(self.s_alpha_confined.value()),
                                  float(self.s_alpha_directed.value())),
            "mobile_d_threshold": float(self.s_mobile_d_threshold.value()),
            "jdd_components":    int(self.s_jdd_components.value()),
            "filter_d_enabled":  bool(self.c_filter_d_enabled.isChecked()),
            "filter_d_min":      float(self.s_filter_d_min.value()),
            "filter_d_max":      float(self.s_filter_d_max.value()),
            "roi_mode":          roi_mode_map.get(
                                    self.c_roi_mode.currentText(), "none"),
            "roi_auto_method":   self.c_roi_auto_method.currentText(),
            "roi_threshold":     float(self.s_roi_threshold.value()),
            "roi_mask_mode":     self.c_roi_mask_mode.currentText(),
            "drift_correct":     bool(self.c_drift_correct.isChecked()),
            "drift_segment":     int(self.s_drift_segment.value()),
            "cluster_eps_nm":      float(self.s_cluster_eps_nm.value()),
            "cluster_min_samples": int(self.s_cluster_min_samples.value()),
            "backend":           self.c_backend.currentText(),
            "workers":           int(self.s_workers.value()),
            "chunk_size":        int(self.s_chunk_size.value()),
        }

    def _on_run_clicked(self):
        # Acting as Stop?
        if self._proc is not None and self._proc.is_alive():
            if self._cancel_event is not None:
                self._cancel_event.set()
            # Record when Stop was requested so the poller can escalate
            # (SIGTERM → SIGKILL) if cooperative cancel doesn't take
            # effect within a few seconds.  Without this, a user clicking
            # Stop during a long uninterruptible region (e.g. trackpy's
            # linker on a high-density chunk) sees nothing happen for
            # minutes.
            self._stop_requested_at = time.time()
            self._stop_escalation_stage = 0   # 0=cooperative, 1=SIGTERM, 2=SIGKILL
            self.btn_run.setText("Stopping…")
            self.btn_run.setEnabled(False)

            # Surface in the active log so the user knows their click registered
            log_widget = (self.batch_log_box
                          if getattr(self, "_is_batch_run", False)
                          else self.log_box)
            log_widget.appendPlainText(
                "\n── Stop requested.  Waiting for the current stage to reach "
                "a checkpoint (up to 5 s); will force-terminate if it doesn't.")
            return

        # Dispatch by active tab.  Sidebar Start button does single-file
        # analysis when the Run Analysis tab is showing, and batch when
        # the Batch tab is showing.
        active_tab_label = self.tabs.tabText(self.tabs.currentIndex())
        if active_tab_label.startswith("Batch"):
            self._start_batch_run()
        else:
            self._start_single_run()

    def _start_single_run(self):
        fpath = self.e_file.text().strip()
        if not fpath or not os.path.isfile(fpath):
            QtWidgets.QMessageBox.warning(
                self, "No file", "Please pick an input file first.")
            return

        params = self._build_params_for_file(
            fpath, self.e_outdir.text().strip() or None)

        # Persist before the long-running task in case of crash/abort.
        try:
            self._save_settings()
        except Exception:
            pass

        # Clear UI for new run
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self._is_batch_run = False

        # Spawn analysis SUBPROCESS (not thread).  Rationale: Qt holds a
        # Metal-backed surface for window compositing on macOS, and that
        # contends with PyTorch's MPS allocator in the same process.  A
        # subprocess gives PyTorch a clean Python interpreter with no Qt
        # loaded — MPS gets the full unified-memory pool to itself.
        self._msg_queue    = multiprocessing.Queue()
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_analysis_in_subprocess,
            args=(params, self._msg_queue, self._cancel_event),
            name="FIREFLY-AnalysisWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.statusBar().showMessage("Running…")

    def _start_batch_run(self):
        """Kick off batch analysis over the checked files in the Batch tab."""
        files = self._batch_checked_files()
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "No files",
                "Pick a folder and check at least one file to process.")
            return

        # Batch outputs go to <input_folder>/batch_results/<stem>/  — same
        # convention as the Tk app.  Build a params dict per file.
        out_root = os.path.join(self.e_batch_folder.text().strip(),
                                "batch_results")
        params_list = []
        for fpath in files:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            file_out = os.path.join(out_root, stem)
            params_list.append(self._build_params_for_file(fpath, file_out))

        try:
            self._save_settings()
        except Exception:
            pass

        # Clear batch UI for new run
        self.batch_log_box.clear()
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Starting…")
        self._is_batch_run = True

        self._msg_queue    = multiprocessing.Queue()
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_batch_in_subprocess,
            args=(params_list, self._msg_queue, self._cancel_event),
            name="FIREFLY-BatchWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.statusBar().showMessage(f"Batch: 0 / {len(files)} files")

    # ── Queue polling (replaces QThread signals) ──────────────────────────
    def _on_poll_queue(self):
        """Drain pending messages from the subprocess's message queue.

        Called on a QTimer at ~30 Hz while a run is active.  We process at
        most a few hundred messages per tick to keep the UI responsive
        when tqdm is spamming progress updates during fast stages.
        """
        if self._msg_queue is None:
            return
        # Drain up to N messages per tick so we don't starve the UI loop
        # on a fast log flood.  Bumped to 1000 because most messages are
        # cheap log lines that we batch into a single appendPlainText.
        budget = 1000
        worker_done = False
        is_batch = getattr(self, "_is_batch_run", False)

        # Route log/progress to whichever tab is "owning" this run.
        log_widget      = self.batch_log_box if is_batch else self.log_box
        progress_widget = self.batch_progress if is_batch else self.progress_bar

        # Buffer log lines and append them in a SINGLE call at end of tick.
        # appendPlainText reflows the document each call; 1000 separate
        # appends on a long document can take seconds.  One append of a
        # newline-joined string completes in milliseconds.
        log_buf: list[str] = []
        last_progress: tuple | None = None  # only the latest progress matters

        while budget > 0:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            budget -= 1
            if kind == "log":
                log_buf.append(payload)
            elif kind == "progress":
                last_progress = payload   # drop earlier intra-tick updates
            elif kind == "done":
                # Single-file completion.  Only valid in non-batch mode;
                # in batch mode the per-file messages are "file_done".
                self._handle_done(payload)
                worker_done = True
            elif kind == "file_done":
                self._handle_file_done(payload)
            elif kind == "file_error":
                self._handle_file_error(payload)
            elif kind == "batch_done":
                self._handle_batch_done(payload)
                worker_done = True
            elif kind == "stopped":
                self._handle_stopped()
                worker_done = True
            elif kind == "error":
                self._handle_failed(payload)
                worker_done = True

        # Flush the per-tick log buffer with ONE append call.  Also
        # coalesce progress: only the most recent value matters for
        # display purposes (it overwrites all earlier ones anyway).
        if log_buf:
            log_widget.appendPlainText("\n".join(log_buf))
        if last_progress is not None:
            pct, msg = last_progress
            progress_widget.setValue(pct)
            progress_widget.setFormat(f"{msg}  ({pct}%)")
            self.statusBar().showMessage(msg)

        # Stop-button escalation: if cancel_event was set N seconds ago
        # and the subprocess is still alive, escalate.  Two-stage SIGTERM
        # → SIGKILL because some torch / native code can ignore SIGTERM.
        stop_at = getattr(self, "_stop_requested_at", None)
        if (stop_at is not None and self._proc is not None
                and self._proc.is_alive()):
            elapsed = time.time() - stop_at
            stage   = getattr(self, "_stop_escalation_stage", 0)
            if stage == 0 and elapsed > 5.0:
                log_widget.appendPlainText(
                    "  Cooperative cancel didn't take effect within 5 s — "
                    "sending SIGTERM to the analysis subprocess.")
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._stop_escalation_stage = 1
                self._stop_requested_at = time.time()  # reset timer for SIGKILL
            elif stage == 1 and elapsed > 3.0:
                log_widget.appendPlainText(
                    "  SIGTERM didn't take effect within 3 s — sending SIGKILL.")
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._stop_escalation_stage = 2

        # Also detect a subprocess that has exited without posting a
        # terminal message (e.g. crashed, SIGTERM'd, or SIGKILL'd).
        if not worker_done and self._proc is not None and not self._proc.is_alive():
            time.sleep(0.05)
            try:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    log_widget.appendPlainText(payload)
            except queue.Empty:
                pass
            # If the user pressed Stop, treat exit as "stopped", not an error
            if getattr(self, "_stop_requested_at", None) is not None:
                self._handle_stopped()
            else:
                self._handle_failed(
                    f"Analysis subprocess exited abnormally "
                    f"(exit code {self._proc.exitcode}).  See log for details.")
            worker_done = True

        if worker_done:
            self._cleanup_after_run()

    # ── Subprocess result handlers ────────────────────────────────────────
    def _handle_done(self, payload: dict):
        self._show_result_figure(payload.get("figure_path"))
        self.statusBar().showMessage(
            f"Analysis complete — output at {payload.get('out_dir')}")

    def _handle_file_done(self, payload: dict):
        """One file in a batch finished successfully — not the terminal msg."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        self.statusBar().showMessage(
            f"Batch: {i} / {total} files complete  ({payload.get('n_tracks', 0):,} tracks)")
        # Overall batch progress: percent of completed files
        if total:
            pct = int(100 * i / total)
            self.batch_progress.setValue(pct)
            self.batch_progress.setFormat(f"Batch  {i}/{total}  ({pct}%)")

    def _handle_file_error(self, payload: dict):
        """One file in a batch failed — log it, batch continues."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        f = payload.get("file", "?")
        self.batch_log_box.appendPlainText(
            f"\n  ⚠ [{i}/{total}] failed: {os.path.basename(f)}")
        self.statusBar().showMessage(f"Batch: file {i} failed (continuing)")

    def _handle_batch_done(self, payload: dict):
        """Batch terminal message — all files attempted."""
        n_total = payload.get("n_total", 0)
        n_ok    = payload.get("n_ok",    0)
        n_fail  = payload.get("n_fail",  0)
        self.batch_progress.setValue(100)
        self.batch_progress.setFormat(
            f"Batch complete  —  {n_ok}/{n_total} succeeded, {n_fail} failed")
        self.statusBar().showMessage(
            f"Batch complete — {n_ok}/{n_total} succeeded, {n_fail} failed")

    def _handle_stopped(self):
        self.statusBar().showMessage("Stopped by user")
        is_batch = getattr(self, "_is_batch_run", False)
        bar = self.batch_progress if is_batch else self.progress_bar
        bar.setFormat("Stopped")

    def _handle_failed(self, tb: str):
        try:
            path = crash_reporter.write_crash_report(
                RuntimeError, RuntimeError("Analysis subprocess raised"),
                None, source="analysis subprocess", context=tb)
            is_batch = getattr(self, "_is_batch_run", False)
            log_widget = self.batch_log_box if is_batch else self.log_box
            log_widget.appendPlainText(f"\nCrash report: {path}")
            self._show_crash_dialog(path)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Analysis error", tb[-1500:])
        self.statusBar().showMessage("Error — see log")

    def _cleanup_after_run(self):
        """Tear down the subprocess + queue after a run ends."""
        self._poll_timer.stop()
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
            except Exception:
                pass
        self._proc                  = None
        self._msg_queue             = None
        self._cancel_event          = None
        self._stop_requested_at     = None
        self._stop_escalation_stage = 0
        self.btn_run.setText("Start")
        self.btn_run.setEnabled(True)

    def _show_result_figure(self, figure_path: str):
        """Display the saved combined-results image on the matplotlib canvas."""
        if not figure_path or not os.path.isfile(figure_path):
            return
        try:
            import numpy as _np
            from PIL import Image as _PILImage
            img = _np.asarray(_PILImage.open(figure_path).convert("RGB"))
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
        except Exception as exc:
            self.log_box.appendPlainText(
                f"  WARN: could not display figure: {exc}")

    # ── Crash reporter integration ────────────────────────────────────────
    def _install_crash_hooks(self):
        """Wire FIREFLY into the global crash reporter.  Same idea as the Tk
        version: capture every uncaught exception, write a self-contained
        text report, surface the path to the user via a dialog."""

        def _log_provider(n: int = 120) -> str:
            try:
                txt = self.log_box.toPlainText()
                return "\n".join(txt.splitlines()[-n:])
            except Exception:
                return ""

        def _state_provider() -> dict:
            try:
                return {
                    "UI":                 "PySide6 (v2.0-dev)",
                    "Current file":       self.e_file.text(),
                    "Output folder":      self.e_outdir.text() or "(default)",
                    "Pixel size":         self.s_pixel_size.value(),
                    "Frame interval":     self.s_frame_interval.value(),
                    "Detection diameter": self.s_diameter.value(),
                    "Min mass":           self.s_minmass.value(),
                    "Detection backend":  self.c_backend.currentText(),
                    "Running":            (self._proc is not None
                                            and self._proc.is_alive()),
                }
            except Exception as e:
                return {"<state error>": repr(e)}

        crash_reporter.set_log_provider(_log_provider)
        crash_reporter.set_app_state_provider(_state_provider)

        def _on_crash(path: str):
            # Marshal to Qt main thread before touching widgets
            QtCore.QMetaObject.invokeMethod(
                self, "_show_crash_dialog",
                Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, path))

        crash_reporter.install_global_handlers(on_crash=_on_crash)

    @QtCore.Slot(str)
    def _show_crash_dialog(self, path: str):
        """Modal dialog with the crash-report path; offers to open the folder."""
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg.setWindowTitle("FIREFLY — Unexpected error")
        msg.setText("FIREFLY hit an unexpected error.")
        msg.setInformativeText(
            f"A detailed crash report has been saved:\n\n"
            f"    {os.path.basename(path)}\n\n"
            f"Location:\n    {os.path.dirname(path)}")
        msg.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Open
            | QtWidgets.QMessageBox.StandardButton.Close)
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Open)
        if msg.exec() == QtWidgets.QMessageBox.StandardButton.Open:
            _open_folder(os.path.dirname(path))

    def _load_icon(self):
        """Best-effort: load assets/icon.png as the window/dock icon."""
        for cand in (
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "icon.png"),
                os.path.join(getattr(sys, "_MEIPASS", ""), "assets", "icon.png"),
        ):
            if os.path.isfile(cand):
                self.setWindowIcon(QtGui.QIcon(cand))
                QtWidgets.QApplication.setWindowIcon(QtGui.QIcon(cand))
                return


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _open_folder(path: str) -> None:
    """Open path in the system file manager."""
    import subprocess
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", os.path.normpath(path)], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def _qt_message_handler(mode, context, message):
    """Forward Qt's own log messages to stderr so they're visible in the
    terminal and end up in the crash report's "Recent log" snapshot."""
    sys.stderr.write(f"[Qt {mode.name}] {message}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Install crash handlers BEFORE creating QApplication so an early failure
    # (e.g. Qt plugin load, OpenGL init) still produces a useful report.
    crash_reporter.install_global_handlers()

    QtCore.qInstallMessageHandler(_qt_message_handler)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FIREFLY")
    app.setOrganizationName("jacoblevers")

    window = MainWindow()
    window.show()

    # CI smoke-test marker (mirrors the Tk app behaviour)
    marker_path = os.environ.get("SPTPALM_READY_MARKER")
    if marker_path:
        try:
            window.repaint()
        except Exception:
            pass
        try:
            with open(marker_path, "w") as f:
                f.write("ready\n")
        except Exception:
            pass

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
