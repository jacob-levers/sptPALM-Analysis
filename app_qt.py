"""
FIREFLY — Fluorescence Inference & Reconstruction Engine.

PySide6 / Qt frontend (v2.0+).  Replaced the original Tkinter UI that
shipped through v1.1.x.

Tabs:
    Run Analysis  — single-file analysis with full parameter coverage.
    Batch         — folder of files processed sequentially in one
                    subprocess (one spawn cost, N analyses).
    Compare       — N-group comparison with drag-and-drop folder loading,
                    multi-panel comparison figure + summary CSVs + PDF
                    report (output of sptpalm_analysis.compare_groups).
    Workspace     — embedded napari viewer for frame scrubbing + track
                    overlay; loads lazily on first activation.

Architecture notes:
    • All long-running analyses execute in a separate subprocess via
      multiprocessing.spawn.  The worker function lives in firefly_worker.py
      (a Qt-free module) so spawn doesn't re-import PySide6 into the child
      process — critical on Apple Silicon to keep Qt's Metal-backed window
      compositor from competing with PyTorch MPS for the unified-memory
      pool.
    • Stop button is three-stage: cooperative cancel → SIGTERM (5 s) →
      SIGKILL (8 s).  Guarantees the analysis halts within ~8 s.
    • Settings persisted via QSettings (per-user, OS-native location).
    • Crash reporter (sys.excepthook + threading.excepthook + Qt's
      qInstallMessageHandler) writes detailed text reports to the OS log
      directory and surfaces them via dialog.

NOTE on MPS environment variables (set below before any imports)
----------------------------------------------------------------
PyTorch's MPS allocator on macOS 26 / Apple M-series can leak memory across
operations even with explicit synchronize() + empty_cache() between stages.
The official mitigation is to disable the high-watermark allocator check
(PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0) and enable graceful CPU fallback for
unimplemented ops (PYTORCH_ENABLE_MPS_FALLBACK=1).  These MUST be set before
torch is imported anywhere in the process — putting them at the very top of
the entry-point module is the only reliable way.
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
_run_compare_in_subprocess  = firefly_worker.run_comparison


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
#  SPINBOX SUBCLASSES — no up/down buttons, no scroll-wheel value changes
# ══════════════════════════════════════════════════════════════════════════════
# Two issues with Qt's default spinboxes that surfaced during user testing:
#   1. The little up/down stepper buttons on the right edge clutter the
#      look at the small sizes used in our compact parameter form.
#   2. Scrolling the mouse wheel over a spinbox silently changes the value
#      — easy to do by accident when scrolling the sidebar past a control,
#      with no visual cue that the value just changed.
#
# These subclasses fix both by setting NoButtons + AlignCenter at construction
# and ignoring wheel events.  Wheel events bubble up to the parent (the
# QScrollArea) so the user can scroll the sidebar past them as expected.
class _QuietSpinBox(QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, e):
        # Pass to parent so the sidebar can scroll; don't change the value
        e.ignore()


class _QuietDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, e):
        e.ignore()


# ══════════════════════════════════════════════════════════════════════════════
#  COLLAPSIBLE SECTION — reusable accordion-style header + content panel
# ══════════════════════════════════════════════════════════════════════════════
class _CollapsibleSection(QtWidgets.QWidget):
    """A section with a clickable header (with ▶/▼ arrow) that toggles
    visibility of its content panel below.

    Usage:
        sec = _CollapsibleSection("My Section")
        form = QtWidgets.QFormLayout()
        sec.content_layout.addLayout(form)
        form.addRow("Field", widget)
        parent_layout.addWidget(sec)
    """
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QtWidgets.QToolButton()
        self._header.setObjectName("section_header")
        self._header.setText(f"▼   {title}")
        self._header.setCheckable(True)
        self._header.setChecked(True)
        self._header.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed)
        self._header.toggled.connect(self._on_toggled)
        outer.addWidget(self._header)

        self._content = QtWidgets.QFrame()
        self._content.setObjectName("section_content")
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 8, 10, 10)
        self._content_layout.setSpacing(6)
        outer.addWidget(self._content)

    def _on_toggled(self, checked: bool):
        self._header.setText(f"{'▼' if checked else '▶'}   {self._title}")
        self._content.setVisible(checked)

    @property
    def content_layout(self) -> QtWidgets.QVBoxLayout:
        return self._content_layout

    def set_expanded(self, expanded: bool):
        if self._header.isChecked() != expanded:
            self._header.setChecked(expanded)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS PANEL — shown after a run completes (replaces the figure canvas)
# ══════════════════════════════════════════════════════════════════════════════
class _ResultsPanel(QtWidgets.QFrame):
    """Compact "results" panel shown below the progress bar on each tab.

    Replaces the in-app matplotlib canvas — figures are now saved to disk
    only, per user preference.  After a run completes, this panel lists
    the saved files and offers a button to open the output folder in the
    system file manager.
    """
    def __init__(self, idle_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("results_panel")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        # Header line — big-ish status text
        self._headline = QtWidgets.QLabel(idle_text)
        self._headline.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 13px;")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._headline.setWordWrap(True)
        v.addWidget(self._headline)

        # Stats grid — populated post-run with key numbers (median D / α,
        # motion-class breakdown, cluster count, etc.).
        self._stats_grid = QtWidgets.QGridLayout()
        self._stats_grid.setColumnStretch(0, 0)
        self._stats_grid.setColumnStretch(1, 1)
        self._stats_grid.setHorizontalSpacing(16)
        self._stats_grid.setVerticalSpacing(4)
        stats_container = QtWidgets.QWidget()
        stats_container.setLayout(self._stats_grid)
        self._stats_container = stats_container
        self._stats_container.hide()
        v.addWidget(self._stats_container)

        # Output-folder row (visible only after a run)
        self._folder_row = QtWidgets.QWidget()
        fr = QtWidgets.QHBoxLayout(self._folder_row)
        fr.setContentsMargins(0, 0, 0, 0)
        self._folder_label = QtWidgets.QLabel("")
        self._folder_label.setStyleSheet(f"color: {_THEME['TXT']};")
        self._folder_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._folder_label.setWordWrap(True)
        fr.addWidget(self._folder_label, 1)
        self._open_btn = QtWidgets.QPushButton("Open folder")
        self._open_btn.clicked.connect(self._on_open_folder)
        fr.addWidget(self._open_btn)
        self._folder_row.hide()
        v.addWidget(self._folder_row)

        # File list (the saved CSVs / PDFs / PNGs)
        self._files = QtWidgets.QListWidget()
        self._files.setObjectName("results_files")
        self._files.setAlternatingRowColors(True)
        self._files.itemDoubleClicked.connect(self._on_file_dbl)
        self._files.hide()
        v.addWidget(self._files, stretch=1)

        # Trailing stretch when idle so the headline centres vertically
        self._stretch_when_idle = True
        v.addStretch(1)

        self._out_dir = ""

    def reset(self, idle_text: str = ""):
        if idle_text:
            self._headline.setText(idle_text)
        self._headline.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 13px;")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._folder_row.hide()
        self._files.clear()
        self._files.hide()
        self._clear_stats()
        self._stats_container.hide()
        self._out_dir = ""

    def _clear_stats(self):
        while self._stats_grid.count():
            item = self._stats_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_stat_row(self, row: int, label: str, value: str,
                      value_colour: str | None = None):
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 12px;")
        val = QtWidgets.QLabel(value)
        col = value_colour or _THEME['TXT']
        val.setStyleSheet(
            f"color: {col}; font-size: 13px; font-weight: 600;")
        val.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._stats_grid.addWidget(lbl, row, 0,
                                   Qt.AlignmentFlag.AlignLeft)
        self._stats_grid.addWidget(val, row, 1,
                                   Qt.AlignmentFlag.AlignLeft)

    def show_stats(self, summary: dict):
        """Populate the stats grid from a worker `summary` dict.

        Expected keys (all optional; missing keys are skipped):
            n_tracks, n_locs, median_d, median_alpha,
            motion_counts (dict), mobile_fraction, n_clusters,
            dwell_tau_s, frames, px_um, fi_s
        """
        self._clear_stats()
        if not summary:
            return
        r = 0

        def _fmt_int(n):    return f"{n:,}" if n is not None else "—"
        def _fmt_pct(f):    return f"{100 * f:.1f} %" if f is not None else "—"
        def _fmt_um2(d):    return f"{d:.4f} µm²/s" if d is not None else "—"
        def _fmt_alpha(a):  return f"{a:.3f}" if a is not None else "—"
        def _fmt_secs(s):   return f"{s:.2f} s" if s is not None else "—"

        # Counts ─────────────────────────────────────────────────────
        self._add_stat_row(r, "Trajectories",
                           _fmt_int(summary.get("n_tracks", 0))); r += 1
        self._add_stat_row(r, "Localisations",
                           _fmt_int(summary.get("n_locs", 0))); r += 1

        # Diffusion ──────────────────────────────────────────────────
        self._add_stat_row(r, "Median D",
                           _fmt_um2(summary.get("median_d"))); r += 1
        self._add_stat_row(r, "Median α",
                           _fmt_alpha(summary.get("median_alpha"))); r += 1
        mf = summary.get("mobile_fraction")
        if mf is not None:
            self._add_stat_row(r, "Mobile fraction (D > threshold)",
                               _fmt_pct(mf)); r += 1

        # Motion-class breakdown ─────────────────────────────────────
        motion_counts = summary.get("motion_counts") or {}
        if motion_counts:
            total = sum(motion_counts.values()) or 1
            # Standard order so the row is predictable
            order = ["Immobile", "Confined", "Brownian", "Directed", "Unknown"]
            colour_map = {
                "Immobile": _THEME['DANGER'],
                "Confined": _THEME['WARN'],
                "Brownian": _THEME['ACC'],
                "Directed": _THEME['SUCCESS'],
                "Unknown":  _THEME['TXT_MUTED'],
            }
            for cls in order:
                if cls in motion_counts:
                    n = motion_counts[cls]
                    self._add_stat_row(
                        r, f"  {cls}",
                        f"{n:,}  ({100 * n / total:.1f} %)",
                        value_colour=colour_map.get(cls)); r += 1

        # Secondary ──────────────────────────────────────────────────
        nc = summary.get("n_clusters", 0)
        if nc:
            self._add_stat_row(r, "DBSCAN clusters",
                               _fmt_int(nc)); r += 1
        dwell = summary.get("dwell_tau_s")
        if dwell is not None:
            self._add_stat_row(r, "Dwell time  τ",
                               _fmt_secs(dwell)); r += 1

        # Imaging metadata footer ────────────────────────────────────
        frames = summary.get("frames")
        if frames:
            self._add_stat_row(
                r, "Source movie",
                f"{frames:,} frames  |  "
                f"px = {summary.get('px_um', 0):.3f} µm  |  "
                f"fi = {summary.get('fi_s', 0):.3f} s",
                value_colour=_THEME['TXT_MUTED']); r += 1

        self._stats_container.show()

    def show_results(self, headline: str, out_dir: str,
                     files: list[str] | None = None):
        """Populate the panel with a completed run's outputs."""
        self._headline.setText(headline)
        self._headline.setStyleSheet(
            f"color: {_THEME['SUCCESS']}; font-size: 14px; font-weight: 600;")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._out_dir = out_dir
        if out_dir:
            self._folder_label.setText(out_dir)
            self._folder_row.show()

        self._files.clear()
        files = files or []
        # Auto-discover saved outputs if the caller didn't pass them
        if out_dir and os.path.isdir(out_dir) and not files:
            for sub in ("data", "firefly_extras", "figures", ""):
                d = os.path.join(out_dir, sub) if sub else out_dir
                if not os.path.isdir(d):
                    continue
                for name in sorted(os.listdir(d)):
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        files.append(full)
        for f in files:
            item = QtWidgets.QListWidgetItem(
                f"  {os.path.relpath(f, out_dir) if out_dir else f}")
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setToolTip(f)
            self._files.addItem(item)
        if files:
            self._files.show()

    def _on_open_folder(self):
        if self._out_dir and os.path.isdir(self._out_dir):
            _open_folder(self._out_dir)

    def _on_file_dbl(self, item: QtWidgets.QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.isfile(path):
            _open_folder(os.path.dirname(path))


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE TAB — folder-drop list + group card
# ══════════════════════════════════════════════════════════════════════════════
class _FolderDropList(QtWidgets.QListWidget):
    """QListWidget that accepts dropped folders (Qt-native, no tkinterdnd2).

    Files dropped onto the widget are added by their full path; non-folders
    are ignored.  Duplicates are silently de-duped.  Behaviour matches the
    Tk app's drag-and-drop on the Compare-tab cards.
    """
    folders_dropped = QtCore.Signal(list)   # list[str] of newly-added folders

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        # Subtle styling cue that this is a drop target
        self.setStyleSheet(
            "QListWidget { border: 1px dashed #777; border-radius: 4px; "
            "padding: 4px; }")

    # Qt drag-and-drop event handlers
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        if not e.mimeData().hasUrls():
            return
        existing = {self.item(i).text() for i in range(self.count())}
        added: list[str] = []
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if not path or not os.path.isdir(path):
                continue
            if path in existing:
                continue
            self.addItem(path)
            added.append(path)
            existing.add(path)
        if added:
            self.folders_dropped.emit(added)
        e.acceptProposedAction()


class _CompareGroupCard(QtWidgets.QGroupBox):
    """One row in the Compare tab — label, colour swatch, folder list,
    +/− buttons.  Emits `changed` whenever its contents change so the
    parent window can persist the state."""
    changed = QtCore.Signal()
    delete_requested = QtCore.Signal(object)   # self

    # Default colour palette — cycled through for new cards
    _DEFAULT_COLORS = ["#3b6ed8", "#f78166", "#56d364", "#d2a8ff",
                       "#ffa657", "#79c0ff"]

    def __init__(self, index: int, label: str = "", color: str = "",
                 parent=None):
        super().__init__(parent)
        if not label:
            label = ["Pre", "Post", "Group C", "Group D",
                     "Group E", "Group F"][min(index, 5)]
        if not color:
            color = self._DEFAULT_COLORS[index % len(self._DEFAULT_COLORS)]
        self._color = color
        self.setTitle(f"Group {index + 1}")

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # Top row: label edit + colour swatch + delete card
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Label"))
        self.e_label = QtWidgets.QLineEdit(label)
        self.e_label.textChanged.connect(lambda _: self.changed.emit())
        row.addWidget(self.e_label, 1)

        self.btn_color = QtWidgets.QPushButton(" ")
        self.btn_color.setFixedSize(28, 22)
        self._refresh_color_button()
        self.btn_color.clicked.connect(self._on_pick_color)
        self.btn_color.setToolTip("Pick a colour for this group's plots.")
        row.addWidget(self.btn_color)

        self.btn_delete = QtWidgets.QToolButton()
        self.btn_delete.setText("×")
        self.btn_delete.setToolTip("Remove this group.")
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self))
        row.addWidget(self.btn_delete)
        v.addLayout(row)

        # Folder list (drop target)
        self.lst_folders = _FolderDropList()
        self.lst_folders.setMinimumHeight(80)
        self.lst_folders.folders_dropped.connect(lambda _: self.changed.emit())
        v.addWidget(self.lst_folders, 1)

        # Add / Remove buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+ Add folder")
        self.btn_add.clicked.connect(self._on_add_folder)
        self.btn_remove = QtWidgets.QPushButton("− Remove")
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        self.lbl_count = QtWidgets.QLabel("0 folders")
        btn_row.addWidget(self.lbl_count)
        v.addLayout(btn_row)

        self.lst_folders.itemSelectionChanged.connect(lambda: self.changed.emit())
        self._refresh_count()

    # ── Helpers ────────────────────────────────────────────────────────────
    @property
    def color(self) -> str:
        return self._color

    def _refresh_color_button(self):
        self.btn_color.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #555;")

    def _refresh_count(self):
        n = self.lst_folders.count()
        self.lbl_count.setText(f"{n} folder{'s' if n != 1 else ''}")

    def _on_pick_color(self):
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._color), self, "Choose group colour")
        if col.isValid():
            self._color = col.name()
            self._refresh_color_button()
            self.changed.emit()

    def _on_add_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add folder to this group", os.path.expanduser("~"))
        if not path:
            return
        existing = {self.lst_folders.item(i).text()
                    for i in range(self.lst_folders.count())}
        if path not in existing:
            self.lst_folders.addItem(path)
            self._refresh_count()
            self.changed.emit()

    def _on_remove_selected(self):
        for it in reversed(self.lst_folders.selectedItems()):
            self.lst_folders.takeItem(self.lst_folders.row(it))
        self._refresh_count()
        self.changed.emit()

    def _on_clear(self):
        self.lst_folders.clear()
        self._refresh_count()
        self.changed.emit()

    def get_state(self) -> dict:
        """Return the group as the dict shape `compare_groups` expects."""
        folders = [self.lst_folders.item(i).text()
                   for i in range(self.lst_folders.count())]
        return {"label":  self.e_label.text().strip() or "Group",
                "color":  self._color,
                "folders": folders}

    def set_state(self, label: str, color: str, folders: list):
        self.e_label.setText(label)
        if color:
            self._color = color
            self._refresh_color_button()
        self.lst_folders.clear()
        for f in folders:
            self.lst_folders.addItem(str(f))
        self._refresh_count()


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

        # Top-level vertical: [header bar] / [sidebar | tabs]
        top = QtWidgets.QVBoxLayout(central)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)

        # ── Header banner ────────────────────────────────────────────────
        top.addWidget(self._build_header_banner())

        # ── Body: sidebar + tabs ─────────────────────────────────────────
        body = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        top.addWidget(body, stretch=1)

        # ── Sidebar ───────────────────────────────────────────────────────
        # Fixed-width left panel; the scrollable parameter list lives
        # inside it so the Start/Stop button can stay pinned at the bottom
        # regardless of how far the user has scrolled.  380 px is wide
        # enough to fit a [QLineEdit + Browse] row at typical font sizes
        # without the button clipping over the line edit's right edge.
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(380)
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
        self.btn_run.setObjectName("primary")  # picks up accent-fill QSS rule
        self.btn_run.setMinimumHeight(36)
        f = self.btn_run.font(); f.setBold(True); f.setPointSize(13)
        self.btn_run.setFont(f)
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_layout.addWidget(self.btn_run)
        sb_outer.addWidget(btn_container)

        layout.addWidget(sidebar)

        # ── Tabs ──────────────────────────────────────────────────────────
        # Tab order: Import → Analysis → Compare → Visualise
        # Import consolidates input/output picking for both single-file
        # and batch mode (replaces the old sidebar Input/Output section +
        # the standalone Batch tab).  Analysis is purely status + result
        # summary.  Compare and Visualise sit at the end.
        self.tabs = QtWidgets.QTabWidget()
        self._build_import_tab()
        self._build_analysis_tab()
        self._build_compare_tab()
        self._build_visualise_tab()
        layout.addWidget(self.tabs, stretch=1)

        # ── Console dock (hidden by default) ──────────────────────────────
        # One shared console for all tabs.  Stays hidden until the user
        # clicks the Console button in the status bar.  Log lines accumulate
        # in the widget even when the dock is hidden, so opening it shows
        # the complete history.
        self._build_console_dock()

        # Status bar with a permanent "Console" toggle button on the right
        self.btn_show_console = QtWidgets.QToolButton()
        self.btn_show_console.setText("Console")
        self.btn_show_console.setCheckable(True)
        self.btn_show_console.setToolTip(
            "Show/hide the debug console.  Captures every log line from "
            "all stages — useful for diagnosing problems but normally not "
            "needed; the progress bar tells you what's happening.")
        self.btn_show_console.clicked.connect(self._toggle_console)
        self.statusBar().addPermanentWidget(self.btn_show_console)
        self.statusBar().showMessage("Ready")

    def _build_header_banner(self) -> QtWidgets.QWidget:
        """Thin header strip:  FIREFLY                Fluorescence Inference & Reconstruction Engine | By Jacob Levers"""
        bar = QtWidgets.QFrame()
        bar.setObjectName("header_bar")
        bar.setStyleSheet(
            f"QFrame#header_bar {{ background-color: {_THEME['PANEL']}; "
            f"border-bottom: 1px solid {_THEME['BORDER']}; }}"
            # The right-side label is a transparent QLabel inside the
            # styled frame; explicitly null out its border so it doesn't
            # inherit the frame's border-bottom rule from the global QSS.
            f"QFrame#header_bar QLabel {{ background: transparent; border: none; }}"
        )
        bar.setFixedHeight(46)
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(18, 0, 18, 0)
        h.setSpacing(8)

        # Left: FIREFLY logo
        logo = QtWidgets.QLabel("FIREFLY")
        logo.setStyleSheet(
            f"color: {_THEME['ACC']}; font-weight: 800; "
            f"font-size: 22px; letter-spacing: 2px;"
        )
        logo.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        h.addWidget(logo)

        h.addStretch(1)

        # Right: tagline + author on ONE line, joined with a pipe.  Using
        # rich-text formatting on a single QLabel sidesteps the nested-
        # container-border issue and looks tidier than two stacked labels.
        right = QtWidgets.QLabel(
            f"<span style='color:{_THEME['TXT']};font-weight:600;'>"
            f"Fluorescence Inference &amp; Reconstruction Engine"
            f"</span>"
            f"<span style='color:{_THEME['TXT_MUTED']};'>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;By Jacob Levers"
            f"</span>"
        )
        right.setTextFormat(Qt.TextFormat.RichText)
        right.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        right.setStyleSheet("font-size: 12px;")
        h.addWidget(right)

        return bar

    def _build_console_dock(self):
        """Create the dockable console.  Hidden by default — toggled via
        the status-bar Console button.  Shared by all tabs, so log lines
        from any analysis stage land in one place.
        """
        self._console_dock = QtWidgets.QDockWidget("Console", self)
        self._console_dock.setObjectName("console_dock")
        self._console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea)

        self.console_log = QtWidgets.QPlainTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setMaximumBlockCount(20000)
        mono = QtGui.QFont("Menlo, Consolas, monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.console_log.setFont(mono)
        self._console_dock.setWidget(self.console_log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self._console_dock)
        # Hidden by default; user has to click Console to open
        self._console_dock.hide()
        # Keep the toggle button in sync when the dock is closed via its
        # own ✕ button
        self._console_dock.visibilityChanged.connect(self._on_console_visibility)

    def _toggle_console(self):
        """Show / hide the console dock from the status-bar button."""
        if self._console_dock.isVisible():
            self._console_dock.hide()
        else:
            self._console_dock.show()

    def _on_console_visibility(self, visible: bool):
        """Keep the status-bar toggle button's checked state in sync."""
        try:
            self.btn_show_console.setChecked(visible)
        except AttributeError:
            pass

    # ── Tiny helpers for compact widget construction ──────────────────────
    @staticmethod
    def _make_form_section(title: str):
        """Return (CollapsibleSection, QFormLayout) for use in the sidebar.
        The form layout is already wired into the section's content; the
        caller just adds rows."""
        sec = _CollapsibleSection(title)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        sec.content_layout.addLayout(form)
        return sec, form

    @staticmethod
    def _make_vbox_section(title: str):
        """Return (CollapsibleSection, QVBoxLayout)."""
        sec = _CollapsibleSection(title)
        vb = QtWidgets.QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(6)
        sec.content_layout.addLayout(vb)
        return sec, vb

    @staticmethod
    def _spin_int(value: int, lo: int, hi: int, step: int = 1,
                  tip: str = "") -> "QtWidgets.QSpinBox":
        s = _QuietSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setValue(value)
        if tip: s.setToolTip(tip)
        return s

    @staticmethod
    def _spin_dbl(value: float, lo: float, hi: float, step: float = 0.01,
                  decimals: int = 3, tip: str = "") -> "QtWidgets.QDoubleSpinBox":
        s = _QuietDoubleSpinBox()
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

        # NOTE: Input/output pickers used to live here but moved to the
        # Import tab in v2.1 — see `_build_import_tab`.  The QLineEdit
        # widgets `self.e_file` and `self.e_outdir` are still owned by
        # this MainWindow (created in the Import tab), so the rest of
        # the worker code that reads them keeps working unchanged.

        # ── Imaging metadata ──────────────────────────────────────────────
        # File-embedded values are used by default; checkbox enables manual
        # override.  Matches the Tk app's behaviour.
        sec, gl = self._make_form_section("Imaging metadata")

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
        layout.addWidget(sec)

        # ── Preprocessing ─────────────────────────────────────────────────
        sec, gl = self._make_form_section("Preprocessing")
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
        layout.addWidget(sec)

        # ── Detection ─────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Detection")
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
        layout.addWidget(sec)

        # ── Linking ───────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Linking")
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
        layout.addWidget(sec)

        # ── Diffusion fit + motion classification ─────────────────────────
        sec, gl = self._make_form_section("Diffusion & motion classification")
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
        layout.addWidget(sec)

        # ── ROI ───────────────────────────────────────────────────────────
        sec, gl = self._make_form_section("ROI")
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
        layout.addWidget(sec)

        # ── Drift correction ──────────────────────────────────────────────
        sec, gl = self._make_form_section("Drift correction")
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
        layout.addWidget(sec)

        # ── Clustering ────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Clustering (DBSCAN)")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_cluster_eps_nm = self._spin_dbl(50.0, 5.0, 1000.0, 5.0, decimals=1,
            tip="DBSCAN neighbourhood radius (nm). Two localisations are in the\n"
                "same cluster if they're within this distance.")
        gl.addRow("eps (nm)", self.s_cluster_eps_nm)
        self.s_cluster_min_samples = self._spin_int(10, 2, 100,
            tip="Minimum localisations to form a DBSCAN cluster. Lower = more\n"
                "clusters detected but noisier; higher = stricter.")
        gl.addRow("min samples", self.s_cluster_min_samples)
        layout.addWidget(sec)

        # ── Performance ───────────────────────────────────────────────────
        sec, gl = self._make_form_section(f"Performance  —  {N_CPUS} cores")
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
        layout.addWidget(sec)

        layout.addStretch(1)

    def _build_import_tab(self):
        """Import tab — single-source-of-truth for input/output config.

        Mode toggle switches the visible sub-panel between:
          • Single file — pick a .czi/.tif + an output folder
          • Batch       — pick a folder of files, choose which to process,
                           output goes to <folder>/batch_results/<stem>/

        The Start button (sidebar) reads from this tab and dispatches to
        the appropriate worker; after starting, the app auto-switches to
        the Analysis tab so the user sees progress immediately.
        """
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # ── Mode toggle ───────────────────────────────────────────────────
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        self.r_mode_single = QtWidgets.QRadioButton("Single file")
        self.r_mode_batch  = QtWidgets.QRadioButton("Batch (folder)")
        self.r_mode_single.setChecked(True)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.addButton(self.r_mode_single, 0)
        self._mode_group.addButton(self.r_mode_batch, 1)
        self.r_mode_single.toggled.connect(self._on_import_mode_changed)
        mode_row.addWidget(self.r_mode_single)
        mode_row.addWidget(self.r_mode_batch)
        mode_row.addStretch(1)
        v.addLayout(mode_row)

        # ── Single-file sub-panel ─────────────────────────────────────────
        self._single_panel = QtWidgets.QGroupBox("Single file")
        sg = QtWidgets.QFormLayout(self._single_panel)
        sg.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        row = QtWidgets.QHBoxLayout()
        self.e_file = QtWidgets.QLineEdit()
        self.e_file.setPlaceholderText("Browse for a .czi / .tif file…")
        b1 = QtWidgets.QPushButton("Browse")
        b1.clicked.connect(self._on_browse_file)
        row.addWidget(self.e_file); row.addWidget(b1)
        w_file = QtWidgets.QWidget(); w_file.setLayout(row)
        sg.addRow("Input file", w_file)

        row = QtWidgets.QHBoxLayout()
        self.e_outdir = QtWidgets.QLineEdit()
        self.e_outdir.setPlaceholderText("Defaults to input file's folder")
        b2 = QtWidgets.QPushButton("Browse")
        b2.clicked.connect(self._on_browse_outdir)
        row.addWidget(self.e_outdir); row.addWidget(b2)
        w_out = QtWidgets.QWidget(); w_out.setLayout(row)
        sg.addRow("Output folder", w_out)
        v.addWidget(self._single_panel)

        # ── Batch sub-panel ───────────────────────────────────────────────
        self._batch_panel = QtWidgets.QGroupBox("Batch")
        bg = QtWidgets.QVBoxLayout(self._batch_panel)
        bg.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Input folder"))
        self.e_batch_folder = QtWidgets.QLineEdit()
        self.e_batch_folder.setPlaceholderText(
            "Pick a folder containing .czi / .tif files…")
        btn_pick = QtWidgets.QPushButton("Browse")
        btn_pick.clicked.connect(self._on_batch_pick_folder)
        btn_refresh = QtWidgets.QPushButton("↻ Rescan")
        btn_refresh.setToolTip("Re-scan the folder for input files.")
        btn_refresh.clicked.connect(self._on_batch_rescan)
        row.addWidget(self.e_batch_folder, 1)
        row.addWidget(btn_pick)
        row.addWidget(btn_refresh)
        bg.addLayout(row)

        bg.addWidget(QtWidgets.QLabel(
            "Files to process  (uncheck to skip):"))
        self.lst_batch_files = QtWidgets.QListWidget()
        self.lst_batch_files.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_batch_files.setMinimumHeight(220)
        bg.addWidget(self.lst_batch_files, stretch=1)

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
        bg.addLayout(sel_row)

        # Where the batch outputs land
        self.lbl_batch_output_path = QtWidgets.QLabel(
            "Output → (pick an input folder first)")
        self.lbl_batch_output_path.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        bg.addWidget(self.lbl_batch_output_path)

        v.addWidget(self._batch_panel, stretch=1)

        # Start visible state: single mode shown, batch hidden
        self._batch_panel.hide()

        self.tabs.addTab(tab, "Import")

    def _on_import_mode_changed(self, single_checked: bool):
        self._single_panel.setVisible(single_checked)
        self._batch_panel.setVisible(not single_checked)

    def _build_analysis_tab(self):
        """Analysis tab — pure status display.  Stage label, progress bar,
        and a results panel that fills in after a run completes."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        self.run_stage_label = QtWidgets.QLabel("Idle")
        self.run_stage_label.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-weight: 600; padding: 2px 0;")
        v.addWidget(self.run_stage_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        v.addWidget(self.progress_bar)

        # Mirror widgets for batch runs.  Single set of widgets — one
        # set of state per tab is excessive when Analysis is universal.
        # Use the same stage label + progress bar for batch by aliasing.
        self.batch_stage_label = self.run_stage_label
        self.batch_progress    = self.progress_bar

        # Per-file mini-progress for batch.  Shown only during a batch
        # run (sits between the overall progress bar and the results
        # panel).  Lets the user see "currently processing X" even when
        # the overall % only ticks once per file.
        self.batch_subprogress = QtWidgets.QProgressBar()
        self.batch_subprogress.setRange(0, 100)
        self.batch_subprogress.setValue(0)
        self.batch_subprogress.setTextVisible(True)
        self.batch_subprogress.setFormat("")
        self.batch_subprogress.hide()
        v.addWidget(self.batch_subprogress)

        self.run_results = _ResultsPanel(
            "Results will appear here after analysis.")
        v.addWidget(self.run_results, stretch=1)

        self.tabs.addTab(tab, "Analysis")

    # ── Batch helpers (Import-tab batch sub-panel) ───────────────────────
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
        # Show the user where the per-file outputs will land
        folder = self.e_batch_folder.text().strip()
        if folder:
            self.lbl_batch_output_path.setText(
                f"Output → {os.path.join(folder, 'batch_results')}/<stem>/")
        else:
            self.lbl_batch_output_path.setText(
                "Output → (pick an input folder first)")

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

    # ══════════════════════════════════════════════════════════════════════
    #  COMPARE TAB
    # ══════════════════════════════════════════════════════════════════════
    COMPARE_PANELS = [
        ("msd",            "Ensemble MSD"),
        ("auc",            "MSD AUC bars"),
        ("logd_dist",      "log10(D) distributions"),
        ("mob_immob",      "Mobile / immobile ratio"),
        ("motion_classes", "Motion-class fractions"),
        ("track_length",   "Track-length distribution"),
        ("jdd",            "Jump-distance distribution"),
        ("dwell_cdf",      "Dwell-time CDF"),
        ("turning_angles", "Turning-angle distribution"),
        ("radial_dist",    "Radial distribution of |θ|"),
    ]
    COMPARE_MAX_GROUPS = 6

    def _build_compare_tab(self):
        """Compare tab: N≥2 groups of analysis-output folders → comparison
        figure + summary CSV + stats CSV + multi-page PDF report."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # ── Comparison settings row ───────────────────────────────────────
        settings = QtWidgets.QGroupBox("Comparison settings")
        sg = QtWidgets.QGridLayout(settings)
        sg.setVerticalSpacing(4)
        sg.setHorizontalSpacing(8)

        sg.addWidget(QtWidgets.QLabel("Output folder"), 0, 0)
        self.e_cmp_outdir = QtWidgets.QLineEdit()
        self.e_cmp_outdir.setPlaceholderText(
            "Where to save the comparison figure + CSVs + PDF report")
        btn_cmp_out = QtWidgets.QPushButton("Browse")
        btn_cmp_out.clicked.connect(self._on_cmp_browse_outdir)
        sg.addWidget(self.e_cmp_outdir, 0, 1)
        sg.addWidget(btn_cmp_out, 0, 2)

        sg.addWidget(QtWidgets.QLabel("Output name"), 1, 0)
        self.e_cmp_stem = QtWidgets.QLineEdit("comparison")
        self.e_cmp_stem.setToolTip(
            "Prefix for the saved files (figure.png, summary.csv, "
            "stats.csv, report.pdf).")
        sg.addWidget(self.e_cmp_stem, 1, 1, 1, 2)

        sg.addWidget(QtWidgets.QLabel("Figure theme"), 2, 0)
        self.c_cmp_theme = QtWidgets.QComboBox()
        self.c_cmp_theme.addItems(["Dark", "Light", "Publication"])
        sg.addWidget(self.c_cmp_theme, 2, 1, 1, 2)

        self.c_cmp_pdf = QtWidgets.QCheckBox(
            "Generate multi-page PDF report (figure + parameters + stats)")
        self.c_cmp_pdf.setChecked(True)
        sg.addWidget(self.c_cmp_pdf, 3, 0, 1, 3)

        v.addWidget(settings)

        # ── Panel selector ────────────────────────────────────────────────
        panels_grp = QtWidgets.QGroupBox("Panels to include")
        pg = QtWidgets.QGridLayout(panels_grp)
        self._cmp_panel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for i, (key, label) in enumerate(self.COMPARE_PANELS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            self._cmp_panel_checkboxes[key] = cb
            pg.addWidget(cb, i // 2, i % 2)
        v.addWidget(panels_grp)

        # ── Group cards (scrollable) ──────────────────────────────────────
        groups_area_label = QtWidgets.QLabel(
            "Groups  —  drop folders directly onto a card to add them, "
            "or use the buttons:")
        v.addWidget(groups_area_label)

        groups_scroll = QtWidgets.QScrollArea()
        groups_scroll.setWidgetResizable(True)
        groups_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        groups_inner = QtWidgets.QWidget()
        self._cmp_groups_layout = QtWidgets.QVBoxLayout(groups_inner)
        self._cmp_groups_layout.setContentsMargins(0, 0, 0, 0)
        self._cmp_groups_layout.setSpacing(6)
        self._cmp_groups_layout.addStretch(1)   # pushes cards to top
        groups_scroll.setWidget(groups_inner)
        v.addWidget(groups_scroll, stretch=1)

        self._cmp_group_cards: list[_CompareGroupCard] = []
        # Seed with the two default groups (Pre / Post) — matches the Tk app
        self._cmp_add_group()
        self._cmp_add_group()

        # ── Action row ────────────────────────────────────────────────────
        actions = QtWidgets.QHBoxLayout()
        self.btn_cmp_add_group = QtWidgets.QPushButton("+ Add group")
        self.btn_cmp_add_group.clicked.connect(self._cmp_add_group)
        actions.addWidget(self.btn_cmp_add_group)
        actions.addStretch(1)
        self.btn_cmp_run = QtWidgets.QPushButton("Generate comparison")
        self.btn_cmp_run.setMinimumHeight(32)
        self.btn_cmp_run.clicked.connect(self._on_run_clicked)
        actions.addWidget(self.btn_cmp_run)
        v.addLayout(actions)

        # ── Stage label + progress bar + figure ───────────────────────────
        self.cmp_stage_label = QtWidgets.QLabel("Idle")
        self.cmp_stage_label.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-weight: 600; padding: 2px 0;")
        v.addWidget(self.cmp_stage_label)

        self.cmp_progress = QtWidgets.QProgressBar()
        self.cmp_progress.setRange(0, 100); self.cmp_progress.setValue(0)
        self.cmp_progress.setFormat("Ready")
        v.addWidget(self.cmp_progress)

        # Results panel (figure is saved to disk only — view it externally
        # or in the Workspace tab if you want to overlay tracks).
        self.cmp_results = _ResultsPanel(
            "Comparison results will appear here after generation.")
        v.addWidget(self.cmp_results, stretch=2)

        self.tabs.addTab(tab, "Compare")

    def _on_cmp_browse_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose comparison output folder",
            self.e_cmp_outdir.text() or os.path.expanduser("~"))
        if path:
            self.e_cmp_outdir.setText(path)

    def _cmp_add_group(self):
        if len(self._cmp_group_cards) >= self.COMPARE_MAX_GROUPS:
            QtWidgets.QMessageBox.information(
                self, "Max groups reached",
                f"At most {self.COMPARE_MAX_GROUPS} groups can be compared "
                "at once.")
            return
        idx = len(self._cmp_group_cards)
        card = _CompareGroupCard(idx)
        card.delete_requested.connect(self._cmp_remove_group)
        # Insert before the stretch element at the end
        self._cmp_groups_layout.insertWidget(idx, card)
        self._cmp_group_cards.append(card)

    def _cmp_remove_group(self, card: _CompareGroupCard):
        if len(self._cmp_group_cards) <= 2:
            QtWidgets.QMessageBox.information(
                self, "Minimum groups",
                "Need at least 2 groups for a comparison.")
            return
        self._cmp_group_cards.remove(card)
        self._cmp_groups_layout.removeWidget(card)
        card.deleteLater()
        # Re-number remaining cards
        for i, c in enumerate(self._cmp_group_cards):
            c.setTitle(f"Group {i + 1}")

    def _cmp_collect_groups(self) -> list[dict]:
        """Return the groups list in the shape `compare_groups` expects."""
        return [card.get_state() for card in self._cmp_group_cards]

    # ══════════════════════════════════════════════════════════════════════
    #  WORKSPACE TAB  (Napari)
    # ══════════════════════════════════════════════════════════════════════
    def _build_visualise_tab(self):
        """Build the Visualise tab — toolbar + lazy-loaded napari viewer.

        Napari is imported lazily on first activation so a missing dep
        doesn't block the rest of FIREFLY from launching.  If the import
        succeeds, the viewer is embedded into this tab.  If it fails, the
        tab shows a clear placeholder with install instructions, and the
        rest of the app keeps working.
        """
        self._napari_viewer = None         # populated lazily
        self._workspace_initialised = False

        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # ── Toolbar row ───────────────────────────────────────────────────
        toolbar = QtWidgets.QHBoxLayout()
        self.btn_ws_load_stack = QtWidgets.QPushButton("Load image stack…")
        self.btn_ws_load_stack.setToolTip(
            "Open a .czi or .tif file as an Image layer in napari.")
        self.btn_ws_load_stack.clicked.connect(self._ws_on_load_stack)
        toolbar.addWidget(self.btn_ws_load_stack)

        self.btn_ws_load_tracks = QtWidgets.QPushButton("Load tracks…")
        self.btn_ws_load_tracks.setToolTip(
            "Open a FIREFLY trajectories CSV as a Tracks layer overlay.")
        self.btn_ws_load_tracks.clicked.connect(self._ws_on_load_tracks)
        toolbar.addWidget(self.btn_ws_load_tracks)

        self.btn_ws_load_run = QtWidgets.QPushButton("Load analysis run…")
        self.btn_ws_load_run.setToolTip(
            "Pick a FIREFLY run output folder.  Auto-loads the original\n"
            "stack and overlays the trajectories.csv as a Tracks layer.")
        self.btn_ws_load_run.clicked.connect(self._ws_on_load_run)
        toolbar.addWidget(self.btn_ws_load_run)

        toolbar.addStretch(1)

        self.c_ws_auto = QtWidgets.QCheckBox("Auto-load after analysis")
        self.c_ws_auto.setChecked(False)
        self.c_ws_auto.setToolTip(
            "When checked, the Workspace tab loads the stack + tracks\n"
            "automatically after a Run-Analysis completes.\n"
            "Off by default — large stacks can use a lot of GPU memory in\n"
            "napari and slow the rest of FIREFLY down.")
        toolbar.addWidget(self.c_ws_auto)
        v.addLayout(toolbar)

        # ── Viewer container ─────────────────────────────────────────────
        # Filled lazily on first tab activation.
        self._ws_container = QtWidgets.QWidget()
        self._ws_container_layout = QtWidgets.QVBoxLayout(self._ws_container)
        self._ws_container_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder until napari is loaded
        self._ws_placeholder = QtWidgets.QLabel(
            "napari viewer will appear here when this tab is first opened.\n\n"
            "If napari isn't installed, run:\n"
            "    pip install \"napari[pyside6]>=0.4.19\"\n"
            "and restart FIREFLY.")
        self._ws_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ws_placeholder.setStyleSheet(
            "color: #888; padding: 40px; font-size: 13px;")
        self._ws_container_layout.addWidget(self._ws_placeholder)

        v.addWidget(self._ws_container, stretch=1)

        self.tabs.addTab(tab, "Visualise")

        # Lazy-init when the tab is first switched to.
        self.tabs.currentChanged.connect(self._ws_maybe_init)

    def _ws_maybe_init(self, idx: int):
        """If the user just switched to the Workspace tab, try to embed
        a napari viewer.  Idempotent — only the first switch actually
        does work."""
        if self._workspace_initialised:
            return
        if self.tabs.tabText(idx) != "Visualise":
            return
        self._workspace_initialised = True   # mark even on failure — don't retry
        self._ws_init_viewer()

    def _ws_init_viewer(self):
        """Try to import napari and embed its viewer into the tab.
        Shows a clear error message in the placeholder on failure."""
        try:
            import napari
        except Exception as exc:
            self._ws_placeholder.setText(
                f"napari failed to import:\n\n  {type(exc).__name__}: {exc}\n\n"
                f"To enable the Workspace tab, run:\n"
                f"    pip install \"napari[pyside6]>=0.4.19\"\n"
                f"and restart FIREFLY.")
            self._ws_placeholder.setStyleSheet(
                "color: #f78166; padding: 40px; font-size: 13px;")
            return

        try:
            # Embedding pattern: create a Viewer with show=False, then
            # take its underlying QtMainWindow as our embedded widget.
            # `viewer.window._qt_window` is the documented internal handle
            # that's been stable across napari 0.4.x.
            viewer = napari.Viewer(show=False)
            qt_window = viewer.window._qt_window
            # Replace the placeholder with the viewer widget
            self._ws_container_layout.removeWidget(self._ws_placeholder)
            self._ws_placeholder.deleteLater()
            self._ws_container_layout.addWidget(qt_window)
            self._napari_viewer = viewer
        except Exception as exc:
            # Replace placeholder text with the real error
            self._ws_placeholder.setText(
                f"napari is installed but the embedded viewer couldn't start:\n\n"
                f"  {type(exc).__name__}: {exc}\n\n"
                f"This is sometimes caused by a napari version mismatch with\n"
                f"PySide6.  Try:\n"
                f"    pip install --upgrade \"napari[pyside6]>=0.4.19,<0.5\"")
            self._ws_placeholder.setStyleSheet(
                "color: #f78166; padding: 40px; font-size: 13px;")
            import traceback as _tb
            print(f"[FIREFLY] napari embed failed:\n{_tb.format_exc()}",
                  file=sys.stderr)

    def _ws_viewer_or_warn(self):
        """Return the embedded napari viewer or None, with a UI warning if
        unavailable."""
        if self._napari_viewer is None:
            QtWidgets.QMessageBox.warning(
                self, "Workspace not ready",
                "The napari viewer hasn't initialised on this machine.\n"
                "See the Workspace tab for details.")
            return None
        return self._napari_viewer

    def _ws_on_load_stack(self):
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load image stack",
            self.e_file.text() or os.path.expanduser("~"),
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if not path:
            return
        self._ws_load_stack_path(path)

    def _ws_load_stack_path(self, path: str):
        """Load `path` as an Image layer using FIREFLY's loader.  Heavy
        ops happen on the GUI thread — fine for small/medium files; large
        stacks block the UI briefly (acceptable for an interactive
        inspect-this-file workflow)."""
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            from sptpalm_analysis import load_file
            self.statusBar().showMessage(f"Loading {os.path.basename(path)} into napari…")
            stack, _, _ = load_file(path, channel=0)
            v.add_image(stack, name=os.path.basename(path),
                        colormap="gray", blending="translucent_no_depth")
            self.statusBar().showMessage(
                f"Loaded {len(stack):,} frames into napari", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load {os.path.basename(path)}:\n\n{exc}")

    def _ws_on_load_tracks(self):
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load tracks CSV",
            self.e_outdir.text() or os.path.expanduser("~"),
            "Tracks CSV (*trajectories.csv);;All CSVs (*.csv)")
        if not path:
            return
        self._ws_load_tracks_path(path)

    def _ws_load_tracks_path(self, csv_path: str):
        """Read a trajectories CSV and add as a napari Tracks layer.

        FIREFLY's trajectories.csv has columns particle, frame, x, y.
        napari Tracks expects (track_id, t, [z,] y, x) per row.
        """
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            need = {"particle", "frame", "x", "y"}
            missing = need - set(df.columns)
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {sorted(missing)}")
            data = df[["particle", "frame", "y", "x"]].values.astype(float)
            v.add_tracks(data, name=os.path.basename(csv_path),
                         blending="opaque")
            self.statusBar().showMessage(
                f"Loaded {df['particle'].nunique():,} tracks "
                f"({len(df):,} points) into napari", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load tracks from {os.path.basename(csv_path)}:\n\n{exc}")

    def _ws_on_load_run(self):
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        run_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pick a FIREFLY run output folder",
            self.e_outdir.text() or os.path.expanduser("~"))
        if not run_dir:
            return
        self._ws_load_run_folder(run_dir)

    def _ws_load_run_folder(self, run_dir: str):
        """Load a complete FIREFLY analysis run:  finds the stack via the
        params.json (if present) and the matching trajectories.csv from
        firefly_extras/."""
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            import json
            extras_dir = os.path.join(run_dir, "firefly_extras")
            if not os.path.isdir(extras_dir):
                raise FileNotFoundError(
                    f"No firefly_extras/ subfolder in {run_dir}")
            # Find the params.json (any *_params.json)
            params_files = [f for f in os.listdir(extras_dir)
                            if f.endswith("_params.json")]
            stack_path = None
            stem = None
            if params_files:
                with open(os.path.join(extras_dir, params_files[0])) as fh:
                    params = json.load(fh)
                stack_path = params.get("input_file") or params.get("stem")
                stem = params_files[0][:-len("_params.json")]
            # Fallback: derive from trajectories filename
            if not stem:
                tr_files = [f for f in os.listdir(extras_dir)
                            if f.endswith("_trajectories.csv")]
                if tr_files:
                    stem = tr_files[0][:-len("_trajectories.csv")]
            if not stem:
                raise FileNotFoundError(
                    "Couldn't determine the run's stem (no params.json or "
                    "trajectories.csv found).")

            tracks_path = os.path.join(extras_dir, f"{stem}_trajectories.csv")
            if not os.path.isfile(tracks_path):
                raise FileNotFoundError(
                    f"Missing {os.path.basename(tracks_path)}")

            # If we have a recorded input-file path that still exists, load
            # it as an image layer.  Otherwise just load the tracks (still
            # useful — user can drop the stack later).
            if stack_path and os.path.isfile(stack_path):
                self._ws_load_stack_path(stack_path)
            else:
                self.statusBar().showMessage(
                    "Tracks loaded; original input stack not found.", 5000)
            self._ws_load_tracks_path(tracks_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load run {os.path.basename(run_dir)}:\n\n{exc}")

    def _ws_auto_load_after_run(self, payload: dict):
        """Called from _handle_done when a single-file run finishes.
        Loads the result into napari if the user has toggled auto-load."""
        if not self.c_ws_auto.isChecked():
            return
        out_dir = payload.get("out_dir")
        if not out_dir:
            return
        # Ensure the viewer is initialised before we try to use it
        if not self._workspace_initialised:
            # Force the lazy init now
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Visualise":
                    self._workspace_initialised = True
                    self._ws_init_viewer()
                    break
        if self._napari_viewer is None:
            return   # napari not available
        try:
            self._ws_load_run_folder(out_dir)
        except Exception:
            pass

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
            ("analysis/batch_folder",    self.e_batch_folder,    "text"),
            ("analysis/mode_batch",      self.r_mode_batch,      "check", _bool_cast),

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

            # ── Compare tab ───────────────────────────────────────────────
            ("compare/outdir",           self.e_cmp_outdir,      "text"),
            ("compare/stem",             self.e_cmp_stem,        "text"),
            ("compare/theme",            self.c_cmp_theme,       "combo"),
            ("compare/pdf_report",       self.c_cmp_pdf,         "check", _bool_cast),

            # ── Workspace tab ─────────────────────────────────────────────
            ("workspace/auto_load",      self.c_ws_auto,         "check", _bool_cast),
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

        # Import-tab mode visibility — show the right sub-panel based on
        # the restored mode_batch flag.  Also re-scan the batch folder
        # if one was previously selected so the file list re-populates.
        try:
            if self.r_mode_batch.isChecked():
                self._single_panel.hide()
                self._batch_panel.show()
            else:
                self._single_panel.show()
                self._batch_panel.hide()
            folder = self.e_batch_folder.text().strip()
            if folder and os.path.isdir(folder):
                self._batch_rescan(folder)
        except Exception:
            pass

        # Compare-tab group cards (label/color/folders) — JSON blob
        try:
            import json
            blob = s.value("compare/groups", type=str)
            if blob:
                data = json.loads(blob)
                if isinstance(data, list) and len(data) >= 2:
                    # Replace existing cards with restored ones
                    while len(self._cmp_group_cards) > 0:
                        card = self._cmp_group_cards.pop()
                        self._cmp_groups_layout.removeWidget(card)
                        card.deleteLater()
                    for i, g in enumerate(data[:self.COMPARE_MAX_GROUPS]):
                        self._cmp_add_group()
                        self._cmp_group_cards[-1].set_state(
                            g.get("label", f"Group {i+1}"),
                            g.get("color", ""),
                            g.get("folders", []))
        except Exception:
            pass

        # Compare-tab panel checkbox states
        try:
            for key, cb in self._cmp_panel_checkboxes.items():
                v = s.value(f"compare/panel_{key}")
                if v is not None:
                    cb.setChecked(_bool_cast(v))
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

        # Compare-tab group cards — serialised as JSON
        try:
            import json
            blob = json.dumps([c.get_state() for c in self._cmp_group_cards])
            s.setValue("compare/groups", blob)
        except Exception:
            pass

        # Compare-tab panel checkbox states
        try:
            for key, cb in self._cmp_panel_checkboxes.items():
                s.setValue(f"compare/panel_{key}", bool(cb.isChecked()))
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

            # Surface in the shared console + status bar so the user
            # knows their click registered (without forcing them to open
            # the console panel).
            self.console_log.appendPlainText(
                "\n── Stop requested.  Waiting for the current stage to reach "
                "a checkpoint (up to 5 s); will force-terminate if it doesn't.")
            self.statusBar().showMessage("Stop requested — waiting for current stage…")
            return

        # Dispatch.  The Compare-tab's own Generate button overrides; the
        # sidebar Start button uses the Import-tab mode (Single / Batch),
        # OR if the active tab is Compare, runs the comparison.
        sender = self.sender()
        if sender is getattr(self, "btn_cmp_run", None):
            self._start_compare_run()
            return
        active_tab_label = self.tabs.tabText(self.tabs.currentIndex())
        if active_tab_label.startswith("Compare"):
            self._start_compare_run()
        elif self.r_mode_batch.isChecked():
            self._start_batch_run()
        else:
            self._start_single_run()

    def _switch_to_tab(self, label: str):
        """Switch the central tab widget to the tab whose visible text
        starts with `label` (e.g. "Analysis" → finds "Analysis")."""
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).startswith(label):
                self.tabs.setCurrentIndex(i)
                return

    def _start_single_run(self):
        fpath = self.e_file.text().strip()
        if not fpath or not os.path.isfile(fpath):
            QtWidgets.QMessageBox.warning(
                self, "No file",
                "Pick an input file on the Import tab first.")
            self._switch_to_tab("Import")
            return
        # Auto-switch to the Analysis tab so the user sees progress
        self._switch_to_tab("Analysis")

        params = self._build_params_for_file(
            fpath, self.e_outdir.text().strip() or None)

        # Persist before the long-running task in case of crash/abort.
        try:
            self._save_settings()
        except Exception:
            pass

        # Clear UI for new run
        self.console_log.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting…")
        self.run_stage_label.setText("Starting…")
        self.run_results.reset("Run in progress…")
        self._is_batch_run   = False
        self._is_compare_run = False

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
        """Kick off batch analysis over the checked files."""
        files = self._batch_checked_files()
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "No files",
                "On the Import tab, switch to Batch mode and pick a "
                "folder + at least one file.")
            self._switch_to_tab("Import")
            return
        self._switch_to_tab("Analysis")

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

        # Clear batch UI for new run.  batch_progress and batch_stage_label
        # are aliased to the Analysis-tab widgets in the new layout.
        self.console_log.clear()
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Starting…")
        self.batch_stage_label.setText("Starting…")
        self.batch_subprogress.setValue(0)
        self.batch_subprogress.setFormat("")
        self.batch_subprogress.show()
        self.run_results.reset("Batch in progress…")
        self._is_batch_run   = True
        self._is_compare_run = False

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

    def _start_compare_run(self):
        """Kick off a comparison over the configured groups."""
        groups = self._cmp_collect_groups()
        # Validation: ≥2 non-empty groups
        non_empty = [g for g in groups if g.get("folders")]
        if len(non_empty) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Not enough groups",
                "Need at least 2 groups, each with at least 1 analysis "
                "folder.")
            return

        outdir = self.e_cmp_outdir.text().strip()
        if not outdir:
            QtWidgets.QMessageBox.warning(
                self, "No output folder",
                "Pick a folder to save the comparison outputs.")
            return
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Cannot create output folder", str(exc))
            return

        # Selected panels
        selected_panels = {key for key, cb in self._cmp_panel_checkboxes.items()
                           if cb.isChecked()}
        if not selected_panels:
            QtWidgets.QMessageBox.warning(
                self, "No panels selected",
                "Pick at least one panel to include in the comparison "
                "figure.")
            return

        comparison_params = {
            "groups":      non_empty,
            "output_dir":  outdir,
            "output_stem": self.e_cmp_stem.text().strip() or "comparison",
            "theme":       self.c_cmp_theme.currentText(),
            "pdf_report":  bool(self.c_cmp_pdf.isChecked()),
            "panels":      list(selected_panels),
            "mobile_d_threshold": float(self.s_mobile_d_threshold.value()),
        }

        try:
            self._save_settings()
        except Exception:
            pass

        # Clear compare UI for new run
        self.console_log.clear()
        self.cmp_progress.setValue(0)
        self.cmp_stage_label.setText("Starting…")
        self.cmp_results.reset("Comparison in progress…")
        self.cmp_progress.setFormat("Starting…")
        self._is_batch_run    = False
        self._is_compare_run  = True

        self._msg_queue    = multiprocessing.Queue()
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_compare_in_subprocess,
            args=(comparison_params, self._msg_queue, self._cancel_event),
            name="FIREFLY-CompareWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.btn_cmp_run.setEnabled(False)
        self.statusBar().showMessage(
            f"Comparing {len(non_empty)} group(s)…")

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
        is_batch   = getattr(self, "_is_batch_run", False)
        is_compare = getattr(self, "_is_compare_run", False)

        # All log lines now land in the shared Console dock — one place
        # for everything.  The per-tab widgets are only the stage label
        # and the progress bar.
        log_widget = self.console_log
        if is_compare:
            progress_widget = self.cmp_progress
            stage_label     = self.cmp_stage_label
        elif is_batch:
            progress_widget = self.batch_progress
            stage_label     = self.batch_stage_label
        else:
            progress_widget = self.progress_bar
            stage_label     = self.run_stage_label

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
            elif kind == "compare_done":
                self._handle_compare_done(payload)
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
            # Progress bar shows just the % (clean look).  Verbose stage
            # info goes above it in the stage label.
            progress_widget.setFormat(f"{pct}%")
            stage_label.setText(msg)
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
        out_dir = payload.get("out_dir", "")
        stem    = payload.get("stem", "")
        summary = payload.get("summary") or {}
        n_tracks = summary.get("n_tracks", payload.get("n_tracks", 0))
        headline = f"{stem}  —  {n_tracks:,} trajectories" if stem else \
                   f"Analysis complete — {n_tracks:,} trajectories"
        self.run_results.show_results(headline, out_dir)
        self.run_results.show_stats(summary)
        self.run_stage_label.setText("Done")
        self.progress_bar.setFormat("Complete")
        self.statusBar().showMessage(f"Analysis complete — output at {out_dir}")
        # Optional: push the result into the Visualise tab's napari viewer
        try:
            self._ws_auto_load_after_run(payload)
        except Exception:
            pass

    def _handle_file_done(self, payload: dict):
        """One file in a batch finished successfully — not the terminal msg."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        n_tracks = payload.get("n_tracks", 0)
        stem     = payload.get("stem", "")
        self.statusBar().showMessage(
            f"Batch: {i} / {total} files complete  ({n_tracks:,} tracks)")
        if total:
            pct = int(100 * i / total)
            self.batch_progress.setValue(pct)
            self.batch_progress.setFormat(f"Batch  {i}/{total}  ({pct}%)")
            self.batch_subprogress.setValue(pct)
            self.batch_subprogress.setFormat(
                f"Last: {stem}  ({n_tracks:,} tracks)")

    def _handle_file_error(self, payload: dict):
        """One file in a batch failed — log it, batch continues."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        f = payload.get("file", "?")
        self.console_log.appendPlainText(
            f"\n  ⚠ [{i}/{total}] failed: {os.path.basename(f)}")
        self.batch_stage_label.setText(
            f"[{i}/{total}] failed: {os.path.basename(f)} — batch continues")
        self.statusBar().showMessage(f"Batch: file {i} failed (continuing)")

    def _handle_batch_done(self, payload: dict):
        """Batch terminal message — all files attempted."""
        n_total = payload.get("n_total", 0)
        n_ok    = payload.get("n_ok",    0)
        n_fail  = payload.get("n_fail",  0)
        self.batch_progress.setValue(100)
        self.batch_progress.setFormat(
            f"Batch complete  —  {n_ok}/{n_total} succeeded, {n_fail} failed")
        self.batch_subprogress.hide()
        self.statusBar().showMessage(
            f"Batch complete — {n_ok}/{n_total} succeeded, {n_fail} failed")

        # Populate the results panel with the batch summary
        headline = (f"Batch complete — {n_ok}/{n_total} succeeded"
                    + (f", {n_fail} failed" if n_fail else ""))
        # Aggregate stats across successful files
        results = payload.get("results") or []
        total_tracks = sum(r.get("n_tracks", 0) for r in results
                           if r.get("ok"))
        total_locs   = sum(r.get("n_locs", 0)   for r in results
                           if r.get("ok"))
        agg_summary = {
            "n_tracks": total_tracks,
            "n_locs":   total_locs,
            "motion_counts": {},
        }
        # Find the common output root (parent of every file's out_dir)
        common_root = ""
        ok_dirs = [r.get("out_dir") for r in results if r.get("ok") and r.get("out_dir")]
        if ok_dirs:
            try:
                common_root = os.path.commonpath(ok_dirs)
            except Exception:
                common_root = os.path.dirname(ok_dirs[0])
        self.run_results.show_results(headline, common_root,
                                      files=None)
        self.run_results.show_stats(agg_summary)

    def _handle_compare_done(self, payload: dict):
        """Compare terminal message — figure + CSVs + PDF have been saved."""
        self.cmp_progress.setValue(100)
        self.cmp_progress.setFormat("Complete")
        out_dir   = payload.get("output_dir", "")
        n_groups  = payload.get("n_groups", 0)
        headline = f"Comparison complete — {n_groups} group(s)"
        self.cmp_results.show_results(headline, out_dir)
        self.cmp_stage_label.setText("Done")
        self.statusBar().showMessage(
            f"Comparison complete — output at {out_dir}")

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
            self.console_log.appendPlainText(f"\nCrash report: {path}")
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
        self._is_batch_run          = False
        self._is_compare_run        = False
        self.btn_run.setText("Start")
        self.btn_run.setEnabled(True)
        try:
            self.btn_cmp_run.setEnabled(True)
        except AttributeError:
            pass
        try:
            self.batch_subprogress.hide()
        except AttributeError:
            pass

    # ── Crash reporter integration ────────────────────────────────────────
    def _install_crash_hooks(self):
        """Wire FIREFLY into the global crash reporter.  Same idea as the Tk
        version: capture every uncaught exception, write a self-contained
        text report, surface the path to the user via a dialog."""

        def _log_provider(n: int = 120) -> str:
            try:
                txt = self.console_log.toPlainText()
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
#  THEME — GitHub-dark-style palette matching the legacy Tk app
# ══════════════════════════════════════════════════════════════════════════════
# Colour constants are duplicated as Python globals here AND injected into
# the QSS template via .format() so they're a single source of truth for
# both stylesheet and any programmatic widget colouring (matplotlib
# canvas backgrounds, error messages, etc.).
_THEME = {
    "BG":          "#0d1117",   # main window background
    "PANEL":       "#161b22",   # cards, group boxes, log boxes
    "PANEL_ALT":   "#1c2128",   # alternating rows / subtle differentiation
    "BORDER":      "#30363d",   # borders, separators
    "BORDER_HI":   "#484f58",   # focused / hovered borders
    "TXT":         "#e6edf3",   # primary text
    "TXT_MUTED":   "#8b949e",   # secondary text (placeholder, labels-of-labels)
    "ACC":         "#58a6ff",   # primary accent (blue)
    "ACC_HOVER":   "#79c0ff",
    "ACC_PRESSED": "#388bfd",
    "ACC_FG":      "#0d1117",   # text on top of an accent fill
    "DANGER":      "#f85149",
    "SUCCESS":     "#56d364",
    "WARN":        "#f78166",
}

_FIREFLY_QSS = """
/* ── Base ────────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG};
    color:            {TXT};
    font-family:      -apple-system, "SF Pro Text", "Segoe UI", "Inter",
                      "Helvetica Neue", Arial, sans-serif;
    font-size:        12px;
}}

QMainWindow, QDialog {{
    background-color: {BG};
}}

/* Sidebar frame gets a slightly different shade so it visually separates
   from the central tab area. */
QMainWindow > QWidget > QFrame[frameShape="6"] {{   /* StyledPanel */
    background-color: {PANEL};
    border-right:     1px solid {BORDER};
}}

/* ── Labels ──────────────────────────────────────────────────────────────── */
QLabel {{
    background:       transparent;
    color:            {TXT};
}}

/* ── Group boxes ─────────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {PANEL};
    border:           1px solid {BORDER};
    border-radius:    6px;
    margin-top:       12px;
    padding:          8px 8px 6px 8px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding:           0 6px;
    margin-left:       8px;
    color:             {TXT_MUTED};
    font-weight:       600;
    font-size:         11px;
    background-color:  {BG};
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {PANEL_ALT};
    color:            {TXT};
    border:           1px solid {BORDER};
    border-radius:    5px;
    padding:          5px 12px;
    min-height:       20px;
}}

QPushButton:hover {{
    background-color: {PANEL};
    border-color:     {BORDER_HI};
}}

QPushButton:pressed {{
    background-color: {BG};
    border-color:     {ACC};
}}

QPushButton:disabled {{
    color:            {TXT_MUTED};
    background-color: {PANEL};
    border-color:     {BORDER};
}}

QPushButton:default,
QPushButton#primary {{
    background-color: {ACC};
    color:            {ACC_FG};
    border:           1px solid {ACC};
    font-weight:      600;
}}

QPushButton:default:hover,
QPushButton#primary:hover {{
    background-color: {ACC_HOVER};
    border-color:     {ACC_HOVER};
}}

QPushButton:default:pressed,
QPushButton#primary:pressed {{
    background-color: {ACC_PRESSED};
    border-color:     {ACC_PRESSED};
}}

QToolButton {{
    background:       transparent;
    color:            {TXT_MUTED};
    border:           1px solid transparent;
    border-radius:    4px;
    padding:          2px 6px;
}}

QToolButton:hover {{
    color:            {DANGER};
    background:       {PANEL_ALT};
    border-color:     {BORDER};
}}

/* ── Line edits / spinboxes / combos ─────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {PANEL_ALT};
    color:            {TXT};
    border:           1px solid {BORDER};
    border-radius:    4px;
    padding:          3px 6px;
    selection-background-color: {ACC};
    selection-color:  {ACC_FG};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color:     {ACC};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {PANEL};
    color:            {TXT_MUTED};
}}

/* Spinbox stepper buttons are removed entirely via NoButtons in the
   _QuietSpinBox subclasses — these QSS rules collapse the cells so any
   stray third-party spinbox (e.g. from napari's plugin UI) also reads
   as a clean borderless number field. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: transparent;
    border:           none;
    width:            0;
    height:           0;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image:  none;
    width:  0;
    height: 0;
}}

QComboBox::drop-down {{
    border: none;
    width:  18px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0; height: 0;
    border-left:  4px solid transparent;
    border-right: 4px solid transparent;
    border-top:    5px solid {TXT_MUTED};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color:            {TXT};
    border:           1px solid {BORDER};
    selection-background-color: {ACC};
    selection-color:  {ACC_FG};
    outline:          0;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    background-color: {BG};
    border:           1px solid {BORDER};
    border-radius:    6px;
    top:              -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color:            {TXT_MUTED};
    border:           1px solid transparent;
    padding:          6px 14px;
    margin-right:     2px;
    border-top-left-radius:  6px;
    border-top-right-radius: 6px;
}}

QTabBar::tab:hover {{
    color:            {TXT};
    background-color: {PANEL};
}}

QTabBar::tab:selected {{
    color:            {TXT};
    background-color: {BG};
    border:           1px solid {BORDER};
    border-bottom:    1px solid {BG};   /* hide bottom border of selected tab */
    font-weight:      600;
}}

/* ── Progress bar ────────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {PANEL_ALT};
    border:           1px solid {BORDER};
    border-radius:    5px;
    text-align:       center;
    color:            {TXT};
    min-height:       18px;
}}
QProgressBar::chunk {{
    background-color: {ACC};
    border-radius:    4px;
}}

/* ── List widgets (folder lists, batch file list) ───────────────────────── */
QListWidget {{
    background-color: {PANEL_ALT};
    color:            {TXT};
    border:           1px solid {BORDER};
    border-radius:    5px;
    outline:          0;
    alternate-background-color: {PANEL};
}}

QListWidget::item {{
    padding:          4px 6px;
    border-radius:    3px;
}}

QListWidget::item:hover {{
    background-color: {PANEL};
}}

QListWidget::item:selected {{
    background-color: {ACC};
    color:            {ACC_FG};
}}

/* ── Checkboxes ─────────────────────────────────────────────────────────── */
QCheckBox, QRadioButton {{
    spacing:          6px;
    background:       transparent;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width:            14px;
    height:           14px;
    border:           1px solid {BORDER_HI};
    background-color: {PANEL_ALT};
    border-radius:    3px;
}}

QRadioButton::indicator {{
    border-radius:    8px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color:     {ACC};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACC};
    border-color:     {ACC};
    image:            none;
}}

/* ── Scrollbars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background:       transparent;
    width:            10px;
    margin:           0;
}}
QScrollBar::handle:vertical {{
    background:       {BORDER};
    min-height:       24px;
    border-radius:    5px;
}}
QScrollBar::handle:vertical:hover {{
    background:       {BORDER_HI};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background:       transparent; border: none; height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background:       transparent;
}}

QScrollBar:horizontal {{
    background:       transparent;
    height:           10px;
    margin:           0;
}}
QScrollBar::handle:horizontal {{
    background:       {BORDER};
    min-width:        24px;
    border-radius:    5px;
}}
QScrollBar::handle:horizontal:hover {{
    background:       {BORDER_HI};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background:       transparent; border: none; width: 0;
}}

/* ── Splitter handle ────────────────────────────────────────────────────── */
QSplitter::handle {{
    background:       {BORDER};
}}
QSplitter::handle:hover {{
    background:       {BORDER_HI};
}}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}

/* ── Status bar ─────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {PANEL};
    color:            {TXT_MUTED};
    border-top:       1px solid {BORDER};
}}

/* ── ScrollArea (sidebar parameter list) ────────────────────────────────── */
QScrollArea {{
    background-color: {PANEL};
    border:           none;
}}

/* ── Tooltips ───────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {PANEL};
    color:            {TXT};
    border:           1px solid {BORDER};
    padding:          4px 6px;
    border-radius:    4px;
}}

/* ── Collapsible section (accordion-style) ───────────────────────────────── */
QToolButton#section_header {{
    background-color:    {PANEL_ALT};
    color:               {TXT};
    border:              1px solid {BORDER};
    border-top-left-radius:  5px;
    border-top-right-radius: 5px;
    padding:             7px 10px;
    margin-top:          4px;
    text-align:          left;
    font-weight:         600;
    font-size:           12px;
}}
QToolButton#section_header:hover {{
    border-color:        {BORDER_HI};
    background-color:    {PANEL};
}}
QToolButton#section_header:checked {{
    border-bottom-left-radius:  0;
    border-bottom-right-radius: 0;
}}
QFrame#section_content {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-top:          none;
    border-bottom-left-radius:  5px;
    border-bottom-right-radius: 5px;
}}

/* ── Results panel ──────────────────────────────────────────────────────── */
QFrame#results_panel {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       6px;
}}
QFrame#results_panel QLabel {{ background: transparent; border: none; }}
QListWidget#results_files {{
    background-color:    {PANEL_ALT};
    border:              1px solid {BORDER};
    border-radius:       4px;
}}

/* ── Menus ──────────────────────────────────────────────────────────────── */
QMenu {{
    background-color: {PANEL};
    color:            {TXT};
    border:           1px solid {BORDER};
    padding:          4px;
}}
QMenu::item {{
    padding:          5px 18px;
    border-radius:    3px;
}}
QMenu::item:selected {{
    background-color: {ACC};
    color:            {ACC_FG};
}}
""".format(**_THEME)


def _apply_firefly_theme(app: QtWidgets.QApplication):
    """Apply the FIREFLY dark theme: QPalette + comprehensive QSS.

    Also nudge the platform style toward "Fusion" — macOS's native style
    ignores most QSS properties (background colours, borders), so without
    Fusion the stylesheet would only partially apply.  Fusion respects
    everything in our QSS and renders identically on macOS / Windows /
    Linux, which is what we want for a cohesive look.
    """
    # Fusion style — required on macOS for our QSS to actually take effect.
    # Without this, the system style overrides background-color etc.
    app.setStyle("Fusion")

    # QPalette — mostly redundant alongside QSS but covers the few widgets
    # that don't read QSS (some native dialogs, scroll bars on some
    # platforms).  Keeps us looking consistent everywhere.
    pal = QtGui.QPalette()
    bg     = QtGui.QColor(_THEME["BG"])
    panel  = QtGui.QColor(_THEME["PANEL"])
    txt    = QtGui.QColor(_THEME["TXT"])
    muted  = QtGui.QColor(_THEME["TXT_MUTED"])
    acc    = QtGui.QColor(_THEME["ACC"])
    border = QtGui.QColor(_THEME["BORDER"])
    pal.setColor(QtGui.QPalette.ColorRole.Window,          bg)
    pal.setColor(QtGui.QPalette.ColorRole.WindowText,      txt)
    pal.setColor(QtGui.QPalette.ColorRole.Base,            panel)
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase,   bg)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     panel)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText,     txt)
    pal.setColor(QtGui.QPalette.ColorRole.Text,            txt)
    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QtGui.QPalette.ColorRole.Button,          panel)
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText,      txt)
    pal.setColor(QtGui.QPalette.ColorRole.Highlight,       acc)
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(_THEME["ACC_FG"]))
    pal.setColor(QtGui.QPalette.ColorRole.Link,            acc)
    pal.setColor(QtGui.QPalette.ColorRole.Mid,             border)
    pal.setColor(QtGui.QPalette.ColorRole.Dark,            bg)
    pal.setColor(QtGui.QPalette.ColorRole.Shadow,          bg)
    app.setPalette(pal)

    app.setStyleSheet(_FIREFLY_QSS)


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

    # Apply the FIREFLY dark theme (Fusion style + QSS + QPalette).
    _apply_firefly_theme(app)

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
