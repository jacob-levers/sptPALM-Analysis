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
#  MODE TILE — big segmented-control button with icon + title + subtitle
# ══════════════════════════════════════════════════════════════════════════════
_NAPARI_WELCOME_PHRASES = (
    "Drag image",
    "open image",
    "key bindings",
    "menu shortcuts",
    "Use the menu",
)


class _UpdateCheckThread(QtCore.QThread):
    """Tiny background thread that hits GitHub's Releases API and emits
    `update_available(tag, html_url)` if the latest tag is newer than
    the running FIREFLY version.  Silent on every other outcome — no
    network, no nag, no error popup."""

    update_available = QtCore.Signal(str, str)

    def __init__(self, api_url: str, current_version: str, parent=None):
        super().__init__(parent)
        self._api_url = api_url
        self._current = current_version

    @staticmethod
    def _parse_version(s: str) -> "tuple[int, ...]":
        """Parse a 'v2.2.0' / '2.2.0-dev3' style tag into a comparable
        tuple of ints.  Non-numeric suffix segments compare as 0."""
        import re
        s = s.lstrip("vV").split("-", 1)[0]
        parts = []
        for chunk in s.split("."):
            m = re.match(r"(\d+)", chunk)
            parts.append(int(m.group(1)) if m else 0)
        # Pad to length 3 for tidy comparisons
        while len(parts) < 3: parts.append(0)
        return tuple(parts)

    def run(self):
        try:
            import json
            import urllib.request
        except Exception:
            return
        try:
            req = urllib.request.Request(
                self._api_url,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": "FIREFLY-app"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                blob = resp.read()
        except Exception:
            return    # offline / rate-limited / etc. — silently no-op
        try:
            data = json.loads(blob)
            tag      = data.get("tag_name") or ""
            html_url = data.get("html_url") or ""
            if not tag:
                return
            if self._parse_version(tag) > self._parse_version(self._current):
                self.update_available.emit(tag, html_url)
        except Exception:
            return


def _hide_napari_chrome(viewer) -> None:
    """Hide napari's bottom-of-canvas viewer-button row (ndisplay / grid /
    home / console / etc.), the new-layer/delete button row under the
    layer list, and the empty-canvas 'Drag image(s) here…' welcome text.

    The welcome text is stripped by emptying the QLabel widgets that
    contain it (rather than hiding the welcome widget itself, which on
    some napari versions also hides the canvas).

    Defensive against napari version changes — attribute names shift
    between minor releases; every access is guarded."""
    try:
        qtv = (getattr(viewer.window, "qt_viewer", None)
               or getattr(viewer.window, "_qt_viewer", None))
    except Exception:
        qtv = None
    if qtv is None:
        return

    # Bottom-of-canvas button strip (ndisplay / grid / transpose / home / console)
    for attr in ("viewerButtons", "_viewer_buttons", "viewer_buttons"):
        w = getattr(qtv, attr, None)
        if w is not None and hasattr(w, "hide"):
            try:    w.hide()
            except Exception: pass

    # New-layer / delete buttons under the layer list
    for attr in ("layerButtons", "_layer_buttons", "layer_buttons"):
        w = getattr(qtv, attr, None)
        if w is not None and hasattr(w, "hide"):
            try:    w.hide()
            except Exception: pass

    # Welcome-overlay text — walk every QLabel under the qt_viewer and
    # blank out any that mention the welcome phrases.  Hiding the
    # parent welcome widget itself takes the canvas with it on some
    # napari versions, so we hide JUST the text content.
    try:
        for lbl in qtv.findChildren(QtWidgets.QLabel):
            try:
                txt = lbl.text() or ""
            except Exception:
                continue
            if any(p in txt for p in _NAPARI_WELCOME_PHRASES):
                try:    lbl.setText("")
                except Exception: pass
    except Exception:
        pass

    # Strip napari's menubar — on macOS its menus (File / View / Plugins /
    # Window / Help) get merged into the system menu bar and ⌘, opens
    # napari's Preferences dialog from "Python → Preferences".  None of
    # those belong to FIREFLY; clear them so the only menus the user sees
    # are the ones the host app actually owns.  Also disable any QActions
    # napari attached to the window — clearing the menubar removes them
    # from the menu, but their global shortcuts (⌘,, ⌘W, ⌘?, etc.) keep
    # firing until the actions themselves are disabled.
    try:
        qt_window = getattr(viewer.window, "_qt_window", None)
        mb = qt_window.menuBar() if qt_window is not None else None
        if mb is not None:
            # NOTE: do NOT call mb.clear() — on macOS the native menu bar
            # is shared with the application, and clearing it nukes the
            # standard QActions including the one ⌘Q routes through.
            # Detaching from the native menu + hiding the widget is
            # enough to keep napari's menus out of sight without taking
            # the host's Quit action down with them.
            try:    mb.setNativeMenuBar(False)
            except Exception: pass
            try:    mb.hide()
            except Exception: pass
        # Disable every QAction owned by the napari window so its global
        # shortcuts stop responding.  We don't delete them — that can
        # crash Qt mid-event — just set them disabled + remove their
        # shortcut binding.
        if qt_window is not None:
            for act in qt_window.findChildren(QtGui.QAction):
                try:    act.setEnabled(False)
                except Exception: pass
                try:    act.setShortcut(QtGui.QKeySequence())
                except Exception: pass
                try:    act.setShortcuts([])
                except Exception: pass
            # QShortcut objects are SEPARATE from QAction — napari uses
            # them for ⌘,, ⌘?, ⌘Y, etc.  Disable + strip them too.
            for sc in qt_window.findChildren(QtGui.QShortcut):
                try:    sc.setEnabled(False)
                except Exception: pass
                try:    sc.setKey(QtGui.QKeySequence())
                except Exception: pass
    except Exception:
        pass

    # Trim the shapes-layer toolbar — for ROI use we only need the
    # polygon tool + vertex edit + select + delete; rectangle / ellipse /
    # line / path / Z-order shuffling are noise.  Walk the WHOLE napari
    # window for buttons (the shape-mode buttons can sit deep inside
    # the layer-controls stack, several parents under `qt_viewer`), and
    # match against tooltip *and* object name + property hints.
    try:
        # Roots we'll walk in order of specificity
        roots = []
        for attr in ("controls", "layer_controls", "_layer_controls",
                     "dockLayerControls"):
            r = getattr(qtv, attr, None)
            if r is not None:
                # docks may carry the widget on .widget()
                inner = r.widget() if hasattr(r, "widget") and callable(r.widget) else r
                if inner is not None: roots.append(inner)
                roots.append(r)
        roots.append(qtv)
        if qt_window is not None:
            roots.append(qt_window)
        # Tooltip / objectName substrings that identify buttons we hide
        _kill = (
            "rectangle", "ellipse", "line ",
            "add lines", "add paths", "path mode", " path ",
            "polygon lasso", "lasso",
            "move to front", "move to back",
            "raise", "lower",   # napari uses these for Z-order too
        )
        seen: set = set()
        for root in roots:
            try:    btns = root.findChildren(QtWidgets.QAbstractButton)
            except Exception: continue
            for btn in btns:
                if id(btn) in seen:
                    continue
                seen.add(id(btn))
                try:
                    tip  = (btn.toolTip() or "").lower()
                    name = (btn.objectName() or "").lower()
                except Exception:
                    continue
                blob = tip + " | " + name
                if any(k in blob for k in _kill):
                    try:    btn.hide()
                    except Exception: pass
    except Exception:
        pass


class _ModeTile(QtWidgets.QFrame):
    """A big card-shaped clickable tile.  Acts like a checkable button:
    clicking toggles its state and emits `toggled(bool)`.  Used by the
    Import-tab Single-file / Batch mode toggle.

    QPushButton can't render rich-text (HTML in setText shows literally),
    so a custom QFrame with child QLabels is the cleanest way to get a
    button with multi-line bold-title + muted-subtitle styling.
    """
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, subtitle: str,
                 icon_char: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("mode_tile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("checked", False)
        self._checked = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(2)

        title_text = f"{icon_char}  {title}" if icon_char else title
        self._title_lbl = QtWidgets.QLabel(title_text)
        self._title_lbl.setObjectName("mode_tile_title")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._title_lbl)

        self._sub_lbl = QtWidgets.QLabel(subtitle)
        self._sub_lbl.setObjectName("mode_tile_subtitle")
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setWordWrap(True)
        v.addWidget(self._sub_lbl)

        self.setMinimumHeight(82)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == bool(checked):
            return
        self._checked = bool(checked)
        self.setProperty("checked", self._checked)
        # Re-evaluate QSS so the :checked-state border kicks in
        self.style().unpolish(self)
        self.style().polish(self)
        self.toggled.emit(self._checked)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self._checked:
            self.setChecked(True)
        super().mousePressEvent(e)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION TILE — landing-page clickable card (not checkable, emits clicked)
# ══════════════════════════════════════════════════════════════════════════════
class _ActionTile(QtWidgets.QFrame):
    """Large clickable card for the landing page.  Title + multi-line
    description + optional icon glyph.  Emits `clicked` when the user
    clicks anywhere on the tile."""
    clicked = QtCore.Signal()

    def __init__(self, title: str, description: str, icon_char: str = "",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("action_tile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(150)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(8)

        if icon_char:
            ico = QtWidgets.QLabel(icon_char)
            ico.setObjectName("action_tile_icon")
            ico.setAlignment(Qt.AlignmentFlag.AlignLeft)
            v.addWidget(ico)

        ttl = QtWidgets.QLabel(title)
        ttl.setObjectName("action_tile_title")
        ttl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v.addWidget(ttl)

        desc = QtWidgets.QLabel(description)
        desc.setObjectName("action_tile_desc")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v.addWidget(desc)
        v.addStretch(1)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


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


class _QuietComboBox(QtWidgets.QComboBox):
    """QComboBox that doesn't change its selection on mouse wheel.

    Same rationale as the spinbox subclasses: when the sidebar's scroll
    area gets a wheel event over a combo, the combo would silently
    change its value before the parent ever sees the wheel.  Override
    wheelEvent to ignore so the wheel bubbles up to the QScrollArea
    instead.  To deliberately change the value, click the dropdown.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Without StrongFocus the combo can also change via arrow keys
        # only after a mouse click anyway — fine for our usage.
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
        # Escape any literal ampersands once at construction.  QToolButton
        # (like every other Qt button-like widget) treats `&` in its text
        # as a keyboard-shortcut marker — "Diffusion & motion classification"
        # would render as "Diffusion _motion classification" (m underlined,
        # Alt+M activates the button).  Doubling the ampersand is the
        # documented way to display a literal `&`.
        self._title = title.replace("&", "&&")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QtWidgets.QToolButton()
        self._header.setObjectName("section_header")
        self._header.setText(f"▼   {self._title}")
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
class _ResourceMonitor(QtWidgets.QFrame):
    """1 Hz strip of system-resource gauges shown at the top of the
    Analysis tab.  Four cells: CPU%, RAM used / total, GPU%, GPU VRAM.

    Catches "why is my run slow" instantly — if the GPU sits at 0% the
    backend silently fell back to CPU; if RAM is pinned the OS is
    paging; etc.  Gracefully degrades when psutil or torch is missing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resource_monitor")
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(14)
        self._cells: "dict[str, QtWidgets.QLabel]" = {}
        for key, label in (("cpu",  "CPU"),
                           ("ram",  "RAM"),
                           ("gpu",  "GPU"),
                           ("vram", "VRAM")):
            cell = QtWidgets.QHBoxLayout()
            cell.setSpacing(4)
            cap = QtWidgets.QLabel(label)
            cap.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']}; font-size: 11px;"
                "font-weight: 600;")
            val = QtWidgets.QLabel("—")
            val.setStyleSheet(
                f"color: {_THEME['TXT']}; font-size: 12px; "
                "font-variant-numeric: tabular-nums;")
            val.setMinimumWidth(70)
            self._cells[key] = val
            cell.addWidget(cap)
            cell.addWidget(val)
            h.addLayout(cell)
        h.addStretch(1)

        # Probe deps once at construction so we don't pay the import cost
        # per tick.  All three are optional — graceful no-op if absent.
        self._psutil = None
        self._torch  = None
        try:
            import psutil as _ps
            self._psutil = _ps
            # Warm the per-process cpu_percent baseline so the first
            # reading isn't a misleading 0.0.
            try:    _ps.cpu_percent(interval=None)
            except Exception: pass
        except Exception:
            pass
        try:
            import torch as _t
            self._torch = _t
        except Exception:
            pass

        # Cached MPS utilisation (Apple Silicon has no Python API — we
        # shell out to `ioreg`, which is fast but worth backgrounding).
        self._mps_util_cache: "int | None" = None
        self._mps_polling: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    @staticmethod
    def _mps_gpu_utilization() -> "int | None":
        """Best-effort macOS GPU utilisation via `ioreg`.  Returns an
        integer percent or None if we can't tell.  Runs subprocess
        — never call from the GUI thread; use `_poll_mps_async`.

        Different macOS / chip combos expose the field under slightly
        different keys, so we try a few.  Cheap regex parse — no
        plistlib needed."""
        import subprocess, re
        try:
            out = subprocess.check_output(
                ["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                stderr=subprocess.DEVNULL, timeout=1.0).decode(
                "utf-8", errors="ignore")
        except Exception:
            return None
        for pattern in (r'"Device Utilization\s*%"\s*=\s*(\d+)',
                        r'"GPU Busy %"\s*=\s*(\d+)',
                        r'"GPU Core Utilization\s*%"\s*=\s*(\d+)'):
            m = re.search(pattern, out)
            if m:
                try:    return int(m.group(1))
                except Exception: pass
        return None

    def _poll_mps_async(self):
        """Run the ioreg parse on a background thread, write the result
        to the cache.  Re-entrancy-guarded so we don't pile up threads."""
        if self._mps_polling:
            return
        self._mps_polling = True
        import threading
        def _run():
            try:    self._mps_util_cache = self._mps_gpu_utilization()
            finally: self._mps_polling = False
        threading.Thread(target=_run, daemon=True).start()

    def _set(self, key: str, txt: str, *, warn: bool = False):
        lbl = self._cells.get(key)
        if lbl is None: return
        col = _THEME['WARN'] if warn else _THEME['TXT']
        lbl.setStyleSheet(
            f"color: {col}; font-size: 12px; "
            "font-variant-numeric: tabular-nums;")
        lbl.setText(txt)

    def _refresh(self):
        # CPU + RAM via psutil
        if self._psutil is not None:
            try:
                pct = self._psutil.cpu_percent(interval=None)
                self._set("cpu", f"{pct:5.1f} %",
                          warn=(pct > 90))
            except Exception:
                self._set("cpu", "—")
            try:
                vm = self._psutil.virtual_memory()
                used_gb = (vm.total - vm.available) / 1e9
                total_gb = vm.total / 1e9
                self._set("ram",
                          f"{used_gb:4.1f} / {total_gb:.0f} GB",
                          warn=(vm.percent > 90))
            except Exception:
                self._set("ram", "—")
        else:
            self._set("cpu", "(psutil)")
            self._set("ram", "(psutil)")

        # GPU usage + VRAM via torch.  CUDA exposes utilisation through
        # nvidia-smi (not torch) so we report VRAM-in-use as the GPU
        # cell when CUDA is active; MPS doesn't expose either cleanly.
        if self._torch is None:
            self._set("gpu",  "(torch)")
            self._set("vram", "(torch)")
            return

        try:
            t = self._torch
            if t.cuda.is_available():
                # GPU utilisation via NVML if available
                util = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                if util is not None:
                    self._set("gpu", f"{util:5.1f} %",
                              warn=(util < 1))
                else:
                    self._set("gpu", "CUDA")
                try:
                    alloc = t.cuda.memory_allocated() / 1e9
                    total = t.cuda.get_device_properties(0).total_memory / 1e9
                    self._set("vram", f"{alloc:4.1f} / {total:.0f} GB")
                except Exception:
                    self._set("vram", "CUDA")
            elif (hasattr(t.backends, "mps")
                  and t.backends.mps.is_available()):
                # MPS exposes "current_allocated_memory" only in recent
                # torch builds; fall back to a stub when missing.
                try:
                    alloc = t.mps.current_allocated_memory() / 1e9
                    self._set("vram", f"{alloc:4.1f} GB")
                except Exception:
                    self._set("vram", "MPS")
                # Kick off the async ioreg poll for the next tick + show
                # the cached value from the previous tick.
                self._poll_mps_async()
                if self._mps_util_cache is not None:
                    self._set("gpu", f"{self._mps_util_cache:5.1f} %",
                              warn=(self._mps_util_cache < 1))
                else:
                    self._set("gpu", "MPS")
            else:
                self._set("gpu",  "CPU only")
                self._set("vram", "—")
        except Exception:
            self._set("gpu",  "—")
            self._set("vram", "—")


class _MassHistogram(QtWidgets.QWidget):
    """Lightweight live histogram of localisation mass values.

    Renders with QPainter — no matplotlib dependency in the GUI process.
    Bars accumulate across chunks; clear via `reset()`.  Designed to
    show during a run so the user can sanity-check minmass on the fly.
    """
    BIN_COUNT = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self._counts = None          # np.ndarray | None
        self._edges  = None          # np.ndarray | None
        self._total  = 0
        self._minmass = None         # float | None (vertical guide line)
        # Throttle repaints — accumulate updates and only repaint at most
        # ~6 Hz to keep the GUI responsive when chunks land in rapid fire.
        self._dirty = False
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(16)   # ~60 Hz
        self._repaint_timer.setSingleShot(False)
        self._repaint_timer.timeout.connect(self._maybe_repaint)
        self._repaint_timer.start()

    def reset(self):
        self._counts = None
        self._edges  = None
        self._total  = 0
        self._dirty  = True

    def set_minmass(self, value):
        try:
            self._minmass = float(value) if value is not None else None
        except (TypeError, ValueError):
            self._minmass = None
        self._dirty = True

    def add_chunk(self, mass_values) -> None:
        """Accept an iterable of mass values and merge into the histogram."""
        try:
            import numpy as _np
            arr = _np.asarray(list(mass_values), dtype=_np.float32)
            arr = arr[_np.isfinite(arr)]
            if arr.size == 0:
                return
            if self._counts is None:
                # First chunk seeds the bin edges.  Range from 0 to 99th-pct
                # of the data, expanded slightly for headroom.
                hi = float(_np.percentile(arr, 99.0)) * 1.3 + 1e-6
                self._edges = _np.linspace(0.0, hi, self.BIN_COUNT + 1)
                self._counts = _np.zeros(self.BIN_COUNT, dtype=_np.int64)
            new_counts, _ = _np.histogram(arr, bins=self._edges)
            self._counts += new_counts
            self._total += int(arr.size)
            self._dirty = True
        except Exception:
            pass

    def _maybe_repaint(self):
        if self._dirty:
            self._dirty = False
            self.update()

    def paintEvent(self, _evt):
        import math
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        r = self.rect()
        # Background
        p.fillRect(r, QtGui.QColor(_THEME['PANEL']))
        # Border
        pen = QtGui.QPen(QtGui.QColor(_THEME['BORDER']), 1)
        p.setPen(pen)
        p.drawRect(r.adjusted(0, 0, -1, -1))

        # Padding
        pad_l, pad_t, pad_r, pad_b = 8, 22, 8, 16
        plot = r.adjusted(pad_l, pad_t, -pad_r, -pad_b)

        # Title
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        f = p.font(); f.setPointSize(10); p.setFont(f)
        if self._counts is None or self._counts.sum() == 0:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "Localisation mass distribution will appear here\n"
                       "as chunks finish.")
            return
        title = f"Localisation mass  ·  n = {self._total:,}"
        p.drawText(QtCore.QRect(r.left() + pad_l, r.top() + 4,
                                r.width() - pad_l - pad_r, 18),
                   Qt.AlignmentFlag.AlignLeft, title)

        # Bars
        n_bins = len(self._counts)
        max_h  = float(self._counts.max()) or 1.0
        bar_w  = plot.width() / n_bins
        bar_pen = QtGui.QPen(QtGui.QColor(_THEME['ACC']))
        bar_pen.setWidth(0)
        p.setPen(bar_pen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(_THEME['ACC'])))
        for i, c in enumerate(self._counts):
            h = (c / max_h) * plot.height()
            x = plot.left() + i * bar_w
            y = plot.bottom() - h
            p.drawRect(QtCore.QRectF(x + 0.5, y, max(1.0, bar_w - 1.0), h))

        # X-axis ticks (just min + max edges)
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        f = p.font(); f.setPointSize(8); p.setFont(f)
        p.drawText(plot.left(),       r.bottom() - 2,
                   f"{self._edges[0]:.2f}")
        p.drawText(plot.right() - 60, r.bottom() - 2,
                   f"{self._edges[-1]:.2f}  mass")

        # Min-mass guide line
        if self._minmass is not None and self._edges is not None:
            lo, hi = float(self._edges[0]), float(self._edges[-1])
            if hi > lo and lo <= self._minmass <= hi * 1.2:
                frac = (self._minmass - lo) / (hi - lo)
                x = plot.left() + min(1.0, max(0.0, frac)) * plot.width()
                pen = QtGui.QPen(QtGui.QColor(_THEME['WARN']))
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidth(1)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(x, plot.top()),
                           QtCore.QPointF(x, plot.bottom()))
                p.setPen(QtGui.QColor(_THEME['WARN']))
                p.drawText(QtCore.QPointF(x + 3, plot.top() + 10),
                           f"min={self._minmass:g}")
        p.end()


class _LiveFrameView(QtWidgets.QWidget):
    """Renders the most-recent preprocessed frame from the localisation
    stream plus the detections found on it.  Pairs with `_MassHistogram`
    to form a 'detection cockpit' on the Analysis tab so the user can
    watch what's actually being detected during a run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self._frame = None           # 2D float32 array
        self._xs = None
        self._ys = None
        self._idx = None
        self._n_frames = None
        self._n_spots = 0
        # Throttle repaints to ~6 Hz so a hot stream of chunks doesn't
        # pin the GUI thread.
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)   # ~60 Hz
        self._timer.timeout.connect(self._maybe_repaint)
        self._timer.start()

    def reset(self):
        self._frame = None
        self._xs = None
        self._ys = None
        self._idx = None
        self._n_frames = None
        self._n_spots = 0
        self._dirty = True

    def set_frame(self, frame, xs, ys, idx, n_frames):
        try:
            import numpy as _np
            self._frame = _np.asarray(frame, dtype=_np.float32)
            self._xs = _np.asarray(xs, dtype=_np.float32)
            self._ys = _np.asarray(ys, dtype=_np.float32)
            self._idx = int(idx)
            self._n_frames = int(n_frames)
            self._n_spots = int(self._xs.size)
            self._dirty = True
        except Exception:
            pass

    def _maybe_repaint(self):
        if self._dirty:
            self._dirty = False
            self.update()

    def paintEvent(self, _evt):
        p = QtGui.QPainter(self)
        r = self.rect()
        p.fillRect(r, QtGui.QColor(_THEME['PANEL']))
        p.setPen(QtGui.QColor(_THEME['BORDER']))
        p.drawRect(r.adjusted(0, 0, -1, -1))

        if self._frame is None:
            p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "Live detection view will appear here\n"
                       "during a run.")
            return

        try:
            import numpy as _np
            f = self._frame
            lo, hi = _np.percentile(f, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            u8 = _np.clip((f - lo) * (255.0 / (hi - lo)),
                          0, 255).astype(_np.uint8, copy=False)
            # Pad to a contiguous buffer that QImage can wrap safely
            u8 = _np.ascontiguousarray(u8)
            h, w = u8.shape
            img = QtGui.QImage(u8.tobytes(), w, h, w,
                                QtGui.QImage.Format.Format_Grayscale8)
        except Exception:
            return

        # Compute the rect inside the widget where we draw the frame
        pad_t = 22; pad = 8
        avail = r.adjusted(pad, pad_t, -pad, -pad)
        if avail.width() <= 0 or avail.height() <= 0:
            return
        scale = min(avail.width() / w, avail.height() / h)
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        disp_x = avail.left() + (avail.width()  - disp_w) // 2
        disp_y = avail.top()  + (avail.height() - disp_h) // 2
        p.drawImage(QtCore.QRect(disp_x, disp_y, disp_w, disp_h), img)

        # Title strip
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        font = p.font(); font.setPointSize(10); p.setFont(font)
        idx = self._idx if self._idx is not None else 0
        total = self._n_frames if self._n_frames else 0
        title = (f"Live detection  ·  frame {idx + 1}/{total}"
                 f"  ·  {self._n_spots} spots")
        p.drawText(QtCore.QRect(r.left() + pad, r.top() + 4,
                                r.width() - 2 * pad, 18),
                   Qt.AlignmentFlag.AlignLeft, title)

        # Detection circles — flat accent for visibility on greyscale
        if self._xs is not None and self._xs.size > 0:
            pen = QtGui.QPen(QtGui.QColor(_THEME['ACC']))
            pen.setWidthF(1.2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            radius = max(3.0, 4.0 * scale)
            for x, y in zip(self._xs, self._ys):
                cx = disp_x + float(x) * scale
                cy = disp_y + float(y) * scale
                p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
        p.end()


class _TrackInspector(QtWidgets.QFrame):
    """Right-side panel for the Visualise tab.  Displays per-particle stats
    for whichever track the user clicked on in the embedded napari viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("track_inspector")
        self.setMinimumWidth(280)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        title = QtWidgets.QLabel("Track inspector")
        title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-size: 14px; font-weight: 700;")
        v.addWidget(title)

        self._hint = QtWidgets.QLabel(
            "Click a track in the viewer to inspect it.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 12px;")
        v.addWidget(self._hint)

        # Stats grid
        self._grid = QtWidgets.QGridLayout()
        self._grid.setColumnStretch(0, 0)
        self._grid.setColumnStretch(1, 1)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(4)
        grid_w = QtWidgets.QWidget()
        grid_w.setLayout(self._grid)
        self._grid_w = grid_w
        v.addWidget(grid_w)
        grid_w.hide()

        v.addStretch(1)

    def clear(self):
        self._hint.show()
        self._grid_w.hide()
        # Wipe the grid
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def show_track(self, *, particle_id: int,
                   length: int | None = None,
                   d: float | None = None,
                   alpha: float | None = None,
                   motion: str | None = None,
                   mean_mass: float | None = None,
                   start_frame: int | None = None,
                   end_frame: int | None = None,
                   net_displacement_um: float | None = None,
                   total_path_um: float | None = None,
                   straightness: float | None = None):
        # Clear and re-populate
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

        def _row(r, label, value, *, color=None):
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']}; font-size: 12px;")
            val = QtWidgets.QLabel(value)
            val.setStyleSheet(
                f"color: {color or _THEME['TXT']}; font-size: 13px; "
                "font-weight: 600;")
            val.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(lbl, r, 0, Qt.AlignmentFlag.AlignLeft)
            self._grid.addWidget(val, r, 1, Qt.AlignmentFlag.AlignLeft)

        motion_colour = {
            "Immobile": _THEME['DANGER'], "Confined": _THEME['WARN'],
            "Brownian": _THEME['ACC'],   "Directed": _THEME['SUCCESS'],
        }
        r = 0
        _row(r, "Particle ID", f"#{particle_id}"); r += 1
        if length is not None:
            _row(r, "Track length", f"{length} frames"); r += 1
        if start_frame is not None and end_frame is not None:
            _row(r, "Frame span", f"{start_frame} → {end_frame}"); r += 1
        if motion:
            _row(r, "Motion class", motion,
                 color=motion_colour.get(motion)); r += 1
        if d is not None:
            _row(r, "Diffusion D",  f"{d:.4f}  µm²/s"); r += 1
        if alpha is not None:
            _row(r, "α (anomalous)", f"{alpha:.3f}"); r += 1
        if net_displacement_um is not None:
            _row(r, "Net displacement",
                 f"{net_displacement_um*1000:.0f} nm"); r += 1
        if total_path_um is not None:
            _row(r, "Total path",
                 f"{total_path_um*1000:.0f} nm"); r += 1
        if straightness is not None:
            _row(r, "Straightness", f"{straightness:.3f}"); r += 1
        if mean_mass is not None:
            _row(r, "Mean mass", f"{mean_mass:.1f}"); r += 1

        self._hint.hide()
        self._grid_w.show()


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

        # ── Quality control ──────────────────────────────────────────
        qc = summary.get("qc") or {}
        if qc:
            # Section header
            hdr = QtWidgets.QLabel("Quality control")
            hdr.setStyleSheet(
                f"color: {_THEME['TXT']}; font-size: 12px; "
                "font-weight: 700; padding-top: 8px;")
            self._stats_grid.addWidget(hdr, r, 0, 1, 2,
                                       Qt.AlignmentFlag.AlignLeft); r += 1

            lr = qc.get("link_ratio")
            if lr is not None:
                col = (_THEME['DANGER'] if lr < 0.10
                       else _THEME['WARN'] if lr < 0.25
                       else _THEME['SUCCESS'])
                self._add_stat_row(r, "Localisations linked",
                                   _fmt_pct(lr),
                                   value_colour=col); r += 1
            avg_pf = qc.get("avg_locs_per_frame")
            if avg_pf is not None:
                col = (_THEME['WARN'] if avg_pf > 800
                       else _THEME['TXT'])
                self._add_stat_row(r, "Locs / frame (avg)",
                                   f"{avg_pf:,.1f}",
                                   value_colour=col); r += 1
            ml = qc.get("median_track_length")
            if ml is not None:
                col = (_THEME['WARN'] if ml < 6 else _THEME['TXT'])
                self._add_stat_row(r, "Median track length",
                                   f"{ml:.1f}  frames",
                                   value_colour=col); r += 1
            gf = qc.get("gap_fraction")
            if gf is not None:
                self._add_stat_row(r, "Tracks with gaps",
                                   _fmt_pct(gf)); r += 1
            sf = qc.get("stuck_fraction")
            if sf is not None:
                col = (_THEME['WARN'] if sf > 0.30 else _THEME['TXT'])
                self._add_stat_row(r, "Stuck tracks  (D < 1e-3)",
                                   _fmt_pct(sf),
                                   value_colour=col); r += 1

            # Flag list — each flag is a colour-coded one-liner
            flags = qc.get("flags") or []
            if flags:
                for f in flags:
                    level = f.get("level", "info")
                    icon  = "⚠" if level == "warn" else "ℹ"
                    col   = (_THEME['WARN'] if level == "warn"
                             else _THEME['TXT_MUTED'])
                    msg = QtWidgets.QLabel(f"  {icon}  {f.get('msg', '')}")
                    msg.setStyleSheet(
                        f"color: {col}; font-size: 12px;")
                    msg.setWordWrap(True)
                    self._stats_grid.addWidget(
                        msg, r, 0, 1, 2,
                        Qt.AlignmentFlag.AlignLeft); r += 1

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
#  ROI DIALOG — embedded napari viewer for drawing per-file polygon ROIs
# ══════════════════════════════════════════════════════════════════════════════
class _RoiDialog(QtWidgets.QDialog):
    """Modal ROI editor.  Loads the mean projection of an input file into
    an embedded napari viewer with a Shapes layer in polygon mode.  User
    draws one or more polygons; clicking Save returns the vertices.

    Vertices are stored as (y, x) coordinate pairs in pixels of the
    original (Y, X) frame — directly consumable by
    skimage.draw.polygon2mask.
    """

    def __init__(self, file_path: str,
                 current_polygons: list | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"ROI — {os.path.basename(file_path)}")
        self.resize(1000, 760)
        self._file_path = file_path
        self._result_polygons: list[list[tuple[float, float]]] = []
        self._viewer = None
        self._shapes_layer = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # ── Instructions ────────────────────────────────────────────────
        hint = QtWidgets.QLabel(
            "Use the <b>polygon</b> tool in the layer controls (top-left) "
            "to draw a region of interest.  Click points to add vertices, "
            "then press <b>Esc</b> or right-click to finish the polygon.  "
            "You can draw multiple polygons — they'll be combined into one "
            "ROI mask.  Save when done; Cancel discards changes."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; padding: 4px 0;")
        v.addWidget(hint)

        # ── Status line ─────────────────────────────────────────────────
        self._status = QtWidgets.QLabel("Loading preview…")
        self._status.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(self._status)

        # ── Embedded napari viewer (placeholder until lazy-init) ────────
        self._viewer_container = QtWidgets.QWidget()
        self._viewer_layout = QtWidgets.QVBoxLayout(self._viewer_container)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._viewer_container, stretch=1)

        # ── Buttons ─────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._b_clear = QtWidgets.QPushButton("Clear ROI")
        self._b_clear.setToolTip("Remove all polygons (file will fall back to "
                                  "the global ROI mode in settings).")
        self._b_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self._b_clear)
        btn_row.addStretch(1)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        btn_row.addWidget(b_cancel)
        b_save = QtWidgets.QPushButton("Save ROI")
        b_save.setObjectName("primary")
        b_save.clicked.connect(self._on_save)
        btn_row.addWidget(b_save)
        v.addLayout(btn_row)

        # Defer the heavy lifting (napari init + file load) so the dialog
        # appears immediately with the "Loading preview…" status.
        QtCore.QTimer.singleShot(50, lambda: self._init_viewer(current_polygons))

    @staticmethod
    def _quick_preview(file_path: str, max_frames: int = 30):
        """Read just enough of `file_path` to render a representative
        background image for ROI drawing.  Returns a 2D float32 array
        of shape (Y, X), or raises.

        This DELIBERATELY does not use `load_file` — that loads the full
        stack (and for multi-file TIF series, concatenates them), which
        on a tight-RAM machine can take minutes and trigger swap.  For
        the ROI editor we only need a clear preview, not the full data,
        so we read just the first `max_frames` pages of the first file
        directly via tifffile / aicspylibczi.
        """
        import os as _os
        import numpy as _np
        ext = _os.path.splitext(file_path)[1].lower()

        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                n_pages = len(tif.pages)
                n = min(max_frames, n_pages)
                # Sample evenly across the (single) file so blinks /
                # bleaches don't dominate the preview
                if n_pages > n:
                    idx = _np.linspace(0, n_pages - 1, n, dtype=int)
                else:
                    idx = _np.arange(n_pages)
                frames = []
                for i in idx:
                    frames.append(tif.pages[int(i)].asarray()
                                  .astype(_np.float32))
                return _np.mean(_np.stack(frames), axis=0)

        if ext == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(file_path)
            # Read first frame.  CZI reads can return shape (1, 1, 1, Y, X)
            # or similar depending on dim order — squeeze to (Y, X).
            try:
                img, _ = czi.read_image(T=0)
            except Exception:
                # Some CZIs have different dim names; fall back to
                # reading the whole thing if T isn't a valid dim
                img = czi.read_mosaic(C=0, scale_factor=1)
            arr = _np.squeeze(_np.asarray(img))
            # If we accidentally got >2D (multichannel etc.) take a mean
            while arr.ndim > 2:
                arr = arr.mean(axis=0)
            return arr.astype(_np.float32)

        raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _quick_preview_stack(file_path: str, max_frames: int = 30):
        """Like `_quick_preview` but returns a 3D (T, Y, X) stack instead of
        the mean.  Used by the embedded ROI viewer so the user can scrub
        through real frames and live-preview detections."""
        import os as _os
        import numpy as _np
        ext = _os.path.splitext(file_path)[1].lower()

        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                n_pages = len(tif.pages)
                n = min(max_frames, n_pages)
                if n_pages > n:
                    idx = _np.linspace(0, n_pages - 1, n, dtype=int)
                else:
                    idx = _np.arange(n_pages)
                frames = [tif.pages[int(i)].asarray().astype(_np.float32)
                          for i in idx]
                return _np.stack(frames), [int(i) for i in idx]

        if ext == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(file_path)
            frames = []
            indices = []
            for t in range(max_frames):
                try:
                    img, _ = czi.read_image(T=t)
                except Exception:
                    break
                arr = _np.squeeze(_np.asarray(img))
                while arr.ndim > 2:
                    arr = arr.mean(axis=0)
                frames.append(arr.astype(_np.float32))
                indices.append(t)
            if not frames:
                raise ValueError("No frames could be read from CZI")
            return _np.stack(frames), indices

        raise ValueError(f"Unsupported file extension: {ext}")

    def _init_viewer(self, current_polygons):
        try:
            import napari
        except Exception as exc:
            self._status.setText(
                f"napari isn't installed: {exc}.\n"
                f"Run `pip install \"napari[pyside6]>=0.4.19\"` and restart.")
            return

        try:
            self._viewer = napari.Viewer(show=False)
            qt_window = self._viewer.window._qt_window
            self._viewer_layout.addWidget(qt_window)
            _hide_napari_chrome(self._viewer)
        except Exception as exc:
            self._status.setText(f"Couldn't embed napari viewer: {exc}")
            return

        # Load just enough to render an ROI background.  No full-stack load,
        # no concat — see _quick_preview's docstring.  Synchronous on the
        # dialog's event loop but the read is tiny (~30 frames).
        try:
            import numpy as _np
            mean_img = self._quick_preview(self._file_path, max_frames=30)
            self._viewer.add_image(mean_img, name="ROI background",
                                    colormap="gray")
            # Shapes layer for the polygon
            initial_shapes = [_np.asarray(poly)
                              for poly in (current_polygons or [])]
            self._shapes_layer = self._viewer.add_shapes(
                data=initial_shapes if initial_shapes else None,
                shape_type="polygon",
                edge_color="#58a6ff",
                face_color="rgba(88,166,255,0.18)",
                edge_width=2,
                name="ROI",
            )
            # Switch to polygon-add mode so the user can start drawing
            try:
                self._shapes_layer.mode = "add_polygon"
            except Exception:
                pass
            self._status.setText(
                f"{mean_img.shape[1]} × {mean_img.shape[0]} px preview "
                f"(quick load).  Draw polygon(s) on the ROI layer; "
                "right-click or Esc to close each polygon.")
        except Exception as exc:
            import traceback as _tb
            self._status.setText(
                f"Couldn't load file preview: {exc}\n\n"
                f"{_tb.format_exc()}")

    def _polygons_from_layer(self) -> list[list[tuple[float, float]]]:
        """Pull current polygon vertices out of the Shapes layer.
        Each entry is a list of (y, x) tuples."""
        polys: list[list[tuple[float, float]]] = []
        if self._shapes_layer is None:
            return polys
        try:
            for shape_data, shape_type in zip(self._shapes_layer.data,
                                              self._shapes_layer.shape_type):
                if shape_type not in ("polygon", "rectangle", "ellipse"):
                    continue
                if shape_type == "polygon":
                    polys.append([(float(y), float(x))
                                  for y, x in shape_data])
                else:
                    # Rectangles / ellipses are stored as 4-vertex bounding
                    # boxes; treat as polygons (rectangle is exact, ellipse
                    # is approximated by its bounding box for now).
                    polys.append([(float(y), float(x))
                                  for y, x in shape_data])
        except Exception:
            pass
        return polys

    def _on_save(self):
        self._result_polygons = self._polygons_from_layer()
        self.accept()

    def _on_clear(self):
        self._result_polygons = []
        self.accept()

    def result_polygons(self) -> list[list[tuple[float, float]]]:
        return self._result_polygons


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDED ROI VIEWER — same idea as _RoiDialog but lives in the Import tab
# ══════════════════════════════════════════════════════════════════════════════
class _RoiViewer(QtWidgets.QWidget):
    """Inline ROI editor for the Import tab.

    Same drawing model as the modal `_RoiDialog` (embedded napari + Shapes
    layer with polygon tool), but lives inside the tab and supports
    switching between files via `set_file`.  Polygons are auto-emitted as
    they change so the host MainWindow can save them per file without
    requiring an explicit Save button.
    """

    polygons_changed = QtCore.Signal(str, list)  # (file_path, polygons)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_file: str = ""
        self._viewer = None
        self._shapes_layer = None
        self._image_layer  = None
        self._points_layer = None
        self._stack = None             # cached 3-D preview stack (raw)
        self._stack_filtered = None    # cached bandpass-filtered version (lazy)
        self._stack_preprocessed = None  # cached pipeline-preprocessed stack
        self._pp_signature = None      # (bg_method, bg_radius) of cached stack
        self._last_mass = None         # mass array from the most-recent locate
        self._roi_mask_layer = None    # auto/manual-threshold overlay layer
        self._roi_mask_params = {"mode": "None", "auto_method": "li",
                                 "threshold": 0.08, "mask_mode": "mean"}
        # When true, _on_layer_removed is a no-op.  Used by set_file
        # while it tears down the previous file's layers — we DON'T want
        # the "user deleted our ROI, recreate it" recovery path to fire
        # during a programmatic teardown, because re-entering
        # add_shapes() mid-clear corrupts napari's layer iterator and
        # freezes the GUI.
        self._suppress_layer_events = False
        self._lazy_init_pending = True
        self._detect_enabled = False
        self._detect_params  = {"diameter": 7, "minmass": 1.0,
                                "bg_method": "uniform_filter",
                                "bg_radius": 50}
        self._dims_connected = False
        self._detect_debounce = QTimer(self)
        self._detect_debounce.setSingleShot(True)
        self._detect_debounce.setInterval(250)
        self._detect_debounce.timeout.connect(self._run_detection)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("Preview viewer")
        self._title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 600; font-size: 13px;")
        header.addWidget(self._title)
        self._status = QtWidgets.QLabel("Pick a file to start")
        self._status.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        header.addWidget(self._status, 1)
        # Turbo-colormap legend bar (low mass → high mass).
        legend_w = QtWidgets.QWidget()
        legend_w.setToolTip(
            "Detections are coloured by integrated mass on a log scale using "
            "the 'turbo' colormap, auto-stretched per frame.  Dim spots "
            "(likely noise) sit at the blue / purple end; bright spots "
            "(likely real PSFs) sit at the red end.  Raise minmass and the "
            "blue end disappears first.")
        lh = QtWidgets.QHBoxLayout(legend_w)
        lh.setContentsMargins(8, 0, 0, 0)
        lh.setSpacing(4)
        lbl_lo = QtWidgets.QLabel("dim")
        lbl_lo.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 10px;")
        lh.addWidget(lbl_lo)
        bar = QtWidgets.QFrame()
        bar.setFixedSize(120, 10)
        # Turbo colormap stops (approximate, perceptually-uniform)
        bar.setStyleSheet(
            "QFrame { border: 1px solid #2d3138; border-radius: 2px; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #30123b, stop:0.15 #4661e0, stop:0.30 #1ce5d5, "
            "stop:0.50 #6cfd62, stop:0.70 #fdbb2d, stop:0.85 #f06b1d, "
            "stop:1 #7a0402); }")
        lh.addWidget(bar)
        lbl_hi = QtWidgets.QLabel("bright")
        lbl_hi.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 10px;")
        lh.addWidget(lbl_hi)
        header.addWidget(legend_w)
        # Bandpass-filtered view toggle — shows what trackpy actually sees
        # after its preprocessing step, which makes real PSFs pop and flat
        # background noise drop away.  Useful for picking a good minmass.
        self._cb_filtered = QtWidgets.QCheckBox("Filtered view")
        self._cb_filtered.setToolTip(
            "Show the bandpass-filtered image (what trackpy sees) instead of\n"
            "the raw frame.  Real PSFs come up bright on a flat dark background;\n"
            "noise stays small.  Detection runs against this same filtering\n"
            "internally, so what you see is closer to what the detector decides.")
        self._cb_filtered.toggled.connect(self._on_filtered_toggled)
        header.addWidget(self._cb_filtered)
        self._b_clear = QtWidgets.QPushButton("Clear polygons")
        self._b_clear.setToolTip(
            "Remove every polygon drawn on the current file's ROI.")
        self._b_clear.clicked.connect(self._on_clear)
        header.addWidget(self._b_clear)
        v.addLayout(header)

        self._viewer_container = QtWidgets.QFrame()
        self._viewer_container.setObjectName("results_panel")
        self._viewer_container.setMinimumHeight(320)
        self._viewer_layout = QtWidgets.QVBoxLayout(self._viewer_container)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        # Placeholder until napari is loaded (lazy)
        self._placeholder = QtWidgets.QLabel(
            "Pick a file above and the ROI viewer will load here."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; padding: 40px;")
        self._viewer_layout.addWidget(self._placeholder)
        v.addWidget(self._viewer_container, stretch=1)

    # ── Lazy napari init ─────────────────────────────────────────────────
    def _ensure_viewer(self) -> bool:
        if self._viewer is not None:
            return True
        try:
            import napari
        except Exception as exc:
            self._status.setText(
                f"napari not available: {exc} — install it to use the preview viewer.")
            return False
        try:
            self._viewer = napari.Viewer(show=False)
            qt_window = self._viewer.window._qt_window
            self._viewer_layout.removeWidget(self._placeholder)
            self._placeholder.hide()
            self._viewer_layout.addWidget(qt_window)
            # Hide napari's "Drag image(s) here" welcome overlay + the
            # bottom-of-canvas viewer-buttons row (ndisplay / grid / home /
            # console / etc.) + the new-layer/delete buttons under the
            # layer list — all visual noise for a viewer driven entirely
            # programmatically by FIREFLY.
            _hide_napari_chrome(self._viewer)
            # Re-run detection when the user scrubs frames (idempotent)
            if not self._dims_connected:
                try:
                    self._viewer.dims.events.current_step.connect(
                        self._on_dims_changed)
                    self._dims_connected = True
                except Exception:
                    pass
            # Auto-recover if the user deletes a layer we own (e.g. the
            # ROI shapes layer via napari's trash button).
            try:
                self._viewer.layers.events.removed.connect(
                    self._on_layer_removed)
            except Exception:
                pass
            return True
        except Exception as exc:
            self._status.setText(f"Couldn't embed napari viewer: {exc}")
            return False

    def _on_layer_removed(self, event):
        """Recover from the user deleting one of our managed layers.

        Shapes layer → recreate empty (so they can keep drawing polygons,
        and let the host know the previous polygons are gone).
        Image / points / mask layers → just null the stored reference;
        the next set_file / detection run / mask-refresh will re-create
        them lazily.
        """
        # Programmatic layer teardown (e.g. set_file clearing the old
        # file's layers before loading a new one) sets this flag so we
        # don't fight napari's iterator by reinserting layers mid-clear.
        if self._suppress_layer_events:
            return
        try:
            removed = getattr(event, "value", None)
        except Exception:
            removed = None
        if removed is None or self._viewer is None:
            return
        if removed is self._shapes_layer:
            try:
                import numpy as _np
                self._shapes_layer = self._viewer.add_shapes(
                    data=None,
                    shape_type="polygon",
                    edge_color="#58a6ff",
                    face_color="rgba(88,166,255,0.18)",
                    edge_width=2,
                    name="ROI",
                )
                try:    self._shapes_layer.mode = "add_polygon"
                except Exception: pass
                try:    self._shapes_layer.events.data.connect(
                            self._on_shapes_changed)
                except Exception: pass
                # Notify host that polygons for the current file have
                # been wiped (matches the napari state on disk).
                if self._current_file:
                    self.polygons_changed.emit(self._current_file, [])
                self._status.setText(
                    "ROI layer was deleted — recreated empty.  "
                    "Draw a new polygon to set the ROI.")
            except Exception:
                self._shapes_layer = None
            return
        if removed is self._image_layer:
            self._image_layer = None
            return
        if removed is self._points_layer:
            self._points_layer = None
            return
        if removed is self._roi_mask_layer:
            self._roi_mask_layer = None
            return

    # ── Public API ───────────────────────────────────────────────────────
    def set_file(self, file_path: str,
                 current_polygons: list | None = None):
        """Switch the viewer to `file_path` and load its preview.

        Auto-emits the current polygons as `polygons_changed` whenever
        the user adds / edits / removes a shape, so the host MainWindow
        can persist the change without an explicit Save button.
        """
        # If we have an active file with edits, flush them before switching.
        self._flush_current_polygons_if_changed()

        self._current_file = file_path or ""

        if not file_path:
            self._title.setText("Preview viewer")
            self._status.setText("Pick a file to start")
            return
        if not os.path.isfile(file_path):
            self._status.setText(f"File not found: {file_path}")
            return

        if not self._ensure_viewer():
            return

        # Clear out any old layers from a previous file.  Wrap the clear
        # in `_suppress_layer_events = True` so _on_layer_removed doesn't
        # try to recreate layers MID-clear — that recursion is what
        # froze the GUI when switching files.
        self._suppress_layer_events = True
        try:
            try:    self._viewer.layers.clear()
            except Exception: pass
        finally:
            self._suppress_layer_events = False
        self._image_layer = None
        self._shapes_layer = None
        self._points_layer = None
        self._stack = None
        self._stack_filtered = None
        self._stack_preprocessed = None
        self._pp_signature = None
        self._last_mass = None
        self._roi_mask_layer = None
        # Reset the toggle silently so it doesn't fight the new image
        try:
            self._cb_filtered.blockSignals(True)
            self._cb_filtered.setChecked(False)
            self._cb_filtered.blockSignals(False)
        except AttributeError:
            pass

        self._title.setText(f"Preview — {os.path.basename(file_path)}")
        self._status.setText("Loading preview…")

        try:
            import numpy as _np
            stack, _idx = _RoiDialog._quick_preview_stack(file_path, max_frames=30)
            self._stack = stack
            # Percentile-based contrast so a few hot pixels don't blow out the
            # display.  Sample a single mid-stack frame for speed.
            sample = stack[stack.shape[0] // 2]
            lo, hi = _np.percentile(sample, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self._image_layer = self._viewer.add_image(
                stack, name="ROI background", colormap="gray",
                contrast_limits=(float(lo), float(hi)))
            initial_shapes = [_np.asarray(poly)
                              for poly in (current_polygons or [])]
            self._shapes_layer = self._viewer.add_shapes(
                data=initial_shapes if initial_shapes else None,
                shape_type="polygon",
                edge_color="#58a6ff",
                face_color="rgba(88,166,255,0.18)",
                edge_width=2,
                name="ROI",
            )
            try:
                self._shapes_layer.mode = "add_polygon"
            except Exception:
                pass
            # Auto-save: emit whenever the shapes layer changes
            try:
                self._shapes_layer.events.data.connect(self._on_shapes_changed)
            except Exception:
                pass
            self._status.setText(
                f"{stack.shape[0]}-frame preview, "
                f"{stack.shape[2]} × {stack.shape[1]} px — "
                "draw polygon(s); right-click or Esc to close each one.")
            # Re-arm detection preview if the host has it enabled.
            if self._detect_enabled:
                self._detect_debounce.start()
            # Re-draw the auto / manual-threshold mask overlay if the
            # host previously set its parameters.
            self._refresh_roi_mask_overlay()
        except Exception as exc:
            import traceback as _tb
            self._status.setText(
                f"Couldn't load preview: {exc}\n{_tb.format_exc()}")

    def current_file(self) -> str:
        return self._current_file

    def current_polygons(self) -> list:
        polys = []
        if self._shapes_layer is None:
            return polys
        try:
            for shape_data, shape_type in zip(self._shapes_layer.data,
                                              self._shapes_layer.shape_type):
                if shape_type in ("polygon", "rectangle", "ellipse"):
                    polys.append([(float(y), float(x))
                                  for y, x in shape_data])
        except Exception:
            pass
        return polys

    # ── Internal ─────────────────────────────────────────────────────────
    def _flush_current_polygons_if_changed(self):
        """Emit a final polygons_changed for the outgoing file, in case
        the user drew something but never triggered the data-changed event."""
        if self._current_file and self._shapes_layer is not None:
            try:
                self.polygons_changed.emit(
                    self._current_file, self.current_polygons())
            except Exception:
                pass

    def _on_shapes_changed(self, _event=None):
        if self._current_file:
            try:
                self.polygons_changed.emit(
                    self._current_file, self.current_polygons())
            except Exception:
                pass

    def _on_clear(self):
        if self._shapes_layer is None:
            return
        try:
            self._shapes_layer.data = []
        except Exception:
            pass
        if self._current_file:
            self.polygons_changed.emit(self._current_file, [])

    # ── Detection preview ────────────────────────────────────────────────
    def enable_detection_preview(self, enabled: bool):
        """Toggle a `tp.locate` overlay on the current frame."""
        self._detect_enabled = bool(enabled)
        if not self._detect_enabled:
            self._remove_points_layer()
            return
        if self._stack is None:
            return
        self._detect_debounce.start()

    def set_detection_params(self, *, diameter: int, minmass: float,
                             bg_method: str = "uniform_filter",
                             bg_radius: int = 50):
        """Update the diameter / minmass / preprocessing used by the
        live overlay.  Matching the pipeline's preprocessing is what makes
        the mass scale here agree with what the run will actually see."""
        new_sig = (str(bg_method), int(bg_radius))
        if self._pp_signature is not None and new_sig != self._pp_signature:
            # Background settings changed → invalidate cached preprocessed stack
            self._stack_preprocessed = None
        self._detect_params = {"diameter": int(diameter),
                               "minmass":  float(minmass),
                               "bg_method": str(bg_method),
                               "bg_radius": int(bg_radius)}
        if self._detect_enabled and self._stack is not None:
            self._detect_debounce.start()

    def _on_dims_changed(self, _evt=None):
        if self._detect_enabled and self._stack is not None:
            self._detect_debounce.start()

    def _current_frame_idx(self) -> int:
        if self._viewer is None or self._stack is None:
            return 0
        try:
            return int(self._viewer.dims.current_step[0])
        except Exception:
            return 0

    def _run_detection(self):
        if not self._detect_enabled or self._stack is None or self._viewer is None:
            return
        idx = max(0, min(self._current_frame_idx(), self._stack.shape[0] - 1))
        # Locate on the PREPROCESSED frame so the mass scale here matches
        # what the pipeline produces during the real run.  The bandpass view
        # toggle is a display aid only and doesn't affect numbers.
        pp = self._ensure_preprocessed_stack()
        if pp is None:
            self._status.setText("Preprocessing failed — falling back to raw frame.")
            frame = self._stack[idx]
        else:
            frame = pp[idx]
        diameter = self._detect_params["diameter"]
        if diameter % 2 == 0:
            diameter += 1
        minmass = self._detect_params["minmass"]
        try:
            import trackpy as tp
            df = tp.locate(frame, diameter=diameter, minmass=minmass)
        except Exception as exc:
            self._status.setText(f"Detection preview failed: {exc}")
            return
        import numpy as _np
        if len(df):
            pts  = df[["y", "x"]].to_numpy()
            mass = df["mass"].to_numpy() if "mass" in df.columns else None
        else:
            pts  = _np.zeros((0, 2), dtype=float)
            mass = None
        self._last_mass = mass
        self._update_points_layer(pts, diameter, mass)
        # Mass-distribution summary helps the user pick a useful minmass.
        if mass is not None and len(mass):
            m_lo, m_med, m_hi = (float(_np.min(mass)),
                                 float(_np.median(mass)),
                                 float(_np.max(mass)))
            mass_summary = f"mass {m_lo:.0f} / med {m_med:.0f} / {m_hi:.0f}"
        else:
            mass_summary = "no spots"
        self._status.setText(
            f"Frame {idx + 1}/{self._stack.shape[0]} — "
            f"{len(df)} spots (d={diameter}, minmass={minmass:g}) — {mass_summary}")

    def _mass_to_rgba(self, mass):
        """Return (N, 4) RGBA in 0..1 using turbo on log(mass) — high
        contrast on a grey background at both ends of the scale."""
        import numpy as _np
        try:
            import matplotlib.cm as _cm
            import matplotlib.colors as _mc
            m = _np.asarray(mass, dtype=float)
            # log scale keeps dim spots distinguishable from bright ones
            logm = _np.log10(_np.clip(m, 1e-3, None))
            if logm.size == 0:
                return _np.zeros((0, 4))
            vmin = float(_np.min(logm))
            vmax = float(_np.max(logm))
            if vmax <= vmin + 1e-9:
                vmax = vmin + 1.0
            norm = _mc.Normalize(vmin=vmin, vmax=vmax)
            try:
                cmap = _cm.get_cmap("turbo")
            except Exception:
                cmap = _cm.viridis
            rgba = cmap(norm(logm))
            return _np.asarray(rgba, dtype=float)
        except Exception:
            n = len(mass) if mass is not None else 0
            return _np.tile([0.0, 1.0, 1.0, 1.0], (n, 1))

    def _update_points_layer(self, pts, diameter: int, mass=None):
        import numpy as _np
        size = max(4, int(diameter) + 6)
        # Build per-point colour array (turbo on log mass) for visibility.
        if mass is not None and len(mass) > 0:
            colours = self._mass_to_rgba(mass)
        else:
            colours = _np.tile([0.0, 1.0, 1.0, 1.0],
                               (len(pts), 1)) if len(pts) else None
        if self._points_layer is None:
            kwargs = dict(
                size=size,
                face_color="transparent",
                symbol="o",
                name="Detections",
                opacity=1.0,
            )
            try:
                self._points_layer = self._viewer.add_points(
                    pts,
                    border_color=(colours if colours is not None else "#00ffff"),
                    border_width=0.30,
                    **kwargs)
            except TypeError:
                # napari < 0.5 — edge_* names
                self._points_layer = self._viewer.add_points(
                    pts,
                    edge_color=(colours if colours is not None else "#00ffff"),
                    edge_width=0.30,
                    **kwargs)
            except Exception as exc:
                self._status.setText(f"Points layer failed: {exc}")
                self._points_layer = None
                return
            try:
                if self._shapes_layer is not None:
                    self._viewer.layers.selection.active = self._shapes_layer
            except Exception:
                pass
        else:
            try:
                self._points_layer.data = pts
                self._points_layer.size = size
                if colours is not None and len(colours):
                    try:
                        self._points_layer.border_color = colours
                    except Exception:
                        try: self._points_layer.edge_color = colours
                        except Exception: pass
            except Exception:
                pass

    # ── Bandpass-filtered view ───────────────────────────────────────────
    def _on_filtered_toggled(self, checked: bool):
        if self._viewer is None or self._image_layer is None or self._stack is None:
            return
        if checked:
            if self._stack_filtered is None:
                self._stack_filtered = self._compute_filtered_stack(self._stack)
            target = self._stack_filtered
        else:
            target = self._stack
        try:
            self._image_layer.data = target
            import numpy as _np
            sample = target[target.shape[0] // 2]
            lo, hi = _np.percentile(sample, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self._image_layer.contrast_limits = (float(lo), float(hi))
        except Exception as exc:
            self._status.setText(f"Couldn't swap image: {exc}")

    def _ensure_preprocessed_stack(self):
        """Lazily build a pipeline-equivalent preprocessed stack so that
        `tp.locate` here sees the same intensities as the real run.  Mirrors
        sptpalm_analysis._preprocess_fast / _preprocess_rolling: background
        subtract → clip ≥0 → gaussian sigma=1 → per-frame normalise to [0,1].
        """
        if self._stack is None:
            return None
        bg_method = self._detect_params.get("bg_method", "uniform_filter")
        bg_radius = int(self._detect_params.get("bg_radius", 50)) or 50
        sig = (str(bg_method), bg_radius)
        if self._stack_preprocessed is not None and self._pp_signature == sig:
            return self._stack_preprocessed
        try:
            import numpy as _np
            from scipy.ndimage import uniform_filter, gaussian_filter
        except Exception:
            return None
        rolling_fn = None
        if bg_method == "rolling_ball":
            try:
                from skimage.restoration import rolling_ball as rolling_fn
            except Exception:
                rolling_fn = None  # fall back to uniform filter silently

        size = int(bg_radius * 2 + 1)
        out = _np.empty(self._stack.shape, dtype=_np.float32)
        for i in range(self._stack.shape[0]):
            f = self._stack[i].astype(_np.float32, copy=False)
            if rolling_fn is not None:
                try:
                    bg = rolling_fn(f, radius=bg_radius)
                except Exception:
                    bg = uniform_filter(f, size=size)
            else:
                bg = uniform_filter(f, size=size)
            corrected = _np.clip(f - bg, 0, None)
            smoothed  = gaussian_filter(corrected, sigma=1.0)
            mn = float(smoothed.min()); mx = float(smoothed.max())
            if mx > mn:
                smoothed = (smoothed - mn) / (mx - mn)
            out[i] = smoothed
        self._stack_preprocessed = out
        self._pp_signature = sig
        return self._stack_preprocessed

    # ── Auto / manual-threshold ROI overlay ──────────────────────────────
    def set_roi_mask_params(self, *, mode: str, auto_method: str,
                            threshold: float, mask_mode: str):
        """Update + redraw the auto/manual-threshold ROI overlay layer.
        `mode` is "None", "Auto threshold", "Manual threshold" or
        "Manual polygon"; the overlay is drawn for the first two."""
        self._roi_mask_params = {"mode": str(mode),
                                 "auto_method": str(auto_method).lower(),
                                 "threshold": float(threshold),
                                 "mask_mode": str(mask_mode).lower()}
        self._refresh_roi_mask_overlay()

    def _refresh_roi_mask_overlay(self):
        if self._viewer is None or self._stack is None:
            return
        mode = self._roi_mask_params.get("mode", "None")
        if mode not in ("Auto threshold", "Manual threshold"):
            self._remove_roi_mask_layer()
            return
        try:
            import numpy as _np
            from scipy.ndimage import gaussian_filter
            from skimage.morphology import binary_closing, disk
        except Exception:
            return
        # Build projection (mean or sum) and renormalise to [0,1]
        proj = (self._stack.sum(axis=0)
                if self._roi_mask_params["mask_mode"] == "sum"
                else self._stack.mean(axis=0)).astype(_np.float32)
        smoothed = gaussian_filter(proj, sigma=5.0)
        mn, mx = float(smoothed.min()), float(smoothed.max())
        if mx > mn:
            smoothed = (smoothed - mn) / (mx - mn)
        # Pick threshold
        if mode == "Manual threshold":
            t = float(self._roi_mask_params["threshold"])
        else:
            t = self._auto_threshold(smoothed,
                                     self._roi_mask_params["auto_method"])
            if t is None:
                t = float(self._roi_mask_params["threshold"])
        try:
            mask = binary_closing(smoothed > t, disk(5))
        except Exception:
            mask = smoothed > t
        self._draw_roi_mask_layer(mask, t)

    @staticmethod
    def _auto_threshold(image_norm, method: str):
        try:
            from skimage.filters import (threshold_otsu, threshold_li,
                                         threshold_triangle)
        except Exception:
            return None
        method = (method or "li").lower()
        try:
            if method == "otsu":     return float(threshold_otsu(image_norm))
            if method == "li":       return float(threshold_li(image_norm))
            if method == "triangle": return float(threshold_triangle(image_norm))
            if method == "mean":     return float(image_norm.mean())
        except Exception:
            pass
        return None

    def _draw_roi_mask_layer(self, mask, threshold: float):
        import numpy as _np
        # Convert bool mask to (Y, X) uint8 so we can colour it via a
        # custom colormap with transparency at 0.
        layer_data = mask.astype(_np.uint8)
        if self._roi_mask_layer is None:
            try:
                # Render through a 2-stop colormap: 0 = transparent,
                # 1 = bright lime so the mask is unmistakable on grey.
                from napari.utils.colormaps import Colormap as _NCmap
                cmap = _NCmap([[0, 0, 0, 0], [0.20, 1.00, 0.30, 1.0]],
                              name="firefly_roi_mask")
                self._roi_mask_layer = self._viewer.add_image(
                    layer_data, name="ROI mask", colormap=cmap,
                    contrast_limits=(0, 1), opacity=0.35,
                    blending="translucent")
            except Exception:
                try:
                    self._roi_mask_layer = self._viewer.add_image(
                        layer_data, name="ROI mask", colormap="green",
                        contrast_limits=(0, 1), opacity=0.35,
                        blending="translucent")
                except Exception as exc:
                    self._status.setText(f"ROI mask layer failed: {exc}")
                    return
            # Re-select shapes layer so polygon drawing keeps working
            try:
                if self._shapes_layer is not None:
                    self._viewer.layers.selection.active = self._shapes_layer
            except Exception:
                pass
        else:
            try:
                self._roi_mask_layer.data = layer_data
            except Exception:
                pass
        # Refresh status with the threshold the user can see and tune
        try:
            n_in = int(_np.sum(mask))
            total = int(mask.size)
            pct = 100.0 * n_in / total if total else 0.0
            self._roi_mask_layer.metadata = {"threshold": threshold,
                                             "fraction": pct}
        except Exception:
            pass

    def _remove_roi_mask_layer(self):
        if self._roi_mask_layer is not None and self._viewer is not None:
            try:
                self._viewer.layers.remove(self._roi_mask_layer)
            except Exception:
                pass
        self._roi_mask_layer = None

    def _compute_filtered_stack(self, stack):
        """Bandpass-filter every frame using trackpy.bandpass (matches what
        tp.locate does internally), so the viewer shows what the detector
        actually sees."""
        import numpy as _np
        diameter = int(self._detect_params.get("diameter", 7)) or 7
        if diameter % 2 == 0:
            diameter += 1
        try:
            import trackpy as tp
            out = _np.empty_like(stack)
            for i in range(stack.shape[0]):
                out[i] = tp.bandpass(stack[i], lshort=1, llong=diameter)
            return out
        except Exception:
            # Fall back to a difference-of-gaussians if trackpy.bandpass is
            # missing or fails — same idea, slightly different kernel.
            try:
                from scipy.ndimage import gaussian_filter
                out = _np.empty_like(stack, dtype=_np.float32)
                short = 1.0
                long  = max(1.5, diameter / 2.0)
                for i in range(stack.shape[0]):
                    f = stack[i].astype(_np.float32)
                    out[i] = gaussian_filter(f, short) - gaussian_filter(f, long)
                return out
            except Exception:
                return stack

    def _remove_points_layer(self):
        if self._points_layer is not None and self._viewer is not None:
            try:
                self._viewer.layers.remove(self._points_layer)
            except Exception:
                pass
        self._points_layer = None


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

        # Elapsed-time tracker for the Analysis tab.  1 Hz tick that
        # updates the "Elapsed: 00:32" label while a run is active.
        self._run_start_time: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        # Per-file polygon ROIs.  Keyed by absolute file path; each value
        # is a list of polygons, each polygon a list of (y, x) vertex
        # tuples in pixel coords.  Persisted to QSettings as JSON.
        self._roi_polygons: dict[str, list] = {}

        self._build_ui()
        self._install_menubar()
        self._install_crash_hooks()
        self._load_icon()
        self._load_roi_polygons()
        self._restore_settings()
        # Initialise ROI status labels now that both settings and polygons
        # are loaded.
        try:
            self._refresh_single_roi_status()
            self._refresh_batch_roi_markers()
        except Exception:
            pass
        # ROI viewer loading is now EXPLICIT — driven by the
        # "Load into ROI viewer" button or a double-click on a batch
        # list item.  The earlier auto-load-on-selection design caused
        # heavy work (napari + file load) on every single mouse click,
        # which raced with the user's checkbox toggles and made some
        # files un-toggleable.
        # All we still do reactively is keep the single-file ROI status
        # label in sync.
        try:
            self.e_file.textChanged.connect(
                lambda _: self._refresh_single_roi_status())
        except Exception:
            pass

        # Auto-load the active file into the embedded ROI viewer whenever
        # the path settles (debounced so we don't fire load-after-every-
        # keystroke while the user is typing or pasting a path).
        self._roi_autoload_timer = QTimer(self)
        self._roi_autoload_timer.setSingleShot(True)
        self._roi_autoload_timer.setInterval(400)
        self._roi_autoload_timer.timeout.connect(
            self._roi_embedded_load_current_file)
        try:
            self.e_file.textChanged.connect(
                lambda _: self._roi_autoload_timer.start())
        except Exception:
            pass

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Top-level vertical: [header bar] / [stack: landing OR main UI]
        top = QtWidgets.QVBoxLayout(central)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)

        # ── Header banner ────────────────────────────────────────────────
        top.addWidget(self._build_header_banner())

        # ── Stacked body: landing page (idx 0) vs main UI (idx 1) ────────
        # Landing is a one-way gateway — once the user picks an action it
        # disappears for the rest of the session.  Main UI rebuilds the
        # sidebar + tab interface from before.
        self._main_stack = QtWidgets.QStackedWidget()
        top.addWidget(self._main_stack, stretch=1)
        self._main_stack.addWidget(self._build_landing_page())

        body = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._main_stack.addWidget(body)

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

        # Mirror each row-widget's tooltip onto its label, so hovering
        # the SETTING NAME (e.g. "Pixel size (µm)") shows the same
        # explanation as hovering the spinbox.  Done in one post-build
        # sweep so individual addRow() calls don't have to remember to
        # do it themselves.
        self._propagate_form_tooltips(scroll_inner)

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
        self._build_figures_tab()
        self._build_compare_tab()
        self._build_visualise_tab()
        layout.addWidget(self.tabs, stretch=1)

        # Start on the landing page; main UI activates only after the user
        # picks an action card.
        self._main_stack.setCurrentIndex(0)

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

        # "Update available" pill — hidden on startup, lit up by the
        # background update-check thread if GitHub Releases reports a
        # newer tag than __version__.  Clicking opens the Releases page.
        self.btn_update_pill = QtWidgets.QPushButton("")
        self.btn_update_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update_pill.setVisible(False)
        self.btn_update_pill.setStyleSheet(
            f"QPushButton {{ background-color: {_THEME['ACC']}; "
            f"color: {_THEME['ACC_FG']}; border: none; "
            "border-radius: 10px; padding: 4px 10px; "
            "font-size: 11px; font-weight: 700; }} "
            f"QPushButton:hover {{ background-color: {_THEME['ACC_HOVER']}; }}")
        self.btn_update_pill.clicked.connect(self._on_update_pill_clicked)
        h.addWidget(self.btn_update_pill)

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

        # Fire off the update check 2 s after startup so it doesn't
        # block the initial paint.
        QtCore.QTimer.singleShot(2000, self._kick_off_update_check)

        return bar

    # ── Auto-update check ─────────────────────────────────────────────────
    _UPDATE_REPO = "jacob-levers/FIREFLY"
    _UPDATE_RELEASES_URL = (
        f"https://github.com/jacob-levers/FIREFLY/releases")
    _UPDATE_API_URL = (
        f"https://api.github.com/repos/jacob-levers/FIREFLY/releases/latest")

    def _kick_off_update_check(self):
        """Hit GitHub Releases asynchronously and show the update pill
        in the header if a newer tag is available than __version__."""
        try:
            import sptpalm_analysis as _sa
            current = str(getattr(_sa, "__version__", "0.0.0"))
        except Exception:
            current = "0.0.0"

        self._update_thread = _UpdateCheckThread(
            self._UPDATE_API_URL, current, parent=self)
        self._update_thread.update_available.connect(self._on_update_available)
        self._update_thread.start()

    def _on_update_available(self, latest_tag: str, html_url: str):
        """Slot called when the background thread finds a newer release."""
        if not hasattr(self, "btn_update_pill"):
            return
        self._update_url = html_url or self._UPDATE_RELEASES_URL
        self.btn_update_pill.setText(f"  ●  Update available: {latest_tag}  ")
        self.btn_update_pill.setToolTip(
            f"FIREFLY {latest_tag} is available on GitHub.  "
            "Click to open the Releases page.")
        self.btn_update_pill.setVisible(True)

    def _on_update_pill_clicked(self):
        url = getattr(self, "_update_url", self._UPDATE_RELEASES_URL)
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        except Exception:
            pass

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
        # Docked at the bottom by default — Qt will shrink the central
        # widget to accommodate, rather than growing the window.  Users
        # can still drag the dock's title bar out to float it, or close
        # it via the × button.  Hidden until the Console toolbar button
        # is clicked.
        self._console_dock.setFloating(False)
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
    def _propagate_form_tooltips(root: "QtWidgets.QWidget") -> None:
        """Walk every QFormLayout in `root`'s subtree and copy each
        row-widget's tooltip onto its label so hovering the SETTING
        NAME (not just the spinbox / combo) reveals the explanation.

        For rows whose field is a wrapper QWidget containing other
        widgets (e.g. the Pixel-size row's `[Override checkbox]
        [spinbox]` pair), we look for the most informative child
        tooltip instead of the wrapper's own (which is empty).
        """
        def _best_tip(widget: "QtWidgets.QWidget") -> str:
            tip = widget.toolTip() or ""
            if tip.strip():
                return tip
            # Wrapper widget — fall through to the most specific child
            # that carries a tooltip.  Prefer spinboxes / combos /
            # sliders, then any other tooltip-bearing widget.
            preferred = (QtWidgets.QAbstractSpinBox,
                         QtWidgets.QComboBox,
                         QtWidgets.QSlider,
                         QtWidgets.QCheckBox,
                         QtWidgets.QLineEdit)
            best = ""
            for cls in preferred:
                for child in widget.findChildren(cls):
                    t = (child.toolTip() or "").strip()
                    if t:
                        return t
            for child in widget.findChildren(QtWidgets.QWidget):
                t = (child.toolTip() or "").strip()
                if t and len(t) > len(best):
                    best = t
            return best

        # findChildren on a QWidget returns every QObject descendant
        # matching the type; QFormLayout is a QObject under the parent
        # widget hierarchy.
        for form in root.findChildren(QtWidgets.QFormLayout):
            try:
                rows = form.rowCount()
            except Exception:
                continue
            label_role = QtWidgets.QFormLayout.ItemRole.LabelRole
            field_role = QtWidgets.QFormLayout.ItemRole.FieldRole
            for r in range(rows):
                lbl_item = form.itemAt(r, label_role)
                fld_item = form.itemAt(r, field_role)
                if lbl_item is None or fld_item is None:
                    continue
                lbl = lbl_item.widget()
                fld = fld_item.widget()
                if lbl is None or fld is None:
                    continue
                if (lbl.toolTip() or "").strip():
                    continue   # caller already set one explicitly
                tip = _best_tip(fld)
                if tip:
                    lbl.setToolTip(tip)
                    # macOS sometimes needs a wider hover region than
                    # the label's tightly-fitted geometry; this attribute
                    # lets the label receive enter/leave events properly.
                    lbl.setAttribute(
                        Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)

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
    def _make_mode_tile(title: str, subtitle: str,
                        icon_char: str = "") -> "_ModeTile":
        """Big segmented-control tile (custom widget — see `_ModeTile`)."""
        return _ModeTile(title, subtitle, icon_char)

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

        # ── Presets ───────────────────────────────────────────────────────
        # Quick switcher for labelled parameter bundles.  Selecting a
        # preset applies its widget snapshot to the rest of the sidebar.
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 6)
        preset_row.setSpacing(6)
        preset_row.addWidget(QtWidgets.QLabel("Preset"))
        self.c_preset = _QuietComboBox()
        self.c_preset.setToolTip(
            "Switch between labelled parameter bundles stored in\n"
            "~/.firefly/presets/.  Two defaults ship out of the box;\n"
            "use the disk icon to save the current sidebar as a new one.")
        preset_row.addWidget(self.c_preset, 1)
        self.btn_preset_save = QtWidgets.QToolButton()
        self.btn_preset_save.setText("Save…")
        self.btn_preset_save.setToolTip(
            "Save the current sidebar values as a new preset.")
        self.btn_preset_save.clicked.connect(self._on_preset_save)
        preset_row.addWidget(self.btn_preset_save)
        self.btn_preset_delete = QtWidgets.QToolButton()
        self.btn_preset_delete.setText("Delete")
        self.btn_preset_delete.setToolTip(
            "Delete the currently-selected preset from\n"
            "~/.firefly/presets/.  Built-in presets are re-seeded on the\n"
            "next launch unless you save your own version with the same\n"
            "name first.")
        self.btn_preset_delete.clicked.connect(self._on_preset_delete)
        preset_row.addWidget(self.btn_preset_delete)
        layout.addLayout(preset_row)
        # Deferred wiring — combobox change must apply only after construction
        # of every widget the preset references.  See `_finalise_presets`.
        QtCore.QTimer.singleShot(0, self._finalise_presets)

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
        self.c_bg_method = _QuietComboBox()
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
        self.c_auto_minmass = QtWidgets.QCheckBox("Auto-detect")
        self.c_auto_minmass.setToolTip(
            "When checked, the pipeline picks the detection threshold from\n"
            "the first chunk's 99th-percentile pixel value × diameter²/8.\n"
            "Heuristic — works on many datasets but may under-shoot; manual\n"
            "tuning is more reliable.\n\n"
            "Equivalent to PALM-Tracer's 'Threshold' field but measured on\n"
            "the integrated raw intensity (trackpy 'mass') rather than on\n"
            "the wavelet domain.")
        self.c_auto_minmass.setChecked(False)
        self.s_minmass = self._spin_dbl(1.0, 0.0, 100.0, 0.05, decimals=2,
            tip="Detection threshold — minimum integrated intensity\n"
                "(trackpy 'mass') for a spot to be kept.\n\n"
                "Too low → many false-positive spots, slow linking, garbage tracks.\n"
                "Too high → real spots filtered out.\n\n"
                "After preprocessing (background subtract + per-frame\n"
                "normalise to [0,1]) values typically land in the 0.5–50\n"
                "range.  Start near 1.0 and sweep the slider — dim points\n"
                "vanish first as you raise it.\n\n"
                "Equivalent role to PALM-Tracer's 'Threshold' field, but the\n"
                "unit is integrated raw intensity here, not k-σ on the\n"
                "wavelet plane — values don't transfer directly between tools.")
        # Slider companion — QSlider is integer-only, and a plain linear
        # mapping over 0..5000 wastes 99% of slider travel on values above
        # the useful range.  Use a square law: minmass = (slider/1000)² × MAX
        # so slider≈100 ↔ minmass 50, slider≈316 ↔ minmass 500.
        _MM_SLD_MAX = 1000
        _MM_VAL_MAX = float(self.s_minmass.maximum())
        def _slider_to_mass(s: int) -> float:
            t = max(0, min(_MM_SLD_MAX, int(s))) / _MM_SLD_MAX
            return float(t * t * _MM_VAL_MAX)
        def _mass_to_slider(m: float) -> int:
            import math as _math
            t = max(0.0, min(_MM_VAL_MAX, float(m))) / _MM_VAL_MAX
            return int(round(_math.sqrt(t) * _MM_SLD_MAX))
        self.sld_minmass = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_minmass.setMinimum(0)
        self.sld_minmass.setMaximum(_MM_SLD_MAX)
        self.sld_minmass.setSingleStep(1)
        self.sld_minmass.setPageStep(20)
        self.sld_minmass.setValue(_mass_to_slider(self.s_minmass.value()))
        self.sld_minmass.setToolTip(
            "Drag to sweep the detection threshold (square-law: fine at the\n"
            "low end, coarse at the high end).  The preview viewer updates\n"
            "spot overlays as you move.  Type into the spinbox for exact values.")
        self._minmass_sync_guard = False
        def _on_slider(v: int):
            if self._minmass_sync_guard: return
            self._minmass_sync_guard = True
            try: self.s_minmass.setValue(_slider_to_mass(v))
            finally: self._minmass_sync_guard = False
        def _on_spin(v: float):
            if self._minmass_sync_guard: return
            self._minmass_sync_guard = True
            try: self.sld_minmass.setValue(_mass_to_slider(v))
            finally: self._minmass_sync_guard = False
        self.sld_minmass.valueChanged.connect(_on_slider)
        self.s_minmass.valueChanged.connect(_on_spin)
        self.c_auto_minmass.toggled.connect(
            lambda checked: (self.s_minmass.setEnabled(not checked),
                             self.sld_minmass.setEnabled(not checked)))
        self.s_minmass.setEnabled(True)

        wmm = QtWidgets.QWidget()
        vmm = QtWidgets.QVBoxLayout(wmm)
        vmm.setContentsMargins(0, 0, 0, 0)
        vmm.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.c_auto_minmass)
        row.addWidget(self.s_minmass, 1)
        vmm.addLayout(row)
        vmm.addWidget(self.sld_minmass)
        gl.addRow("Threshold", wmm)

        # Push spinbox / combo edits into the live preview.  Background
        # widgets are wired here too because the preview re-preprocesses
        # frames using these settings to match the pipeline's mass scale.
        self.s_diameter.valueChanged.connect(
            lambda _=None: self._push_detection_preview_params())
        self.s_minmass.valueChanged.connect(
            lambda _=None: self._push_detection_preview_params())
        self.c_bg_method.currentTextChanged.connect(
            lambda _=None: self._push_detection_preview_params())
        self.s_bg_radius.valueChanged.connect(
            lambda _=None: self._push_detection_preview_params())
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
        self.c_roi_mode = _QuietComboBox()
        self.c_roi_mode.addItems(
            ["None", "Auto threshold", "Manual threshold", "Manual polygon"])
        self.c_roi_mode.setCurrentText("Auto threshold")
        self.c_roi_mode.setToolTip(
            "Restrict analysis to a region of interest in the field of view.\n"
            "• None — analyse the whole image.\n"
            "• Auto threshold — pick a threshold from the mean projection.\n"
            "• Manual threshold — use the value below.\n"
            "• Manual polygon — draw a polygon per file on the Import tab\n"
            "  (Set ROI… buttons).  Files without a saved polygon fall back\n"
            "  to the global Auto-threshold behaviour.")
        gl.addRow("Mode", self.c_roi_mode)
        self.c_roi_auto_method = _QuietComboBox()
        self.c_roi_auto_method.addItems(["Li", "Otsu", "Triangle", "Mean"])
        self.c_roi_auto_method.setToolTip(
            "Auto-thresholding method (from scikit-image).  Li is robust for\n"
            "low-contrast SMLM data; Otsu for bimodal histograms.")
        gl.addRow("Auto method", self.c_roi_auto_method)
        self.s_roi_threshold = self._spin_dbl(0.08, 0.0, 1.0, 0.005, decimals=3,
            tip="Manual threshold on the normalised mean projection [0, 1].\n"
                "Drag the slider below to sweep — the green mask overlay in\n"
                "the ROI viewer updates as you move.")
        # Slider companion — linear ×1000 mapping (range 0..1.000, step 0.001).
        self.sld_roi_threshold = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_roi_threshold.setMinimum(0)
        self.sld_roi_threshold.setMaximum(1000)
        self.sld_roi_threshold.setSingleStep(5)
        self.sld_roi_threshold.setPageStep(50)
        self.sld_roi_threshold.setValue(int(round(self.s_roi_threshold.value() * 1000)))
        self.sld_roi_threshold.setToolTip(
            "Drag to sweep manual threshold (0.000 – 1.000).  The green mask\n"
            "in the ROI viewer redraws live, so you can see exactly which\n"
            "pixels end up inside / outside the ROI.")
        self._roi_thresh_sync_guard = False
        def _on_roi_sld(v: int):
            if self._roi_thresh_sync_guard: return
            self._roi_thresh_sync_guard = True
            try: self.s_roi_threshold.setValue(v / 1000.0)
            finally: self._roi_thresh_sync_guard = False
        def _on_roi_spin(v: float):
            if self._roi_thresh_sync_guard: return
            self._roi_thresh_sync_guard = True
            try: self.sld_roi_threshold.setValue(int(round(v * 1000)))
            finally: self._roi_thresh_sync_guard = False
        self.sld_roi_threshold.valueChanged.connect(_on_roi_sld)
        self.s_roi_threshold.valueChanged.connect(_on_roi_spin)

        wrt = QtWidgets.QWidget()
        vrt = QtWidgets.QVBoxLayout(wrt)
        vrt.setContentsMargins(0, 0, 0, 0)
        vrt.setSpacing(4)
        vrt.addWidget(self.s_roi_threshold)
        vrt.addWidget(self.sld_roi_threshold)
        gl.addRow("Manual threshold", wrt)
        self.c_roi_mask_mode = _QuietComboBox()
        self.c_roi_mask_mode.addItems(["Mean", "Sum"])
        self.c_roi_mask_mode.setToolTip(
            "Which projection is used to compute the ROI mask.\n"
            "Mean is appropriate when signal density is uniform; Sum\n"
            "emphasises bright sparse spots.")
        gl.addRow("Projection for ROI", self.c_roi_mask_mode)

        # Grey out threshold-related controls when the mode doesn't use
        # them, AND show/hide the embedded ROI viewer on the Import tab
        # when "Manual polygon" is selected.
        self.c_roi_mode.currentTextChanged.connect(self._on_roi_mode_changed)
        # Push ROI mask updates to the embedded viewer whenever the user
        # changes any of these knobs.
        self.c_roi_auto_method.currentTextChanged.connect(
            lambda _=None: self._push_roi_mask_params())
        self.s_roi_threshold.valueChanged.connect(
            lambda _=None: self._push_roi_mask_params())
        self.c_roi_mask_mode.currentTextChanged.connect(
            lambda _=None: self._push_roi_mask_params())
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
        self.c_backend = _QuietComboBox()
        self.c_backend.addItems(self._available_backends())
        self.c_backend.setToolTip(
            "Which implementation to use for spot localisation.\n"
            "• Auto                — pick the fastest healthy backend on this machine.\n"
            "• Trackpy (CPU)       — reference CPU implementation (battle-tested).\n"
            "• Torch (auto)        — PyTorch, device auto-selected.\n"
            "• Torch — Apple MPS   — force Apple GPU.  Fast when stable; on some\n"
            "                        macOS/M-chip combinations may hit memory-\n"
            "                        allocator issues at very low minmass.\n"
            "• Torch — NVIDIA CUDA — force NVIDIA GPU.\n"
            "• Torch — CPU         — force PyTorch on CPU (for benchmarking).")
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

    # ── Landing page (one-way gateway, not a tab) ─────────────────────────
    def _build_landing_page(self) -> QtWidgets.QWidget:
        """Full-window welcome screen shown on launch.  Once the user picks
        an action card, the QStackedWidget swaps to the main sidebar+tabs
        UI and there's no way back to this page for the rest of the
        session."""
        page = QtWidgets.QWidget()
        page.setObjectName("landing_page")

        # Use a horizontal centring wrapper so the content column is capped
        # at ~860 px wide regardless of window width — keeps the hero text
        # readable and the cards from stretching to absurd widths.
        wrap = QtWidgets.QHBoxLayout(page)
        wrap.setContentsMargins(40, 28, 40, 28)
        wrap.addStretch(1)

        column = QtWidgets.QWidget()
        column.setMaximumWidth(860)
        outer = QtWidgets.QVBoxLayout(column)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(18)

        # Hero block.  Title uses rich text so we can colour "FIREFLY" in
        # the accent blue while keeping "Welcome to " in the default text
        # colour.
        title = QtWidgets.QLabel(
            f"Welcome to "
            f"<span style='color:{_THEME['ACC']};'>FIREFLY</span>")
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-size: 28px; font-weight: 700;")
        outer.addWidget(title)
        sub = QtWidgets.QLabel(
            "Fluorescence Inference & Reconstruction Engine — Framework "
            "for Localization Yields.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 13px;")
        outer.addWidget(sub)
        prompt = QtWidgets.QLabel("What would you like to do?")
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setStyleSheet(
            f"color: {_THEME['TXT']}; font-size: 15px; "
            "font-weight: 600; padding-top: 8px;")
        outer.addWidget(prompt)

        # Card grid — 2x2
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(14)
        grid.setContentsMargins(0, 4, 0, 0)

        def _go(target_tab: str, *, batch: bool | None = None):
            def _fn():
                if batch is True:
                    try: self.r_mode_batch.setChecked(True)
                    except AttributeError: pass
                elif batch is False:
                    try: self.r_mode_single.setChecked(True)
                    except AttributeError: pass
                self._enter_main_ui(target_tab)
            return _fn

        tiles = [
            ("Analyse a sample",
             "Pick one .czi or .tif and run the full sptPALM pipeline.",
             "▶", _go("Import", batch=False)),
            ("Batch a folder",
             "Process every file in a folder, one after another, with shared settings.",
             "⊞", _go("Import", batch=True)),
            ("Compare groups",
             "Load existing analysis outputs and produce a side-by-side comparison figure.",
             "⇄", _go("Compare")),
            ("Visualise tracks",
             "Open a previous run in an embedded napari viewer to scrub frames and explore tracks.",
             "◉", _go("Visualise")),
        ]
        for i, (ttl, desc, icon, slot) in enumerate(tiles):
            tile = _ActionTile(ttl, desc, icon_char=icon)
            tile.clicked.connect(slot)
            grid.addWidget(tile, i // 2, i % 2)
        outer.addLayout(grid, stretch=1)

        # Footer row — secondary jump-link to Figures
        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(20)
        footer.addStretch(1)
        btn = QtWidgets.QPushButton("Customise figures →")
        btn.setFlat(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ color: {_THEME['ACC']}; "
            "background: transparent; border: none; padding: 6px 8px; }} "
            f"QPushButton:hover {{ color: {_THEME['ACC_HOVER']}; "
            "text-decoration: underline; }}")
        btn.clicked.connect(lambda _=None: self._enter_main_ui("Figures"))
        footer.addWidget(btn)
        footer.addStretch(1)
        outer.addLayout(footer)

        wrap.addWidget(column, stretch=0)
        wrap.addStretch(1)
        return page

    def _enter_main_ui(self, target_tab: str):
        """Swap the QStackedWidget from landing → main UI and activate the
        named tab.  Called once per session, on action-card click."""
        self._main_stack.setCurrentIndex(1)
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == target_tab:
                self.tabs.setCurrentIndex(i)
                return

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
        # Segmented control: two big tile buttons, exclusive.  Looks like
        # a pair of cards — fills the available width and makes the choice
        # feel deliberate rather than incidental.
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(12)

        self.r_mode_single = self._make_mode_tile(
            "Single file",
            "Analyse one .czi / .tif file end-to-end")
        self.r_mode_batch = self._make_mode_tile(
            "Batch (folder)",
            "Process every file in a folder, one after another")
        self.r_mode_csv = self._make_mode_tile(
            "External CSV",
            "Skip detection — load localisations from PALM-Tracer /\n"
            "ThunderSTORM / Picasso and run linking + downstream\n"
            "analyses only")
        self.r_mode_single.setChecked(True)

        # Manual exclusivity (these custom tiles aren't QAbstractButtons,
        # so QButtonGroup can't manage them).  Clicking one unchecks the
        # others and fires the mode-change handler.  Modes are tracked
        # by string for clarity now that we have three of them.
        def _set_mode(name: str):
            self.r_mode_single.setChecked(name == "single")
            self.r_mode_batch.setChecked(name == "batch")
            self.r_mode_csv.setChecked(name == "csv")
            self._on_import_mode_changed(name)

        self.r_mode_single.toggled.connect(
            lambda checked: _set_mode("single") if checked else None)
        self.r_mode_batch.toggled.connect(
            lambda checked: _set_mode("batch")  if checked else None)
        self.r_mode_csv.toggled.connect(
            lambda checked: _set_mode("csv")    if checked else None)

        mode_row.addWidget(self.r_mode_single, 1)
        mode_row.addWidget(self.r_mode_batch,  1)
        mode_row.addWidget(self.r_mode_csv,    1)
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

        # Replay-from-manifest row — load a previous run's parameters
        # from its <stem>_run_manifest.json so you can reproduce it.
        row = QtWidgets.QHBoxLayout()
        self.btn_load_manifest = QtWidgets.QPushButton(
            "Load run manifest…")
        self.btn_load_manifest.setToolTip(
            "Open a previous run's <stem>_run_manifest.json and apply its\n"
            "parameters to the sidebar.  Useful for reproducing a run\n"
            "exactly or starting a new analysis from a known-good config.")
        self.btn_load_manifest.clicked.connect(self._on_load_manifest)
        row.addStretch(1)
        row.addWidget(self.btn_load_manifest)
        w_manifest = QtWidgets.QWidget(); w_manifest.setLayout(row)
        sg.addRow("", w_manifest)

        # ROI status + explicit "Load into ROI viewer" button.  The viewer
        # is the embedded _RoiViewer below — always visible, auto-loads
        # whenever the input path settles.
        self.lbl_single_roi_status = QtWidgets.QLabel(
            "ROI: using global setting")
        self.lbl_single_roi_status.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        sg.addRow("Region of interest", self.lbl_single_roi_status)

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
            "Series to process  (expand a series to deselect individual files):"))
        self.tree_batch_files = QtWidgets.QTreeWidget()
        self.tree_batch_files.setHeaderHidden(True)
        self.tree_batch_files.setColumnCount(1)
        self.tree_batch_files.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_batch_files.setMinimumHeight(200)
        self.tree_batch_files.setRootIsDecorated(True)
        self.tree_batch_files.setUniformRowHeights(True)
        self.tree_batch_files.setIndentation(18)
        bg.addWidget(self.tree_batch_files, stretch=1)

        sel_row = QtWidgets.QHBoxLayout()
        for label, fn in (("Select all",     self._on_batch_select_all),
                          ("Select none",    self._on_batch_select_none),
                          ("Invert selection", self._on_batch_select_inverse)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            sel_row.addWidget(b)
        sel_row.addStretch(1)
        # Explicit "open in preview viewer" — file loading is heavy
        # (reads ~30 frames + embeds in napari) and was previously
        # triggered on every checkbox toggle via itemClicked, which froze
        # the UI when the user was rapidly de-selecting files.  Now the
        # load only fires when the user explicitly asks for it.
        self.btn_batch_open_in_viewer = QtWidgets.QPushButton(
            "Open in viewer")
        self.btn_batch_open_in_viewer.setToolTip(
            "Load the highlighted series (or file) into the preview\n"
            "viewer below.  Double-clicking a row in the tree does the\n"
            "same thing.")
        self.btn_batch_open_in_viewer.clicked.connect(
            self._on_batch_open_in_viewer)
        sel_row.addWidget(self.btn_batch_open_in_viewer)
        self.lbl_batch_summary = QtWidgets.QLabel("0 series / 0 selected")
        sel_row.addWidget(self.lbl_batch_summary)
        bg.addLayout(sel_row)

        # Power-user shortcut: double-click a row to load it without
        # using the toolbar button.  Single-click only highlights —
        # no heavy work happens on checkbox toggles.
        self.tree_batch_files.itemDoubleClicked.connect(
            self._on_batch_tree_item_double_clicked)
        # Parent ↔ child check-state propagation.  Wired in _batch_rescan
        # after population to avoid spurious fires during seeding.

        # Where the batch outputs land
        self.lbl_batch_output_path = QtWidgets.QLabel(
            "Output → (pick an input folder first)")
        self.lbl_batch_output_path.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        bg.addWidget(self.lbl_batch_output_path)

        v.addWidget(self._batch_panel, stretch=1)

        # ── External-CSV sub-panel ────────────────────────────────────────
        # "Skip detection" mode: load localisations from PALM-Tracer /
        # ThunderSTORM / Picasso and run linking + downstream analyses
        # only.  No image is loaded, so the live preview viewer below
        # shows a placeholder unless the user supplies a background image
        # (optional, used for the figure's max-projection panel only).
        self._csv_panel = QtWidgets.QGroupBox("External CSV")
        cg = QtWidgets.QFormLayout(self._csv_panel)
        cg.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        # Input CSV
        row = QtWidgets.QHBoxLayout()
        self.e_csv_path = QtWidgets.QLineEdit()
        self.e_csv_path.setPlaceholderText(
            "Pick a localisations CSV from PALM-Tracer / "
            "ThunderSTORM / Picasso…")
        btn_csv = QtWidgets.QPushButton("Browse")
        btn_csv.clicked.connect(self._on_browse_csv)
        row.addWidget(self.e_csv_path, 1); row.addWidget(btn_csv)
        w_csv = QtWidgets.QWidget(); w_csv.setLayout(row)
        cg.addRow("Localisations CSV", w_csv)
        # Preset combo
        self.c_csv_preset = _QuietComboBox()
        self.c_csv_preset.addItems(
            ["Auto-detect", "PALM-Tracer", "ThunderSTORM",
             "Picasso", "Custom"])
        self.c_csv_preset.setToolTip(
            "Source-tool preset.  Tells FIREFLY how to interpret the\n"
            "CSV's columns (frame indexing, x/y units, mass column).\n"
            "Auto-detect sniffs the header; pick a specific preset if\n"
            "auto-detect picks the wrong one.")
        cg.addRow("Source preset", self.c_csv_preset)
        # Output folder
        row = QtWidgets.QHBoxLayout()
        self.e_csv_outdir = QtWidgets.QLineEdit()
        self.e_csv_outdir.setPlaceholderText(
            "Output folder for figure + CSV / JSON artifacts")
        btn_csv_out = QtWidgets.QPushButton("Browse")
        btn_csv_out.clicked.connect(self._on_browse_csv_outdir)
        row.addWidget(self.e_csv_outdir, 1); row.addWidget(btn_csv_out)
        w_csv_out = QtWidgets.QWidget(); w_csv_out.setLayout(row)
        cg.addRow("Output folder", w_csv_out)
        # Optional background image
        row = QtWidgets.QHBoxLayout()
        self.e_csv_bg = QtWidgets.QLineEdit()
        self.e_csv_bg.setPlaceholderText(
            "Optional — used only for the figure's max-projection panel")
        btn_csv_bg = QtWidgets.QPushButton("Browse")
        btn_csv_bg.clicked.connect(self._on_browse_csv_bg)
        row.addWidget(self.e_csv_bg, 1); row.addWidget(btn_csv_bg)
        w_csv_bg = QtWidgets.QWidget(); w_csv_bg.setLayout(row)
        cg.addRow("Background image", w_csv_bg)
        # Helpful note about which sidebar settings still matter
        hint = QtWidgets.QLabel(
            "External CSV mode uses:  Pixel size · Frame interval · "
            "Linking · Diffusion · ROI · Drift correction · Clustering · "
            "Figures.  Detection / preprocessing sidebar sections are "
            "ignored.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 11px;")
        cg.addRow("", hint)
        v.addWidget(self._csv_panel, stretch=1)

        # Start visible state: single mode shown, others hidden
        self._batch_panel.hide()
        self._csv_panel.hide()
        self._import_mode = "single"

        # ── Embedded ROI viewer (always visible) ──────────────────────────
        self._roi_viewer_container = QtWidgets.QFrame()
        # Reserve a min height so the panel doesn't grow from nothing the
        # first time a file is loaded — that resize is what macOS animates
        # as a "slide".
        self._roi_viewer_container.setMinimumHeight(280)
        rvl = QtWidgets.QVBoxLayout(self._roi_viewer_container)
        rvl.setContentsMargins(0, 8, 0, 0)
        self._roi_viewer = _RoiViewer()
        self._roi_viewer.polygons_changed.connect(self._on_roi_polygons_changed)
        rvl.addWidget(self._roi_viewer)
        v.addWidget(self._roi_viewer_container, stretch=2)
        # Pre-init the napari viewer right after construction so the very
        # first file load doesn't have to embed napari + load data + grow
        # the layout in one step (that triple causes the macOS slide).
        QtCore.QTimer.singleShot(0, lambda: self._roi_viewer._ensure_viewer())

        self.tabs.addTab(tab, "Import")

    def _on_import_mode_changed(self, mode):
        """Show whichever sub-panel matches the new mode and hide the
        others.  Accepts a string ("single" / "batch" / "csv") for the
        new tri-state mode toggle; falls back to bool for the legacy
        two-mode call sites."""
        # Legacy callers pass a bool (single=True / batch=False).
        if isinstance(mode, bool):
            mode = "single" if mode else "batch"
        self._import_mode = mode
        try:    self._single_panel.setVisible(mode == "single")
        except AttributeError: pass
        try:    self._batch_panel.setVisible(mode == "batch")
        except AttributeError: pass
        try:    self._csv_panel.setVisible(mode == "csv")
        except AttributeError: pass

    def _build_analysis_tab(self):
        """Analysis tab — pure status display.  Stage label, progress bar,
        and a results panel that fills in after a run completes."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        # Stage label on the left, elapsed-time counter on the right.
        # Both updated by the polling timer (stage) and a 1 Hz elapsed
        # timer (clock).
        stage_row = QtWidgets.QHBoxLayout()
        self.run_stage_label = QtWidgets.QLabel("Idle")
        self.run_stage_label.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-weight: 600; padding: 2px 0;")
        stage_row.addWidget(self.run_stage_label, 1)
        self.lbl_elapsed = QtWidgets.QLabel("")
        self.lbl_elapsed.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-variant-numeric: tabular-nums;")
        self.lbl_elapsed.setAlignment(Qt.AlignmentFlag.AlignRight)
        stage_row.addWidget(self.lbl_elapsed)
        v.addLayout(stage_row)

        # Resource monitor — CPU / RAM / GPU / VRAM at 1 Hz
        self.resource_monitor = _ResourceMonitor()
        v.addWidget(self.resource_monitor)

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

        # Detection cockpit (during a run) vs results panel (post-run)
        # share the same bottom slot via a QStackedWidget.
        self._analysis_stack = QtWidgets.QStackedWidget()

        # Page 0 — cockpit: narrow mass histogram across the top (its
        # original position / aspect), live frame view below it filling
        # all remaining vertical space.
        cockpit_w = QtWidgets.QWidget()
        cockpit   = QtWidgets.QVBoxLayout(cockpit_w)
        cockpit.setContentsMargins(0, 0, 0, 0)
        cockpit.setSpacing(6)
        self.mass_hist = _MassHistogram()
        self.live_view = _LiveFrameView()
        cockpit.addWidget(self.mass_hist)             # narrow, no stretch
        cockpit.addWidget(self.live_view, 1)          # fills the rest
        self._analysis_stack.addWidget(cockpit_w)

        # Page 1 — results: same _ResultsPanel as before.
        self.run_results = _ResultsPanel(
            "Results will appear here after analysis.")
        self._analysis_stack.addWidget(self.run_results)

        v.addWidget(self._analysis_stack, stretch=1)
        # Start on the results page (cockpit only shows during runs)
        self._analysis_stack.setCurrentIndex(1)

        self.tabs.addTab(tab, "Analysis")

    # ── Figures tab ───────────────────────────────────────────────────────
    def _build_figures_tab(self):
        """Customisation for figure outputs — single-sample (Analysis tab)
        and comparison (Compare tab) — plus a live preview that updates
        as the user changes theme / colormap settings."""
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(tab)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # ── Settings column ──────────────────────────────────────────────
        settings_col = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(settings_col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Style and output format for the figures produced by the "
            "Analysis and Compare tabs.  Preview on the right updates as "
            "you change the theme / colormap.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(intro)

        # ── Single-sample figure ──────────────────────────────────────────
        sec, gl = self._make_form_section("Single-sample figure (Analysis tab)")
        self.c_fig_theme = _QuietComboBox()
        self.c_fig_theme.addItems(["Dark", "Light", "Publication"])
        self.c_fig_theme.setToolTip(
            "Overall colour scheme for figure backgrounds, axes, and text.\n"
            "• Dark         — GitHub-dark (matches the GUI).\n"
            "• Light        — GitHub-light, sans-serif.\n"
            "• Publication  — White background, black axes, serif font.")
        gl.addRow("Theme", self.c_fig_theme)
        self.c_fig_proj_cmap = _QuietComboBox()
        self.c_fig_proj_cmap.addItems(
            ["Inferno", "Hot", "Viridis", "Plasma", "Greys"])
        self.c_fig_proj_cmap.setToolTip(
            "Colormap for the max-projection panel.  Inferno is the\n"
            "default — perceptually uniform with deep blacks for dark\n"
            "backgrounds.  Greys flips automatically for light themes.")
        gl.addRow("Projection colormap", self.c_fig_proj_cmap)
        self.s_fig_dpi = self._spin_int(150, 72, 600, step=10,
            tip="Pixel density for the combined PNG.  150 DPI matches the\n"
                "default print size; bump to 300 for posters / publications.")
        gl.addRow("PNG DPI", self.s_fig_dpi)
        self.c_fig_save_pdf = QtWidgets.QCheckBox(
            "Also save vector PDF alongside the PNG")
        self.c_fig_save_pdf.setToolTip(
            "Write a vector PDF copy of the figure.  Same content as the\n"
            "PNG but infinitely zoomable — recommended for talks and papers.")
        gl.addRow("", self.c_fig_save_pdf)
        self.c_fig_per_panel = QtWidgets.QCheckBox(
            "Also save each panel as a separate PNG")
        self.c_fig_per_panel.setToolTip(
            "Export each labelled panel (A, B, C, …) of the combined figure\n"
            "to figures/panels/.  Useful when you want a single chart for a\n"
            "talk without cropping the full grid.")
        gl.addRow("", self.c_fig_per_panel)
        v.addWidget(sec)

        # Single-sample panel selector — only affects per-panel PNG exports
        # (combined figure always contains every panel that has data).
        single_panels_grp = QtWidgets.QGroupBox(
            "Single-sample panels to export individually")
        spg = QtWidgets.QGridLayout(single_panels_grp)
        self._single_panel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for i, (key, label) in enumerate(self.SINGLE_PANELS):
            cb = QtWidgets.QCheckBox(f"{key}.  {label}")
            cb.setChecked(True)
            cb.setToolTip(
                f"Include panel {key} ({label}) when 'Also save each panel\n"
                "as a separate PNG' is on.  The combined figure always shows\n"
                "every panel that has data.")
            self._single_panel_checkboxes[key] = cb
            spg.addWidget(cb, i // 2, i % 2)
        v.addWidget(single_panels_grp)

        # ── Comparison figure (moved from Compare tab) ────────────────────
        sec, gl = self._make_form_section("Comparison figure (Compare tab)")
        self.c_cmp_theme = _QuietComboBox()
        self.c_cmp_theme.addItems(["Dark", "Light", "Publication"])
        self.c_cmp_theme.setToolTip(
            "Theme for the multi-group comparison figure.  Independent\n"
            "from the single-sample theme so you can mix and match.")
        gl.addRow("Theme", self.c_cmp_theme)
        self.c_cmp_pdf = QtWidgets.QCheckBox(
            "Generate multi-page PDF report (figure + parameters + stats)")
        self.c_cmp_pdf.setChecked(True)
        gl.addRow("", self.c_cmp_pdf)
        v.addWidget(sec)

        # Comparison panels (which sub-panels to include in the figure)
        panels_grp = QtWidgets.QGroupBox("Comparison panels to include")
        pg = QtWidgets.QGridLayout(panels_grp)
        self._cmp_panel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for i, (key, label) in enumerate(self.COMPARE_PANELS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            self._cmp_panel_checkboxes[key] = cb
            pg.addWidget(cb, i // 2, i % 2)
        v.addWidget(panels_grp)
        v.addStretch(1)

        # ── Preview column (two stacked previews) ────────────────────────
        preview_col = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(preview_col)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)

        def _make_preview_label(caption: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel("Rendering preview…")
            lbl.setMinimumSize(560, 320)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Ignored policy in both directions → layout sizes the label
            # from the stretch / minimum hints only, NOT from the pixmap's
            # natural size.  Without this, every theme change produces a
            # slightly different matplotlib output → the label's sizeHint
            # bumps up → layout reallocates → bigger label → bigger render…
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                              QtWidgets.QSizePolicy.Policy.Ignored)
            lbl.setStyleSheet(
                f"QLabel {{ border: 1px solid {_THEME['BORDER']}; "
                f"background: {_THEME['PANEL']}; color: {_THEME['TXT_MUTED']}; "
                "border-radius: 4px; }}")
            return lbl

        cap_single = QtWidgets.QLabel("Single-sample figure")
        cap_single.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 600;")
        pv.addWidget(cap_single)
        self.lbl_fig_preview_single = _make_preview_label("single")
        pv.addWidget(self.lbl_fig_preview_single, stretch=1)

        cap_compare = QtWidgets.QLabel("Comparison figure")
        cap_compare.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 600;")
        pv.addWidget(cap_compare)
        self.lbl_fig_preview_compare = _make_preview_label("comparison")
        pv.addWidget(self.lbl_fig_preview_compare, stretch=1)

        # Cache of unscaled preview pixmaps so we can re-fit them when the
        # labels resize (e.g. on window resize) without re-rendering.
        self._fig_preview_pixmaps: dict[QtWidgets.QLabel, QtGui.QPixmap] = {}
        # Install a resize filter on both labels — re-scales the cached
        # raw pixmap to fit the new label dimensions.
        for _lbl in (self.lbl_fig_preview_single, self.lbl_fig_preview_compare):
            _lbl.installEventFilter(self)

        hint = QtWidgets.QLabel(
            "Rendered on a synthetic dataset — actual figures will use "
            "your data but keep these style choices.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 11px;")
        pv.addWidget(hint)

        outer.addWidget(settings_col, 1)
        outer.addWidget(preview_col, 2)

        # ── Debounced preview refresh ────────────────────────────────────
        self._figpreview_timer = QTimer(self)
        self._figpreview_timer.setSingleShot(True)
        self._figpreview_timer.setInterval(120)
        self._figpreview_timer.timeout.connect(self._refresh_figures_preview)
        for w in (self.c_fig_theme, self.c_fig_proj_cmap, self.c_cmp_theme):
            w.currentTextChanged.connect(
                lambda _=None: self._figpreview_timer.start())
        # First render after construction settles
        QtCore.QTimer.singleShot(80, self._refresh_figures_preview)

        self.tabs.addTab(tab, "Figures")

    @staticmethod
    def _figure_theme_palette(theme: str):
        """Return (BG, PNL, TXT, GRD, ACC, font) for a theme name —
        mirrors what make_figure() and compare_groups() do internally."""
        if theme == "Light":
            return ("#ffffff", "#f6f8fa", "#24292f",
                    "#d0d7de", "#0969da", "sans-serif")
        if theme == "Publication":
            return ("#ffffff", "#ffffff", "#000000",
                    "#cccccc", "#333333", "serif")
        return ("#0d1117", "#161b22", "#e6edf3",
                "#30363d", "#58a6ff", "monospace")

    def _refresh_figures_preview(self):
        """Render both single-sample and comparison previews in the user's
        chosen styles and push them into their respective QLabels.  Runs
        in-process (Agg backend) so it can't interfere with the analysis
        subprocess."""
        if not hasattr(self, "lbl_fig_preview_single"):
            return
        try:
            import io
            import numpy as np
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.lbl_fig_preview_single.setText(f"Preview unavailable: {exc}")
            self.lbl_fig_preview_compare.setText(f"Preview unavailable: {exc}")
            return

        proj_cmap_name = self.c_fig_proj_cmap.currentText()

        def _rc(theme):
            BG, PNL, TXT, GRD, _ACC, font = self._figure_theme_palette(theme)
            return {
                "text.color": TXT, "axes.labelcolor": TXT,
                "xtick.color": TXT, "ytick.color": TXT,
                "axes.edgecolor": GRD, "axes.facecolor": PNL,
                "grid.color": GRD, "grid.alpha": 0.4,
                "font.family": font,
            }

        def _render(kind: str, theme: str) -> "QtGui.QPixmap | None":
            BG, PNL, TXT, GRD, ACC, _font = self._figure_theme_palette(theme)
            cmap_map = {"Inferno": "inferno", "Hot": "hot",
                        "Viridis": "viridis", "Plasma": "plasma",
                        "Greys": "Greys" if theme in ("Light", "Publication")
                                         else "Greys_r"}
            proj = cmap_map.get(proj_cmap_name, "inferno")
            rng = np.random.default_rng(0 if kind == "single" else 1)
            buf = io.BytesIO()
            try:
                with plt.rc_context(_rc(theme)):
                    if kind == "comparison":
                        fig = self._render_comparison_preview(
                            plt, np, rng, BG, PNL, TXT, GRD)
                    else:
                        fig = self._render_single_sample_preview(
                            plt, np, rng, BG, PNL, TXT, GRD, ACC, proj)
                    fig.savefig(buf, format="png", facecolor=BG, dpi=440,
                                bbox_inches="tight")
                    plt.close(fig)
            except Exception as exc:
                return None, str(exc)
            buf.seek(0)
            pix = QtGui.QPixmap()
            if not pix.loadFromData(buf.read()):
                return None, "decode failed"
            return pix, None

        for label_widget, kind, theme in (
                (self.lbl_fig_preview_single,
                 "single", self.c_fig_theme.currentText()),
                (self.lbl_fig_preview_compare,
                 "comparison", self.c_cmp_theme.currentText())):
            pix, err = _render(kind, theme)
            if pix is None:
                label_widget.setText(f"Preview render failed: {err}")
                continue
            # Cache raw, then scale once for the current label size.
            self._fig_preview_pixmaps[label_widget] = pix
            self._fit_preview_pixmap(label_widget)

    def _fit_preview_pixmap(self, label: QtWidgets.QLabel) -> None:
        """Scale the cached raw pixmap for `label` to its current size."""
        pix = self._fig_preview_pixmaps.get(label)
        if pix is None or pix.isNull():
            return
        size = label.size()
        if size.width() <= 1 or size.height() <= 1:
            return
        scaled = pix.scaled(size,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    def eventFilter(self, obj, event):
        # Re-scale cached preview pixmaps when their labels resize.
        if (event.type() == QtCore.QEvent.Type.Resize
                and isinstance(obj, QtWidgets.QLabel)
                and hasattr(self, "_fig_preview_pixmaps")
                and obj in self._fig_preview_pixmaps):
            self._fit_preview_pixmap(obj)
        return super().eventFilter(obj, event)

    def _render_single_sample_preview(self, plt, np, rng,
                                       BG, PNL, TXT, GRD, ACC, proj_cmap):
        """Two-panel mock-up: projection + MSD curves."""
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                                  facecolor=BG, dpi=110)
        # Panel A — fake max projection (Gaussian blob + noise)
        H = W = 48
        Y, X = np.mgrid[0:H, 0:W]
        img = (np.exp(-((X - 26)**2 + (Y - 22)**2) / 70) * 0.9
               + np.exp(-((X - 12)**2 + (Y - 30)**2) / 30) * 0.5
               + rng.random((H, W)) * 0.08)
        ax = axes[0]
        ax.set_facecolor(PNL)
        ax.imshow(img, cmap=proj_cmap)
        ax.set_title("  A   Max projection", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)

        # Panel B — MSD-like curves for three motion classes
        ax = axes[1]
        ax.set_facecolor(PNL)
        t = np.linspace(0.02, 1.0, 25)
        for alpha, label, col in ((1.0, "Brownian", ACC),
                                  (0.55, "Confined", "#f78166"),
                                  (1.45, "Directed", "#56d364")):
            msd = 0.05 * t**alpha + rng.normal(0, 0.004, t.size)
            ax.plot(t, msd, marker="o", markersize=3, linewidth=1.4,
                    color=col, label=label)
        ax.set_xlabel("τ (s)", fontsize=9)
        ax.set_ylabel("MSD (μm²)", fontsize=9)
        ax.set_title("  B   Ensemble MSD", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(fontsize=8, frameon=False)
        for txt in leg.get_texts(): txt.set_color(TXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        fig.tight_layout()
        return fig

    def _render_comparison_preview(self, plt, np, rng, BG, PNL, TXT, GRD):
        """Two-panel mock-up resembling the Compare-tab figure: grouped
        MSD lines + bar chart for two synthetic groups."""
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                                  facecolor=BG, dpi=110)
        groups = [("Pre",  "#3b6ed8", 1.00),
                  ("Post", "#f78166", 0.70)]
        # Panel 1 — MSD per group
        ax = axes[0]
        ax.set_facecolor(PNL)
        t = np.linspace(0.02, 1.0, 25)
        for label, col, scale in groups:
            msd = 0.06 * scale * t**0.95 + rng.normal(0, 0.003, t.size)
            ax.plot(t, msd, marker="o", markersize=3, linewidth=1.6,
                    color=col, label=label)
        ax.set_xlabel("τ (s)", fontsize=9)
        ax.set_ylabel("MSD (μm²)", fontsize=9)
        ax.set_title("  Ensemble MSD", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(fontsize=8, frameon=False)
        for txt in leg.get_texts(): txt.set_color(TXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        # Panel 2 — mobile fraction bar chart
        ax = axes[1]
        ax.set_facecolor(PNL)
        labels = [g[0] for g in groups]
        cols   = [g[1] for g in groups]
        vals   = [0.62, 0.41]
        errs   = [0.04, 0.05]
        ax.bar(labels, vals, yerr=errs, color=cols, edgecolor=GRD,
               capsize=5, linewidth=1.0)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mobile fraction", fontsize=9)
        ax.set_title("  Mobile fraction", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        fig.tight_layout()
        return fig

    # ── Batch helpers (Import-tab batch sub-panel) ───────────────────────
    @staticmethod
    def _looks_like_input_file(name: str) -> bool:
        n = name.lower()
        return n.endswith(".czi") or n.endswith(".tif") or n.endswith(".tiff")

    @staticmethod
    def _series_key(filename: str) -> str:
        """Return the series key — filename stem with any trailing '(N)'
        stripped.  Zeiss splits long recordings across files named like
        `experiment.tif`, `experiment(1).tif`, `experiment(2).tif`, …
        all of which belong to the same continuous time series and are
        joined by the loader.  This function maps each of those back to
        the common stem ('experiment'), so the batch UI can group them.
        """
        import re as _re
        stem = os.path.splitext(filename)[0]
        return _re.sub(r"\(\d+\)\s*$", "", stem).rstrip()

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

    # Custom data roles for tree items
    _ROLE_PATH        = Qt.ItemDataRole.UserRole       # full file path
    _ROLE_KIND        = Qt.ItemDataRole.UserRole + 1   # "series" or "file"
    _ROLE_SERIES_KEY  = Qt.ItemDataRole.UserRole + 2   # series identifier
    _ROLE_FILE_COUNT  = Qt.ItemDataRole.UserRole + 3   # series count (series items only)

    def _batch_rescan(self, folder: str):
        """Populate the tree with one parent per file SERIES + one child
        per sister file inside the series.

        Each series's parent toggles all of its file children at once;
        the children can be individually deselected to exclude specific
        sister files from the loader concat.  When the run starts, the
        worker receives the per-series checked-file list and overrides
        the auto-discovery in `load_tif` / `load_czi` accordingly.
        """
        # Disconnect itemChanged so populating doesn't fire a cascade
        try:
            self.tree_batch_files.itemChanged.disconnect()
        except (TypeError, RuntimeError):
            pass
        # Re-entrancy guard for parent ↔ child propagation
        self._tree_propagation_guard = False

        self.tree_batch_files.blockSignals(True)
        self.tree_batch_files.clear()
        self._batch_series_map: dict[str, list[tuple[str, str]]] = {}

        if not os.path.isdir(folder):
            self.tree_batch_files.blockSignals(False)
            self._batch_update_summary()
            return
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            self.tree_batch_files.blockSignals(False)
            self._batch_update_summary()
            return

        # Phase 1 — group files into series by the loader's keying.
        for name in names:
            if not self._looks_like_input_file(name):
                continue
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            key = self._series_key(name)
            self._batch_series_map.setdefault(key, []).append((name, full))

        # Phase 2 — build the tree.
        for key in sorted(self._batch_series_map.keys()):
            sisters = sorted(self._batch_series_map[key])
            primary_name, primary_full = sisters[0]
            for nm, pth in sisters:
                if os.path.splitext(nm)[0] == key:
                    primary_name, primary_full = nm, pth
                    break
            n = len(sisters)
            parent_label = (primary_name if n == 1
                            else f"{primary_name}   ×  {n} files")
            parent = QtWidgets.QTreeWidgetItem([parent_label])
            parent.setFlags(parent.flags()
                            | Qt.ItemFlag.ItemIsUserCheckable
                            | Qt.ItemFlag.ItemIsAutoTristate)
            parent.setCheckState(0, Qt.CheckState.Checked)
            parent.setData(0, self._ROLE_PATH, primary_full)
            parent.setData(0, self._ROLE_KIND, "series")
            parent.setData(0, self._ROLE_SERIES_KEY, key)
            parent.setData(0, self._ROLE_FILE_COUNT, n)
            # Highlight the parent slightly to distinguish from children
            f = parent.font(0); f.setBold(True); parent.setFont(0, f)
            # Add one child per sister file (in display order)
            for nm, pth in sisters:
                child = QtWidgets.QTreeWidgetItem([nm])
                child.setFlags(child.flags()
                               | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
                child.setData(0, self._ROLE_PATH, pth)
                child.setData(0, self._ROLE_KIND, "file")
                child.setData(0, self._ROLE_SERIES_KEY, key)
                parent.addChild(child)
            self.tree_batch_files.addTopLevelItem(parent)
            # Single-file series collapse — no point expanding a one-row group.
            parent.setExpanded(n > 1)

        self.tree_batch_files.blockSignals(False)
        self.tree_batch_files.itemChanged.connect(self._on_tree_item_changed)
        self._batch_update_summary()
        # Mark series + files that already have a saved polygon ROI
        self._refresh_batch_roi_markers()

    # ── Tree iteration helpers ───────────────────────────────────────────
    def _batch_iter_series(self):
        """Yield each top-level (series) item in the batch tree."""
        if not hasattr(self, "tree_batch_files"):
            return
        for i in range(self.tree_batch_files.topLevelItemCount()):
            yield self.tree_batch_files.topLevelItem(i)

    def _batch_iter_files(self):
        """Yield every (series_item, file_item) pair in the batch tree."""
        for ser in self._batch_iter_series():
            for j in range(ser.childCount()):
                yield ser, ser.child(j)

    def _on_tree_item_changed(self, item: "QtWidgets.QTreeWidgetItem",
                              _col: int):
        """Keep parent ↔ child check states in sync, then refresh the
        summary.  Re-entrancy-guarded because every setCheckState we
        make in here would otherwise re-fire this slot."""
        if self._tree_propagation_guard:
            return
        self._tree_propagation_guard = True
        try:
            kind = item.data(0, self._ROLE_KIND)
            if kind == "series":
                # Push the parent's new state down to children
                state = item.checkState(0)
                if state != Qt.CheckState.PartiallyChecked:
                    for j in range(item.childCount()):
                        item.child(j).setCheckState(0, state)
            elif kind == "file":
                parent = item.parent()
                if parent is not None:
                    n  = parent.childCount()
                    on = sum(1 for j in range(n)
                             if parent.child(j).checkState(0)
                             == Qt.CheckState.Checked)
                    if on == 0:
                        parent.setCheckState(0, Qt.CheckState.Unchecked)
                    elif on == n:
                        parent.setCheckState(0, Qt.CheckState.Checked)
                    else:
                        parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
        finally:
            self._tree_propagation_guard = False
            self._batch_update_summary()

    def _on_batch_tree_item_double_clicked(
            self, item: "QtWidgets.QTreeWidgetItem", _col: int):
        """Double-click on a tree row loads it into the preview viewer.
        Single-click only highlights — checkbox toggles are cheap and
        the heavy file load happens exclusively from here or the
        toolbar button."""
        if item is None:
            return
        path = item.data(0, self._ROLE_PATH)
        if path:
            self._roi_load_specific_path(path)

    def _on_batch_open_in_viewer(self):
        """Load the currently-highlighted tree item into the preview
        viewer.  Triggered by the "Open in viewer" toolbar button."""
        if not hasattr(self, "tree_batch_files"):
            return
        it = self.tree_batch_files.currentItem()
        if it is None:
            QtWidgets.QMessageBox.information(
                self, "Open in viewer",
                "Click a series or file in the tree first to highlight "
                "it, then press 'Open in viewer'.")
            return
        path = it.data(0, self._ROLE_PATH)
        if path:
            self._roi_load_specific_path(path)

    def _roi_load_specific_path(self, path: str):
        """Load `path` into the embedded preview viewer.  Defers the
        actual work one event-loop tick so the UI repaints first
        (highlight / button-press feedback), then runs the heavy
        napari load.  Status is surfaced via the viewer's own status
        line so the user sees that something is happening."""
        if not (path and os.path.isfile(path)):
            return
        try:
            self.statusBar().showMessage(
                f"Loading {os.path.basename(path)} into viewer…", 2000)
        except Exception:
            pass

        def _go():
            try:
                existing = self._roi_polygons.get(os.path.abspath(path))
                self._roi_viewer.set_file(path, current_polygons=existing)
                self._push_detection_preview_params()
                self._push_roi_mask_params()
                self._roi_viewer.enable_detection_preview(True)
            except Exception as exc:
                try:
                    self.statusBar().showMessage(
                        f"Couldn't load preview: {exc}", 8000)
                except Exception:
                    pass
        QtCore.QTimer.singleShot(0, _go)

    def _batch_update_summary(self):
        n_series     = sum(1 for _ in self._batch_iter_series())
        n_sel_series = sum(1 for s in self._batch_iter_series()
                           if s.checkState(0) != Qt.CheckState.Unchecked)
        n_total_files = sum(1 for _ in self._batch_iter_files())
        n_sel_files   = sum(1 for _, c in self._batch_iter_files()
                            if c.checkState(0) == Qt.CheckState.Checked)
        if n_total_files == n_series:
            self.lbl_batch_summary.setText(
                f"{n_series} series / {n_sel_series} selected")
        else:
            self.lbl_batch_summary.setText(
                f"{n_series} series ({n_total_files} files) / "
                f"{n_sel_series} series selected ({n_sel_files} files)")
        folder = self.e_batch_folder.text().strip()
        if folder:
            self.lbl_batch_output_path.setText(
                f"Output → {os.path.join(folder, 'batch_results')}/<stem>/")
        else:
            self.lbl_batch_output_path.setText(
                "Output → (pick an input folder first)")

    def _on_batch_select_all(self):
        for s in self._batch_iter_series():
            s.setCheckState(0, Qt.CheckState.Checked)

    def _on_batch_select_none(self):
        for s in self._batch_iter_series():
            s.setCheckState(0, Qt.CheckState.Unchecked)

    def _on_batch_select_inverse(self):
        for s in self._batch_iter_series():
            cur = s.checkState(0)
            s.setCheckState(0,
                Qt.CheckState.Unchecked
                if cur == Qt.CheckState.Checked
                else Qt.CheckState.Checked)

    def _batch_checked_series(self) -> "list[dict]":
        """Return one entry per series the user wants processed, each a
        dict {primary, key, files}: the primary file path (for stem +
        outdir naming), the series key, and the explicit list of
        checked sister files (in display order)."""
        out: list[dict] = []
        for s in self._batch_iter_series():
            if s.checkState(0) == Qt.CheckState.Unchecked:
                continue
            files = [s.child(j).data(0, self._ROLE_PATH)
                     for j in range(s.childCount())
                     if s.child(j).checkState(0)
                     == Qt.CheckState.Checked]
            if not files:
                continue
            out.append({
                "primary": s.data(0, self._ROLE_PATH),
                "key":     s.data(0, self._ROLE_SERIES_KEY),
                "files":   files,
            })
        return out

    def _batch_checked_files(self) -> list[str]:
        """Backwards-compatible flat list of primary paths for any series
        that has at least one checked file.  Kept for callers that only
        need to count what's selected."""
        return [g["primary"] for g in self._batch_checked_series()]

    # ── Embedded ROI viewer (Import tab) ─────────────────────────────────
    def _roi_embedded_load_current_file(self):
        """Load whichever file is currently 'active' into the embedded
        ROI viewer.  In single mode that's `e_file`; in batch mode it's
        the currently-highlighted item in the file list."""
        if not hasattr(self, "_roi_viewer"):
            return
        path = ""
        if self.r_mode_batch.isChecked():
            it = self.tree_batch_files.currentItem() \
                if hasattr(self, "tree_batch_files") else None
            if it is not None:
                path = it.data(0, self._ROLE_PATH) or ""
        else:
            path = self.e_file.text().strip()
        if path and os.path.isfile(path):
            existing = self._roi_polygons.get(os.path.abspath(path))
            self._roi_viewer.set_file(path, current_polygons=existing)
            # Always push current parameters and turn the live overlay on —
            # the viewer is the only detection-preview surface now, so it
            # may as well be on whenever a file is loaded.
            self._push_detection_preview_params()
            self._push_roi_mask_params()
            self._roi_viewer.enable_detection_preview(True)
        else:
            self._roi_viewer.set_file("", None)

    def _push_roi_mask_params(self):
        """Forward the current ROI-mode settings to the embedded viewer
        so its auto/manual-threshold overlay reflects the sidebar in real
        time."""
        if not hasattr(self, "_roi_viewer"):
            return
        try:
            mode = self.c_roi_mode.currentText()
            method = self.c_roi_auto_method.currentText().lower()  # otsu/li/triangle/mean
            threshold = float(self.s_roi_threshold.value())
            mask_mode = self.c_roi_mask_mode.currentText().lower()  # mean/sum
            self._roi_viewer.set_roi_mask_params(
                mode=mode, auto_method=method,
                threshold=threshold, mask_mode=mask_mode)
        except Exception:
            pass

    def _push_detection_preview_params(self):
        """Forward the current diameter / minmass / bg settings to the
        embedded viewer.  Bg settings matter because the pipeline runs
        detection on background-subtracted, renormalised frames — the
        preview must do the same or the mass scale won't match."""
        if not hasattr(self, "_roi_viewer"):
            return
        bg_method_map = {"Uniform Filter": "uniform_filter",
                         "Rolling Ball":   "rolling_ball"}
        try:
            self._roi_viewer.set_detection_params(
                diameter=int(self.s_diameter.value()),
                minmass=float(self.s_minmass.value()),
                bg_method=bg_method_map.get(
                    self.c_bg_method.currentText(), "uniform_filter"),
                bg_radius=int(self.s_bg_radius.value()),
            )
        except Exception:
            pass

    def _on_roi_polygons_changed(self, file_path: str, polys: list):
        """The embedded viewer emits this whenever the user adds/edits/
        removes a polygon.  Auto-persist to QSettings."""
        if not file_path:
            return
        key = os.path.abspath(file_path)
        if polys:
            self._roi_polygons[key] = polys
        else:
            self._roi_polygons.pop(key, None)
        self._save_roi_polygons()
        # Refresh status indicators
        self._refresh_single_roi_status()
        self._refresh_batch_roi_markers()

    # ── ROI mode enabled-state + embedded viewer visibility ───────────────
    def _on_roi_mode_changed(self, text: str):
        """Grey out threshold/projection controls that don't apply to the
        active mode and push the mask overlay to the viewer."""
        is_auto    = text == "Auto threshold"
        is_manual  = text == "Manual threshold"

        # Auto method only meaningful when in Auto-threshold mode
        try: self.c_roi_auto_method.setEnabled(is_auto)
        except AttributeError: pass
        # Manual threshold spinbox only used in Manual-threshold mode
        try: self.s_roi_threshold.setEnabled(is_manual)
        except AttributeError: pass
        try: self.sld_roi_threshold.setEnabled(is_manual)
        except AttributeError: pass
        # Projection (mean vs sum) is irrelevant in polygon mode (we use
        # a sample mean for the preview regardless) and in None mode.
        try: self.c_roi_mask_mode.setEnabled(is_auto or is_manual)
        except AttributeError: pass

        self._push_roi_mask_params()

    # ── Per-file ROI editor ────────────────────────────────────────────────
    _ROI_MARKER = "  ◉"

    def _decorate_filename_with_roi(self, name: str, has_roi: bool) -> str:
        """Add/remove the ◉ marker on a batch file-list item name."""
        base = name[:-len(self._ROI_MARKER)] \
               if name.endswith(self._ROI_MARKER) else name
        return f"{base}{self._ROI_MARKER}" if has_roi else base

    def _refresh_batch_roi_markers(self):
        """Walk the batch tree and refresh ◉ markers.

        File children get the marker when their own path has a saved
        ROI; the series parent gets it when *any* of its files do.
        Wrapped in blockSignals so the resulting itemChanged cascade
        doesn't re-trigger the parent/child propagation logic."""
        if not hasattr(self, "tree_batch_files"):
            return
        self.tree_batch_files.blockSignals(True)
        try:
            for ser in self._batch_iter_series():
                any_roi = False
                for j in range(ser.childCount()):
                    child = ser.child(j)
                    path = child.data(0, self._ROLE_PATH)
                    has_roi = bool(self._roi_polygons.get(
                        os.path.abspath(path))) if path else False
                    any_roi = any_roi or has_roi
                    new = self._decorate_filename_with_roi(
                        child.text(0), has_roi)
                    if new != child.text(0):
                        child.setText(0, new)
                new_parent = self._decorate_filename_with_roi(
                    ser.text(0), any_roi)
                if new_parent != ser.text(0):
                    ser.setText(0, new_parent)
        finally:
            self.tree_batch_files.blockSignals(False)

    def _refresh_single_roi_status(self):
        """Update the single-file ROI status label."""
        path = self.e_file.text().strip()
        if not path:
            self.lbl_single_roi_status.setText("Pick an input file first")
        elif self._roi_polygons.get(os.path.abspath(path)):
            n = len(self._roi_polygons[os.path.abspath(path)])
            self.lbl_single_roi_status.setText(
                f"✓ Custom polygon ROI saved ({n} shape{'s' if n != 1 else ''})")
            self.lbl_single_roi_status.setStyleSheet(
                f"color: {_THEME['SUCCESS']};")
        else:
            self.lbl_single_roi_status.setText("No custom ROI for this file")
            self.lbl_single_roi_status.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']};")

    def _open_roi_dialog(self, file_path: str) -> bool:
        """Open the ROI editor for `file_path`.  Returns True if the user
        saved (including saving an empty / cleared polygon), False on
        Cancel."""
        if not os.path.isfile(file_path):
            QtWidgets.QMessageBox.warning(
                self, "ROI editor", f"File not found:\n{file_path}")
            return False
        existing = self._roi_polygons.get(os.path.abspath(file_path))
        dlg = _RoiDialog(file_path, current_polygons=existing, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        polys = dlg.result_polygons()
        key = os.path.abspath(file_path)
        if polys:
            self._roi_polygons[key] = polys
        else:
            self._roi_polygons.pop(key, None)
        self._save_roi_polygons()
        return True

    def _on_single_set_roi(self):
        path = self.e_file.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(
                self, "ROI editor",
                "Pick an input file first on the Import tab.")
            return
        if self._open_roi_dialog(path):
            self._refresh_single_roi_status()

    def _on_batch_set_roi(self):
        # Use the highlighted item (currentItem), not the checked ones,
        # so the user can edit ROIs without changing what's selected
        # for processing.
        it = self.tree_batch_files.currentItem() \
            if hasattr(self, "tree_batch_files") else None
        if it is None:
            QtWidgets.QMessageBox.warning(
                self, "ROI editor",
                "Click a file or series in the tree to highlight it, "
                "then click Set ROI… again.")
            return
        path = it.data(0, self._ROLE_PATH)
        if self._open_roi_dialog(path):
            self._refresh_batch_roi_markers()

    def _on_batch_clear_roi(self):
        it = self.tree_batch_files.currentItem() \
            if hasattr(self, "tree_batch_files") else None
        if it is None:
            return
        path = it.data(0, self._ROLE_PATH)
        key = os.path.abspath(path) if path else None
        if key and key in self._roi_polygons:
            del self._roi_polygons[key]
            self._save_roi_polygons()
            self._refresh_batch_roi_markers()

    def _save_roi_polygons(self):
        """Persist all per-file ROIs to QSettings as a JSON blob."""
        try:
            import json
            payload = {k: v for k, v in self._roi_polygons.items()}
            self._settings.setValue("roi/polygons", json.dumps(payload))
        except Exception:
            pass

    def _load_roi_polygons(self):
        """Load all saved per-file ROIs from QSettings."""
        try:
            import json
            raw = self._settings.value("roi/polygons", type=str)
            if raw:
                data = json.loads(raw)
                # Vertices come back as lists-of-lists; that's fine
                self._roi_polygons = {k: v for k, v in data.items()
                                       if isinstance(v, list) and v}
        except Exception:
            self._roi_polygons = {}

    # ══════════════════════════════════════════════════════════════════════
    #  COMPARE TAB
    # ══════════════════════════════════════════════════════════════════════
    SINGLE_PANELS = [
        ("A", "Max projection"),
        ("B", "Trajectories"),
        ("C", "Trajectories by D"),
        ("D", "MSD curves"),
        ("E", "Diffusion coefficient distribution"),
        ("F", "Motion classification"),
        ("G", "α (anomalous exponent) distribution"),
        ("H", "Position density map"),
        ("I", "Turning-angle distribution"),
        ("J", "Mobile fraction over time"),
        ("K", "Jump-distance distribution"),
        ("L", "Cluster map (DBSCAN)"),
        ("M", "Dwell-time distribution"),
        ("N", "Moment-scaling spectrum"),
        ("O", "Radial distribution"),
    ]
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

        # Pointer to where style settings now live
        style_hint = QtWidgets.QLabel(
            "<i>Figure theme, panel selection and PDF report toggle now "
            "live on the <b>Figures</b> tab.</i>")
        style_hint.setTextFormat(Qt.TextFormat.RichText)
        style_hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        sg.addWidget(style_hint, 2, 0, 1, 3)

        v.addWidget(settings)

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

        # The status widgets (stage label, progress bar, results panel)
        # used to live below the action row, but they were visually noisy
        # for a tab that mostly just configures + kicks off the comparison.
        # They're still constructed and parented to the tab so the rest of
        # the run-machinery can call .setText / .setValue / .reset on them
        # — but they're hidden so they don't show in the UI.  Progress is
        # surfaced via the status bar instead.
        self.cmp_stage_label = QtWidgets.QLabel("Idle", tab)
        self.cmp_progress    = QtWidgets.QProgressBar(tab)
        self.cmp_progress.setRange(0, 100)
        self.cmp_results     = _ResultsPanel("", parent=tab)
        for w in (self.cmp_stage_label, self.cmp_progress, self.cmp_results):
            w.hide()

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

        # ── Inspector panel (right side, populated on track click) ───────
        self._ws_inspector = _TrackInspector()
        # Per-run state for click→stats lookup
        self._ws_tracks_df: "pd.DataFrame | None" = None
        self._ws_diff_df:   "pd.DataFrame | None" = None
        self._ws_tracks_layer = None

        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._ws_container)
        split.addWidget(self._ws_inspector)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 320])
        v.addWidget(split, stretch=1)

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
            _hide_napari_chrome(viewer)
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

    def _ws_load_tracks_path(self, csv_path: str,
                              diff_csv_path: "str | None" = None):
        """Read a trajectories CSV and add as a napari Tracks layer.

        FIREFLY's trajectories.csv has columns particle, frame, x, y.
        napari Tracks expects (track_id, t, [z,] y, x) per row.  If a
        diffusion-summary CSV is also supplied, the tracks are coloured
        by motion class and stored on `self` for click→stats lookup.
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

            # Optional sidecar — diffusion summary keyed by particle id
            diff_df = None
            if diff_csv_path and os.path.isfile(diff_csv_path):
                try:    diff_df = pd.read_csv(diff_csv_path)
                except Exception: diff_df = None
            else:
                # Auto-detect: same folder, <prefix>_diffusion_summary.csv
                guess = csv_path.replace("_trajectories.csv",
                                          "_diffusion_summary.csv")
                if guess != csv_path and os.path.isfile(guess):
                    try:    diff_df = pd.read_csv(guess)
                    except Exception: diff_df = None

            data = df[["particle", "frame", "y", "x"]].values.astype(float)

            # Per-track features so we can colour by motion class if available
            features = None
            color_by = None
            if diff_df is not None and "motion" in diff_df.columns:
                motion_to_int = {"Immobile": 0, "Confined": 1,
                                 "Brownian": 2, "Directed": 3,
                                 "Unknown":  4}
                motion_map = dict(zip(diff_df["particle"],
                                      diff_df["motion"]))
                # napari Tracks features are indexed by the rows of `data`
                col = [motion_to_int.get(motion_map.get(int(pid), "Unknown"), 4)
                       for pid in df["particle"].values]
                features = {"motion_int": col}
                color_by = "motion_int"

            layer = v.add_tracks(
                data, name=os.path.basename(csv_path),
                blending="opaque",
                **({"features": features, "color_by": color_by,
                    "colormap": "turbo"} if features is not None else {}))

            # Cache for the click handler
            self._ws_tracks_df    = df
            self._ws_diff_df      = diff_df
            self._ws_tracks_layer = layer
            self._attach_track_click_handler(layer)
            self._ws_inspector.clear()

            self.statusBar().showMessage(
                f"Loaded {df['particle'].nunique():,} tracks "
                f"({len(df):,} points) into napari — "
                "click a track to inspect.", 6000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load tracks from {os.path.basename(csv_path)}:\n\n{exc}")

    def _attach_track_click_handler(self, layer):
        """Hook a mouse-drag callback onto the Tracks layer so clicking a
        track populates the inspector panel.  Idempotent — replaces any
        previous handler on the same layer."""
        if layer is None:
            return
        try:
            # Wipe previous callbacks we attached
            keep = [cb for cb in layer.mouse_drag_callbacks
                    if getattr(cb, "_firefly_inspector", False) is False]
            layer.mouse_drag_callbacks.clear()
            for cb in keep:
                layer.mouse_drag_callbacks.append(cb)
        except Exception:
            pass

        def _on_click(_layer, event):
            if event.type != "mouse_press":
                return
            try:
                pid = self._track_id_at(event.position)
            except Exception:
                pid = None
            if pid is None:
                return
            self._show_track_in_inspector(int(pid))
        _on_click._firefly_inspector = True
        try:
            layer.mouse_drag_callbacks.append(_on_click)
        except Exception:
            pass

    def _track_id_at(self, world_pos) -> "int | None":
        """Return the particle id of the track whose nearest localisation
        is closest to `world_pos`.

        We search ALL localisations globally rather than only the
        current frame — napari draws each track's trail across many
        past frames, so a click on a trail line will rarely land on a
        frame where that particular track has a point.  A small
        temporal penalty (~weighted) prefers hits at or near the
        current time when there's a tie, but doesn't exclude trails.
        """
        if self._ws_tracks_df is None:
            return None
        import numpy as _np
        # world_pos comes from napari's Tracks layer; for a 3-D data
        # array (track_id, t, y, x) the position tuple is (t, y, x).
        if len(world_pos) < 3:
            return None
        t = float(world_pos[0])
        y = float(world_pos[-2])
        x = float(world_pos[-1])
        df = self._ws_tracks_df
        xs = df["x"].values
        ys = df["y"].values
        fs = df["frame"].values
        # Spatial distance² + a tiny temporal penalty so ties go to
        # localisations near the current time.  The temporal weight
        # is in *px²-per-frame²* units — set so ~10 frames away costs
        # the same as ~1 pixel away.
        d2 = (xs - x) ** 2 + (ys - y) ** 2 + 0.01 * (fs - t) ** 2
        idx = int(_np.argmin(d2))
        # Tolerance is on the SPATIAL part only — generous (≤ ~16 px)
        # so clicks on the trail line (between vertices) still register.
        sp_d2 = (xs[idx] - x) ** 2 + (ys[idx] - y) ** 2
        if sp_d2 > 256.0:        # > 16 px from any track point
            return None
        return int(df["particle"].values[idx])

    def _show_track_in_inspector(self, particle_id: int):
        """Look up per-particle stats and push into the inspector panel."""
        if self._ws_tracks_df is None:
            return
        df = self._ws_tracks_df
        rows = df[df["particle"] == particle_id]
        if rows.empty:
            return
        kwargs: dict = {"particle_id": particle_id}
        kwargs["length"] = int(len(rows))
        kwargs["start_frame"] = int(rows["frame"].min())
        kwargs["end_frame"]   = int(rows["frame"].max())
        # Net displacement + path length in PIXELS — convert to µm if we
        # know the pixel size (overridden in the sidebar or 1.0 fallback).
        try:
            px = (float(self.s_pixel_size.value())
                  if self.c_override_px.isChecked() else 1.0)
        except AttributeError:
            px = 1.0
        try:
            import numpy as _np
            xs = rows["x"].values; ys = rows["y"].values
            if len(xs) >= 2:
                net = float(_np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) * px
                seg = _np.hypot(_np.diff(xs), _np.diff(ys)).sum() * px
                kwargs["net_displacement_um"] = net
                kwargs["total_path_um"]       = float(seg)
                if seg > 0:
                    kwargs["straightness"] = net / float(seg)
        except Exception:
            pass
        if "mass" in rows.columns:
            try:    kwargs["mean_mass"] = float(rows["mass"].mean())
            except Exception: pass
        # Diff-summary lookups
        diff = self._ws_diff_df
        if diff is not None and "particle" in diff.columns:
            d_row = diff[diff["particle"] == particle_id]
            if not d_row.empty:
                r = d_row.iloc[0]
                if "D" in d_row.columns:
                    try:    kwargs["d"] = float(r["D"])
                    except Exception: pass
                if "alpha" in d_row.columns:
                    try:    kwargs["alpha"] = float(r["alpha"])
                    except Exception: pass
                if "motion" in d_row.columns:
                    kwargs["motion"] = str(r["motion"])
        self._ws_inspector.show_track(**kwargs)

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
            diff_path = os.path.join(extras_dir,
                                       f"{stem}_diffusion_summary.csv")
            self._ws_load_tracks_path(
                tracks_path,
                diff_csv_path=diff_path if os.path.isfile(diff_path) else None)
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

            # ── Figures tab ───────────────────────────────────────────────
            ("figures/theme",            self.c_fig_theme,       "combo"),
            ("figures/proj_cmap",        self.c_fig_proj_cmap,   "combo"),
            ("figures/dpi",              self.s_fig_dpi,         "spin",  int),
            ("figures/save_pdf",         self.c_fig_save_pdf,    "check", _bool_cast),
            ("figures/per_panel",        self.c_fig_per_panel,   "check", _bool_cast),

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
        # Clamp the window to the available screen rect so a saved
        # geometry from a previous session with a bigger / external
        # monitor doesn't push the window off-screen.  Leaves a small
        # margin so the title bar + bottom edge stay visible.
        try:
            screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                w = min(self.width(),  max(900, avail.width()  - 20))
                h = min(self.height(), max(640, avail.height() - 40))
                self.resize(w, h)
                # If the top-left is off-screen (negative or beyond the
                # edges), re-centre instead of restoring.
                x = self.x(); y = self.y()
                if (x < avail.left() or y < avail.top()
                        or x + w > avail.right()
                        or y + h > avail.bottom()):
                    self.move(avail.left() + (avail.width()  - w) // 2,
                              avail.top()  + (avail.height() - h) // 2)
        except Exception:
            pass

        # Path entries are deliberately NOT restored — every launch starts
        # with empty file / folder fields so the user always picks fresh
        # inputs.  Clear any previously-saved values from the QSettings
        # store too, so they don't linger on disk.
        _skip_paths = {"analysis/file", "analysis/outdir",
                       "analysis/batch_folder"}
        for _k in _skip_paths:
            try: s.remove(_k)
            except Exception: pass

        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            if key in _skip_paths:
                continue
            try:
                v = s.value(key)
                if v is None or v == "":
                    continue
                if kind == "text":
                    widget.setText(str(v))
                elif kind == "combo":
                    v_str = str(v)
                    items = [widget.itemText(i)
                             for i in range(widget.count())]
                    if v_str in items:
                        widget.setCurrentText(v_str)
                    # Migration: old saved backend values were stored as
                    # internal strings ("torch-mps") but the combo now
                    # shows labels ("Torch — Apple MPS").  Translate if
                    # this widget is the backend combo and the saved
                    # value is a recognised internal value.
                    elif widget is getattr(self, "c_backend", None):
                        lbl = self._BACKEND_VALUE_TO_LABEL.get(v_str)
                        if lbl and lbl in items:
                            widget.setCurrentText(lbl)
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

        # Single-sample panel checkbox states (Figures tab)
        try:
            for key, cb in self._single_panel_checkboxes.items():
                v = s.value(f"figures/single_panel_{key}")
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
        # Path entries are intentionally not persisted — see _restore_settings.
        _skip_paths = {"analysis/file", "analysis/outdir",
                       "analysis/batch_folder"}
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            if key in _skip_paths:
                continue
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

        # Single-sample panel checkbox states (Figures tab)
        try:
            for key, cb in self._single_panel_checkboxes.items():
                s.setValue(f"figures/single_panel_{key}", bool(cb.isChecked()))
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

        # Explicitly close any embedded napari viewers.  Each one is a
        # QMainWindow under the hood — without calling `viewer.close()`
        # it stays alive as a "still-visible top-level window", which
        # is enough to stop QApplication from quitting even after our
        # own MainWindow closes (the "⌘Q does nothing after a run"
        # symptom we saw).
        for attr_chain in (
                ("_roi_viewer", "_viewer"),       # Import-tab preview
                ("_napari_viewer",),              # Visualise-tab viewer
        ):
            try:
                obj = self
                for a in attr_chain:
                    obj = getattr(obj, a, None)
                    if obj is None:
                        break
                if obj is not None and hasattr(obj, "close"):
                    obj.close()
            except Exception:
                pass

        super().closeEvent(event)
        # Belt-and-braces: ask the QApplication to quit.  This is a
        # no-op when QApplication.quitOnLastWindowClosed already
        # triggered, but covers cases where a stray hidden window
        # (e.g. napari's docked plugin manager) keeps the event loop
        # alive on macOS.
        try:    QtWidgets.QApplication.instance().quit()
        except Exception: pass

    # ── Backend availability helper ───────────────────────────────────────
    # Two-way mapping between GUI labels and internal backend strings.
    # The pipeline's `_resolve_backend` understands the hyphenated forms
    # (auto / trackpy / torch / torch-mps / torch-cuda / torch-cpu); the
    # GUI shows them as proper grammar so users don't see lowercase
    # snake-case-y identifiers in their face.
    _BACKEND_LABEL_TO_VALUE = {
        "Auto":               "auto",
        "Trackpy (CPU)":      "trackpy",
        "Torch (auto)":       "torch",
        "Torch — Apple MPS":  "torch-mps",
        "Torch — NVIDIA CUDA": "torch-cuda",
        "Torch — CPU":        "torch-cpu",
    }
    _BACKEND_VALUE_TO_LABEL = {v: k for k, v in _BACKEND_LABEL_TO_VALUE.items()}

    def _available_backends(self) -> list[str]:
        """Return the static list of selectable backend LABELS (display
        strings) for the dropdown.  Internal values are resolved via
        `_backend_value_from_label` before being sent to the worker.

        IMPORTANT: we deliberately do NOT probe torch here.  On some macOS
        / PyTorch / Apple-Silicon combinations, just importing torch and
        calling `torch.backends.mps.is_available()` is enough to trigger
        noisy MPS command-buffer errors on stderr and, in the worst case,
        kill the process before the GUI is fully up.  Probing happens
        lazily inside the analysis worker only when actually selected.
        """
        return list(self._BACKEND_LABEL_TO_VALUE.keys())

    def _backend_value_from_label(self, label: str) -> str:
        """Translate a dropdown label to the internal pipeline string.
        Falls through to the label itself so old saved-settings values
        (`torch-mps` etc.) still work after upgrading."""
        if label in self._BACKEND_LABEL_TO_VALUE:
            return self._BACKEND_LABEL_TO_VALUE[label]
        if label in self._BACKEND_VALUE_TO_LABEL:
            return label   # already an internal value
        return label or "auto"

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

    # ── External-CSV pickers ──────────────────────────────────────────────
    def _on_browse_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select localisations CSV",
            self.e_csv_outdir.text() or os.path.expanduser("~"),
            "CSV (*.csv);;All files (*)")
        if path:
            self.e_csv_path.setText(path)
            if not self.e_csv_outdir.text():
                self.e_csv_outdir.setText(os.path.dirname(path))

    def _on_browse_csv_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder",
            self.e_csv_outdir.text() or os.path.expanduser("~"))
        if path:
            self.e_csv_outdir.setText(path)

    def _on_browse_csv_bg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select background image (optional)",
            self.e_csv_outdir.text() or os.path.expanduser("~"),
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if path:
            self.e_csv_bg.setText(path)

    def _on_load_manifest(self):
        """Open a `<stem>_run_manifest.json` and apply its widget_state
        snapshot to the sidebar, plus repopulate the input/output paths."""
        import json
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open run manifest",
            self.e_outdir.text() or os.path.expanduser("~"),
            "Manifest (*_run_manifest.json);;JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Couldn't load manifest", str(exc))
            return
        # Apply widget snapshot (most important)
        state = manifest.get("widget_state") or {}
        self._apply_widget_state(state)
        # Path fields: try the original input path; if missing, leave alone
        inp = (manifest.get("input") or {}).get("path", "") or ""
        if inp and os.path.isfile(inp):
            self.e_file.setText(inp)
        # Output folder
        outd = manifest.get("output_dir", "")
        if outd:
            self.e_outdir.setText(outd)
        # Status feedback
        v = manifest.get("firefly_version", "?")
        when = manifest.get("created_at", "?")
        self.statusBar().showMessage(
            f"Loaded manifest from {os.path.basename(path)}  "
            f"(FIREFLY {v}, {when})", 8000)

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
            "Manual polygon":    "polygon",
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
            # Per-file polygon ROI lookup.  If this file has a saved
            # polygon, it's sent regardless of the ROI-mode setting and
            # the worker treats it as if mode were "polygon".  Files
            # without a saved polygon fall back to the global ROI mode.
            "roi_polygon":       self._roi_polygons.get(
                                    os.path.abspath(fpath)) or None,
            "drift_correct":     bool(self.c_drift_correct.isChecked()),
            "drift_segment":     int(self.s_drift_segment.value()),
            "cluster_eps_nm":      float(self.s_cluster_eps_nm.value()),
            "cluster_min_samples": int(self.s_cluster_min_samples.value()),
            "backend":           self._backend_value_from_label(
                                    self.c_backend.currentText()),
            "workers":           int(self.s_workers.value()),
            "chunk_size":        int(self.s_chunk_size.value()),
            # ── Figures-tab knobs (single-sample figure output) ───────────
            "fig_theme":         self.c_fig_theme.currentText(),
            "fig_proj_cmap":     self.c_fig_proj_cmap.currentText(),
            "fig_dpi":           int(self.s_fig_dpi.value()),
            "fig_save_pdf":      bool(self.c_fig_save_pdf.isChecked()),
            "fig_per_panel":     bool(self.c_fig_per_panel.isChecked()),
            "fig_single_panels": [k for k, cb in
                                  self._single_panel_checkboxes.items()
                                  if cb.isChecked()],
            # Full widget-state snapshot — written into the run manifest
            # so the run can be exactly replayed later via "Load manifest…"
            "widget_state":      self._widget_state_dict(),
        }

    def _widget_state_dict(self) -> dict:
        """Return the current sidebar widget values keyed by their QSettings
        path (e.g. 'analysis/diameter').  Mirrors `_save_settings` but
        in-memory so we can embed it in run manifests."""
        out: dict = {}
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            try:
                if   kind == "text":  out[key] = widget.text()
                elif kind == "combo": out[key] = widget.currentText()
                elif kind == "spin":  out[key] = widget.value()
                elif kind == "check": out[key] = bool(widget.isChecked())
            except Exception:
                pass
        # Panel-checkbox selections (single + compare) too
        try:
            out["figures/single_panels"] = [
                k for k, cb in self._single_panel_checkboxes.items()
                if cb.isChecked()]
        except AttributeError: pass
        try:
            out["compare/panels"] = [
                k for k, cb in self._cmp_panel_checkboxes.items()
                if cb.isChecked()]
        except AttributeError: pass
        return out

    def _apply_widget_state(self, state: dict) -> None:
        """Push a widget-state dict (produced by `_widget_state_dict`) back
        into the sidebar widgets.  Used by the manifest 'Replay' button."""
        if not isinstance(state, dict):
            return
        # Cast helpers (mirror _restore_settings)
        def _bool_cast(v):
            if isinstance(v, bool): return v
            if isinstance(v, str):  return v.lower() in ("1", "true", "yes")
            return bool(v)
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            if key not in state:
                continue
            v = state[key]
            try:
                if kind == "text":
                    widget.setText(str(v))
                elif kind == "combo":
                    items = [widget.itemText(i)
                             for i in range(widget.count())]
                    if str(v) in items:
                        widget.setCurrentText(str(v))
                elif kind == "spin":
                    caster = spec[3]
                    widget.setValue(caster(v))
                elif kind == "check":
                    caster = spec[3]
                    widget.setChecked(caster(v))
            except Exception:
                pass
        # Panel checkboxes
        try:
            wanted = set(state.get("figures/single_panels", []))
            if wanted and hasattr(self, "_single_panel_checkboxes"):
                for k, cb in self._single_panel_checkboxes.items():
                    cb.setChecked(k in wanted)
        except Exception: pass
        try:
            wanted = set(state.get("compare/panels", []))
            if wanted and hasattr(self, "_cmp_panel_checkboxes"):
                for k, cb in self._cmp_panel_checkboxes.items():
                    cb.setChecked(k in wanted)
        except Exception: pass

    # ── Parameter presets ────────────────────────────────────────────────
    _BUILTIN_PRESETS_TAG = "__firefly_builtin__"

    @staticmethod
    def _presets_dir() -> str:
        d = os.path.expanduser("~/.firefly/presets")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def _builtin_presets(cls) -> "dict[str, dict]":
        """Default presets seeded on first launch.  Two reasonable
        starting points for common rigs in the lab — users can override
        or extend them via the 'Save…' button."""
        return {
            "PC12 Cells": {
                cls._BUILTIN_PRESETS_TAG: True,
                # Imaging metadata — 100x oil, fast PALM acquisition
                "analysis/override_px":     True,
                "analysis/pixel_size":      0.106,
                "analysis/override_fi":     True,
                "analysis/frame_interval":  0.020,
                # Preprocessing — flat well-spread cytoplasm
                "analysis/bg_method":       "Uniform Filter",
                "analysis/bg_radius":       20,
                # Detection — typical PALM PSF after preprocessing
                "analysis/diameter":        7,
                "analysis/auto_minmass":    False,
                "analysis/minmass":         1.5,
                # Linking — small per-step displacements at 50 fps
                "analysis/search_range":    5,
                "analysis/memory":          3,
                "analysis/min_track_len":   8,
                "analysis/max_track_len":   0,
                # Diffusion + motion classification — standard sptPALM
                "analysis/max_lagtime":     20,
                "analysis/n_fit":           5,
                "analysis/alpha_immobile":  0.5,
                "analysis/alpha_confined":  0.9,
                "analysis/alpha_directed":  1.1,
                "analysis/mobile_d":        0.05,
                "analysis/jdd_components":  2,
                # ROI — let auto-threshold handle the cell outline
                "analysis/roi_mode":        "Auto threshold",
                "analysis/roi_auto_method": "Li",
                "analysis/roi_threshold":   0.08,
                "analysis/roi_mask_mode":   "Mean",
                # Drift correction — segment length tuned for ~10k frames
                "analysis/drift_correct":   True,
                "analysis/drift_segment":   500,
                # Clustering — receptor-nanodomain defaults
                "analysis/cluster_eps_nm":  50.0,
                "analysis/cluster_min_samples": 10,
            },
            "Drosophila Neurons": {
                cls._BUILTIN_PRESETS_TAG: True,
                # Imaging metadata
                "analysis/override_px":     True,
                "analysis/pixel_size":      0.106,
                "analysis/override_fi":     True,
                "analysis/frame_interval":  0.030,
                # Preprocessing — narrower processes, smaller bg radius
                "analysis/bg_method":       "Uniform Filter",
                "analysis/bg_radius":       15,
                # Detection — typically sparser labelling than PC12
                "analysis/diameter":        7,
                "analysis/auto_minmass":    False,
                "analysis/minmass":         1.0,
                # Linking — slower diffusion in axons / dendrites
                "analysis/search_range":    4,
                "analysis/memory":          2,
                "analysis/min_track_len":   10,
                "analysis/max_track_len":   0,
                # Diffusion + motion classification
                "analysis/max_lagtime":     20,
                "analysis/n_fit":           5,
                "analysis/alpha_immobile":  0.5,
                "analysis/alpha_confined":  0.9,
                "analysis/alpha_directed":  1.1,
                "analysis/mobile_d":        0.03,
                "analysis/jdd_components":  2,
                # ROI — small / branched cells; manual polygon is more robust
                "analysis/roi_mode":        "Manual polygon",
                "analysis/roi_auto_method": "Li",
                "analysis/roi_threshold":   0.10,
                "analysis/roi_mask_mode":   "Mean",
                # Drift correction
                "analysis/drift_correct":   True,
                "analysis/drift_segment":   400,
                # Clustering — synaptic-density defaults
                "analysis/cluster_eps_nm":  40.0,
                "analysis/cluster_min_samples": 8,
            },
        }

    def _finalise_presets(self) -> None:
        """Called once on startup (deferred so every sidebar widget is
        constructed first).  Seeds the two built-in presets to disk if
        the user hasn't already saved overrides, then populates the
        combobox."""
        try:
            self._seed_builtin_presets()
            self._refresh_preset_combo()
            try:
                self.c_preset.currentTextChanged.connect(
                    self._on_preset_picked)
            except Exception:
                pass
        except Exception:
            pass

    def _seed_builtin_presets(self) -> None:
        """Write the built-in presets to ~/.firefly/presets/ on first
        launch.  Skips any name the user has already saved their own
        version of, so user-customised presets aren't overwritten."""
        import json
        d = self._presets_dir()
        for name, payload in self._builtin_presets().items():
            path = os.path.join(d, f"{name}.json")
            if os.path.isfile(path):
                # Only overwrite if our previous write also tagged it as
                # built-in (i.e. user hasn't customised it).
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        prev = json.load(fh)
                    if not prev.get(self._BUILTIN_PRESETS_TAG, False):
                        continue
                except Exception:
                    continue
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
            except Exception:
                pass

    def _list_presets(self) -> "list[str]":
        d = self._presets_dir()
        try:
            return sorted(
                os.path.splitext(f)[0] for f in os.listdir(d)
                if f.endswith(".json"))
        except Exception:
            return []

    def _refresh_preset_combo(self) -> None:
        if not hasattr(self, "c_preset"):
            return
        names = self._list_presets()
        self.c_preset.blockSignals(True)
        try:
            self.c_preset.clear()
            self.c_preset.addItem("— Current settings —")
            for n in names:
                self.c_preset.addItem(n)
        finally:
            self.c_preset.blockSignals(False)

    def _on_preset_picked(self, name: str) -> None:
        """Apply a preset to the sidebar when the user picks one from the
        combobox.  Ignores the leading '— Current settings —' sentinel."""
        if not name or name.startswith("—"):
            return
        import json
        path = os.path.join(self._presets_dir(), f"{name}.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Couldn't load preset", str(exc))
            return
        # Drop our own internal tag before applying
        state.pop(self._BUILTIN_PRESETS_TAG, None)
        self._apply_widget_state(state)
        self.statusBar().showMessage(f"Applied preset: {name}", 5000)

    def _on_preset_delete(self) -> None:
        """Remove the currently-selected preset from disk after
        confirmation."""
        name = self.c_preset.currentText() if hasattr(self, "c_preset") else ""
        if not name or name.startswith("—"):
            QtWidgets.QMessageBox.information(
                self, "No preset selected",
                "Pick a preset from the dropdown first, then click Delete.")
            return
        path = os.path.join(self._presets_dir(), f"{name}.json")
        if not os.path.isfile(path):
            return
        # Heads-up if the user is about to delete a built-in: it'll come
        # back on next launch from the seeding logic.
        import json
        is_builtin = False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                is_builtin = bool(json.load(fh).get(
                    self._BUILTIN_PRESETS_TAG, False))
        except Exception:
            pass
        msg = f"Delete preset '{name}'?"
        if is_builtin:
            msg += ("\n\nThis is a built-in preset — it will be re-created "
                    "on the next FIREFLY launch unless you save your own "
                    "version with the same name first.")
        ret = QtWidgets.QMessageBox.question(
            self, "Delete preset", msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No)
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:    os.remove(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Delete failed", str(exc))
            return
        self._refresh_preset_combo()
        # Park selection on the "current settings" sentinel
        try:    self.c_preset.setCurrentIndex(0)
        except Exception: pass
        self.statusBar().showMessage(f"Deleted preset: {name}", 5000)

    def _on_preset_save(self) -> None:
        """Prompt for a name and write the current sidebar to disk."""
        import json, re
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save preset",
            "Name this preset (use letters, numbers, spaces, '-' or '_'):")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            return
        # Sanitise the filename — no path separators, control chars, etc.
        if not re.match(r"^[A-Za-z0-9 _\-]+$", name):
            QtWidgets.QMessageBox.warning(
                self, "Invalid name",
                "Preset names can only contain letters, numbers, "
                "spaces, '-' and '_'.")
            return
        path = os.path.join(self._presets_dir(), f"{name}.json")
        if os.path.isfile(path):
            ret = QtWidgets.QMessageBox.question(
                self, "Overwrite preset?",
                f"'{name}' already exists.  Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        state = self._widget_state_dict()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", str(exc))
            return
        self._refresh_preset_combo()
        # Surface the new preset as the current selection
        try:
            self.c_preset.setCurrentText(name)
        except Exception:
            pass
        self.statusBar().showMessage(f"Saved preset: {name}", 5000)

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
        elif getattr(self, "r_mode_csv", None) and self.r_mode_csv.isChecked():
            self._start_csv_run()
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

    def _start_csv_run(self):
        """External-CSV mode: skip detection, treat the CSV as the
        localisations source and run the rest of the pipeline."""
        csv_path = self.e_csv_path.text().strip()
        if not csv_path or not os.path.isfile(csv_path):
            QtWidgets.QMessageBox.warning(
                self, "No CSV",
                "Pick a localisations CSV on the Import tab first.")
            self._switch_to_tab("Import")
            return
        out_dir = self.e_csv_outdir.text().strip() or os.path.dirname(csv_path)
        self._switch_to_tab("Analysis")
        self._start_elapsed_timer()

        # Build params the same way as a normal run, then override the
        # source-related fields.  Using the same widget snapshot means
        # ROI / drift / linking / MSD / figure settings are honoured.
        params = self._build_params_for_file(csv_path, out_dir)
        # Pixel size / frame interval default off the sidebar even when
        # the Override checkboxes are unticked — there's no file metadata
        # to fall back on for a CSV.
        if not params.get("pixel_size"):
            params["pixel_size"] = float(self.s_pixel_size.value())
        if not params.get("frame_interval"):
            params["frame_interval"] = float(self.s_frame_interval.value())
        preset = self.c_csv_preset.currentText()
        params["source"] = "external_csv"
        params["csv_preset"] = (
            "auto" if preset == "Auto-detect" else preset)
        bg = self.e_csv_bg.text().strip()
        if bg and os.path.isfile(bg):
            params["bg_image_path"] = bg

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
        try:
            self.mass_hist.reset()
            self.live_view.reset()
            self._analysis_stack.setCurrentIndex(0)
        except AttributeError:
            pass
        self._is_batch_run   = False
        self._is_compare_run = False

        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_analysis_in_subprocess,
            args=(params, self._msg_queue, self._cancel_event),
            name="FIREFLY-AnalysisWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()
        self.btn_run.setText("Stop")
        self.statusBar().showMessage("Running (external CSV)…")

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
        self._start_elapsed_timer()

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
        try:
            self.mass_hist.reset()
            self.mass_hist.set_minmass(float(self.s_minmass.value())
                                      if not self.c_auto_minmass.isChecked()
                                      else None)
            self.live_view.reset()
            self._analysis_stack.setCurrentIndex(0)   # show cockpit
        except AttributeError:
            pass
        self._is_batch_run   = False
        self._is_compare_run = False

        # Spawn analysis SUBPROCESS (not thread).  Rationale: Qt holds a
        # Metal-backed surface for window compositing on macOS, and that
        # contends with PyTorch's MPS allocator in the same process.  A
        # subprocess gives PyTorch a clean Python interpreter with no Qt
        # loaded — MPS gets the full unified-memory pool to itself.
        # Bounded queue — a 60 FPS live preview at ~590 KB/frame can
        # push ~35 MB/s through this pipe.  If the GUI ever stalls for
        # a few seconds the queue grows unbounded and pushes the
        # system into swap (we had a user hard-freeze caused by that).
        # 2000 messages is enough headroom for normal jitter; the
        # worker drops preview/mass messages when full and keeps only
        # the analysis-critical ones (log/progress/done/etc.).
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
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
        """Kick off batch analysis over the checked series.  Each series
        contributes one analysis run; the loader uses the per-series
        list of checked files (rather than auto-discovering siblings)."""
        groups = self._batch_checked_series()
        if not groups:
            QtWidgets.QMessageBox.warning(
                self, "No files",
                "On the Import tab, switch to Batch mode and pick a "
                "folder + at least one file.")
            self._switch_to_tab("Import")
            return
        self._switch_to_tab("Analysis")
        self._start_elapsed_timer()

        # Batch outputs go to <input_folder>/batch_results/<stem>/  — same
        # convention as the Tk app.  Build a params dict per series.
        out_root = os.path.join(self.e_batch_folder.text().strip(),
                                "batch_results")
        params_list = []
        for g in groups:
            fpath = g["primary"]
            stem = os.path.splitext(os.path.basename(fpath))[0]
            file_out = os.path.join(out_root, stem)
            p = self._build_params_for_file(fpath, file_out)
            # Override loader auto-discovery: pass the exact list of
            # checked sister files for this series.
            p["series_files"] = list(g.get("files") or [])
            params_list.append(p)

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
        try:
            self.mass_hist.reset()
            self.mass_hist.set_minmass(float(self.s_minmass.value())
                                      if not self.c_auto_minmass.isChecked()
                                      else None)
            self.live_view.reset()
            self._analysis_stack.setCurrentIndex(0)   # show cockpit
        except AttributeError:
            pass
        self._is_batch_run   = True
        self._is_compare_run = False

        # Bounded queue — a 60 FPS live preview at ~590 KB/frame can
        # push ~35 MB/s through this pipe.  If the GUI ever stalls for
        # a few seconds the queue grows unbounded and pushes the
        # system into swap (we had a user hard-freeze caused by that).
        # 2000 messages is enough headroom for normal jitter; the
        # worker drops preview/mass messages when full and keeps only
        # the analysis-critical ones (log/progress/done/etc.).
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_batch_in_subprocess,
            args=(params_list, self._msg_queue, self._cancel_event),
            name="FIREFLY-BatchWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.statusBar().showMessage(
            f"Batch: 0 / {len(groups)} series")

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

        # Bounded queue — a 60 FPS live preview at ~590 KB/frame can
        # push ~35 MB/s through this pipe.  If the GUI ever stalls for
        # a few seconds the queue grows unbounded and pushes the
        # system into swap (we had a user hard-freeze caused by that).
        # 2000 messages is enough headroom for normal jitter; the
        # worker drops preview/mass messages when full and keeps only
        # the analysis-critical ones (log/progress/done/etc.).
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
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
            elif kind == "mass_chunk":
                # Live histogram update from the localisation stream
                try:    self.mass_hist.add_chunk(payload)
                except AttributeError: pass
            elif kind == "preview_frame":
                # Live detection-view update.  Payload carries a flat
                # bytes blob + shape so we can reconstruct the frame
                # array without round-tripping through numpy in the
                # queue (lighter and works in subprocess-spawned land).
                try:
                    import numpy as _np
                    shape = payload.get("shape") or [0, 0]
                    blob  = payload.get("frame")
                    if blob and shape[0] and shape[1]:
                        arr = _np.frombuffer(blob, dtype=_np.float32) \
                                 .reshape(shape[0], shape[1])
                        self.live_view.set_frame(
                            arr,
                            payload.get("xs", []),
                            payload.get("ys", []),
                            payload.get("idx", 0),
                            payload.get("n_frames", 0))
                except (AttributeError, ValueError, KeyError):
                    pass
            elif kind == "done":
                # Single-file completion.  Only valid in non-batch mode;
                # in batch mode the per-file messages are "file_done".
                self._handle_done(payload)
                worker_done = True
            elif kind == "file_starting":
                # New file in a batch — wipe the mass histogram so it
                # doesn't accumulate values from the previous file's
                # localisations.  Live view is fine — preview_frame
                # messages naturally overwrite as they arrive.
                try:    self.mass_hist.reset()
                except AttributeError: pass
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
        """One series in a batch finished successfully — not the terminal msg.
        ('file' here = 'series' in the GUI sense — the batch list now has
        one entry per series, and the worker calls it once per series.)
        """
        i, total = payload.get("index", 0), payload.get("total", 0)
        n_tracks = payload.get("n_tracks", 0)
        stem     = payload.get("stem", "")
        self.statusBar().showMessage(
            f"Batch: {i} / {total} series complete  ({n_tracks:,} tracks)")
        if total:
            pct = int(100 * i / total)
            self.batch_progress.setValue(pct)
            self.batch_progress.setFormat(f"Batch  {i}/{total}  ({pct}%)")
            self.batch_subprogress.setValue(pct)
            self.batch_subprogress.setFormat(
                f"Last: {stem}  ({n_tracks:,} tracks)")

    def _handle_file_error(self, payload: dict):
        """One series in a batch failed — log it, batch continues."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        f = payload.get("file", "?")
        self.console_log.appendPlainText(
            f"\n  ⚠ [{i}/{total}] failed: {os.path.basename(f)}")
        self.batch_stage_label.setText(
            f"[{i}/{total}] failed: {os.path.basename(f)} — batch continues")
        self.statusBar().showMessage(f"Batch: series {i} failed (continuing)")

    def _handle_batch_done(self, payload: dict):
        """Batch terminal message — all series attempted."""
        n_total = payload.get("n_total", 0)
        n_ok    = payload.get("n_ok",    0)
        n_fail  = payload.get("n_fail",  0)
        self.batch_progress.setValue(100)
        self.batch_progress.setFormat(
            f"Batch complete  —  {n_ok}/{n_total} series succeeded, "
            f"{n_fail} failed")
        self.batch_subprogress.hide()
        self.statusBar().showMessage(
            f"Batch complete — {n_ok}/{n_total} series succeeded, "
            f"{n_fail} failed")

        # Populate the results panel with the batch summary
        headline = (f"Batch complete — {n_ok}/{n_total} series succeeded"
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

    # ── Elapsed-time tracker for the Analysis tab ─────────────────────────
    @staticmethod
    def _format_elapsed(secs: float) -> str:
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _start_elapsed_timer(self):
        import time as _time
        self._run_start_time = _time.monotonic()
        try:
            self.lbl_elapsed.setText("Elapsed: 00:00")
        except AttributeError:
            return
        self._elapsed_timer.start()

    def _on_elapsed_tick(self):
        if self._run_start_time is None:
            return
        import time as _time
        try:
            self.lbl_elapsed.setText(
                f"Elapsed: {self._format_elapsed(_time.monotonic() - self._run_start_time)}")
        except AttributeError:
            pass

    def _stop_elapsed_timer(self):
        self._elapsed_timer.stop()
        if self._run_start_time is not None:
            import time as _time
            final = self._format_elapsed(_time.monotonic() - self._run_start_time)
            try:
                self.lbl_elapsed.setText(f"Elapsed: {final}")
            except AttributeError:
                pass
        self._run_start_time = None

    def _cleanup_after_run(self):
        """Tear down the subprocess + queue after a run ends."""
        self._poll_timer.stop()
        self._stop_elapsed_timer()
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
        # Run is over — swap from cockpit to results panel.
        try:
            self._analysis_stack.setCurrentIndex(1)
        except AttributeError:
            pass

    # ── Crash reporter integration ────────────────────────────────────────
    def _install_menubar(self):
        """Give FIREFLY its own QMenuBar with at minimum a File → Quit
        action.  On macOS the menubar is global; if we don't own one,
        any embedded napari Viewer will claim it, and clearing napari's
        menubar later (we no longer do that, but defensively) used to
        take ⌘Q down with it.  Adding our own keeps Quit reliable."""
        mb = self.menuBar()
        # Use native (system) menu bar on macOS so the entries show in
        # the system bar instead of inside the window.
        try:    mb.setNativeMenuBar(True)
        except Exception: pass

        file_menu = mb.addMenu("File")

        # Quit — ⌘Q on macOS, Ctrl+Q elsewhere.  Qt's StandardKey.Quit
        # maps to the right shortcut per platform.  We bind the shortcut
        # with ApplicationShortcut context so it fires no matter which
        # QMainWindow currently has focus (the embedded napari viewer
        # is also a QMainWindow, and without this the shortcut only
        # fires when *our* window is key — hence the flaky ⌘Q).
        act_quit = QtGui.QAction("Quit FIREFLY", self)
        act_quit.setMenuRole(QtGui.QAction.MenuRole.QuitRole)
        act_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        act_quit.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Belt-and-braces backup — a standalone QShortcut at the same
        # application-wide context.  If something downstream resets
        # the action's shortcut (some napari versions reach into the
        # global QAction list), this still catches ⌘Q.
        self._sc_quit = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Quit, self)
        self._sc_quit.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_quit.activated.connect(self.close)

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
                    "Threshold":          self.s_minmass.value(),
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
/* Note: we deliberately DO NOT set a background on the bare QWidget rule.
   Doing so paints every transparent wrapper widget (e.g. the QWidgets used
   as containers for QHBoxLayout rows inside a QGroupBox) in the darkest
   shade, which then shows through as a dark rectangle against the lighter
   panel background.  Widgets that need an explicit background get one
   from their own rule (QMainWindow, QGroupBox, sidebar frame, etc.). */
QWidget {{
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
    subcontrol-origin:    padding;
    subcontrol-position:  top right;
    background:           transparent;
    background-color:     transparent;
    border:               none;
    width:                22px;
}}
/* Drop-down indicator rendered as a small light circle instead of the
   default arrow / rectangle.  Width = height + border-radius half-of-side
   keeps it perfectly round. */
QComboBox::down-arrow {{
    image:                none;
    width:                8px;
    height:               8px;
    border-radius:        4px;
    background-color:     {TXT_MUTED};
    margin:               0 8px 0 0;
}}
QComboBox::down-arrow:on,
QComboBox::down-arrow:hover {{
    background-color:     {ACC};
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

/* ── Mode-toggle tiles (Import tab) ─────────────────────────────────────── */
/* Big segmented-control cards.  Custom QFrame subclass (_ModeTile) with
   a 'checked' Qt property — QSS uses property selectors to switch the
   border between border-color: BORDER (unchecked) and ACC (checked). */
QFrame#mode_tile {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       8px;
}}
QFrame#mode_tile:hover {{
    background-color:    {PANEL_ALT};
    border-color:        {BORDER_HI};
}}
QFrame#mode_tile[checked="true"] {{
    background-color:    {PANEL_ALT};
    border:              2px solid {ACC};
}}
QFrame#mode_tile[checked="true"]:hover {{
    border-color:        {ACC_HOVER};
}}

QLabel#mode_tile_title {{
    font-size:           14px;
    font-weight:         700;
    color:               {TXT};
    background:          transparent;
    border:              none;
}}
QLabel#mode_tile_subtitle {{
    font-size:           11px;
    color:               {TXT_MUTED};
    background:          transparent;
    border:              none;
}}

/* ── Action tiles (Home / landing tab) ──────────────────────────────────── */
QFrame#action_tile {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       10px;
}}
QFrame#action_tile:hover {{
    background-color:    {PANEL_ALT};
    border:              1px solid {ACC};
}}
QLabel#action_tile_icon {{
    font-size:           28px;
    color:               {ACC};
    background:          transparent;
    border:              none;
}}
QLabel#action_tile_title {{
    font-size:           18px;
    font-weight:         700;
    color:               {TXT};
    background:          transparent;
    border:              none;
}}
QLabel#action_tile_desc {{
    font-size:           12px;
    color:               {TXT_MUTED};
    background:          transparent;
    border:              none;
}}

/* ── Results panel ──────────────────────────────────────────────────────── */
QFrame#resource_monitor {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       4px;
}}

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
