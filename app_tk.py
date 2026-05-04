#!/usr/bin/env python3
"""
sptPALM Analysis Pipeline — Zeiss Elyra — By Jacob Levers
Tkinter GUI | Cross-platform, PyInstaller-ready
Run with:  python app_tk.py
"""
import os
import sys

# ══════════════════════════════════════════════════════════════════════════════
#  FIRST-TIME SETUP BOOTSTRAP
#  Runs before any non-stdlib import.  If required packages are missing it
#  shows a dark-themed setup window, creates a venv, installs everything,
#  then re-execs this script with the venv Python so the app starts normally.
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap():
    _REQUIRED = [
        ("numpy",      "numpy"),
        ("pandas",     "pandas"),
        ("scipy",      "scipy"),
        ("matplotlib", "matplotlib"),
        ("PIL",        "Pillow"),
        ("skimage",    "scikit-image"),
        ("trackpy",    "trackpy"),
        ("tifffile",   "tifffile"),
        ("czifile",    "czifile"),
        ("joblib",     "joblib"),
        ("tqdm",       "tqdm"),
    ]
    _INSTALL = [
        "numpy", "pandas", "scipy", "matplotlib", "Pillow",
        "scikit-image", "trackpy", "tifffile", "czifile", "imagecodecs",
        "aicspylibczi", "joblib", "tqdm", "imageio", "psutil",
    ]

    missing = []
    for _mod, _pkg in _REQUIRED:
        try:
            __import__(_mod)
        except ImportError:
            missing.append(_pkg)
    if not missing:
        return  # All packages present — proceed normally

    # ── packages are missing: show setup UI ───────────────────────────────────
    import tkinter as tk
    from tkinter import ttk
    import subprocess
    import threading

    _FOLDER = os.path.dirname(os.path.abspath(__file__))
    _VENV   = os.path.join(_FOLDER, "sptpalm-env")
    _VENV_PY = (os.path.join(_VENV, "Scripts", "python.exe")
                if sys.platform == "win32"
                else os.path.join(_VENV, "bin", "python3"))
    _PIP     = (os.path.join(_VENV, "Scripts", "pip.exe")
                if sys.platform == "win32"
                else os.path.join(_VENV, "bin", "pip"))

    # Prefer Python 3.12 for the venv — it has pre-built wheels for all
    # scientific packages (aicspylibczi, scipy, etc.) and avoids compilation
    # errors seen on Python 3.13+ with newer compilers.
    import shutil as _shutil
    _PREFERRED_PY = (
        _shutil.which("python3.12") or
        _shutil.which("python3.13") or
        sys.executable
    )

    # Palette (duplicated here so bootstrap needs no app-level globals)
    _BG     = "#09090e"
    _CARD   = "#181d27"
    _BORDER = "#252d3a"
    _TXT    = "#dce6f0"
    _MUTED  = "#7a8899"
    _ACC    = "#4ea8ff"
    _GREEN  = "#2ea043"
    _RED    = "#f85149"
    _FF     = "Helvetica Neue" if sys.platform == "darwin" else (
              "Segoe UI"       if sys.platform == "win32"  else "Helvetica")
    _FM     = "Menlo"    if sys.platform == "darwin" else (
              "Consolas" if sys.platform == "win32"  else "DejaVu Sans Mono")

    root = tk.Tk()
    root.title("sptPALM — First-time Setup")
    root.configure(bg=_BG)
    root.resizable(False, False)
    W, H = 520, 400
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")

    tk.Label(root, text="sptPALM", bg=_BG, fg=_ACC,
             font=(_FF, 22, "bold")).pack(pady=(28, 2))
    tk.Label(root, text="First-time setup  —  installing required libraries",
             bg=_BG, fg=_MUTED, font=(_FF, 11)).pack(pady=(0, 14))

    log_outer = tk.Frame(root, bg=_BORDER)
    log_outer.pack(fill="both", expand=True, padx=24, pady=(0, 10))
    log = tk.Text(log_outer, bg=_CARD, fg=_TXT, font=(_FM, 10),
                  relief="flat", bd=0, state="disabled", wrap="word",
                  highlightthickness=0, padx=10, pady=8)
    sb = tk.Scrollbar(log_outer, command=log.yview)
    log.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    log.pack(fill="both", expand=True, padx=1, pady=1)

    log.tag_configure("acc",   foreground=_ACC)
    log.tag_configure("green", foreground=_GREEN)
    log.tag_configure("red",   foreground=_RED)
    log.tag_configure("muted", foreground=_MUTED)

    status_var = tk.StringVar(value="Starting…")
    tk.Label(root, textvariable=status_var, bg=_BG, fg=_MUTED,
             font=(_FF, 10)).pack()

    sty = ttk.Style(root)
    sty.theme_use("clam")
    sty.configure("S.Horizontal.TProgressbar",
                  troughcolor=_CARD, background=_ACC,
                  borderwidth=0, thickness=4)
    pbar = ttk.Progressbar(root, style="S.Horizontal.TProgressbar",
                           mode="indeterminate", length=472)
    pbar.pack(pady=(4, 20))
    pbar.start(12)

    def _log(text, tag=None):
        root.after(0, lambda: _log_now(text, tag))

    def _log_now(text, tag):
        log.configure(state="normal")
        log.insert("end", text + "\n", tag or "")
        log.configure(state="disabled")
        log.see("end")

    def _set_status(msg):
        root.after(0, lambda: status_var.set(msg))

    def _run():
        try:
            _set_status("Creating virtual environment…")
            _log(f"Creating virtual environment (Python {_PREFERRED_PY})…", "muted")
            subprocess.run(
                [_PREFERRED_PY, "-m", "venv", _VENV],
                check=True, capture_output=True)
            _log("  ✓ Virtual environment ready", "acc")

            _set_status("Upgrading pip…")
            subprocess.run(
                [_PIP, "install", "--upgrade", "pip", "--quiet"],
                check=True, capture_output=True)

            _log(f"\nInstalling {len(_INSTALL)} packages  "
                 f"(3–5 minutes on first run)…", "muted")

            for pkg in _INSTALL:
                _set_status(f"Installing {pkg}…")
                _log(f"  {pkg}…")
                r = subprocess.run(
                    [_PIP, "install", pkg, "--quiet"],
                    capture_output=True, text=True)
                if r.returncode != 0:
                    _log(f"  ✗ {pkg} — {r.stderr.strip()}", "red")
                else:
                    _log(f"  ✓ {pkg}", "acc")

            _log("\nAll done!  Launching sptPALM…", "green")
            _set_status("Done — launching…")
            root.after(0, lambda: pbar.configure(mode="determinate",
                                                  value=100))
            root.after(900, lambda: _relaunch())

        except Exception as exc:
            _log(f"\nSetup failed: {exc}", "red")
            _set_status("Setup failed — see log above")
            root.after(0, pbar.stop)

    def _relaunch():
        root.destroy()
        if sys.platform == "win32":
            subprocess.Popen([_VENV_PY, os.path.abspath(__file__)])
            sys.exit(0)
        else:
            os.execv(_VENV_PY, [_VENV_PY, os.path.abspath(__file__)])

    threading.Thread(target=_run, daemon=True).start()
    root.mainloop()
    sys.exit(0)


# sys.frozen is True inside a PyInstaller bundle — all packages are already
# embedded so the setup UI is never needed.
if not getattr(sys, "frozen", False):
    _bootstrap()

# ══════════════════════════════════════════════════════════════════════════════
#  All required packages are present — continue with normal startup
# ══════════════════════════════════════════════════════════════════════════════

import time
import threading
import queue
import traceback
import multiprocessing
import subprocess

# Cap BLAS/OpenBLAS/MKL internal threads to 1 before numpy is imported.
# Without this, each joblib worker thread also spawns N_CPUS BLAS threads
# internally, causing N_CPUS² threads competing for N_CPUS cores — which
# collapses multi-core performance (most visible on Windows).
for _blas_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_blas_env, "1")

# Must be set before any other imports on macOS
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

N_CPUS = multiprocessing.cpu_count()

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#09090e"      # near-black canvas
SIDEBAR = "#111318"      # left panel — visibly distinct from BG
CARD    = "#181d27"      # parameter card face
BORDER  = "#252d3a"      # visible dividers
TXT     = "#dce6f0"
MUTED   = "#7a8899"
ACC     = "#4ea8ff"
ACC2    = "#79c0ff"      # hover/lighter accent
GREEN   = "#1f7a3a"
GREEN2  = "#2ea043"      # button face
RED     = "#f85149"

# Keep PNL/PNL2 as aliases so old references still work
PNL  = SIDEBAR
PNL2 = CARD
MOTION_COLORS = {
    "Immobile": "#e05252", "Confined": "#f5a623",
    "Brownian": "#4a90d9", "Directed": "#7ed321",
}

# ── System fonts (closest to native on each OS) ────────────────────────────────
if sys.platform == "darwin":
    FONT_FAM, FONT_MONO = "Helvetica Neue", "Menlo"
elif sys.platform == "win32":
    FONT_FAM, FONT_MONO = "Segoe UI", "Consolas"
else:
    FONT_FAM, FONT_MONO = "Helvetica", "DejaVu Sans Mono"

def F(size=10, weight="normal"):
    return (FONT_FAM, size, weight)
def FM(size=10, weight="normal"):
    return (FONT_MONO, size, weight)


# ══════════════════════════════════════════════════════════════════════════════
#  ICON  (generated with PIL — no external image file needed)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_icon(size: int = 256):
    """
    App icon: a palm tree made of glowing particle traces on a dark circle.
    Trunk and fronds are drawn as localisation dots connected by trajectory
    lines, matching the UI colour palette.
    Returns a PIL Image, or None if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import math
    except ImportError:
        return None

    s  = size
    m  = s // 18
    AC = (88, 166, 255)       # accent blue
    BR = (210, 230, 255)      # bright white-blue for dots

    def p(fx, fy):
        return (int(fx * s), int(fy * s))

    def bezier(p0, p1, p2, n=7):
        """Sample n points along a quadratic bezier curve."""
        pts = []
        for i in range(n):
            t  = i / (n - 1)
            u  = 1 - t
            x  = u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0]
            y  = u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1]
            pts.append((int(x), int(y)))
        return pts

    # Trunk: gentle S-curve from base to crown
    trunk = [
        p(0.500, 0.900),
        p(0.498, 0.820),
        p(0.496, 0.740),
        p(0.497, 0.660),
        p(0.500, 0.580),
        p(0.504, 0.510),
        p(0.508, 0.450),
        p(0.510, 0.390),   # crown
    ]
    crown = trunk[-1]

    # Fronds: quadratic bezier (start=crown, control, tip)
    # each tuple is (control_frac, tip_frac)
    frond_defs = [
        (p(0.36, 0.29), p(0.14, 0.33)),   # far left, drooping
        (p(0.40, 0.22), p(0.27, 0.16)),   # mid left, upswept
        (p(0.50, 0.20), p(0.50, 0.10)),   # straight up centre
        (p(0.61, 0.22), p(0.74, 0.16)),   # mid right, upswept
        (p(0.65, 0.29), p(0.87, 0.33)),   # far right, drooping
    ]
    fronds = [bezier(crown, ctrl, tip, n=7) for ctrl, tip in frond_defs]

    # Collect all particle positions
    all_pts = list(trunk)
    for frond in fronds:
        all_pts.extend(frond[1:])  # crown already in trunk

    # ── Base circle ──────────────────────────────────────────────────────────
    base = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d    = ImageDraw.Draw(base)
    d.ellipse([m, m, s - m, s - m], fill=(13, 17, 23, 255))

    # ── Glow layer ───────────────────────────────────────────────────────────
    glow   = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    g      = ImageDraw.Draw(glow)
    spot_r = max(4, s // 30)
    for x, y in all_pts:
        for gr in range(spot_r * 4, 0, -1):
            a = int(110 * (1 - gr / (spot_r * 4)) ** 2.0)
            g.ellipse([x - gr, y - gr, x + gr, y + gr], fill=(*AC, a))
    glow_blur = glow.filter(ImageFilter.GaussianBlur(radius=max(2, s // 36)))

    # ── Sharp layer ──────────────────────────────────────────────────────────
    sharp = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sh    = ImageDraw.Draw(sharp)

    tlw = max(2, s // 56)   # trunk line width
    flw = max(1, s // 80)   # frond line width

    # Trunk lines (slightly brighter, thicker)
    for i in range(len(trunk) - 1):
        sh.line([trunk[i], trunk[i + 1]], fill=(*AC, 200), width=tlw)

    # Frond lines (fade toward tip)
    for frond in fronds:
        n = len(frond)
        for i in range(n - 1):
            a = int(220 - 120 * (i / (n - 1)))
            sh.line([frond[i], frond[i + 1]], fill=(*AC, a), width=flw)

    # Particle dots
    cr = max(2, s // 56)
    for x, y in all_pts:
        sh.ellipse([x - cr, y - cr, x + cr, y + cr], fill=(*BR, 255))

    # Border ring
    bw = max(2, s // 70)
    sh.ellipse([m, m, s - m, s - m], outline=(*AC, 210), width=bw)

    # ── Composite ────────────────────────────────────────────────────────────
    final = base.copy()
    final = Image.alpha_composite(final, glow_blur)
    final = Image.alpha_composite(final, sharp)
    return final


# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════

def _apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    # Base defaults — everything inherits from here
    style.configure(".",
        background=BG, foreground=TXT,
        fieldbackground=CARD, troughcolor=CARD,
        bordercolor=BORDER, selectbackground=ACC,
        selectforeground="white",
        relief="flat", font=F(12))

    # Frames
    style.configure("TFrame",        background=BG)
    style.configure("Panel.TFrame",  background=SIDEBAR)
    style.configure("Card.TFrame",   background=CARD)

    # Labels
    style.configure("TLabel",        background=BG,      foreground=TXT,   font=F(12))
    style.configure("Panel.TLabel",  background=SIDEBAR, foreground=TXT,   font=F(12))
    style.configure("Muted.TLabel",  background=BG,      foreground=MUTED, font=F(11))
    style.configure("PMuted.TLabel", background=SIDEBAR, foreground=MUTED, font=F(11))
    # Big title in header
    style.configure("Acc.TLabel",    background=BG,      foreground=ACC,   font=F(20, "bold"))
    style.configure("Section.TLabel",background=CARD,    foreground=ACC,   font=F(12, "bold"))

    style.configure("TLabelframe",
        background=CARD, foreground=MUTED, bordercolor=BORDER, relief="solid")
    style.configure("TLabelframe.Label",
        background=CARD, foreground=ACC, font=F(12, "bold"))

    # Input widgets — sit on CARD background, bright insert caret
    style.configure("TEntry",
        fieldbackground=CARD, foreground=TXT, insertcolor=ACC,
        bordercolor=BORDER, padding=(6, 5))
    style.configure("TSpinbox",
        fieldbackground=CARD, foreground=TXT, insertcolor=ACC,
        arrowcolor=ACC2, bordercolor=BORDER, padding=(6, 4))
    style.configure("TCombobox",
        fieldbackground=CARD, foreground=TXT,
        arrowcolor=ACC2, bordercolor=BORDER,
        lightcolor=BORDER, darkcolor=BORDER,
        borderwidth=1, padding=(6, 4))
    style.map("TCombobox",
        fieldbackground=[("readonly", CARD), ("disabled", SIDEBAR)],
        foreground=[("disabled", MUTED)],
        selectbackground=[("readonly", CARD)],
        selectforeground=[("readonly", TXT)])
    style.map("TSpinbox",
        fieldbackground=[("disabled", SIDEBAR)],
        foreground=[("disabled", MUTED)])

    style.configure("TCheckbutton",
        background=CARD, foreground=TXT,
        indicatorcolor=SIDEBAR, font=F(12))
    style.map("TCheckbutton",
        background=[("active", CARD)],
        indicatorcolor=[("selected", ACC)])
    style.configure("Card.TCheckbutton",
        background=CARD, foreground=TXT,
        indicatorcolor=SIDEBAR, font=F(12))
    style.map("Card.TCheckbutton",
        background=[("active", CARD)],
        indicatorcolor=[("selected", ACC)])

    # Buttons
    style.configure("Run.TButton",
        background=GREEN2, foreground="white",
        padding=(24, 12), font=F(14, "bold"), borderwidth=0)
    style.map("Run.TButton",
        background=[("active", "#3fb950"), ("disabled", BORDER)],
        foreground=[("disabled", MUTED)])

    style.configure("Stop.TButton",
        background="#a12d2d", foreground="white",
        padding=(24, 12), font=F(14, "bold"), borderwidth=0)
    style.map("Stop.TButton",
        background=[("active", "#c0392b"), ("disabled", BORDER)],
        foreground=[("disabled", MUTED)])

    style.configure("TButton",
        background=CARD, foreground=TXT,
        padding=(12, 7), font=F(11), borderwidth=0)
    style.map("TButton",
        background=[("active", BORDER)],
        foreground=[("active", ACC2)])

    # Tabs — tabmargins=[0,0,0,0] prevents the selected tab from "popping up",
    # which is what causes the unequal height in the clam theme.
    style.configure("TNotebook",
        background=BG, bordercolor=BORDER, tabmargins=[0, 0, 0, 0])
    style.configure("TNotebook.Tab",
        background=CARD, foreground=MUTED,
        padding=[20, 10], font=F(11, "bold"), borderwidth=0,
        focuscolor=BG)
    style.map("TNotebook.Tab",
        background=[("selected", BG), ("active", BORDER)],
        foreground=[("selected", ACC), ("active", TXT)],
        padding=[("selected", [20, 10])])

    # Progress bar — taller, more visible
    style.configure("Horizontal.TProgressbar",
        background=ACC, troughcolor=CARD,
        bordercolor=BORDER, lightcolor=ACC, darkcolor=ACC, thickness=8)

    style.configure("TSeparator", background=BORDER)
    # Minimal scrollbar: blue thumb, no visible arrows
    style.configure("Vertical.TScrollbar",
        background=ACC, troughcolor=SIDEBAR,
        bordercolor=SIDEBAR, arrowcolor=SIDEBAR,
        gripcount=0, arrowsize=1, width=5)
    style.map("Vertical.TScrollbar",
        background=[("active", ACC2), ("disabled", SIDEBAR)])


# ══════════════════════════════════════════════════════════════════════════════
#  SCROLLABLE FRAME HELPER
# ══════════════════════════════════════════════════════════════════════════════

class _ScrollFrame(ttk.Frame):
    """
    Vertical-scroll frame. Place children in self.inner.

    Each instance registers itself in _ScrollFrame._instances so the global
    scroll dispatcher can find it by screen-geometry hit-testing.  Widget
    hierarchy walking (via .master) is NOT used because widgets embedded in a
    tk.Canvas via create_window are not reliably found by winfo_containing.
    """
    _instances: list = []   # all live _ScrollFrame objects

    @classmethod
    def find_under(cls, x_root: int, y_root: int):
        """Return the _ScrollFrame whose screen rect contains (x_root, y_root)."""
        for sf in cls._instances:
            try:
                rx = sf.winfo_rootx()
                ry = sf.winfo_rooty()
                if (rx <= x_root <= rx + sf.winfo_width() and
                        ry <= y_root <= ry + sf.winfo_height()):
                    return sf
            except Exception:
                pass
        return None

    def __init__(self, parent, bg=None, **kw):
        super().__init__(parent, **kw)
        _ScrollFrame._instances.append(self)
        _bg = bg or SIDEBAR
        self._canvas = tk.Canvas(self, bg=_bg, bd=0, highlightthickness=0)
        # Scrollbar is created (needed for yscrollcommand) but NOT packed — scrolling
        # is gesture-only via the global dispatcher installed on SPTPalmApp.
        self._sb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview,
                                bg=ACC, troughcolor=SIDEBAR, activebackground=ACC2,
                                relief="flat", bd=0, highlightthickness=0, width=14)
        self.inner = tk.Frame(self._canvas, bg=_bg)

        self._win_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
            lambda e: self._canvas.itemconfig(self._win_id, width=e.width))
        self._canvas.configure(yscrollcommand=self._sb.set)

        # Minimal scrollbar on right edge, then canvas fills the rest
        self._sb.pack(side="right", fill="y")
        self._canvas.pack(fill="both", expand=True)

    def do_scroll(self, e):
        """Platform-aware scroll. macOS trackpad delta is ~1–20, not 120-unit ticks."""
        if sys.platform == "darwin":
            self._canvas.yview_scroll(int(-1 * e.delta), "units")
        else:
            self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def scroll_up(self):   self._canvas.yview_scroll(-1, "units")
    def scroll_down(self): self._canvas.yview_scroll(1, "units")


# ══════════════════════════════════════════════════════════════════════════════
#  FLAT BUTTON  (Label-based — fully themeable on macOS)
# ══════════════════════════════════════════════════════════════════════════════

class _FlatButton(tk.Frame):
    """
    Fully themeable flat button for macOS and all platforms.

    Structure: tk.Frame (1 px BORDER outline) → tk.Label (clickable face).
    The frame background shows through as a 1 px border — identical to how
    ttk draws combobox / entry borders in the dark theme.

    Colours mirror the ttk.Combobox style map:
      enabled  → face bg = CARD,    fg = TXT
      disabled → face bg = SIDEBAR, fg = MUTED
    Hover      → face bg = BORDER,  fg = ACC2

    Supports .configure(state="normal"/"disabled") and .configure(text=…).
    """
    _BG      = CARD
    _BG_HOV  = BORDER
    _BG_DIS  = SIDEBAR
    _FG      = TXT
    _FG_DIS  = MUTED

    def __init__(self, parent, text, command, **kw):
        self._cmd     = command
        self._enabled = True
        # Frame provides the 1 px border
        super().__init__(parent, bg=BORDER,
                         padx=1, pady=1,
                         highlightthickness=0, bd=0)
        # Inner label is the visible face
        self._lbl = tk.Label(self, text=text,
                             bg=self._BG, fg=self._FG,
                             font=F(11), padx=10, pady=5,
                             cursor="hand2", relief="flat",
                             highlightthickness=0)
        self._lbl.pack(fill="both", expand=True)
        for w in (self, self._lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>",    self._hover_on)
            w.bind("<Leave>",    self._hover_off)

    def _click(self, _e=None):
        if self._enabled:
            self._cmd()

    def _hover_on(self, _e=None):
        if self._enabled:
            self._lbl.configure(bg=self._BG_HOV, fg=ACC2)

    def _hover_off(self, _e=None):
        if self._enabled:
            self._lbl.configure(bg=self._BG, fg=self._FG)

    def configure(self, **kw):
        if "state" in kw:
            state = kw.pop("state")
            self._enabled = state != "disabled"
            if self._enabled:
                self._lbl.configure(bg=self._BG,     fg=self._FG,     cursor="hand2")
            else:
                self._lbl.configure(bg=self._BG_DIS, fg=self._FG_DIS, cursor="")
        if "text" in kw:
            self._lbl.configure(text=kw.pop("text"))
        if kw:
            super().configure(**kw)

    config = configure   # alias used by Tkinter internally


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLTIP
# ══════════════════════════════════════════════════════════════════════════════

class _Tooltip:
    """
    Dark-themed tooltip that appears after a short hover delay.
    Attach to any widget via _Tooltip(widget, text).
    """
    DELAY_MS  = 400
    MAX_WIDTH = 320   # wrap width in pixels

    def __init__(self, widget, text: str):
        self._widget  = widget
        self._text    = text
        self._tip_win = None
        self._after_id = None
        widget.bind("<Enter>",  self._on_enter,  add="+")
        widget.bind("<Leave>",  self._on_leave,  add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        self._cancel()
        self._after_id = self._widget.after(self.DELAY_MS, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._after_id:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip_win:
            return
        # Position: just below and right of the widget
        wx = self._widget.winfo_rootx()
        wy = self._widget.winfo_rooty() + self._widget.winfo_height() + 4

        win = tk.Toplevel(self._widget)
        win.wm_overrideredirect(True)
        win.wm_attributes("-topmost", True)
        # macOS: avoid shadow / rounded corners artefacts
        if sys.platform == "darwin":
            try:
                win.tk.call("::tk::unsupported::MacWindowStyle", "style",
                            win._w, "help", "noactivates")
            except Exception:
                pass

        # Outer border frame
        border = tk.Frame(win, bg=BORDER, bd=0)
        border.pack(fill="both", expand=True, padx=0, pady=0)

        lbl = tk.Label(
            border, text=self._text,
            bg="#1a2030", fg=TXT,
            font=F(10), justify="left",
            wraplength=self.MAX_WIDTH,
            padx=10, pady=8,
        )
        lbl.pack(fill="both", expand=True, padx=1, pady=1)

        win.update_idletasks()
        # Keep within screen width
        sw = win.winfo_screenwidth()
        tw = win.winfo_reqwidth()
        if wx + tw > sw - 8:
            wx = sw - tw - 8
        win.wm_geometry(f"+{wx}+{wy}")
        self._tip_win = win

    def _hide(self):
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None


# ══════════════════════════════════════════════════════════════════════════════
#  ROI EDITOR DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class _ROIEditorDialog(tk.Toplevel):
    """
    Dialog for drawing a freehand polygon ROI on the file's mean projection.
    Calls on_apply(mask) with a boolean numpy array when the user confirms.
    """
    def __init__(self, parent, fpath, channel, on_apply):
        super().__init__(parent)
        self.title("ROI Editor  —  Draw polygon")
        self.geometry("960x680")
        self.minsize(600, 440)
        self.configure(bg=BG)
        self.transient(parent)

        self._fpath    = fpath
        self._channel  = channel
        self._on_apply = on_apply
        self._points   = []       # polygon vertices in image coords [(x,y), …]
        self._closed   = False
        self._proj_rgb = None     # PIL RGB image (inferno-coloured projection)
        self._scale    = 1.0      # canvas px per image px
        self._offset   = (0, 0)   # (ox, oy): canvas origin of displayed image
        self._photo    = None     # ImageTk ref — must stay alive
        self._cursor   = None     # current canvas cursor pos for preview line

        self._build_ui()
        self._load_projection()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=SIDEBAR)
        hdr.pack(fill="x")
        tk.Label(hdr,
                 text="Left-click to add vertices  ·  Double-click to close  ·  Right-click to undo",
                 bg=SIDEBAR, fg=MUTED, font=F(10)).pack(side="left", padx=14, pady=8)

        self._canvas = tk.Canvas(self, bg=BG, bd=0, highlightthickness=0,
                                 cursor="crosshair")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>",        self._on_click)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Button-2>",        self._on_undo)   # macOS middle
        self._canvas.bind("<Button-3>",        self._on_undo)
        self._canvas.bind("<Motion>",          self._on_motion)
        self._canvas.bind("<Configure>",       lambda e: self._full_redraw())

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        btm = tk.Frame(self, bg=SIDEBAR)
        btm.pack(fill="x", side="bottom")
        btm.columnconfigure(1, weight=1)

        self._apply_btn = ttk.Button(btm, text="Apply ROI",
                                     style="Run.TButton", command=self._apply,
                                     state="disabled", width=14)
        self._apply_btn.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self._status_lbl = tk.Label(btm, text="Loading image…",
                                    bg=SIDEBAR, fg=MUTED, font=F(10))
        self._status_lbl.grid(row=0, column=1, padx=8, sticky="w")

        ttk.Button(btm, text="Clear",  command=self._clear ).grid(row=0, column=2, padx=(0,  6), pady=10)
        ttk.Button(btm, text="Cancel", command=self.destroy).grid(row=0, column=3, padx=(0, 14), pady=10)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_projection(self):
        def _worker():
            try:
                import sys as _sys
                base = (_sys._MEIPASS if getattr(_sys, "frozen", False)
                        else os.path.dirname(os.path.abspath(__file__)))
                if base not in _sys.path:
                    _sys.path.insert(0, base)
                from sptpalm_analysis import load_projection_fast
                # Reads only ~100 evenly-spaced frames — fast even for 16K files
                proj = load_projection_fast(self._fpath,
                                            channel=self._channel,
                                            max_frames=100)
                try:
                    import matplotlib.cm as _cm
                    from PIL import Image
                    rgb = (_cm.inferno(proj)[..., :3] * 255).astype("uint8")
                    self._proj_rgb = Image.fromarray(rgb)
                except Exception:
                    from PIL import Image
                    self._proj_rgb = Image.fromarray(
                        (proj * 255).astype("uint8")).convert("RGB")
                self.after(0, self._on_loaded)
            except Exception as exc:
                self.after(0, lambda: self._status_lbl.configure(
                    text=f"Error loading file: {exc}"))
        threading.Thread(target=_worker, daemon=True).start()

    def _on_loaded(self):
        self._status_lbl.configure(
            text="Click to place vertices  ·  Double-click to close polygon")
        self._full_redraw()

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _compute_transform(self):
        if self._proj_rgb is None:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih       = self._proj_rgb.size
        self._scale  = min(cw / iw, ch / ih)
        dw           = int(iw * self._scale)
        dh           = int(ih * self._scale)
        self._offset = ((cw - dw) // 2, (ch - dh) // 2)

    def _img_to_cv(self, x, y):
        ox, oy = self._offset
        return ox + x * self._scale, oy + y * self._scale

    def _cv_to_img(self, cx, cy):
        ox, oy = self._offset
        return (cx - ox) / self._scale, (cy - oy) / self._scale

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _full_redraw(self):
        """Re-render PIL base image then overlay polygon canvas items."""
        self._canvas.delete("all")
        if self._proj_rgb is None:
            w = max(self._canvas.winfo_width(), 1)
            h = max(self._canvas.winfo_height(), 1)
            self._canvas.create_text(w // 2, h // 2, text="Loading…",
                                     fill=MUTED, font=F(14))
            return

        self._compute_transform()
        from PIL import Image, ImageDraw, ImageTk
        iw, ih = self._proj_rgb.size
        dw = int(iw * self._scale)
        dh = int(ih * self._scale)
        display = self._proj_rgb.resize((dw, dh), Image.LANCZOS)

        # Bake filled polygon overlay into base image when closed
        if self._closed and len(self._points) >= 3:
            local_pts = [
                (int(cx - self._offset[0]), int(cy - self._offset[1]))
                for cx, cy in (self._img_to_cv(x, y) for x, y in self._points)
            ]
            ov = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
            ImageDraw.Draw(ov).polygon(local_pts,
                                       fill=(78, 168, 255, 65),
                                       outline=(78, 168, 255, 220))
            display = Image.alpha_composite(display.convert("RGBA"), ov).convert("RGB")

        self._photo = ImageTk.PhotoImage(display)
        ox, oy = self._offset
        self._canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        self._draw_poly_items()

    def _draw_poly_items(self):
        """
        Delete and redraw only the 'poly' tagged items (edges, dots, cursor line).
        Called on mouse motion — no PIL re-render needed.
        """
        self._canvas.delete("poly")
        if not self._points:
            return
        cv   = [self._img_to_cv(x, y) for x, y in self._points]
        dash = () if self._closed else (6, 3)

        # Edges
        for i in range(len(cv) - 1):
            self._canvas.create_line(*cv[i], *cv[i + 1],
                                     fill=ACC, width=2, dash=dash, tags="poly")
        if self._closed:
            self._canvas.create_line(*cv[-1], *cv[0], fill=ACC, width=2, tags="poly")

        # Cursor preview line while polygon is open
        if not self._closed and self._cursor and len(self._points) >= 1:
            self._canvas.create_line(*cv[-1], *self._cursor,
                                     fill=ACC2, width=1, dash=(4, 4), tags="poly")

        # Vertex dots
        r = 5
        for i, (cx, cy) in enumerate(cv):
            col = ACC2 if i == 0 else ACC
            self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill=col, outline="white", width=1, tags="poly")

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_click(self, e):
        if self._proj_rgb is None or self._closed:
            return
        ix, iy = self._cv_to_img(e.x, e.y)
        iw, ih = self._proj_rgb.size
        if 0 <= ix <= iw and 0 <= iy <= ih:
            self._points.append((ix, iy))
            self._full_redraw()

    def _on_double_click(self, e):
        if self._proj_rgb is None or self._closed or len(self._points) < 3:
            return
        # Tk fires single-click before double-click — pop the duplicate point
        if len(self._points) > 3:
            self._points.pop()
        self._closed = True
        self._cursor = None
        self._apply_btn.configure(state="normal")
        self._status_lbl.configure(
            text=f"Polygon closed — {len(self._points)} vertices.  "
                 f"Click Apply ROI to confirm.")
        self._full_redraw()

    def _on_undo(self, e):
        if self._closed:
            return
        if self._points:
            self._points.pop()
            self._full_redraw()

    def _on_motion(self, e):
        if self._closed or self._proj_rgb is None or not self._points:
            return
        self._cursor = (e.x, e.y)
        self._draw_poly_items()    # fast: only updates 'poly' items, no PIL

    def _clear(self):
        self._points = []
        self._closed = False
        self._cursor = None
        self._apply_btn.configure(state="disabled")
        self._status_lbl.configure(
            text="Click to place vertices  ·  Double-click to close polygon")
        self._full_redraw()

    def _apply(self):
        if len(self._points) < 3 or not self._closed:
            return
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            iw, ih = self._proj_rgb.size
            mask_img = Image.new("L", (iw, ih), 0)
            ImageDraw.Draw(mask_img).polygon(
                [(int(x), int(y)) for x, y in self._points], fill=255)
            self._on_apply(np.array(mask_img) > 0)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Error",
                                 f"Could not create ROI mask:\n{exc}", parent=self)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class SPTPalmApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("sptPALM Analysis Pipeline — Zeiss Elyra — By Jacob Levers")
        self.geometry("1280x820")
        self.minsize(960, 640)
        self.configure(bg=BG)

        _apply_theme(self)
        self._q: queue.Queue = queue.Queue()
        self._running    = False
        self._stop_event = threading.Event()
        self._img_refs: list = []
        self._preview_canvas      = None   # set when progress panel is shown
        self._preview_fig         = None
        self._preview_ax          = None
        self._preview_image_array = None   # last RGB frame for resize redraws
        self._flash_id            = None   # after() id for dot flash
        self._elapsed_id          = None   # after() id for elapsed counter
        self._init_vars()
        self._build_ui()
        self._load_settings()   # overwrite defaults with any saved preferences
        self._install_scroll_dispatcher()
        self._set_icon()

    # ── Global scroll dispatcher ──────────────────────────────────────────────
    #
    # Why this exists:
    #   On macOS, ttk Spinbox/Combobox have class-level <MouseWheel> handlers
    #   that change their values on scroll. Per-instance bindings that return
    #   "break" do not reliably block the class handler. The only thing that
    #   does is REPLACING the class binding itself via bind_class().
    #
    # How it works:
    #   _ScrollFrame.find_under(x_root, y_root) finds the scroll panel under
    #   the cursor using screen-geometry bounding-box checks — much more
    #   reliable than winfo_containing + .master walk-up, which breaks for
    #   widgets embedded inside a tk.Canvas via create_window.

    def _install_scroll_dispatcher(self):

        def _route_wheel(e):
            sf = _ScrollFrame.find_under(e.x_root, e.y_root)
            if sf is not None:
                sf.do_scroll(e)

        def _route_up(e):
            sf = _ScrollFrame.find_under(e.x_root, e.y_root)
            if sf is not None:
                sf.scroll_up()

        def _route_down(e):
            sf = _ScrollFrame.find_under(e.x_root, e.y_root)
            if sf is not None:
                sf.scroll_down()

        # Spinbox / Combobox: replace the class binding so their value never
        # changes on scroll. Return "break" to block the default handler.
        def _spin_wheel(e):
            _route_wheel(e)
            return "break"

        def _spin_up(e):
            _route_up(e)
            return "break"

        def _spin_down(e):
            _route_down(e)
            return "break"

        for cls in ("TSpinbox", "Spinbox", "TCombobox", "Combobox"):
            self.bind_class(cls, "<MouseWheel>",       _spin_wheel)
            self.bind_class(cls, "<Shift-MouseWheel>", _spin_wheel)
            self.bind_class(cls, "<Button-4>",         _spin_up)
            self.bind_class(cls, "<Button-5>",         _spin_down)

        # Catch scroll anywhere else in the app
        self.bind_all("<MouseWheel>",       _route_wheel)
        self.bind_all("<Shift-MouseWheel>", _route_wheel)
        self.bind_all("<Button-4>",         _route_up)
        self.bind_all("<Button-5>",         _route_down)

    # ── Variable initialisation ───────────────────────────────────────────────

    def _init_vars(self):
        # File paths
        self.v_file   = tk.StringVar()
        self.v_outdir = tk.StringVar()
        self._drawn_roi_mask = None   # numpy bool array set by ROI editor

        # Acquisition
        # Defaults are tuned for Drosophila neurons on Zeiss Elyra 7.
        # CZI files override pixel_size and frame_interval automatically.
        self.v_pixel_size     = tk.DoubleVar(value=0.104)
        self.v_frame_interval = tk.DoubleVar(value=0.020)
        self.v_override_px    = tk.BooleanVar(value=False)
        self.v_override_fi    = tk.BooleanVar(value=False)
        self.v_channel        = tk.IntVar(value=0)

        # Preprocessing
        self.v_bg_method = tk.StringVar(value="Uniform Filter")
        self.v_bg_radius = tk.IntVar(value=10)   # small for thin neurites

        # Localisation
        self.v_diameter = tk.IntVar(value=7)
        self.v_auto_mm  = tk.BooleanVar(value=True)
        self.v_minmass  = tk.DoubleVar(value=0.3)

        # Linking
        self.v_search_range  = tk.IntVar(value=5)
        self.v_memory        = tk.IntVar(value=3)
        self.v_min_track_len = tk.IntVar(value=8)
        self.v_max_track_len = tk.IntVar(value=0)   # 0 = disabled

        # MSD & diffusion
        self.v_max_lagtime = tk.IntVar(value=20)
        self.v_n_fit       = tk.IntVar(value=5)

        # ROI — auto-threshold with Li algorithm by default (best for neurites)
        self.v_roi_mode        = tk.StringVar(value="Auto threshold")
        self.v_roi_auto_method = tk.StringVar(value="Li")
        self.v_roi_threshold   = tk.DoubleVar(value=0.08)
        self.v_roi_mask_mode   = tk.StringVar(value="Mean")

        # Drift correction — enabled by default (recommended for neurite data)
        self.v_drift_correct = tk.BooleanVar(value=True)
        self.v_drift_segment = tk.IntVar(value=500)   # sparse labelling

        # Performance
        self.v_workers    = tk.IntVar(value=N_CPUS)
        self.v_chunk_size = tk.IntVar(value=500)

        # Figure style
        self.v_fig_theme      = tk.StringVar(value="Dark")
        self.v_proj_cmap      = tk.StringVar(value="Inferno")
        self.v_jdd_components = tk.IntVar(value=2)

        # Track filtering by D (applied after MSD fitting)
        self.v_filter_d_enabled = tk.BooleanVar(value=False)
        self.v_filter_d_min     = tk.DoubleVar(value=0.0)
        self.v_filter_d_max     = tk.DoubleVar(value=1.0)

        # Cluster Analysis (DBSCAN)
        self.v_cluster_eps_nm      = tk.DoubleVar(value=50.0)
        self.v_cluster_min_samples = tk.IntVar(value=10)

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        # Header — taller, two-tone strip
        hdr = tk.Frame(self, bg=SIDEBAR, pady=0)
        hdr.pack(fill="x")
        inner_hdr = tk.Frame(hdr, bg=SIDEBAR)
        inner_hdr.pack(fill="x", padx=18, pady=12)
        tk.Label(inner_hdr, text="sptPALM",
                 bg=SIDEBAR, fg=ACC, font=F(22, "bold")).pack(side="left")
        tk.Label(inner_hdr, text=" Analysis Pipeline",
                 bg=SIDEBAR, fg=TXT, font=F(18)).pack(side="left")
        tk.Label(inner_hdr, text="Zeiss Elyra  ·  Jacob Levers",
                 bg=SIDEBAR, fg=MUTED, font=F(11)).pack(side="right")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")  # separator

        # File bar
        fb = tk.Frame(self, bg=SIDEBAR)
        fb.pack(fill="x")
        self._build_file_bar(fb)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")  # separator

        # Main content area — left panel + right panel side by side
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        # Left: fixed-width scrollable parameters
        self._left_panel = tk.Frame(content, bg=SIDEBAR, width=340)
        self._left_panel.pack(side="left", fill="y")
        self._left_panel.pack_propagate(False)

        # Collapse toggle strip (sits between left panel and right area)
        self._toggle_strip = tk.Frame(content, bg=BORDER, width=16, cursor="hand2")
        self._toggle_strip.pack(side="left", fill="y")
        self._toggle_strip.pack_propagate(False)
        self._toggle_lbl = tk.Label(self._toggle_strip, text="◀", bg=BORDER,
                                    fg=MUTED, font=F(9), cursor="hand2")
        self._toggle_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self._toggle_strip.bind("<Button-1>", lambda e: self._toggle_params())
        self._toggle_lbl.bind("<Button-1>",  lambda e: self._toggle_params())
        self._panel_visible = True

        self._scroll_frame = _ScrollFrame(self._left_panel, bg=SIDEBAR)
        self._scroll_frame.pack(fill="both", expand=True)
        self._build_params(self._scroll_frame.inner)

        # Vertical separator
        tk.Frame(content, bg=BORDER, width=1).pack(side="left", fill="y")

        # Right: results area
        self._right = tk.Frame(content, bg=BG)
        self._right.pack(side="left", fill="both", expand=True)
        self._show_placeholder()

        # Bottom bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        btm = tk.Frame(self, bg=SIDEBAR)
        btm.pack(fill="x", side="bottom")
        self._build_bottom_bar(btm)

    def _build_file_bar(self, parent):
        pad = dict(padx=6, pady=11)
        tk.Label(parent, text="Input file", bg=SIDEBAR, fg=MUTED, font=F(11)).pack(
            side="left", padx=(14, 6), pady=11)
        tk.Entry(parent, textvariable=self.v_file, width=55,
                 bg=CARD, fg=TXT, insertbackground=ACC,
                 relief="flat", bd=0, font=F(11),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACC).pack(side="left", **pad)
        ttk.Button(parent, text="Browse",
                   command=self._browse_file).pack(side="left", padx=(0, 20))

        tk.Label(parent, text="Output folder", bg=SIDEBAR, fg=MUTED, font=F(11)).pack(
            side="left", padx=(0, 6), pady=11)
        tk.Entry(parent, textvariable=self.v_outdir, width=30,
                 bg=CARD, fg=TXT, insertbackground=ACC,
                 relief="flat", bd=0, font=F(11),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACC).pack(side="left", **pad)
        ttk.Button(parent, text="Browse",
                   command=self._browse_outdir).pack(side="left", padx=(0, 14))
        self._batch_btn = ttk.Button(parent, text="Batch",
                                     command=self._on_batch, width=7)
        self._batch_btn.pack(side="left", padx=(0, 14))

    def _build_bottom_bar(self, parent):
        # Grid layout so the button is always in column 0 and cannot be pushed off
        parent.columnconfigure(1, weight=1)

        self._run_btn = ttk.Button(parent, text="▶  Run Analysis",
                                   style="Run.TButton", command=self._on_run,
                                   width=16)
        self._run_btn.grid(row=0, column=0, padx=(18, 6), pady=14, sticky="w")

        self._status_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._status_var,
                 bg=SIDEBAR, fg=MUTED, font=F(11)).grid(
            row=0, column=1, padx=(0, 12), sticky="w")

    def _on_stop(self):
        """Signal the worker to stop. Button stays in place, text changes."""
        self._stop_event.set()
        self._run_btn.configure(state="disabled", text="■  Stopping…",
                                style="Stop.TButton")

    def _toggle_params(self):
        if self._panel_visible:
            self._collapse_panel()
        else:
            self._expand_panel()

    def _collapse_panel(self, callback=None):
        if not self._panel_visible:
            if callback: callback()
            return
        self._left_panel.pack_forget()
        self._panel_visible = False
        self._toggle_lbl.configure(text="▶")
        if callback: callback()

    def _expand_panel(self, callback=None):
        if self._panel_visible:
            if callback: callback()
            return
        DURATION = 0.22
        PANEL_W  = 340
        self._left_panel.configure(width=1)
        self._left_panel.pack(side="left", fill="y", before=self._toggle_strip)
        t0 = time.monotonic()

        def _step():
            progress = min((time.monotonic() - t0) / DURATION, 1.0)
            ease = progress * progress * (3.0 - 2.0 * progress)
            if progress < 1.0:
                self._left_panel.configure(width=max(1, int(PANEL_W * ease)))
                self.after(4, _step)
            else:
                self._left_panel.configure(width=PANEL_W)
                self._panel_visible = True
                self._toggle_lbl.configure(text="◀")
                if callback: callback()

        _step()

    # ── Parameter panel ───────────────────────────────────────────────────────

    def _section(self, parent, title: str) -> tk.Frame:
        """Card with a 3-px blue left strip + title bar + parameter rows."""
        wrapper = tk.Frame(parent, bg=SIDEBAR)
        wrapper.pack(fill="x", padx=12, pady=(14, 0))

        # Blue accent strip (thicker = more visible)
        tk.Frame(wrapper, bg=ACC, width=3).pack(side="left", fill="y")

        # Card body
        card = tk.Frame(wrapper, bg=CARD)
        card.pack(side="left", fill="both", expand=True)
        card.columnconfigure(1, weight=1)

        # Title bar with a slightly lighter background to separate it from rows
        title_bar = tk.Frame(card, bg=CARD)
        title_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(title_bar, text=title, bg=CARD, fg=ACC,
                 font=F(12, "bold"), anchor="w",
                 padx=14, pady=10).pack(side="left")
        # Bottom border of title
        tk.Frame(card, bg=BORDER, height=1).grid(
            row=1, column=0, columnspan=2, sticky="ew")

        card._row_count = 2      # rows 0 and 1 are title + divider
        return card

    def _row(self, frame, label, widget_factory, r=None, info=None, sticky="ew"):
        """Add a label + widget pair as one grid row; returns the widget.

        If *info* is given a small ⓘ label is appended to the row label.
        Hovering over it (or the main label) shows a dark tooltip.
        *sticky* controls how the widget fills its grid cell (default "ew").
        Pass sticky="w" for buttons so they don't stretch full-width.
        """
        if r is None:
            r = getattr(frame, "_row_count", 0)
            frame._row_count = r + 1

        # Container for label text + optional ⓘ badge
        lbl_frame = tk.Frame(frame, bg=CARD)
        lbl_frame.grid(row=r, column=0, sticky="w", padx=(16, 4), pady=7)

        lbl = tk.Label(lbl_frame, text=label, bg=CARD, fg=MUTED,
                       font=F(11), anchor="w")
        lbl.pack(side="left")

        if info:
            badge = tk.Label(lbl_frame, text=" ⓘ", bg=CARD, fg=ACC,
                             font=F(10), anchor="w", cursor="question_arrow")
            badge.pack(side="left")
            _Tooltip(badge, info)
            _Tooltip(lbl,   info)   # also trigger from the label itself

        w = widget_factory(frame)
        w.grid(row=r, column=1, sticky=sticky, padx=(0, 14), pady=7)
        return w

    def _spin_int(self, parent, var, lo, hi, inc=1):
        return tk.Entry(parent, textvariable=var, width=10,
                        bg=CARD, fg=TXT, insertbackground=ACC,
                        relief="flat", bd=0, font=F(11), justify="center",
                        highlightthickness=1, highlightbackground=BORDER,
                        highlightcolor=ACC)

    def _spin_flt(self, parent, var, lo, hi, inc, fmt="%.3f"):
        return tk.Entry(parent, textvariable=var, width=10,
                        bg=CARD, fg=TXT, insertbackground=ACC,
                        relief="flat", bd=0, font=F(11), justify="center",
                        highlightthickness=1, highlightbackground=BORDER,
                        highlightcolor=ACC)

    def _combo(self, parent, var, values, w=18):
        cb = ttk.Combobox(parent, textvariable=var, values=values,
                          state="readonly", width=w, justify="center",
                          takefocus=0)
        return cb

    def _build_params(self, p):
        # ── Acquisition ───────────────────────────────────────────────────────
        f = self._section(p, "Acquisition")

        # Pixel size row: spinbox + independent override checkbox
        self._px_widget = self._row(f, "Pixel size (µm/px)",
                  lambda P: self._spin_flt(P, self.v_pixel_size, 0.001, 2.0, 0.001),
                  info=(
                      "Physical size of one camera pixel in µm.\n\n"
                      "Auto-read from file metadata when available:\n"
                      "  • CZI  — always present in Zeiss metadata\n"
                      "  • OME-TIFF — PhysicalSizeX from OME-XML\n"
                      "  • ImageJ TIFF — XResolution + unit tag\n\n"
                      "Typical values:\n"
                      "  • Zeiss Elyra 7, 63× oil (1.4 NA): ~0.106 µm/px\n"
                      "  • 100× oil (1.46 NA): ~0.067 µm/px\n\n"
                      "Incorrect values scale all diffusion coefficients and "
                      "displacements proportionally — verify before running."
                  ))
        self._px_cb = self._row(f, "  Set manually",
                  lambda P: ttk.Checkbutton(P, variable=self.v_override_px,
                                            style="Card.TCheckbutton",
                                            command=self._toggle_px),
                  info=(
                      "Tick to override the pixel size read from file metadata.\n\n"
                      "Use this when the file has no calibration metadata, or "
                      "when the embedded value is incorrect."
                  ))

        # Frame interval row: spinbox + independent override checkbox
        self._fi_widget = self._row(f, "Frame interval (s)",
                  lambda P: self._spin_flt(P, self.v_frame_interval, 0.001, 60.0, 0.001),
                  info=(
                      "Time between consecutive frames in seconds.\n\n"
                      "Auto-read from file metadata when available:\n"
                      "  • CZI  — always present in Zeiss metadata\n"
                      "  • OME-TIFF — TimeIncrement from OME-XML\n"
                      "  • ImageJ TIFF — finterval / fps tag\n\n"
                      "Typical values:\n"
                      "  • Fast sptPALM (50 Hz): 0.020 s\n"
                      "  • Standard (10 Hz): 0.100 s\n\n"
                      "Directly scales all diffusion coefficients — "
                      "D is proportional to 1/Δt."
                  ))
        self._fi_cb = self._row(f, "  Set manually",
                  lambda P: ttk.Checkbutton(P, variable=self.v_override_fi,
                                            style="Card.TCheckbutton",
                                            command=self._toggle_fi),
                  info=(
                      "Tick to override the frame interval read from file metadata.\n\n"
                      "Use this when the file has no timing metadata, or "
                      "when the embedded value is incorrect."
                  ))

        self._row(f, "Channel index",
                  lambda P: self._spin_int(P, self.v_channel, 0, 10),
                  info=(
                      "Zero-based index of the channel to analyse.\n\n"
                      "0 = first channel (most common for single-colour sptPALM).\n"
                      "Increase if your photoactivatable fluorophore is recorded "
                      "on a later channel in a multi-colour acquisition."
                  ))
        self._toggle_px()
        self._toggle_fi()

        # ── Preprocessing ─────────────────────────────────────────────────────
        f = self._section(p, "Preprocessing")
        self._row(f, "Background method",
                  lambda P: self._combo(P, self.v_bg_method,
                                        ["Uniform Filter", "Rolling Ball"]),
                  info=(
                      "Algorithm used to estimate and subtract the background "
                      "fluorescence before localisation.\n\n"
                      "Uniform Filter — recommended. Uses a fast box-filter (~1700× "
                      "faster). Produces clean results for even illumination.\n\n"
                      "Rolling Ball — FIJI default. More accurate for uneven "
                      "illumination (e.g. TIRF gradients) but much slower."
                  ))
        self._row(f, "BG radius (px)",
                  lambda P: self._spin_int(P, self.v_bg_radius, 5, 300),
                  info=(
                      "Radius of the background estimation kernel in pixels.\n\n"
                      "Should be larger than the largest feature you want to keep "
                      "(i.e. larger than your PSF).\n\n"
                      "Typical values:\n"
                      "  • Fast uniform filter: 5–15 px\n"
                      "  • Rolling ball: 20–100 px\n\n"
                      "Too small → over-subtracts signal. "
                      "Too large → background not removed."
                  ))

        # ── Localisation ──────────────────────────────────────────────────────
        f = self._section(p, "Localisation")
        self._row(f, "PSF diameter (px, odd)",
                  lambda P: self._spin_int(P, self.v_diameter, 3, 21, 2),
                  info=(
                      "Expected diameter of a single fluorophore's point-spread "
                      "function (PSF) in pixels. Must be an odd integer.\n\n"
                      "Estimated as: diameter ≈ 2.4 × λ / (NA × pixel_size)\n\n"
                      "Typical values (Zeiss Elyra 7, 0.106 µm/px, 1.4 NA):\n"
                      "  • 488 nm / 561 nm channel: 7 px\n"
                      "  • 642 nm channel: 9 px\n\n"
                      "Too small → misses dim particles. "
                      "Too large → poor sub-pixel precision."
                  ))
        self._row(f, "Auto-detect minmass",
                  lambda P: ttk.Checkbutton(P, variable=self.v_auto_mm,
                                            style="Card.TCheckbutton",
                                            command=self._toggle_minmass),
                  info=(
                      "When ticked, the pipeline estimates the minimum integrated "
                      "intensity (minmass) threshold automatically using the "
                      "background noise level.\n\n"
                      "Recommended for most datasets. Disable and set manually "
                      "if auto-detection picks up too much noise or misses dim "
                      "single molecules."
                  ))
        self._mm_widget = self._row(f, "Minmass (manual)",
                  lambda P: self._spin_flt(P, self.v_minmass, 0.0, 20.0, 0.05),
                  info=(
                      "Minimum integrated brightness (in normalised units) that "
                      "a candidate spot must have to be accepted as a localisation.\n\n"
                      "Only active when Auto-detect minmass is OFF.\n\n"
                      "Start with 1.0 and adjust while checking the preview: "
                      "increase to reject noise, decrease to recover dim particles. "
                      "Typical range: 0.5 – 5.0."
                  ))
        self._toggle_minmass()

        # ── Linking ───────────────────────────────────────────────────────────
        f = self._section(p, "Linking")
        self._row(f, "Search range (px)",
                  lambda P: self._spin_int(P, self.v_search_range, 1, 30),
                  info=(
                      "Maximum distance (pixels) a particle may travel between "
                      "consecutive frames and still be linked into the same "
                      "trajectory.\n\n"
                      "Rule of thumb: set to ~2–3 × expected single-step displacement.\n"
                      "Expected displacement ≈ √(4D·Δt) / pixel_size\n\n"
                      "Typical values:\n"
                      "  • Slow membrane protein (D ≈ 0.01 µm²/s, Δt = 20 ms): "
                      "~2–3 px\n"
                      "  • Fast cytoplasmic (D ≈ 1 µm²/s): ~10–15 px\n\n"
                      "Too large → false links between different particles. "
                      "Too small → trajectories broken unnecessarily."
                  ))
        self._row(f, "Memory (frames)",
                  lambda P: self._spin_int(P, self.v_memory, 0, 20),
                  info=(
                      "Number of frames a particle is allowed to disappear "
                      "(blink / move out of focus) and still be reconnected "
                      "to the same trajectory.\n\n"
                      "Typical values: 1–5 frames.\n\n"
                      "0 = no gap-closing (strictest). Increase for fluorophores "
                      "prone to blinking (e.g. mEos, PA-GFP in low activation "
                      "conditions).\n\n"
                      "Higher values increase run time and the risk of "
                      "merging separate molecules."
                  ))
        self._row(f, "Min track length",
                  lambda P: self._spin_int(P, self.v_min_track_len, 3, 50),
                  info=(
                      "Minimum number of frames a trajectory must span to be "
                      "included in the analysis.\n\n"
                      "Typical values: 8–20 frames.\n\n"
                      "Rule of thumb: set to at least 2–3 × 'Fit first N lag points' "
                      "so MSD fitting is well-constrained.\n\n"
                      "Short tracks contribute poorly determined D values and "
                      "inflate the immobile population."
                  ))
        self._row(f, "Max track length (0=off)",
                  lambda P: self._spin_int(P, self.v_max_track_len, 0, 10000, 10),
                  info=(
                      "Truncate trajectories longer than this many frames. "
                      "0 = no upper limit (recommended).\n\n"
                      "Useful only if very long tracks from persistent "
                      "non-activated molecules are contaminating the dataset, "
                      "or to cap memory use on extremely dense acquisitions."
                  ))

        # ── Drift Correction ──────────────────────────────────────────────────
        f = self._section(p, "Drift Correction")
        self._row(f, "Enable",
                  lambda P: ttk.Checkbutton(P, variable=self.v_drift_correct,
                                            style="Card.TCheckbutton",
                                            command=self._toggle_drift),
                  info=(
                      "Apply reference-free redundant cross-correlation (RCC) "
                      "drift correction before trajectory linking.\n\n"
                      "Recommended whenever stage drift is suspected — even small "
                      "drift (<50 nm) significantly biases diffusion coefficients "
                      "toward higher values.\n\n"
                      "No fiducial markers required. Uses the spatial distribution "
                      "of all localisations to estimate drift between time segments."
                  ))
        self._drift_seg_w = self._row(f, "Segment size (frames)",
                  lambda P: self._spin_int(P, self.v_drift_segment, 50, 2000, 50),
                  info=(
                      "Number of consecutive frames grouped into one segment for "
                      "RCC drift estimation.\n\n"
                      "Each segment needs enough localisations to produce a "
                      "reliable density map. Aim for > 200 localisations/segment.\n\n"
                      "Typical values:\n"
                      "  • Sparse labelling: 400–1000 frames\n"
                      "  • Dense labelling: 100–200 frames\n\n"
                      "Smaller → finer temporal resolution but noisier estimates. "
                      "Larger → smoother but may miss fast drift events."
                  ))
        self._toggle_drift()

        # ── MSD & Diffusion ───────────────────────────────────────────────────
        f = self._section(p, "MSD & Diffusion")
        self._row(f, "Max lag time (points)",
                  lambda P: self._spin_int(P, self.v_max_lagtime, 5, 100),
                  info=(
                      "Number of lag-time points computed for each trajectory's "
                      "mean squared displacement (MSD) curve.\n\n"
                      "Typical values: 15–25.\n\n"
                      "Rule of thumb: use at most 1/4 of the shortest accepted "
                      "track length — MSD estimates become unreliable beyond that "
                      "because they are averaged over very few trajectory segments.\n\n"
                      "More points give a better view of confined motion plateaus "
                      "but don't improve the diffusion coefficient fit."
                  ))
        self._row(f, "Fit first N lag points",
                  lambda P: self._spin_int(P, self.v_n_fit, 3, 30),
                  info=(
                      "Number of short lag-time points used to fit the diffusion "
                      "coefficient D and anomalous exponent α (MSD = 4D·τ^α).\n\n"
                      "Typical values: 4–6.\n\n"
                      "Fewer points → less noise but may miss curvature indicating "
                      "confinement. More points → captures confinement in the fit "
                      "but biases D downward for confined particles.\n\n"
                      "For Brownian diffusion classification, 4 points is usually "
                      "optimal."
                  ))

        self._row(f, "JDD populations",
                  lambda P: self._combo(P, self.v_jdd_components,
                                        [1, 2, 3]),
                  info=(
                      "Number of diffusion populations to fit in the Jump "
                      "Distance Distribution (JDD).\n\n"
                      "2 = slow + fast (most common for sptPALM).\n"
                      "3 = immobile + confined + free (if 3 populations visible).\n"
                      "1 = single-population control.\n\n"
                      "JDD fits the distribution of single-frame step sizes "
                      "across ALL tracks simultaneously, giving population-level "
                      "D values and fractions — complementary to the per-track "
                      "MSD analysis above."
                  ))

        # D filter
        self._row(f, "Filter by D value",
                  lambda P: ttk.Checkbutton(P, variable=self.v_filter_d_enabled,
                                            style="Card.TCheckbutton",
                                            command=self._toggle_d_filter),
                  info=(
                      "After MSD fitting, discard tracks whose diffusion "
                      "coefficient D falls outside the Min–Max range below.\n\n"
                      "Useful for isolating a specific population "
                      "(e.g. only free-diffusing, or only confined).\n\n"
                      "JDD, motion classification, and the figure are all "
                      "recomputed on the filtered set.\n\n"
                      "To replicate PALMTracer's immobile cutoff, enable this "
                      "and set D min to ~0.01–0.05 µm²/s (PALMTracer typically "
                      "uses 0.02 µm²/s by default). Leave D max at a high value "
                      "to pass all mobile tracks through."
                  ))
        self._d_min_w = self._row(f, "  D min (µm²/s)",
                  lambda P: self._spin_flt(P, self.v_filter_d_min,
                                           0.0, 100.0, 0.001),
                  info="Tracks with D below this value are excluded.")
        self._d_max_w = self._row(f, "  D max (µm²/s)",
                  lambda P: self._spin_flt(P, self.v_filter_d_max,
                                           0.0001, 100.0, 0.001),
                  info="Tracks with D above this value are excluded.")
        self._toggle_d_filter()

        # ── Cluster Analysis (DBSCAN) ─────────────────────────────────────────
        f = self._section(p, "Cluster Analysis  (DBSCAN)")
        self._row(f, "DBSCAN radius (nm)",
                  lambda P: self._spin_flt(P, self.v_cluster_eps_nm, 10.0, 500.0, 5.0),
                  info=("Search radius for DBSCAN clustering of raw localisations (nm). "
                        "Typical values: 30–80 nm for membrane receptor clusters. "
                        "Smaller = tighter/more clusters, larger = merges nearby clusters."))
        self._row(f, "Min localisations",
                  lambda P: self._spin_int(P, self.v_cluster_min_samples, 3, 200),
                  info=("Minimum number of localisations in a neighbourhood to form a cluster. "
                        "Typical values: 5–20. Increase to reject noise, decrease to find small clusters."))

        # ── ROI Masking ───────────────────────────────────────────────────────
        f = self._section(p, "ROI Masking")
        cb = self._row(f, "ROI mode",
                  lambda P: self._combo(P, self.v_roi_mode,
                                        ["Disabled", "Auto threshold",
                                         "Manual threshold", "Draw ROI"]),
                  info=(
                      "Restrict analysis to a region of interest (ROI).\n\n"
                      "Disabled — use the full frame.\n\n"
                      "Auto threshold — threshold the mean projection automatically; "
                      "keeps bright regions (e.g. cell bodies in TIRF).\n\n"
                      "Manual threshold — set the threshold yourself [0–1].\n\n"
                      "Draw ROI — freehand polygon drawn in the ROI Editor."
                  ))
        cb.bind("<<ComboboxSelected>>", lambda e: self._toggle_roi())

        self._roi_auto_w = self._row(f, "Auto method",
                  lambda P: self._combo(P, self.v_roi_auto_method,
                                        ["Auto", "Triangle", "Li", "Otsu"]),
                  info=(
                      "Automatic thresholding algorithm applied to the mean "
                      "projection to generate the ROI mask.\n\n"
                      "Auto — tries Triangle, falls back to Li.\n"
                      "Triangle — good for images with a prominent background peak.\n"
                      "Li — minimum cross-entropy; robust for fluorescence images.\n"
                      "Otsu — minimises intra-class variance; best when foreground "
                      "and background are similar in area."
                  ))
        self._roi_thresh_w = self._row(f, "Threshold [0–1]",
                  lambda P: self._spin_flt(P, self.v_roi_threshold,
                                           0.01, 0.99, 0.01, "%.2f"),
                  info=(
                      "Manual threshold applied to the normalised [0–1] mean "
                      "projection intensity.\n\n"
                      "Pixels above this value are included in the ROI.\n\n"
                      "Typical range: 0.05–0.30. Start at 0.10 and adjust by "
                      "inspecting the preview overlay."
                  ))
        self._roi_mmode_w = self._row(f, "Mask mode",
                  lambda P: self._combo(P, self.v_roi_mask_mode,
                                        ["Mean", "Per Frame"]),
                  info=(
                      "How the ROI mask is applied over time.\n\n"
                      "Mean — one mask from the time-averaged projection, applied "
                      "to every frame. Fast and robust for static cells.\n\n"
                      "Per Frame — mask recomputed for each frame from its own "
                      "intensity. Useful for drifting samples or moving cells, "
                      "but significantly slower."
                  ))
        self._roi_draw_btn = self._row(f, "Draw ROI",
                  lambda P: _FlatButton(P, text="Open ROI Editor",
                                        command=self._open_roi_editor),
                  sticky="w",
                  info=(
                      "Open the interactive ROI Editor to draw a freehand polygon "
                      "on the mean projection.\n\n"
                      "Left-click to add vertices. Double-click to close the "
                      "polygon. Right-click to undo the last vertex.\n\n"
                      "Click Apply to save the mask — the button label will "
                      "update to show the percentage of the frame included."
                  ))
        self._toggle_roi()

        # ── Performance ───────────────────────────────────────────────────────
        f = self._section(p, f"Performance  —  {N_CPUS} cores detected")
        self._row(f, "CPU workers",
                  lambda P: self._spin_int(P, self.v_workers, 1, N_CPUS),
                  info=(
                      f"Number of parallel CPU workers used for localisation "
                      f"and MSD computation. Max on this machine: {N_CPUS}.\n\n"
                      f"Recommended: {max(1, N_CPUS - 1)} (leave one core free "
                      f"for the OS and UI).\n\n"
                      f"Setting this higher than the number of physical cores "
                      f"rarely helps and can increase memory pressure."
                  ))
        self._row(f, "Chunk size (frames)",
                  lambda P: self._spin_int(P, self.v_chunk_size, 50, 5000, 100),
                  info=(
                      "Number of frames loaded into memory at once during "
                      "localisation. Larger chunks = fewer I/O reads = faster, "
                      "but higher RAM usage.\n\n"
                      "Typical values:\n"
                      "  • 16 GB RAM, 512×512 px: 500–1000 frames\n"
                      "  • 8 GB RAM or large images: 100–300 frames\n\n"
                      "Reduce if you see out-of-memory errors."
                  ))

        # ── Figure Style ──────────────────────────────────────────────────────
        f = self._section(p, "Figure Style")
        self._row(f, "Theme",
                  lambda P: self._combo(P, self.v_fig_theme,
                                        ["Dark", "Light", "Publication"]),
                  info=(
                      "Visual style of the output figure.\n\n"
                      "Dark — dark background; ideal for presentations.\n\n"
                      "Light — white background with light panels; good for "
                      "reports and slides.\n\n"
                      "Publication — white background, serif font, minimal "
                      "gridlines; suitable for journal figures."
                  ))
        self._row(f, "Projection colourmap",
                  lambda P: self._combo(P, self.v_proj_cmap,
                                        ["Inferno", "Hot", "Viridis",
                                         "Plasma", "Greys"]),
                  info=(
                      "Colourmap applied to the max-projection image (panel A).\n\n"
                      "Inferno — warm yellow-orange; high contrast. Default.\n"
                      "Hot — red-orange; classic fluorescence look.\n"
                      "Viridis — blue-green-yellow; perceptually uniform and "
                      "colourblind-safe.\n"
                      "Plasma — purple-pink-yellow; vivid.\n"
                      "Greys — greyscale; clean for publication."
                  ))

        # ── Save Settings ─────────────────────────────────────────────────────
        # Thin divider
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(12, 4))
        _FlatButton(p, text="Save",
                    command=self._save_settings
                    ).pack(fill="x", padx=16, pady=(4, 14))

    # ── Toggle callbacks ──────────────────────────────────────────────────────

    def _toggle_px(self):
        """Pixel-size spinbox is active only when its override checkbox is ticked."""
        self._px_widget.configure(
            state="normal" if self.v_override_px.get() else "disabled")

    def _toggle_fi(self):
        """Frame-interval spinbox is active only when its override checkbox is ticked."""
        self._fi_widget.configure(
            state="normal" if self.v_override_fi.get() else "disabled")

    def _toggle_minmass(self):
        state = "disabled" if self.v_auto_mm.get() else "normal"
        self._mm_widget.configure(state=state)

    def _toggle_drift(self):
        state = "normal" if self.v_drift_correct.get() else "disabled"
        self._drift_seg_w.configure(state=state)

    def _toggle_d_filter(self):
        state = "normal" if self.v_filter_d_enabled.get() else "disabled"
        self._d_min_w.configure(state=state)
        self._d_max_w.configure(state=state)

    def _toggle_roi(self):
        mode     = self.v_roi_mode.get()
        auto_s   = "readonly" if mode == "Auto threshold"                      else "disabled"
        thresh_s = "normal"   if mode == "Manual threshold"                    else "disabled"
        mmode_s  = "readonly" if mode in ("Auto threshold", "Manual threshold") else "disabled"
        draw_s   = "normal"   if mode == "Draw ROI"                            else "disabled"
        self._roi_auto_w.configure(state=auto_s)
        self._roi_thresh_w.configure(state=thresh_s)
        self._roi_mmode_w.configure(state=mmode_s)
        self._roi_draw_btn.configure(state=draw_s)

    # ── Settings save / load ──────────────────────────────────────────────────

    _SETTINGS_PATH = os.path.join(os.path.expanduser("~"),
                                  ".sptpalm_settings.json")

    def _settings_dict(self) -> dict:
        """Serialise every settings variable to a plain dict."""
        return {
            "pixel_size":      self.v_pixel_size.get(),
            "frame_interval":  self.v_frame_interval.get(),
            "override_px":     self.v_override_px.get(),
            "override_fi":     self.v_override_fi.get(),
            "channel":         self.v_channel.get(),
            "bg_method":       self.v_bg_method.get(),
            "bg_radius":       self.v_bg_radius.get(),
            "diameter":        self.v_diameter.get(),
            "auto_mm":         self.v_auto_mm.get(),
            "minmass":         self.v_minmass.get(),
            "search_range":    self.v_search_range.get(),
            "memory":          self.v_memory.get(),
            "min_track_len":   self.v_min_track_len.get(),
            "max_track_len":   self.v_max_track_len.get(),
            "max_lagtime":     self.v_max_lagtime.get(),
            "n_fit":           self.v_n_fit.get(),
            "jdd_components":  self.v_jdd_components.get(),
            "filter_d_enabled": self.v_filter_d_enabled.get(),
            "filter_d_min":    self.v_filter_d_min.get(),
            "filter_d_max":    self.v_filter_d_max.get(),
            "roi_mode":        self.v_roi_mode.get(),
            "roi_auto_method": self.v_roi_auto_method.get(),
            "roi_threshold":   self.v_roi_threshold.get(),
            "roi_mask_mode":   self.v_roi_mask_mode.get(),
            "drift_correct":   self.v_drift_correct.get(),
            "drift_segment":   self.v_drift_segment.get(),
            "workers":         self.v_workers.get(),
            "chunk_size":      self.v_chunk_size.get(),
            "fig_theme":       self.v_fig_theme.get(),
            "proj_cmap":       self.v_proj_cmap.get(),
            "cluster_eps_nm":      self.v_cluster_eps_nm.get(),
            "cluster_min_samples": self.v_cluster_min_samples.get(),
        }

    def _apply_settings_dict(self, d: dict):
        """Apply a settings dict loaded from disk to all Tk variables."""
        def _s(var, key):
            if key in d:
                var.set(d[key])
        _s(self.v_pixel_size,     "pixel_size")
        _s(self.v_frame_interval, "frame_interval")
        _s(self.v_override_px,    "override_px")
        _s(self.v_override_fi,    "override_fi")
        _s(self.v_channel,        "channel")
        _s(self.v_bg_method,      "bg_method")
        _s(self.v_bg_radius,      "bg_radius")
        _s(self.v_diameter,       "diameter")
        _s(self.v_auto_mm,        "auto_mm")
        _s(self.v_minmass,        "minmass")
        _s(self.v_search_range,   "search_range")
        _s(self.v_memory,         "memory")
        _s(self.v_min_track_len,  "min_track_len")
        _s(self.v_max_track_len,  "max_track_len")
        _s(self.v_max_lagtime,    "max_lagtime")
        _s(self.v_n_fit,          "n_fit")
        _s(self.v_jdd_components,   "jdd_components")
        _s(self.v_filter_d_enabled, "filter_d_enabled")
        _s(self.v_filter_d_min,     "filter_d_min")
        _s(self.v_filter_d_max,     "filter_d_max")
        _s(self.v_roi_mode,       "roi_mode")
        _s(self.v_roi_auto_method,"roi_auto_method")
        _s(self.v_roi_threshold,  "roi_threshold")
        _s(self.v_roi_mask_mode,  "roi_mask_mode")
        _s(self.v_drift_correct,  "drift_correct")
        _s(self.v_drift_segment,  "drift_segment")
        _s(self.v_workers,        "workers")
        _s(self.v_chunk_size,     "chunk_size")
        _s(self.v_fig_theme,      "fig_theme")
        _s(self.v_proj_cmap,      "proj_cmap")
        _s(self.v_cluster_eps_nm,      "cluster_eps_nm")
        _s(self.v_cluster_min_samples, "cluster_min_samples")
        # Refresh dependent widget states
        self._toggle_px()
        self._toggle_fi()
        self._toggle_minmass()
        self._toggle_drift()
        self._toggle_roi()

    def _save_settings(self):
        import json
        try:
            with open(self._SETTINGS_PATH, "w") as fh:
                json.dump(self._settings_dict(), fh, indent=2)
            self._status_var.set("✓  Settings saved as default")
            self.after(2500, lambda: self._status_var.set(""))
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _load_settings(self):
        import json
        if not os.path.exists(self._SETTINGS_PATH):
            return
        try:
            with open(self._SETTINGS_PATH) as fh:
                d = json.load(fh)
            self._apply_settings_dict(d)
        except Exception:
            pass  # Corrupt / old file — silently use built-in defaults

    def _open_roi_editor(self):
        fpath = self.v_file.get().strip()
        if not fpath or not os.path.isfile(fpath):
            messagebox.showerror("No file loaded",
                                 "Please load a CZI or TIF file before opening the ROI editor.")
            return
        _ROIEditorDialog(self, fpath, self.v_channel.get(),
                         on_apply=self._apply_drawn_roi)

    def _apply_drawn_roi(self, mask):
        self._drawn_roi_mask = mask
        h, w = mask.shape
        pct  = 100.0 * int(mask.sum()) / (h * w)
        self._roi_draw_btn.configure(
            text=f"✓ ROI set ({pct:.0f}% of image) — Edit")

    # ── Right-panel helpers ───────────────────────────────────────────────────

    def _clear_right(self):
        for w in self._right.winfo_children():
            w.destroy()
        self._img_refs.clear()

    def _show_placeholder(self, ready=False):
        self._clear_right()
        ph = ttk.Frame(self._right)
        ph.place(relx=0.5, rely=0.5, anchor="center")
        if ready:
            ttk.Label(ph, text="Ready to analyse",
                      style="Acc.TLabel").pack()
            ttk.Label(ph, text="Click Run Analysis to start.",
                      style="Muted.TLabel").pack(pady=(6, 2))
            ttk.Label(ph, text="Live particle tracking will appear here once analysis begins.",
                      style="Muted.TLabel").pack()
        else:
            ttk.Label(ph, text="Upload a file to get started",
                      style="Acc.TLabel").pack()
            ttk.Label(ph, text="Select a CZI or TIF file using Browse… above.",
                      style="Muted.TLabel").pack(pady=(6, 2))
            ttk.Label(ph, text="Live particle tracking will appear here once analysis begins.",
                      style="Muted.TLabel").pack()

    def _show_progress_panel(self):
        """Progress panel with header + progress bar + LIVE preview canvas + log."""
        self._clear_right()
        f = ttk.Frame(self._right, padding=18)
        f.pack(fill="both", expand=True)

        # Header
        ttk.Label(f, text="Analysis running",
                  style="Acc.TLabel").pack(anchor="w")
        self._prog_label = ttk.Label(f, text="Starting…", style="Muted.TLabel")
        self._prog_label.pack(anchor="w", pady=(4, 8))
        self._prog_bar = ttk.Progressbar(f, mode="determinate", maximum=100)
        self._prog_bar.pack(fill="x", pady=(0, 14))

        # Split: live preview (top) + log (bottom)
        split = ttk.PanedWindow(f, orient="vertical")
        split.pack(fill="both", expand=True)

        # ── Live tracking preview ──────────────────────────────────────────────
        prev_card = tk.Frame(split, bg=PNL2, bd=0)
        split.add(prev_card, weight=3)

        prev_hdr = tk.Frame(prev_card, bg=PNL2)
        prev_hdr.pack(fill="x", padx=12, pady=(8, 0))
        self._live_dot_label = tk.Label(prev_hdr, text="●",
                                        bg=PNL2, fg=ACC, font=F(10, "bold"))
        self._live_dot_label.pack(side="left")
        tk.Label(prev_hdr, text=" LIVE  Particle Tracking",
                 bg=PNL2, fg=ACC, font=F(10, "bold")).pack(side="left")
        self._elapsed_label = tk.Label(prev_hdr, text="0s",
                                       bg=PNL2, fg=MUTED, font=F(9))
        self._elapsed_label.pack(side="right")
        self._preview_info = tk.Label(prev_hdr, text="waiting for first frame…",
                                      bg=PNL2, fg=MUTED, font=F(9))
        self._preview_info.pack(side="right", padx=(0, 12))
        self._start_live_timers()

        self._preview_holder = tk.Frame(prev_card, bg=PNL2)
        self._preview_holder.pack(fill="both", expand=True, padx=10, pady=10)
        self._init_preview_canvas(self._preview_holder)

        # ── Log ────────────────────────────────────────────────────────────────
        log_card = tk.Frame(split, bg=PNL2)
        split.add(log_card, weight=2)

        tk.Label(log_card, text="Log output", bg=PNL2, fg=MUTED,
                 font=F(10, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        self._log_box = scrolledtext.ScrolledText(
            log_card, bg=CARD, fg=MUTED, insertbackground=TXT, wrap="word",
            font=FM(9), height=10, bd=0, relief="flat",
            highlightthickness=0)
        self._log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._log_box.configure(state="disabled")

    def _init_preview_canvas(self, parent):
        """Plain tk.Canvas for live preview — no nested matplotlib figure."""
        self._preview_canvas = tk.Canvas(parent, bg=BG, bd=0, highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)
        self._preview_canvas.bind("<Configure>", lambda e: self._draw_preview_image())
        self._spinner_angle  = 0
        self._spinner_id     = None   # after() handle so we can cancel it
        self._preview_canvas.after(120, self._preview_spinner_tick)

    def _preview_spinner_tick(self):
        """Animate a revolving arc placeholder until the first frame arrives."""
        if self._preview_image_array is not None or not self._preview_canvas:
            return  # first frame arrived — stop spinning
        try:
            w = self._preview_canvas.winfo_width()
            h = self._preview_canvas.winfo_height()
            if w <= 1 or h <= 1:
                self._spinner_id = self._preview_canvas.after(
                    120, self._preview_spinner_tick)
                return

            cx, cy = w // 2, h // 2
            r = 18

            self._preview_canvas.delete("spinner")

            self._preview_canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=self._spinner_angle, extent=260,
                outline=ACC, width=3, style="arc",
                tags="spinner")

            self._spinner_angle = (self._spinner_angle + 12) % 360
        except Exception:
            pass

        self._spinner_id = self._preview_canvas.after(
            50, self._preview_spinner_tick)

    def _update_preview(self, rgb_array, label: str = ""):
        """Display a pre-rendered RGB numpy array on the preview canvas."""
        if self._preview_canvas is None:
            return
        # Stop the spinner now that we have a real frame
        if self._spinner_id is not None:
            try:
                self._preview_canvas.after_cancel(self._spinner_id)
            except Exception:
                pass
            self._spinner_id = None
        self._preview_image_array = rgb_array
        self._draw_preview_image()
        if hasattr(self, "_preview_info"):
            self._preview_info.configure(text=label)

    def _draw_preview_image(self):
        """Resize and blit the stored RGB frame onto the preview canvas."""
        if self._preview_image_array is None or self._preview_canvas is None:
            return
        try:
            from PIL import Image, ImageTk
            w = self._preview_canvas.winfo_width()
            h = self._preview_canvas.winfo_height()
            if w <= 1 or h <= 1:
                return
            img = Image.fromarray(self._preview_image_array)
            # thumbnail() preserves aspect ratio, fitting within (w, h)
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            # Keep reference so GC doesn't collect it
            self._preview_photo = photo
            self._preview_canvas.delete("all")
            self._preview_canvas.create_image(w // 2, h // 2,
                                              anchor="center", image=photo)
        except Exception:
            pass

    def _start_live_timers(self):
        """Begin the flashing dot and elapsed-time counter."""
        self._dot_on = True
        self._flash_live_dot()
        self._update_elapsed()

    def _flash_live_dot(self):
        if not self._running:
            return
        self._dot_on = not self._dot_on
        try:
            self._live_dot_label.configure(fg=ACC if self._dot_on else PNL2)
        except Exception:
            pass
        self._flash_id = self.after(600, self._flash_live_dot)

    def _update_elapsed(self):
        if not self._running:
            return
        elapsed = time.monotonic() - self._analysis_start
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        text = f"{m}m {s:02d}s" if m else f"{s}s"
        try:
            self._elapsed_label.configure(text=text)
        except Exception:
            pass
        self._elapsed_id = self.after(1000, self._update_elapsed)

    def _stop_live_timers(self):
        """Cancel dot-flash and elapsed-counter after() callbacks."""
        if self._flash_id is not None:
            self.after_cancel(self._flash_id)
            self._flash_id = None
        if self._elapsed_id is not None:
            self.after_cancel(self._elapsed_id)
            self._elapsed_id = None

    def _append_log(self, text: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _show_results(self, data: dict):
        self._clear_right()

        diff_df  = data["diff_df"]
        fig_path = data["fig_path"]
        roi_path = data.get("roi_path")
        out_dir  = data["out_dir"]
        stem     = data["stem"]

        # ── helpers ──────────────────────────────────────────────────────────
        def _bordered_card(parent):
            """1 px BORDER outline → CARD face, same pattern as _FlatButton."""
            outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1,
                             highlightthickness=0)
            inner = tk.Frame(outer, bg=CARD, highlightthickness=0)
            inner.pack(fill="both", expand=True)
            return outer, inner

        def _divider(parent):
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x",
                                                        padx=20, pady=12)

        def _section_label(parent, text):
            tk.Label(parent, text=text.upper(), bg=BG, fg=MUTED,
                     font=F(8, "bold")).pack(anchor="w", padx=20, pady=(14, 6))

        def _embed_image(parent, path, max_w=900, max_h=560):
            try:
                from PIL import Image, ImageTk
                img = Image.open(path)
                img.thumbnail((max_w, max_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._img_refs.append(photo)
                tk.Label(parent, image=photo, bg=BG).pack(padx=12, pady=12)
                return True
            except ImportError:
                tk.Label(parent,
                         text=f"Install Pillow to view figure here.\n{path}",
                         bg=BG, fg=MUTED, font=F(9), justify="center"
                         ).pack(expand=True)
            except Exception as exc:
                tk.Label(parent, text=f"Could not load image:\n{exc}",
                         bg=BG, fg=RED, font=F(9)).pack(expand=True)
            return False

        # ── Tab strip (manual, themed) ────────────────────────────────────────
        # We avoid ttk.Notebook entirely — it pulls in native OS chrome that
        # doesn't respect our dark palette.  Instead we build a simple
        # button-strip + stacked Frame approach.
        outer = tk.Frame(self._right, bg=BG)
        outer.pack(fill="both", expand=True)

        tab_bar = tk.Frame(outer, bg=SIDEBAR, height=36)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)

        content = tk.Frame(outer, bg=BG)
        content.pack(fill="both", expand=True)

        _tabs   = {}   # name → Frame
        _btns   = {}
        _active = tk.StringVar(value="")

        def _switch(name):
            prev = _active.get()
            if prev and prev in _btns:
                _btns[prev].configure(bg=SIDEBAR, fg=MUTED)
            _active.set(name)
            _btns[name].configure(bg=BG, fg=TXT)
            for n, f in _tabs.items():
                if n == name:
                    f.place(relx=0, rely=0, relwidth=1, relheight=1)
                else:
                    f.place_forget()

        def _add_tab(name):
            f = tk.Frame(content, bg=BG)
            _tabs[name] = f
            btn = tk.Label(tab_bar, text=name, bg=SIDEBAR, fg=MUTED,
                           font=F(10), padx=18, pady=8, cursor="hand2")
            btn.pack(side="left")
            btn.bind("<Button-1>", lambda e, n=name: _switch(n))
            _btns[name] = btn
            return f

        # ── Tab: Summary ──────────────────────────────────────────────────────
        t1 = _add_tab("Summary")
        t1_scroll = _ScrollFrame(t1, bg=BG)
        t1_scroll.pack(fill="both", expand=True)
        p1 = t1_scroll.inner

        n_tracks  = diff_df.shape[0]
        med_D     = diff_df["D"].dropna().median()
        mean_D    = diff_df["D"].dropna().mean()
        med_alpha = diff_df["alpha"].dropna().median()
        mc        = diff_df["motion"].value_counts()
        med_lp    = (diff_df["loc_precision_nm"].dropna().median()
                     if "loc_precision_nm" in diff_df.columns else None)

        _section_label(p1, "Key metrics")
        mf = tk.Frame(p1, bg=BG)
        mf.pack(fill="x", padx=16, pady=(0, 4))
        _metrics = [
            ("Trajectories", f"{n_tracks:,}",    ""),
            ("Median D",     f"{med_D:.4f}",      "µm²/s"),
            ("Mean D",       f"{mean_D:.4f}",     "µm²/s"),
            ("Median α",     f"{med_alpha:.3f}",  ""),
        ]
        if med_lp is not None:
            _metrics.append(("Loc. precision", f"{med_lp:.1f}", "nm"))
        for label, val, unit in _metrics:
            outer_c, inner_c = _bordered_card(mf)
            outer_c.pack(side="left", expand=True, fill="x", padx=4, pady=4)
            tk.Label(inner_c, text=label, bg=CARD, fg=MUTED,
                     font=F(9)).pack(pady=(10, 2))
            tk.Label(inner_c, text=val, bg=CARD, fg=ACC,
                     font=F(20, "bold")).pack()
            tk.Label(inner_c, text=unit or " ", bg=CARD, fg=MUTED,
                     font=F(8)).pack(pady=(0, 10))

        _section_label(p1, "Motion classification")
        mof = tk.Frame(p1, bg=BG)
        mof.pack(fill="x", padx=16, pady=(0, 4))
        for mot in ["Immobile", "Confined", "Brownian", "Directed"]:
            cnt = mc.get(mot, 0)
            pct = 100 * cnt / max(n_tracks, 1)
            col = MOTION_COLORS[mot]
            outer_c, inner_c = _bordered_card(mof)
            outer_c.pack(side="left", expand=True, fill="x", padx=4, pady=4)
            # coloured top accent strip
            tk.Frame(inner_c, bg=col, height=3).pack(fill="x")
            tk.Label(inner_c, text=mot, bg=CARD, fg=col,
                     font=F(9, "bold")).pack(pady=(8, 2))
            tk.Label(inner_c, text=f"{cnt:,}", bg=CARD, fg=TXT,
                     font=F(17, "bold")).pack()
            tk.Label(inner_c, text=f"{pct:.1f}%", bg=CARD, fg=MUTED,
                     font=F(9)).pack(pady=(0, 10))

        # JDD section
        jdd = data.get("jdd")
        if jdd:
            _divider(p1)
            _section_label(p1, "Jump Distance Distribution")
            jf = tk.Frame(p1, bg=BG)
            jf.pack(fill="x", padx=16, pady=(0, 4))
            _jdd_colors = ["#4ea8ff", "#f78166", "#3fb950"]
            _jdd_labels = ["Slow", "Medium", "Fast"]
            for k, (D, f) in enumerate(
                    zip(jdd["D_values"], jdd["fractions"])):
                col = _jdd_colors[k]
                outer_c, inner_c = _bordered_card(jf)
                outer_c.pack(side="left", expand=True, fill="x", padx=4, pady=4)
                tk.Frame(inner_c, bg=col, height=3).pack(fill="x")
                tk.Label(inner_c, text=_jdd_labels[k], bg=CARD, fg=col,
                         font=F(9, "bold")).pack(pady=(8, 2))
                tk.Label(inner_c, text=f"{D:.4f}", bg=CARD, fg=TXT,
                         font=F(14, "bold")).pack()
                tk.Label(inner_c, text="µm²/s", bg=CARD, fg=MUTED,
                         font=F(8)).pack()
                tk.Label(inner_c, text=f"{f*100:.1f}%", bg=CARD, fg=MUTED,
                         font=F(9)).pack(pady=(2, 10))
            tk.Label(p1, text=f"{jdd['n_jumps']:,} single-frame jumps fitted",
                     bg=BG, fg=MUTED, font=F(8)).pack(anchor="w", padx=20)

        _divider(p1)
        _FlatButton(p1, text="Open Output Folder",
                    command=lambda: _open_folder(out_dir)
                    ).pack(anchor="w", padx=20, pady=(0, 16))

        # ── Tab: Figure ───────────────────────────────────────────────────────
        t2 = _add_tab("Figure")
        fig_scroll = _ScrollFrame(t2, bg=BG)
        fig_scroll.pack(fill="both", expand=True)
        _embed_image(fig_scroll.inner, fig_path)
        if roi_path and os.path.exists(roi_path):
            _section_label(fig_scroll.inner, "ROI mask preview")
            _embed_image(fig_scroll.inner, roi_path, max_w=900, max_h=320)

        # ── Tab: Output Files ─────────────────────────────────────────────────
        t3 = _add_tab("Output Files")
        t3_scroll = _ScrollFrame(t3, bg=BG)
        t3_scroll.pack(fill="both", expand=True)
        p3 = t3_scroll.inner

        _section_label(p3, "Saved to")
        tk.Label(p3, text=out_dir, bg=BG, fg=ACC,
                 font=FM(9), wraplength=680, justify="left",
                 anchor="w").pack(anchor="w", padx=20, pady=(0, 4))

        fig_dir2  = os.path.join(out_dir, "figures")
        data_dir2 = os.path.join(out_dir, "data")

        _section_label(p3, "Files")
        file_info = [
            (f"{stem}_sptpalm_figure.png",    "Analysis figure",              fig_dir2),
            (f"{stem}_sptpalm_figure.pdf",    "Analysis figure (PDF/vector)", fig_dir2),
            (f"{stem}_diffusion_summary.csv", "Per-trajectory D, α, motion",  data_dir2),
            (f"{stem}_trajectories.csv",      "Full trajectory table",         data_dir2),
            (f"{stem}_localisations.csv",     "Raw localisations",             data_dir2),
            (f"{stem}_ensemble_msd.csv",      "Ensemble MSD curve",            data_dir2),
            (f"{stem}_cluster_stats.csv",     "DBSCAN cluster statistics",     data_dir2),
        ]
        if roi_path and os.path.exists(roi_path):
            file_info.append((os.path.basename(roi_path), "ROI mask preview", data_dir2))

        for fname, desc, fdir in file_info:
            fpath  = os.path.join(fdir, fname)
            exists = os.path.exists(fpath)
            outer_r, inner_r = _bordered_card(p3)
            outer_r.pack(fill="x", padx=20, pady=3)
            row = tk.Frame(inner_r, bg=CARD)
            row.pack(fill="x", padx=12, pady=8)
            tk.Label(row, text="✓" if exists else "✗",
                     bg=CARD, fg=GREEN2 if exists else RED,
                     font=F(11, "bold")).pack(side="left", padx=(0, 8))
            tk.Label(row, text=fname, bg=CARD,
                     fg=TXT if exists else MUTED,
                     font=FM(9)).pack(side="left")
            tk.Label(row, text=f"  —  {desc}", bg=CARD, fg=MUTED,
                     font=F(9)).pack(side="left")

        _divider(p3)
        _FlatButton(p3, text="Open Output Folder",
                    command=lambda: _open_folder(out_dir)
                    ).pack(anchor="w", padx=20, pady=(0, 16))

        # activate first tab
        _switch("Summary")

    # ── Icon ─────────────────────────────────────────────────────────────────

    def _set_icon(self):
        icon_img = _generate_icon(256)
        if icon_img is None:
            return
        try:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(icon_img)
            self._img_refs.append(photo)          # prevent GC
            self.iconphoto(True, photo)            # window + taskbar icon

            # macOS: also set the Dock icon via AppKit (requires pyobjc)
            if sys.platform == "darwin":
                try:
                    import tempfile, os as _os
                    from AppKit import NSApplication, NSImage
                    tmp = tempfile.mktemp(suffix=".png")
                    icon_img.save(tmp)
                    ns_img = NSImage.alloc().initWithContentsOfFile_(tmp)
                    NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
                    _os.unlink(tmp)
                except Exception:
                    pass  # pyobjc not available — window icon still set above
        except Exception:
            pass

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _browse_file(self):
        self.lift(); self.focus_force(); self.update()
        path = filedialog.askopenfilename(
            parent=self,
            title="Select CZI or TIF microscopy file",
            filetypes=[
                ("Microscopy files", "*.czi;*.tif;*.tiff"),
                ("CZI files", "*.czi"),
                ("TIFF files", "*.tif;*.tiff"),
                ("All files", "*.*"),
            ])
        if path:
            self.v_file.set(path)
            if not self.v_outdir.get():
                self.v_outdir.set(os.path.dirname(os.path.abspath(path)))
            # Clear any drawn ROI — it belongs to the previous file's dimensions
            self._drawn_roi_mask = None
            self._roi_draw_btn.configure(text="Open ROI Editor")
            self._status_var.set("Ready to analyse")
            # Only refresh placeholder if we're not already showing results/progress
            if not self._running:
                self._show_placeholder(ready=True)

    def _browse_outdir(self):
        self.lift(); self.focus_force(); self.update()
        path = filedialog.askdirectory(parent=self, title="Select output folder")
        if path:
            self.v_outdir.set(path)

    # ── Batch processing ──────────────────────────────────────────────────────

    def _on_batch(self):
        if self._running:
            return
        self.lift(); self.focus_force(); self.update()
        folder = filedialog.askdirectory(parent=self, title="Select folder to batch-process")
        if not folder:
            return

        exts = (".czi", ".tif", ".tiff")
        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
            and not f.startswith(".")
        )
        if not files:
            messagebox.showinfo("Batch", "No CZI or TIF files found in that folder.")
            return

        if not messagebox.askyesno(
                "Batch Process",
                f"Found {len(files)} file(s) in:\n{folder}\n\n"
                f"Run analysis on all of them with current settings?"):
            return

        self._running = True
        self._analysis_start = time.monotonic()
        self._stop_event.clear()
        self._run_btn.configure(text="■  Stop", command=self._on_stop,
                                style="Stop.TButton")
        self._batch_btn.configure(state="disabled")
        self._show_progress_panel()
        self._status_var.set(f"Batch: 0 / {len(files)}")

        def _start():
            threading.Thread(
                target=self._batch_worker,
                args=(folder, files),
                daemon=True).start()
            self.after(100, self._poll)

        self._collapse_panel(callback=_start)

    def _batch_worker(self, folder, files):
        import numpy as np
        import time as _time

        stop  = self._stop_event
        n     = len(files)
        rows  = []

        def _emit_log(text):
            self._q.put(("log", text))

        def _emit_progress(msg, pct):
            self._q.put(("progress", (msg, pct)))

        def _check_stop():
            if stop.is_set():
                raise Exception("Stopped")

        old_stdout = sys.stdout

        class _Cap:
            def write(self, text):
                if text.strip():
                    self._q_ref.put(("log", text.rstrip()))
            def flush(self): pass

        cap = _Cap()
        cap._q_ref = self._q
        sys.stdout = cap

        try:
            base = (sys._MEIPASS if getattr(sys, "frozen", False)
                    else os.path.dirname(os.path.abspath(__file__)))
            if base not in sys.path:
                sys.path.insert(0, base)

            from sptpalm_analysis import (
                load_file, preprocess_and_localise_adaptive,
                link_trajectories, compute_msd_and_fit, compute_jdd,
                compute_turning_angles, compute_mobile_fraction_over_time,
                compute_clusters, compute_dwell_times, compute_mss,
                build_roi_mask, apply_roi_mask, make_figure,
                correct_drift, _Cancelled,
            )

            px_arg = self.v_pixel_size.get()     if self.v_override_px.get() else None
            fi_arg = self.v_frame_interval.get() if self.v_override_fi.get() else None
            _bg_map  = {"Uniform Filter": "uniform_filter", "Rolling Ball": "rolling_ball"}
            _roi_map = {"Auto": "auto", "Triangle": "triangle", "Li": "li", "Otsu": "otsu"}
            bg_method_raw       = _bg_map.get(self.v_bg_method.get(), "uniform_filter")
            roi_auto_method_raw = _roi_map.get(self.v_roi_auto_method.get(), "auto")
            roi_mode   = self.v_roi_mode.get()
            roi_auto   = roi_mode == "Auto threshold"
            roi_manual = roi_mode == "Manual threshold"
            diameter   = self.v_diameter.get()
            if diameter % 2 == 0:
                diameter += 1
            workers    = self.v_workers.get()
            chunk_size = self.v_chunk_size.get()

            for i, fname in enumerate(files):
                _check_stop()
                fpath = os.path.join(folder, fname)
                stem  = os.path.splitext(fname)[0]
                out_dir  = os.path.join(folder, "batch_results", stem)
                fig_dir  = os.path.join(out_dir, "figures")
                data_dir = os.path.join(out_dir, "data")
                os.makedirs(fig_dir,  exist_ok=True)
                os.makedirs(data_dir, exist_ok=True)

                pct_base = int(i / n * 90)
                _emit_progress(f"[{i+1}/{n}]  {fname}", pct_base)
                _emit_log(f"\n{'='*50}")
                _emit_log(f"  [{i+1}/{n}]  {fname}")

                try:
                    stack, meta_px, meta_fi = load_file(
                        fpath, channel=self.v_channel.get(), stop_event=stop)
                    n_frames = len(stack)
                    px = px_arg or meta_px or 0.104
                    fi = fi_arg or meta_fi or 0.050

                    proj_idx    = np.linspace(0, n_frames - 1, min(200, n_frames), dtype=int)
                    proj_sample = stack[proj_idx].copy()

                    roi_mask = None
                    if roi_auto or roi_manual:
                        raw_mean = stack.mean(axis=0)
                        mn, mx   = raw_mean.min(), raw_mean.max()
                        raw_mean_norm = (raw_mean - mn) / (mx - mn) if mx > mn else raw_mean
                        thresh = self.v_roi_threshold.get() if roi_manual else None
                        roi_mask = build_roi_mask(
                            precomputed_mean_proj=raw_mean_norm,
                            threshold=thresh, mode="mean",
                            threshold_method=roi_auto_method_raw if roi_auto else "auto",
                            save_path=os.path.join(data_dir, f"{stem}_roi_mask.png"))

                    minmass_arg = None if self.v_auto_mm.get() else self.v_minmass.get()
                    locs, _, _ = preprocess_and_localise_adaptive(
                        stack, diameter=diameter, minmass=minmass_arg,
                        bg_radius=self.v_bg_radius.get(),
                        bg_method=bg_method_raw,
                        workers=workers, chunk_size=chunk_size,
                        stop_event=stop)
                    del stack

                    if roi_mask is not None:
                        locs = apply_roi_mask(locs, roi_mask)

                    if self.v_drift_correct.get() and len(locs) > 0:
                        locs, drift_df = correct_drift(
                            locs, n_seg_frames=self.v_drift_segment.get())
                        drift_df.to_csv(
                            os.path.join(data_dir, f"{stem}_drift.csv"), index=False)

                    max_tl = self.v_max_track_len.get()
                    tracks = link_trajectories(
                        locs,
                        search_range=self.v_search_range.get(),
                        memory=self.v_memory.get(),
                        min_len=self.v_min_track_len.get(),
                        max_len=max_tl if max_tl > 0 else None)

                    imsd_df, emsd_df, diff_df = compute_msd_and_fit(
                        tracks, px, fi,
                        max_lagtime=self.v_max_lagtime.get(),
                        n_fit=self.v_n_fit.get(),
                        workers=workers)

                    if self.v_filter_d_enabled.get():
                        d_min, d_max = self.v_filter_d_min.get(), self.v_filter_d_max.get()
                        mask      = diff_df["D"].between(d_min, d_max)
                        keep_pids = set(diff_df.loc[mask, "particle"])
                        diff_df   = diff_df[mask].reset_index(drop=True)
                        tracks    = tracks[tracks["particle"].isin(keep_pids)]

                    jdd = compute_jdd(tracks, px, fi,
                                      n_components=self.v_jdd_components.get())

                    b_ta = compute_turning_angles(tracks)
                    b_mf = compute_mobile_fraction_over_time(
                        tracks, diff_df, fi,
                        window_frames=max(50, int(0.1 * tracks["frame"].max()))
                        if len(tracks) else 100)

                    b_cluster_labels, b_cluster_stats_df, b_n_clusters, b_cluster_xy = compute_clusters(
                        locs, px,
                        eps_um=self.v_cluster_eps_nm.get() / 1000.0,
                        min_samples=self.v_cluster_min_samples.get())
                    _emit_log(f"  Clusters: {b_n_clusters} found")

                    b_dwell_df, b_dwell_tau = compute_dwell_times(tracks, diff_df, fi)

                    b_mss_df = compute_mss(tracks, px, fi,
                                           max_lagtime=self.v_max_lagtime.get())
                    if len(b_mss_df):
                        diff_df = diff_df.merge(
                            b_mss_df[["particle", "mss_slope"]], on="particle", how="left")

                    # Save CSVs
                    for df, suf in [(locs, "localisations"),
                                    (tracks, "trajectories"),
                                    (diff_df, "diffusion_summary")]:
                        df.to_csv(os.path.join(data_dir, f"{stem}_{suf}.csv"), index=False)

                    if len(b_cluster_stats_df):
                        b_cluster_stats_df.to_csv(
                            os.path.join(data_dir, f"{stem}_cluster_stats.csv"), index=False)

                    fig_path = os.path.join(fig_dir, f"{stem}_sptpalm_figure.png")
                    make_figure(proj_sample, tracks, imsd_df, emsd_df, diff_df,
                                px, fi, fig_path, roi_mask=roi_mask,
                                fig_theme=self.v_fig_theme.get(),
                                proj_cmap=self.v_proj_cmap.get(), jdd=jdd,
                                turning_angles=b_ta, mobile_frac_df=b_mf,
                                cluster_labels=b_cluster_labels,
                                cluster_locs=b_cluster_xy,
                                dwell_df=b_dwell_df,
                                dwell_tau=b_dwell_tau)

                    mc_  = diff_df["motion"].value_counts()
                    row  = {
                        "file":         fname,
                        "n_tracks":     diff_df.shape[0],
                        "median_D":     diff_df["D"].dropna().median(),
                        "mean_D":       diff_df["D"].dropna().mean(),
                        "median_alpha": diff_df["alpha"].dropna().median(),
                        "immobile_pct": 100 * mc_.get("Immobile", 0) / max(diff_df.shape[0], 1),
                        "confined_pct": 100 * mc_.get("Confined", 0) / max(diff_df.shape[0], 1),
                        "brownian_pct": 100 * mc_.get("Brownian", 0) / max(diff_df.shape[0], 1),
                        "directed_pct": 100 * mc_.get("Directed", 0) / max(diff_df.shape[0], 1),
                        "pixel_size":   px,
                        "frame_interval": fi,
                        "status":       "OK",
                    }
                    if jdd:
                        for k, (D, f) in enumerate(
                                zip(jdd["D_values"], jdd["fractions"])):
                            row[f"jdd_D{k+1}"]  = D
                            row[f"jdd_f{k+1}"]  = f
                    if "loc_precision_nm" in diff_df.columns:
                        row["median_loc_precision_nm"] = diff_df["loc_precision_nm"].dropna().median()
                    rows.append(row)
                    _emit_log(f"  ✓ Done — {diff_df.shape[0]:,} tracks, "
                              f"median D={diff_df['D'].dropna().median():.4f} µm²/s")

                except Exception as exc:
                    if stop.is_set():
                        raise
                    _emit_log(f"  ✗ Error: {exc}")
                    rows.append({"file": fname, "status": f"ERROR: {exc}"})

            # Save batch summary
            batch_root     = os.path.join(folder, "batch_results")
            batch_data_dir = os.path.join(batch_root, "data")
            batch_fig_dir  = os.path.join(batch_root, "figures")
            os.makedirs(batch_data_dir, exist_ok=True)
            os.makedirs(batch_fig_dir,  exist_ok=True)
            if rows:
                import pandas as _pd
                summary_path = os.path.join(batch_data_dir, "batch_summary.csv")
                _pd.DataFrame(rows).to_csv(summary_path, index=False)
                _emit_log(f"\nBatch summary -> {summary_path}")

            # Batch summary comparison plot
            try:
                ok_rows = [r for r in rows if r.get("status") == "OK"]
                if len(ok_rows) >= 2:
                    import matplotlib.pyplot as _bplt
                    _bplt.rcParams.update({"font.family": "sans-serif"})
                    names    = [os.path.splitext(r["file"])[0] for r in ok_rows]
                    mean_Ds  = [r.get("mean_D", 0) for r in ok_rows]
                    mob_pcts = [r.get("brownian_pct", 0) + r.get("directed_pct", 0)
                                for r in ok_rows]
                    x = np.arange(len(names))
                    fig_b, ax1 = _bplt.subplots(figsize=(max(6, len(names) * 1.2), 5))
                    fig_b.patch.set_facecolor("#161b22")
                    ax1.set_facecolor("#161b22")
                    ax1.bar(x, mean_Ds, color="#58a6ff", alpha=0.8, label="Mean D (µm²/s)")
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(names, rotation=30, ha="right",
                                        fontsize=9, color="#e6edf3")
                    ax1.set_ylabel("Mean D  (µm²/s)", color="#e6edf3", fontsize=10)
                    ax1.tick_params(axis="y", colors="#e6edf3")
                    for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")
                    ax2 = ax1.twinx()
                    ax2.plot(x, mob_pcts, "o-", color="#3fb950", lw=2, ms=6,
                             label="Mobile %")
                    ax2.set_ylabel("Mobile fraction (%)", color="#3fb950", fontsize=10)
                    ax2.tick_params(axis="y", colors="#3fb950")
                    ax2.set_ylim(0, 100)
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1 + h2, l1 + l2, fontsize=8, facecolor="#161b22",
                               edgecolor="#30363d", labelcolor="#e6edf3",
                               loc="upper right")
                    _bplt.title("Batch Summary", color="#e6edf3", fontsize=12)
                    _bplt.tight_layout()
                    _bplt.savefig(os.path.join(batch_fig_dir, "batch_summary_plot.png"),
                                  dpi=150, bbox_inches="tight",
                                  facecolor=fig_b.get_facecolor())
                    _bplt.savefig(os.path.join(batch_fig_dir, "batch_summary_plot.pdf"),
                                  bbox_inches="tight",
                                  facecolor=fig_b.get_facecolor())
                    _bplt.close(fig_b)
                    _emit_log(f"Batch summary plot -> {batch_fig_dir}")
            except Exception as _be:
                _emit_log(f"  Batch plot error: {_be}")

            _emit_progress(f"Batch complete — {len(files)} files", 100)
            self._q.put(("batch_done", {"out_dir": batch_root}))

        except Exception:
            self._q.put(("stopped", None))
        finally:
            sys.stdout = old_stdout

    # ── Run / worker / queue ──────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        fpath = self.v_file.get().strip()
        if not fpath or not os.path.isfile(fpath):
            messagebox.showerror(
                "No file selected",
                "Please select a valid CZI or TIF file before running.")
            return
        if self.v_roi_mode.get() == "Draw ROI" and self._drawn_roi_mask is None:
            messagebox.showerror(
                "No ROI drawn",
                "ROI mode is set to 'Draw ROI' but no polygon has been drawn.\n"
                "Open the ROI Editor and draw a region, or switch the ROI mode.")
            return

        self._running = True
        self._analysis_start = time.monotonic()
        self._stop_event.clear()
        self._run_btn.configure(text="■  Stop", command=self._on_stop,
                                style="Stop.TButton")
        self._show_progress_panel()
        self._status_var.set("Analysis running…")

        # Collapse settings panel then start worker
        def _start():
            threading.Thread(target=self._worker, daemon=True).start()
            self.after(100, self._poll)

        self._collapse_panel(callback=_start)

    def _worker(self):
        """Background thread — full analysis pipeline with stop support."""
        import numpy as np
        import time as _time
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        stop = self._stop_event  # local alias for speed

        class _Stopped(Exception):
            pass

        def _check_stop():
            if stop.is_set():
                raise _Stopped()

        def _emit_log(text: str):
            self._q.put(("log", text))

        def _emit_progress(msg: str, pct: int):
            self._q.put(("progress", (msg, pct)))

        # ── Helper: render any matplotlib Figure to an RGB numpy array ────────
        def _fig_to_rgb(fig):
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            return buf.reshape(h, w, 4)[..., :3]

        # ── Helper: send any matplotlib Figure as a preview ───────────────────
        _prev_last = [0.0]
        def _send_fig(fig, label=""):
            try:
                rgb = _fig_to_rgb(fig)
                self._q.put(("preview", (rgb, label)))
            except Exception:
                pass

        # Throttled localisation preview (per-chunk, max 3 fps)
        def _preview_cb(frame_idx, frame_img, xs, ys, total):
            now = _time.monotonic()
            if now - _prev_last[0] < 0.33:
                return
            _prev_last[0] = now
            try:
                img = np.asarray(frame_img, dtype=np.float32)
                h, w = img.shape
                step = max(1, int(np.ceil(max(h, w) / 512)))
                img  = img[::step, ::step]
                xs2  = np.asarray(xs, dtype=np.float32) / step
                ys2  = np.asarray(ys, dtype=np.float32) / step
                lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99.5))
                if hi > lo:
                    img = np.clip((img - lo) / (hi - lo), 0, 1)

                fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                ax  = fig.add_axes([0, 0, 1, 1])
                ax.set_facecolor("#09090e")
                ax.imshow(img, cmap="inferno", origin="upper",
                          vmin=0, vmax=1, interpolation="nearest")
                if len(xs2) > 0:
                    ax.scatter(xs2, ys2, s=55, facecolors="none",
                               edgecolors="#4ea8ff", linewidths=1.0, alpha=0.85)
                ax.set_axis_off()
                ax.set_title(f"Localising  —  frame {frame_idx:,} / {total:,}   "
                             f"({len(xs)} particles)",
                             color="#79c0ff", fontsize=9, pad=4)
                _send_fig(fig, f"frame {frame_idx:,} / {total:,}")
                fig.clear()
            except Exception:
                pass

        class _StdoutCapture:
            def write(self, text):
                if text.strip():
                    _emit_log(text.rstrip())
            def flush(self): pass

        old_stdout = sys.stdout
        sys.stdout = _StdoutCapture()

        try:
            _check_stop()
            base = (sys._MEIPASS if getattr(sys, "frozen", False)
                    else os.path.dirname(os.path.abspath(__file__)))
            if base not in sys.path:
                sys.path.insert(0, base)

            import gc

            from sptpalm_analysis import (
                load_file, preprocess_and_localise_adaptive,
                link_trajectories, compute_msd_and_fit, compute_jdd,
                compute_turning_angles, compute_mobile_fraction_over_time,
                compute_clusters, compute_dwell_times, compute_mss,
                build_roi_mask, apply_roi_mask, make_figure,
                correct_drift, _Cancelled,
            )

            fpath   = self.v_file.get()
            stem    = os.path.splitext(os.path.basename(fpath))[0]
            out_dir = (self.v_outdir.get().strip()
                       or os.path.dirname(os.path.abspath(fpath)))
            os.makedirs(out_dir, exist_ok=True)
            fig_dir  = os.path.join(out_dir, "figures")
            data_dir = os.path.join(out_dir, "data")
            os.makedirs(fig_dir,  exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)

            # Independent overrides
            px_arg = self.v_pixel_size.get()     if self.v_override_px.get() else None
            fi_arg = self.v_frame_interval.get() if self.v_override_fi.get() else None

            # Map display labels back to internal values expected by analysis code
            _bg_map   = {"Uniform Filter": "uniform_filter", "Rolling Ball": "rolling_ball"}
            _roi_map  = {"Auto": "auto", "Triangle": "triangle", "Li": "li", "Otsu": "otsu"}
            _mask_map = {"Mean": "mean", "Per Frame": "perframe"}
            bg_method_raw      = _bg_map.get(self.v_bg_method.get(), "uniform_filter")
            roi_auto_method_raw = _roi_map.get(self.v_roi_auto_method.get(), "auto")
            roi_mask_mode_raw  = _mask_map.get(self.v_roi_mask_mode.get(), "mean")

            roi_mode   = self.v_roi_mode.get()
            roi_auto   = roi_mode == "Auto threshold"
            roi_manual = roi_mode == "Manual threshold"
            roi_draw   = roi_mode == "Draw ROI"

            diameter = self.v_diameter.get()
            if diameter % 2 == 0:
                diameter += 1

            workers    = self.v_workers.get()
            chunk_size = self.v_chunk_size.get()

            # ── 1. Load ───────────────────────────────────────────────────────
            _emit_progress("Loading file…", 5)
            stack, meta_px, meta_fi = load_file(
                fpath, channel=self.v_channel.get(), stop_event=stop)
            n_frames = len(stack)
            px = px_arg or meta_px or 0.104
            fi = fi_arg or meta_fi or 0.050
            _emit_log(f"  Frames: {n_frames:,}  |  px={px} µm  fi={fi} s")

            # Preview: first raw frame
            try:
                raw0 = stack[0].astype(np.float32)
                lo, hi = float(np.percentile(raw0, 1)), float(np.percentile(raw0, 99.9))
                if hi > lo:
                    raw0 = np.clip((raw0 - lo) / (hi - lo), 0, 1)
                fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                ax  = fig.add_axes([0, 0, 1, 1])
                ax.set_facecolor("#09090e")
                ax.imshow(raw0, cmap="inferno", origin="upper", interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"Loaded  —  {n_frames:,} frames  |  {stack.shape[2]}×{stack.shape[1]} px",
                             color="#79c0ff", fontsize=9, pad=4)
                _send_fig(fig, "raw stack")
                fig.clear()
            except Exception:
                pass

            _check_stop()

            # ── 2. Sample frames for figure projection ─────────────────────────
            proj_idx    = np.linspace(0, n_frames - 1, min(200, n_frames), dtype=int)
            proj_sample = stack[proj_idx].copy()

            # ── 2b. ROI ────────────────────────────────────────────────────────
            roi_mask = None
            roi_path = None
            if roi_draw and self._drawn_roi_mask is not None:
                _emit_progress("Applying drawn ROI…", 12)
                roi_mask = self._drawn_roi_mask
                roi_path = os.path.join(data_dir, f"{stem}_roi_mask.png")
                try:
                    from PIL import Image as _PILImg
                    _PILImg.fromarray((roi_mask * 255).astype("uint8")).save(roi_path)
                except Exception:
                    roi_path = None
                # Preview: first raw frame + ROI overlay
                try:
                    raw0 = stack[0].astype(np.float32)
                    lo, hi = float(np.percentile(raw0, 1)), float(np.percentile(raw0, 99.9))
                    if hi > lo:
                        raw0 = np.clip((raw0 - lo) / (hi - lo), 0, 1)
                    fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                    ax  = fig.add_axes([0, 0, 1, 1])
                    ax.set_facecolor("#09090e")
                    ax.imshow(raw0, cmap="inferno", origin="upper")
                    ax.imshow(roi_mask, cmap="Blues", alpha=0.35,
                              origin="upper", vmin=0, vmax=1)
                    ax.set_axis_off()
                    ax.set_title("Drawn ROI  —  blue = included region",
                                 color="#79c0ff", fontsize=9, pad=4)
                    _send_fig(fig, "drawn ROI")
                    fig.clear()
                except Exception:
                    pass
            elif roi_auto or roi_manual:
                _emit_progress("Building ROI mask…", 12)
                raw_mean = stack.mean(axis=0)
                mn, mx   = raw_mean.min(), raw_mean.max()
                raw_mean_norm = (raw_mean - mn) / (mx - mn) if mx > mn else raw_mean
                del raw_mean; gc.collect()

                roi_path = os.path.join(data_dir, f"{stem}_roi_mask.png")
                thresh   = self.v_roi_threshold.get() if roi_manual else None
                roi_mask = build_roi_mask(
                    precomputed_mean_proj=raw_mean_norm,
                    threshold=thresh,
                    mode="mean",
                    threshold_method=(roi_auto_method_raw if roi_auto else "auto"),
                    save_path=roi_path)

                # Preview: mean projection + ROI overlay
                try:
                    fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                    ax  = fig.add_axes([0, 0, 1, 1])
                    ax.set_facecolor("#09090e")
                    ax.imshow(raw_mean_norm, cmap="inferno", origin="upper")
                    ax.imshow(roi_mask, cmap="Blues", alpha=0.35,
                              origin="upper", vmin=0, vmax=1)
                    ax.set_axis_off()
                    ax.set_title("ROI Mask  —  blue = included region",
                                 color="#79c0ff", fontsize=9, pad=4)
                    _send_fig(fig, "ROI mask")
                    fig.clear()
                except Exception:
                    pass

                del raw_mean_norm; gc.collect()
            _check_stop()

            # ── 3. Preprocess + localise ───────────────────────────────────────
            _emit_progress("Localising particles…", 20)
            minmass_arg = None if self.v_auto_mm.get() else self.v_minmass.get()
            locs, mean_proj, _minmass = preprocess_and_localise_adaptive(
                stack,
                diameter=diameter,
                minmass=minmass_arg,
                bg_radius=self.v_bg_radius.get(),
                bg_method=bg_method_raw,
                workers=workers,
                chunk_size=chunk_size,
                preview_cb=_preview_cb,
                stop_event=stop)

            del stack; gc.collect()
            _check_stop()

            if len(locs) == 0:
                raise RuntimeError(
                    "No particles found.\n"
                    "Try lowering Minmass or disabling Auto-detect.")

            if roi_mask is not None:
                locs = apply_roi_mask(locs, roi_mask)
                if len(locs) == 0:
                    raise RuntimeError(
                        "No localisations inside ROI.\n"
                        "Lower the ROI threshold or disable ROI masking.")

            # ── 3b. Drift correction ──────────────────────────────────────────
            if self.v_drift_correct.get() and len(locs) > 0:
                _emit_progress("Correcting drift…", 50)
                locs, drift_df = correct_drift(
                    locs, n_seg_frames=self.v_drift_segment.get())
                # Save drift trajectory alongside other outputs
                drift_df.to_csv(
                    os.path.join(data_dir, f"{stem}_drift.csv"), index=False)
                # Preview: drift trajectory plot
                try:
                    t_arr = drift_df["frame"].values * fi
                    fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                    ax  = fig.add_subplot(111)
                    ax.set_facecolor("#111318")
                    ax.plot(t_arr, drift_df["dx"].values,
                            color="#4ea8ff", lw=1.5, label="X drift")
                    ax.plot(t_arr, drift_df["dy"].values,
                            color="#f5a623", lw=1.5, label="Y drift")
                    ax.axhline(0, color="#252d3a", lw=0.8, ls="--")
                    ax.set_xlabel("Time (s)", color="#7a8899", fontsize=9)
                    ax.set_ylabel("Drift (px)", color="#7a8899", fontsize=9)
                    ax.tick_params(colors="#7a8899", labelsize=8)
                    for s in ax.spines.values():
                        s.set_color("#252d3a")
                    ax.legend(fontsize=8, facecolor="#111318",
                              edgecolor="#252d3a", labelcolor="#dce6f0")
                    ax.set_title("Drift trajectory (corrected)",
                                 color="#79c0ff", fontsize=9)
                    fig.tight_layout(pad=1.0)
                    _send_fig(fig, "drift correction")
                    fig.clear()
                except Exception:
                    pass
            _check_stop()

            # ── 4. Link ───────────────────────────────────────────────────────
            _emit_progress("Linking trajectories…", 55)
            max_tl = self.v_max_track_len.get()
            tracks = link_trajectories(
                locs,
                search_range=self.v_search_range.get(),
                memory=self.v_memory.get(),
                min_len=self.v_min_track_len.get(),
                max_len=max_tl if max_tl > 0 else None)

            if tracks["particle"].nunique() == 0:
                raise RuntimeError(
                    "No trajectories after filtering.\n"
                    "Try lowering Min track length.")

            # Preview: trajectory overlay on mean projection
            try:
                fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                ax  = fig.add_axes([0, 0, 1, 1])
                ax.set_facecolor("#09090e")
                ax.imshow(mean_proj, cmap="inferno", origin="upper",
                          vmin=0, vmax=1, interpolation="nearest")
                n_show = min(500, tracks["particle"].nunique())
                pids   = tracks["particle"].unique()[:n_show]
                cmap   = __import__("matplotlib").colormaps["cool"]
                for i, pid in enumerate(pids):
                    t = tracks[tracks["particle"] == pid]
                    col = cmap(i / max(len(pids) - 1, 1))
                    ax.plot(t["x"].values, t["y"].values,
                            lw=0.7, alpha=0.6, color=col)
                ax.set_axis_off()
                n_total = tracks["particle"].nunique()
                ax.set_title(
                    f"Linked  —  {n_total:,} trajectories  (showing {n_show})",
                    color="#79c0ff", fontsize=9, pad=4)
                _send_fig(fig, f"{n_total:,} trajectories")
                fig.clear()
            except Exception:
                pass

            _check_stop()

            # ── 5. MSD + diffusion ─────────────────────────────────────────────
            _emit_progress("Computing MSD & diffusion…", 70)
            imsd_df, emsd_df, diff_df = compute_msd_and_fit(
                tracks, px, fi,
                max_lagtime=self.v_max_lagtime.get(),
                n_fit=self.v_n_fit.get(),
                workers=workers)

            # Preview: ensemble MSD curve
            try:
                lag_times = emsd_df.index.values * fi
                fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                ax  = fig.add_subplot(111)
                ax.set_facecolor("#111318")
                ax.plot(lag_times, emsd_df.values, color="#4ea8ff", lw=2)
                ax.set_xlabel("Lag time (s)", color="#7a8899", fontsize=9)
                ax.set_ylabel("MSD (µm²)", color="#7a8899", fontsize=9)
                ax.tick_params(colors="#7a8899", labelsize=8)
                for s in ax.spines.values():
                    s.set_color("#252d3a")
                fig.patch.set_facecolor("#09090e")
                ax.set_title("Ensemble MSD", color="#79c0ff", fontsize=9)
                fig.tight_layout(pad=1.0)
                _send_fig(fig, "ensemble MSD")
                fig.clear()
            except Exception:
                pass

            # ── 5b. D-value filter ────────────────────────────────────────────
            if self.v_filter_d_enabled.get():
                d_min = self.v_filter_d_min.get()
                d_max = self.v_filter_d_max.get()
                mask      = diff_df["D"].between(d_min, d_max, inclusive="both")
                keep_pids = set(diff_df.loc[mask, "particle"])
                n_before  = diff_df.shape[0]
                diff_df   = diff_df[mask].reset_index(drop=True)
                tracks    = tracks[tracks["particle"].isin(keep_pids)]
                _emit_log(f"  D filter [{d_min:.4f}–{d_max:.4f} µm²/s]: "
                          f"{len(keep_pids):,} of {n_before:,} tracks kept")
                if diff_df.empty:
                    raise RuntimeError(
                        "No tracks remain after D-value filter.\n"
                        "Widen the D min/max range or disable the filter.")

            _check_stop()

            # ── 5c. Jump Distance Distribution ────────────────────────────────
            _emit_progress("Jump Distance Distribution…", 80)
            jdd = compute_jdd(tracks, px, fi,
                              n_components=self.v_jdd_components.get())
            if jdd:
                _emit_log(f"  JDD  ({jdd['n_components']} populations, "
                          f"{jdd['n_jumps']:,} jumps):")
                labels = ["Slow", "Medium", "Fast"]
                for k, (D, f) in enumerate(
                        zip(jdd["D_values"], jdd["fractions"])):
                    _emit_log(f"    {labels[k]:6s}  D={D:.4f} µm²/s  "
                              f"({f*100:.1f}%)")
            else:
                _emit_log("  JDD: too few jumps to fit.")
                jdd = None

            turning_angles = compute_turning_angles(tracks)
            mobile_frac_df = compute_mobile_fraction_over_time(
                tracks, diff_df, fi,
                window_frames=max(50, int(0.1 * tracks["frame"].max())) if len(tracks) else 100)

            _emit_progress("Cluster analysis…", 88)
            cluster_labels, cluster_stats_df, n_clusters, cluster_xy = compute_clusters(
                locs, px,
                eps_um=self.v_cluster_eps_nm.get() / 1000.0,
                min_samples=self.v_cluster_min_samples.get())
            _emit_log(f"  Clusters: {n_clusters} found")

            dwell_df, dwell_tau = compute_dwell_times(tracks, diff_df, fi)
            if len(dwell_df):
                _emit_log(f"  Dwell time: τ = {dwell_tau:.3f} s  (n={len(dwell_df)} confined/immobile tracks)"
                          if np.isfinite(dwell_tau) else
                          f"  Dwell time: {len(dwell_df)} confined/immobile tracks (too few to fit τ)")

            mss_df = compute_mss(tracks, px, fi, max_lagtime=self.v_max_lagtime.get())
            if len(mss_df):
                diff_df = diff_df.merge(mss_df[["particle", "mss_slope"]], on="particle", how="left")
                _emit_log(f"  MSS: computed for {mss_df.shape[0]:,} tracks")

            _check_stop()

            # ── 6. Save ───────────────────────────────────────────────────────
            _emit_progress("Saving outputs…", 92)
            fig_path = os.path.join(fig_dir, f"{stem}_sptpalm_figure.png")
            make_figure(proj_sample, tracks, imsd_df, emsd_df, diff_df,
                        px, fi, fig_path, roi_mask=roi_mask,
                        fig_theme=self.v_fig_theme.get(),
                        proj_cmap=self.v_proj_cmap.get(),
                        jdd=jdd,
                        turning_angles=turning_angles,
                        mobile_frac_df=mobile_frac_df,
                        cluster_labels=cluster_labels,
                        cluster_locs=cluster_xy,
                        dwell_df=dwell_df,
                        dwell_tau=dwell_tau)
            del proj_sample; gc.collect()

            # Preview: the final saved figure
            try:
                from PIL import Image as _PILImage
                final_img = np.asarray(_PILImage.open(fig_path).convert("RGB"))
                fig = Figure(figsize=(5, 4), dpi=90, facecolor="#09090e")
                ax  = fig.add_axes([0, 0, 1, 1])
                ax.imshow(final_img, interpolation="lanczos")
                ax.set_axis_off()
                ax.set_title("Analysis complete", color="#2ea043", fontsize=9, pad=4)
                _send_fig(fig, "complete")
                fig.clear()
            except Exception:
                pass

            locs.to_csv(
                os.path.join(data_dir, f"{stem}_localisations.csv"), index=False)
            tracks.to_csv(
                os.path.join(data_dir, f"{stem}_trajectories.csv"), index=False)
            diff_df.to_csv(
                os.path.join(data_dir, f"{stem}_diffusion_summary.csv"), index=False)
            (emsd_df.to_frame("msd_um2")
                    .reset_index(names="lag_frame")
                    .to_csv(os.path.join(data_dir, f"{stem}_ensemble_msd.csv"),
                            index=False))
            if len(cluster_stats_df):
                cluster_stats_df.to_csv(
                    os.path.join(data_dir, f"{stem}_cluster_stats.csv"), index=False)

            _emit_progress("Complete!", 100)
            self._q.put(("done", {
                "diff_df": diff_df, "fig_path": fig_path,
                "roi_path": roi_path, "out_dir": out_dir, "stem": stem,
                "jdd": jdd,
            }))

        except (_Stopped, _Cancelled):
            self._q.put(("stopped", None))
        except Exception:
            self._q.put(("error", traceback.format_exc()))
        finally:
            sys.stdout = old_stdout

    def _poll(self):
        try:
            latest_preview = None
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    msg, pct = payload
                    self._prog_bar["value"] = pct
                    self._prog_label.configure(text=msg)
                    self._status_var.set(msg)
                elif kind == "preview":
                    latest_preview = payload   # coalesce: only render the newest
                elif kind == "done":
                    if latest_preview:
                        self._update_preview(*latest_preview)
                    self._on_done(payload)
                    return
                elif kind == "batch_done":
                    self._on_batch_done(payload)
                    return
                elif kind == "stopped":
                    self._on_stopped()
                    return
                elif kind == "error":
                    self._on_error(payload)
                    return
        except queue.Empty:
            pass

        if latest_preview is not None:
            self._update_preview(*latest_preview)

        if self._running:
            self.after(100, self._poll)

    def _reset_run_btn(self):
        """Restore button to Run mode after analysis finishes or is stopped."""
        self._run_btn.configure(state="normal",
                                text="▶  Run Analysis",
                                style="Run.TButton",
                                command=self._on_run)
        self._batch_btn.configure(state="normal")

    def _on_batch_done(self, data: dict):
        self._running = False
        self._stop_live_timers()
        self._reset_run_btn()
        out_dir = data.get("out_dir", "")
        self._status_var.set("Batch complete!")
        self._prog_label.configure(text="Batch processing finished.")
        self._expand_panel()
        messagebox.showinfo(
            "Batch Complete",
            f"All files processed.\n\nResults saved to:\n{out_dir}")

    def _on_done(self, data: dict):
        self._running = False
        self._stop_live_timers()
        self._reset_run_btn()
        self._status_var.set("Analysis complete!")
        self._show_results(data)
        self._expand_panel()

    def _on_stopped(self):
        self._running = False
        self._stop_live_timers()
        self._reset_run_btn()
        self._status_var.set("Stopped by user")
        self._prog_label.configure(text="Analysis cancelled.")
        self._expand_panel()

    def _on_error(self, tb: str):
        self._running = False
        self._stop_live_timers()
        self._reset_run_btn()
        self._status_var.set("Error — see details below")
        self._expand_panel()
        messagebox.showerror(
            "Analysis Error",
            f"An error occurred during analysis:\n\n"
            + tb.strip()[-600:])


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _open_folder(path: str) -> None:
    """Open path in the system file manager."""
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", os.path.normpath(path)], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def main():
    app = SPTPalmApp()
    app.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()   # must be first — prevents double-window on PyInstaller + spawn
    main()
