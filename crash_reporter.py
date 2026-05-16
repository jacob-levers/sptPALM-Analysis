"""
FIREFLY crash reporter.

Writes detailed crash reports for any uncaught exception (main thread,
background threads, Tk callbacks, analysis worker) so the user can hand the
file straight to the developer when something blows up.  Reports include:

    • Timestamp, FIREFLY version, Python version, OS / arch, frozen?
    • CPU count and total / available RAM (via psutil if installed)
    • Versions of all relevant scientific-Python deps
    • Whether PyTorch and an MPS / CUDA device are visible
    • Last N lines from the in-app log (if the app handed them to us)
    • Snapshot of app state (current file, pixel size, frame interval, etc.)
    • Full traceback with local variables in the deepest frame

Reports are saved to:

    macOS    ~/Library/Logs/FIREFLY/crash_reports/
    Windows  %LOCALAPPDATA%/FIREFLY/crash_reports/
    Linux    ~/.local/share/FIREFLY/crash_reports/

The user is shown the path via a non-blocking message and can copy / open it.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import sys
import threading
import traceback
import typing as _t

__all__ = [
    "crash_report_dir",
    "write_crash_report",
    "install_global_handlers",
    "set_app_state_provider",
    "set_log_provider",
]

# ── Configurable hooks the host app installs ──────────────────────────────────
# `_app_state_provider()`  → returns a dict of free-form key/value snapshots
# `_log_provider(n=80)`    → returns the last n lines of in-app log text
_app_state_provider: _t.Optional[_t.Callable[[], dict]] = None
_log_provider:       _t.Optional[_t.Callable[[int], str]] = None


def set_app_state_provider(fn: _t.Callable[[], dict]) -> None:
    """Register a callable returning the current app-state snapshot dict."""
    global _app_state_provider
    _app_state_provider = fn


def set_log_provider(fn: _t.Callable[[int], str]) -> None:
    """Register a callable returning the last `n` lines of the in-app log."""
    global _log_provider
    _log_provider = fn


# ── Where reports live ────────────────────────────────────────────────────────
def crash_report_dir() -> str:
    """Return the platform-appropriate crash-report directory; create on first use."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Logs/FIREFLY/crash_reports")
    elif sys.platform == "win32":
        base = os.path.join(
            os.environ.get("LOCALAPPDATA",
                           os.path.expanduser("~\\AppData\\Local")),
            "FIREFLY", "crash_reports")
    else:
        base = os.path.expanduser("~/.local/share/FIREFLY/crash_reports")
    os.makedirs(base, exist_ok=True)
    return base


# ── Helpers ───────────────────────────────────────────────────────────────────
def _firefly_version() -> str:
    # Single source of truth: read from a VERSION constant if present
    try:
        import sptpalm_analysis as _sa
        v = getattr(_sa, "__version__", None)
        if v: return str(v)
    except Exception:
        pass
    return "unknown"


def _package_versions() -> dict[str, str]:
    """Best-effort import + read of __version__ for relevant scientific deps."""
    names = ["numpy", "scipy", "pandas", "trackpy", "matplotlib",
             "skimage", "sklearn", "PIL", "tifffile", "aicspylibczi",
             "imagecodecs", "joblib", "psutil", "torch", "tkinter",
             "tkinterdnd2"]
    out: dict[str, str] = {}
    for n in names:
        try:
            mod = __import__(n)
            v = getattr(mod, "__version__", None)
            if v is None and hasattr(mod, "VERSION"):
                v = str(mod.VERSION)
            if v is None:
                v = "installed"
            out[n] = str(v)
        except Exception:
            out[n] = "(not installed)"
    return out


def _torch_devices() -> str:
    try:
        import torch
        bits = [f"PyTorch {torch.__version__}"]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            bits.append("MPS available")
        if torch.cuda.is_available():
            bits.append(f"CUDA available ({torch.cuda.device_count()} dev)")
        if len(bits) == 1:
            bits.append("CPU only")
        return ", ".join(bits)
    except Exception:
        return "(PyTorch not installed)"


def _memory_info() -> str:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"{vm.total/1024**3:.1f} GB total, {vm.available/1024**3:.1f} GB available"
    except Exception:
        return "(psutil not installed)"


def _format_locals(tb: _t.Optional[object]) -> str:
    """Return repr() of locals at the deepest frame.  Best-effort, never raises."""
    if tb is None:
        return "(no traceback)"
    try:
        # Walk to the deepest frame
        last = tb
        while getattr(last, "tb_next", None) is not None:
            last = last.tb_next
        frame = last.tb_frame
        rows = []
        for name, val in list(frame.f_locals.items())[:50]:
            try:
                rep = repr(val)
            except Exception as e:
                rep = f"<unreprable: {e!r}>"
            if len(rep) > 400:
                rep = rep[:400] + " …(truncated)"
            rows.append(f"  {name} = {rep}")
        return "\n".join(rows) if rows else "  (no locals)"
    except Exception as e:
        return f"  (failed to read locals: {e!r})"


# ── Main entrypoint ───────────────────────────────────────────────────────────
def write_crash_report(exc_type, exc_value, exc_tb,
                       *,
                       source: str = "uncaught exception",
                       context: _t.Optional[str] = None) -> str:
    """Write a crash report file and return its absolute path."""
    ts = _dt.datetime.now()
    fname = f"firefly_crash_{ts.strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(crash_report_dir(), fname)

    lines: list[str] = []
    a = lines.append

    a("FIREFLY Crash Report")
    a("=" * 72)
    a(f"Time:           {ts.isoformat(timespec='seconds')}")
    a(f"Source:         {source}")
    a(f"FIREFLY:        {_firefly_version()}")
    a(f"Python:         {platform.python_version()} ({platform.python_implementation()})")
    a(f"Platform:       {platform.platform()}")
    a(f"Arch:           {platform.machine()}")
    a(f"Frozen:         {'yes (PyInstaller)' if getattr(sys, 'frozen', False) else 'no (source)'}")
    a(f"Executable:     {sys.executable}")
    a(f"PID:            {os.getpid()}")
    a(f"CWD:            {os.getcwd()}")
    a("")

    a("Hardware")
    a("-" * 72)
    a(f"CPUs:           {os.cpu_count()}")
    a(f"Memory:         {_memory_info()}")
    a(f"PyTorch:        {_torch_devices()}")
    a("")

    a("Packages")
    a("-" * 72)
    for name, ver in _package_versions().items():
        a(f"  {name:<18} {ver}")
    a("")

    # App state snapshot (if provided)
    if _app_state_provider is not None:
        try:
            state = _app_state_provider() or {}
        except Exception as e:
            state = {"<provider error>": repr(e)}
        if state:
            a("App State")
            a("-" * 72)
            for k, v in state.items():
                a(f"  {k:<22} {v}")
            a("")

    # Recent log
    if _log_provider is not None:
        try:
            log_text = _log_provider(120)
        except Exception as e:
            log_text = f"(log provider error: {e!r})"
        if log_text:
            a("Recent log (last 120 lines)")
            a("-" * 72)
            a(log_text.rstrip())
            a("")

    if context:
        a("Context")
        a("-" * 72)
        a(context.rstrip())
        a("")

    # Traceback
    a("Traceback")
    a("-" * 72)
    if exc_tb is None and exc_value is not None:
        exc_tb = getattr(exc_value, "__traceback__", None)
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    a(tb_str.rstrip())
    a("")

    a("Locals at deepest frame")
    a("-" * 72)
    a(_format_locals(exc_tb))
    a("")

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except Exception:
        # Fallback: write to /tmp so we never lose the report
        import tempfile
        path = os.path.join(tempfile.gettempdir(), fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    # Echo to stderr for terminal users / log capture
    try:
        sys.stderr.write(f"\n[FIREFLY] Crash report written to:\n  {path}\n\n")
        sys.stderr.write(tb_str)
        sys.stderr.flush()
    except Exception:
        pass

    return path


# ── Global hooks ──────────────────────────────────────────────────────────────
_original_excepthook        = sys.excepthook
_original_threading_hook    = getattr(threading, "excepthook", None)
_on_crash_callback: _t.Optional[_t.Callable[[str], None]] = None


def _main_excepthook(exc_type, exc_value, exc_tb):
    # Don't bother for clean Ctrl-C
    if issubclass(exc_type, KeyboardInterrupt):
        _original_excepthook(exc_type, exc_value, exc_tb)
        return
    try:
        path = write_crash_report(exc_type, exc_value, exc_tb,
                                  source="main thread")
        if _on_crash_callback is not None:
            try: _on_crash_callback(path)
            except Exception: pass
    finally:
        _original_excepthook(exc_type, exc_value, exc_tb)


def _thread_excepthook(args):
    try:
        path = write_crash_report(
            args.exc_type, args.exc_value, args.exc_traceback,
            source=f"thread '{getattr(args.thread, 'name', '?')}'")
        if _on_crash_callback is not None:
            try: _on_crash_callback(path)
            except Exception: pass
    finally:
        if _original_threading_hook is not None:
            _original_threading_hook(args)


def install_global_handlers(on_crash: _t.Optional[_t.Callable[[str], None]] = None) -> None:
    """Install sys.excepthook and threading.excepthook.

    `on_crash(path)`  is invoked with the written report path after each
    crash — use it to show a dialog from the host application.
    """
    global _on_crash_callback
    _on_crash_callback = on_crash
    sys.excepthook = _main_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
