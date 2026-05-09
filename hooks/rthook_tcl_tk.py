"""
Runtime hook: ensure Tcl/Tk data is locatable before the default
pyi_rthtkinter hook runs.

The default PyInstaller tkinter rthook hard-codes the path
sys._MEIPASS/_tcl_data and crashes with FileNotFoundError if that exact
directory is missing. This hook runs FIRST and:

  1. Scans the bundle for any Tcl/Tk library (looks for init.tcl / tk.tcl)
  2. If found at a non-standard location, creates _tcl_data / _tk_data
     pointing at it (junction on Windows, symlink elsewhere, copy as fallback)
  3. Sets TCL_LIBRARY / TK_LIBRARY env vars as a belt-and-suspenders measure

This makes the app resilient to packaging variations across PyInstaller
versions, build environments, and any future changes to the layout.
"""
import os
import sys


def _find_dir_containing(base, marker_filename):
    """Walk `base` and return the first directory containing `marker_filename`."""
    if not os.path.isdir(base):
        return None
    for root, dirs, files in os.walk(base):
        if marker_filename in files:
            return root
    return None


def _ensure_link(src, dst):
    """Make `dst` resolve to `src`. Try junction (Windows), then symlink, then copy."""
    if os.path.isdir(dst):
        return True
    if not os.path.isdir(src):
        return False

    # Windows: directory junction (no admin needed, no symlink permission)
    if sys.platform == "win32":
        try:
            import subprocess
            subprocess.check_call(
                ["cmd", "/c", "mklink", "/J", dst, src],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.isdir(dst):
                return True
        except Exception:
            pass

    # POSIX or Windows fallback: symlink
    try:
        os.symlink(src, dst, target_is_directory=True)
        if os.path.isdir(dst):
            return True
    except Exception:
        pass

    # Last resort: full directory copy
    try:
        import shutil
        shutil.copytree(src, dst)
        return os.path.isdir(dst)
    except Exception:
        return False


def _setup():
    base = getattr(sys, "_MEIPASS", None)
    if not base or not os.path.isdir(base):
        return  # not a frozen app

    tcl_data = os.path.join(base, "_tcl_data")
    tk_data = os.path.join(base, "_tk_data")

    # If _tcl_data is missing, search the bundle for init.tcl
    if not os.path.isdir(tcl_data):
        found = _find_dir_containing(base, "init.tcl")
        if found and found != tcl_data:
            _ensure_link(found, tcl_data)

    # Same for _tk_data → tk.tcl
    if not os.path.isdir(tk_data):
        found = _find_dir_containing(base, "tk.tcl")
        # tk.tcl can also live inside the tcl tree — make sure we pick the tk one
        if found and "tk" in os.path.basename(found).lower() and found != tk_data:
            _ensure_link(found, tk_data)
        elif found and found != tk_data:
            # Fallback: any tk.tcl
            _ensure_link(found, tk_data)

    # Belt-and-suspenders: env vars for any code path that respects them
    if os.path.isdir(tcl_data):
        os.environ["TCL_LIBRARY"] = tcl_data
    if os.path.isdir(tk_data):
        os.environ["TK_LIBRARY"] = tk_data


_setup()
