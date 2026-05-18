"""
FIREFLY CUDA-torch sidecar installer.

The Windows .exe ships with a CPU-only PyTorch build because the CUDA
torch wheel (~2.5 GB) exceeds GitHub Releases' 2 GiB asset cap.  On
first launch on Windows we detect an NVIDIA GPU and offer to download
the matching CUDA torch wheel into %LOCALAPPDATA%\\FIREFLY\\torch-cuda
on demand.  On subsequent launches we prepend the extracted sidecar to
sys.path so `import torch` resolves to the CUDA build, shadowing the
bundled CPU build.

This module is pure stdlib — no PySide6 dependency — so it can be
imported safely from anywhere in the app (including before the Qt
event loop exists and before any torch import).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Callable, Optional


# ── Platform helpers ──────────────────────────────────────────────────────────
def is_windows() -> bool:
    return sys.platform == "win32"


def _no_window_kwargs() -> dict:
    """subprocess kwargs that suppress the brief cmd.exe flash on Windows."""
    if not is_windows():
        return {}
    # CREATE_NO_WINDOW = 0x08000000 — defined in subprocess only on Windows
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return {"creationflags": flags}


# ── GPU detection ─────────────────────────────────────────────────────────────
def detect_nvidia_gpu() -> Optional[str]:
    """Return the first NVIDIA GPU name reported by nvidia-smi, or None.

    Uses a 5 s timeout.  Suppresses the cmd.exe window flash on Windows.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            **_no_window_kwargs(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    name = line[0].strip()
    return name or None


# ── Filesystem layout ─────────────────────────────────────────────────────────
def sidecar_dir() -> str:
    """%LOCALAPPDATA%\\FIREFLY\\torch-cuda on Windows, ~/.firefly/torch-cuda
    elsewhere (dev/testing only).  Parent dirs are created on demand."""
    if is_windows():
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        path = os.path.join(base, "FIREFLY", "torch-cuda")
    else:
        path = os.path.join(os.path.expanduser("~"), ".firefly", "torch-cuda")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def sidecar_extracted_dir() -> str:
    return os.path.join(sidecar_dir(), "extracted")


def is_installed() -> bool:
    try:
        return os.path.isfile(
            os.path.join(sidecar_extracted_dir(), "torch", "__init__.py"))
    except Exception:
        return False


# ── User-declined flag ────────────────────────────────────────────────────────
def settings_path() -> str:
    return os.path.join(sidecar_dir(), "state.json")


def _read_state() -> dict:
    try:
        with open(settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
        with open(settings_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except Exception:
        pass


def user_declined() -> bool:
    return bool(_read_state().get("declined", False))


def mark_declined() -> None:
    state = _read_state()
    state["declined"] = True
    _write_state(state)


def clear_declined() -> None:
    state = _read_state()
    state.pop("declined", None)
    _write_state(state)


# ── Torch version / URL building ──────────────────────────────────────────────
def bundled_torch_version() -> Optional[str]:
    """Return the base version of the currently-imported torch (e.g. '2.5.1'),
    stripping any '+cpu' / '+cu124' local-version suffix.  None if torch
    can't be imported."""
    try:
        import torch  # noqa: F401  — safe; we just read __version__
        ver = getattr(torch, "__version__", "") or ""
    except Exception:
        return None
    if not ver:
        return None
    # PEP 440 local-version separator is '+'
    base = ver.split("+", 1)[0].strip()
    return base or None


def cuda_wheel_url(torch_version: str, cuda_tag: str = "cu124",
                   python_tag: Optional[str] = None) -> str:
    """Build the CUDA wheel URL on download.pytorch.org.

    Example:
        torch_version='2.5.1', cuda_tag='cu124', python_tag='cp312'
        → https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl
    """
    if python_tag is None:
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # '+' must be URL-encoded as %2B in the local-version segment.
    filename = (
        f"torch-{torch_version}%2B{cuda_tag}"
        f"-{python_tag}-{python_tag}-win_amd64.whl"
    )
    return f"https://download.pytorch.org/whl/{cuda_tag}/{filename}"


# ── Download / extract ────────────────────────────────────────────────────────
def download_wheel(url: str,
                   dest_path: str,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   cancel_cb: Optional[Callable[[], bool]] = None) -> None:
    """Stream-download `url` to `dest_path` with optional progress and
    cancellation callbacks.  Raises RuntimeError with user-facing wording
    on failure.

    progress_cb(downloaded, total) is called every ~64 KB.  total may be
    0 if the server omits Content-Length.
    cancel_cb() is polled regularly; returning True aborts the download
    and removes the partial file.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    chunk_size = 64 * 1024
    # Remove any stale partial from a prior attempt
    try:
        if os.path.exists(dest_path):
            os.remove(dest_path)
    except Exception:
        pass

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "FIREFLY-CUDA-installer/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            try:
                total = int(resp.headers.get("Content-Length") or 0)
            except Exception:
                total = 0
            downloaded = 0
            with open(dest_path, "wb") as out:
                while True:
                    if cancel_cb is not None:
                        try:
                            if cancel_cb():
                                # Cancelled — clean up the partial file
                                try:
                                    out.close()
                                except Exception:
                                    pass
                                try:
                                    os.remove(dest_path)
                                except Exception:
                                    pass
                                raise RuntimeError(
                                    "Download cancelled by user.")
                        except RuntimeError:
                            raise
                        except Exception:
                            pass
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        try:
                            progress_cb(downloaded, total)
                        except Exception:
                            pass
    except urllib.error.HTTPError as exc:
        try:
            os.remove(dest_path)
        except Exception:
            pass
        raise RuntimeError(
            f"Server returned HTTP {exc.code} when downloading the CUDA "
            f"PyTorch wheel.  The exact build for this Python/torch "
            f"combination may not be available — please report this URL: "
            f"{url}"
        ) from exc
    except urllib.error.URLError as exc:
        try:
            os.remove(dest_path)
        except Exception:
            pass
        raise RuntimeError(
            "Could not reach download.pytorch.org.  Check your internet "
            "connection or proxy settings and try again."
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        try:
            os.remove(dest_path)
        except Exception:
            pass
        raise RuntimeError(
            f"Unexpected error while downloading the CUDA PyTorch wheel: "
            f"{exc}"
        ) from exc


def extract_wheel(wheel_path: str,
                  dest_dir: str,
                  progress_cb: Optional[Callable[[int, int], None]] = None
                  ) -> None:
    """Extract the .whl (a zip) into dest_dir.  progress_cb(done, total)
    is invoked every ~50 files."""
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            total = len(names)
            for i, name in enumerate(names, start=1):
                zf.extract(name, dest_dir)
                if progress_cb is not None and (i % 50 == 0 or i == total):
                    try:
                        progress_cb(i, total)
                    except Exception:
                        pass
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "The downloaded CUDA PyTorch wheel is corrupt.  Please try "
            "again."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not extract the CUDA PyTorch wheel: {exc}"
        ) from exc


# ── End-to-end installer ──────────────────────────────────────────────────────
def install_cuda_torch(cuda_tag: str = "cu124",
                       download_progress_cb=None,
                       extract_progress_cb=None,
                       cancel_cb=None) -> None:
    """Download + extract the CUDA torch wheel matching the currently-
    bundled torch version into the sidecar directory.

    Raises RuntimeError with user-facing wording on any failure.
    """
    ver = bundled_torch_version()
    if not ver:
        raise RuntimeError(
            "Could not determine the bundled PyTorch version.  Cannot "
            "install CUDA acceleration without a matching version.")

    url = cuda_wheel_url(ver, cuda_tag=cuda_tag)
    sd = sidecar_dir()
    wheel_path = os.path.join(sd, f"torch-{ver}+{cuda_tag}.whl")
    extracted = sidecar_extracted_dir()

    # If a previous partial extraction is sitting in place, blow it away
    # so we start clean.
    try:
        if os.path.isdir(extracted):
            shutil.rmtree(extracted, ignore_errors=True)
    except Exception:
        pass

    download_wheel(url, wheel_path,
                   progress_cb=download_progress_cb, cancel_cb=cancel_cb)

    # Honour cancellation between phases too
    if cancel_cb is not None:
        try:
            if cancel_cb():
                try:
                    os.remove(wheel_path)
                except Exception:
                    pass
                raise RuntimeError("Installation cancelled by user.")
        except RuntimeError:
            raise
        except Exception:
            pass

    extract_wheel(wheel_path, extracted, progress_cb=extract_progress_cb)

    # Wheel is no longer needed once extracted — reclaim ~2.5 GB
    try:
        os.remove(wheel_path)
    except Exception:
        pass

    if not is_installed():
        raise RuntimeError(
            "Extraction completed but torch/__init__.py is missing in "
            "the sidecar directory.  The wheel layout may be unexpected.")


# ── sys.path injection ────────────────────────────────────────────────────────
def inject_sidecar_into_sys_path() -> None:
    """Prepend the sidecar's extracted directory to sys.path so subsequent
    `import torch` resolves to the CUDA build.  No-op on non-Windows or
    when the sidecar isn't installed.  Idempotent.

    MUST be called BEFORE any `import torch` anywhere in the process.
    """
    if not is_windows():
        return
    try:
        if not is_installed():
            return
        target = sidecar_extracted_dir()
        # Drop any stale entry first, then put it at the very front so it
        # shadows the bundled CPU torch.
        try:
            while target in sys.path:
                sys.path.remove(target)
        except Exception:
            pass
        sys.path.insert(0, target)
    except Exception:
        # Disk / permissions errors here must NOT crash FIREFLY at startup.
        pass


# ── Uninstall ─────────────────────────────────────────────────────────────────
def uninstall() -> None:
    """Remove the sidecar directory.  Used to clean up or change CUDA
    versions."""
    try:
        sd = sidecar_dir()
        if os.path.isdir(sd):
            shutil.rmtree(sd, ignore_errors=True)
    except Exception:
        pass
