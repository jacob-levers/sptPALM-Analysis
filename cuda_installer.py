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
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable, Optional


# ── Diagnostic log plumbing ───────────────────────────────────────────────────
# When a user reports "it gets stuck", we need a step-by-step breadcrumb
# trail of what the installer was doing.  Modules outside cuda_installer
# can register a callback via set_log_callback(); every call to _log()
# inside this module forwards there in addition to stdout.
_log_cb: Optional[Callable[[str], None]] = None
_log_t0: float = 0.0


def set_log_callback(cb: Optional[Callable[[str], None]]) -> None:
    """Register a callable that receives each diagnostic line.  Pass
    None to clear.  Lines are also always printed to stdout."""
    global _log_cb, _log_t0
    _log_cb = cb
    _log_t0 = time.monotonic()


def _log(msg: str) -> None:
    """Emit a timestamped diagnostic line."""
    elapsed = time.monotonic() - _log_t0 if _log_t0 else 0.0
    line = f"[+{elapsed:5.2f}s] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    if _log_cb is not None:
        try:
            _log_cb(line)
        except Exception:
            pass


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
    # Bigger chunks → fewer read() syscalls AND fewer progress signals
    # queued to the GUI thread.  At ~50 MB/s on a 64 KB chunk that's
    # ~800 signal emissions per second, which overwhelms Qt's event
    # queue and starves paint events — Windows then marks the app
    # "Not Responding" even though the download is fine.  256 KB cuts
    # that to ~200/s and we additionally throttle progress_cb to
    # ~10 Hz below.
    chunk_size = 256 * 1024
    progress_throttle_s = 0.1   # 10 Hz cap on progress callbacks
    last_progress_t = 0.0
    # Remove any stale partial from a prior attempt
    try:
        if os.path.exists(dest_path):
            os.remove(dest_path)
    except Exception:
        pass

    # Surface the URL in the FIREFLY console log — when the dialog
    # appears stuck, the user (and we) can read the log to see if the
    # URL itself is 404 (wrong torch version → no matching cu wheel)
    # vs a real network problem.
    _log(f"GET {url}")
    _log(f"  dest: {dest_path}")

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "FIREFLY-CUDA-installer/1.0"})
        # 20 s timeout (was 30) so a dead URL fails fast instead of
        # leaving the user staring at a frozen-looking dialog.
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=20) as resp:
            _log(f"  HTTP {getattr(resp, 'status', '?')} "
                 f"in {time.monotonic()-t0:.2f}s")
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
                    # Throttle progress emissions to ~10 Hz.  Without
                    # this, a fast connection floods the main Qt event
                    # queue with thousands of queued slot calls per
                    # second, paint events get starved, and Windows
                    # marks the app "Not Responding".
                    if progress_cb is not None:
                        now = time.monotonic()
                        if (now - last_progress_t) >= progress_throttle_s:
                            last_progress_t = now
                            try:
                                progress_cb(downloaded, total)
                            except Exception:
                                pass
                # Final 100 % tick so the bar visibly hits the end.
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
    is invoked at ~10 Hz."""
    os.makedirs(dest_dir, exist_ok=True)
    last_t = 0.0
    progress_throttle_s = 0.1
    try:
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            total = len(names)
            for i, name in enumerate(names, start=1):
                zf.extract(name, dest_dir)
                if progress_cb is not None:
                    now = time.monotonic()
                    if (now - last_t) >= progress_throttle_s or i == total:
                        last_t = now
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


def url_exists(url: str, timeout: float = 8.0) -> bool:
    """HEAD request to check whether a wheel URL is reachable.

    Wrapped in a hard wall-clock watchdog: urllib's `timeout` is not
    reliably honored on Windows when the TLS handshake or DNS stage
    stalls (observed: cu118 HEAD hung indefinitely on Windows 11).
    The watchdog runs the actual request on a daemon thread and gives
    up after `timeout + 2 s`, so a stuck HEAD can never wedge the
    worker thread (which was making the whole app look frozen).

    Returns True on 2xx, False on anything else.  Never raises.
    """
    _log(f"HEAD {url}")
    _log(f"  (timeout={timeout}s, watchdog={timeout + 2}s)")

    import threading
    result_holder = {"ok": False, "done": False}
    t0 = time.monotonic()

    def _do_head():
        try:
            req = urllib.request.Request(
                url, method="HEAD",
                headers={"User-Agent": "FIREFLY-CUDA-installer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                dt = time.monotonic() - t0
                code = int(getattr(resp, "status", 0) or 0)
                _log(f"  → HTTP {code} in {dt:.2f}s")
                result_holder["ok"] = 200 <= code < 300
        except urllib.error.HTTPError as exc:
            _log(f"  → HTTPError {exc.code}: {exc.reason} "
                 f"in {time.monotonic()-t0:.2f}s")
            result_holder["ok"] = False
        except urllib.error.URLError as exc:
            _log(f"  → URLError: {exc.reason} "
                 f"in {time.monotonic()-t0:.2f}s")
            result_holder["ok"] = False
        except Exception as exc:
            _log(f"  → {type(exc).__name__}: {exc} "
                 f"in {time.monotonic()-t0:.2f}s")
            result_holder["ok"] = False
        finally:
            result_holder["done"] = True

    t = threading.Thread(target=_do_head, daemon=True,
                          name="cuda-head-watchdog")
    t.start()
    t.join(timeout=timeout + 2)
    if not result_holder["done"]:
        _log(f"  → WATCHDOG: HEAD hung past {timeout + 2}s "
             f"(urllib timeout not honored — likely Windows TLS "
             f"handshake stall).  Treating as unreachable, moving on.")
        # Daemon thread will keep running but won't block process exit
        # or the worker thread.  Critical: we DON'T close the socket
        # here — that would race with the daemon thread.  It'll time
        # out eventually and exit on its own.
        return False
    return result_holder["ok"]


def install_cuda_torch_auto(torch_version: str,
                             cuda_tags: tuple = ("cu124", "cu121", "cu118"),
                             download_progress_cb=None,
                             extract_progress_cb=None,
                             cancel_cb=None,
                             status_cb: Optional[Callable[[str], None]] = None
                             ) -> str:
    """Try each CUDA tag in `cuda_tags` until one is reachable, then
    download.  Returns the cuda_tag that worked.

    Strategy: cheap HEAD requests to pick the first tag whose wheel
    actually exists (each HEAD is <1 s on a normal connection), THEN
    one full GET.  Avoids the 60-second triple-timeout stall the user
    hit when the bundled torch version doesn't have any CUDA wheel.

    `status_cb(msg)` is called between attempts so the GUI can update
    its label ("Checking cu121…", "Found cu121, downloading…").
    """
    _log(f"install_cuda_torch_auto starting — torch_version={torch_version}, "
         f"cuda_tags={cuda_tags}")
    chosen_tag: Optional[str] = None
    tried_urls = []
    for tag in cuda_tags:
        url = cuda_wheel_url(torch_version, cuda_tag=tag)
        tried_urls.append(url)
        _log(f"--- Checking {tag} ---")
        if status_cb is not None:
            try: status_cb(f"Checking torch {torch_version} + {tag}…")
            except Exception: pass
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Installation cancelled by user.")
        if url_exists(url):
            _log(f"  ✓ {tag} is available, will download")
            chosen_tag = tag
            break
        _log(f"  ✗ {tag} not available, trying next")

    if chosen_tag is None:
        _log("✗ No CUDA tag returned a working wheel URL")
        # All three HEAD-checks said "not found" — make the failure
        # actionable instead of mysterious.  Most likely cause: the
        # bundled torch version isn't a real release on PyTorch's
        # index (e.g. a pre-release or test version).
        url_lines = "\n  ".join(tried_urls)
        raise RuntimeError(
            f"No CUDA wheel exists for torch {torch_version} at "
            f"download.pytorch.org.\n\n"
            f"Tried:\n  {url_lines}\n\n"
            f"The bundled torch version may be a pre-release or a "
            f"version PyTorch hasn't shipped CUDA builds for.  To get "
            f"GPU acceleration on this machine, install FIREFLY from "
            f"source and follow the 'Enabling CUDA' section of the "
            f"README — that path lets pip resolve the latest matching "
            f"CUDA torch wheel against your local Python."
        )

    url = cuda_wheel_url(torch_version, cuda_tag=chosen_tag)
    if status_cb is not None:
        try: status_cb(f"Found cu{chosen_tag[2:]}, downloading…")
        except Exception: pass
    install_cuda_torch_from_url(
        url, torch_version=torch_version, cuda_tag=chosen_tag,
        download_progress_cb=download_progress_cb,
        extract_progress_cb=extract_progress_cb,
        cancel_cb=cancel_cb)
    return chosen_tag


def install_cuda_torch_from_url(url: str,
                                 *,
                                 torch_version: str,
                                 cuda_tag: str = "cu124",
                                 download_progress_cb=None,
                                 extract_progress_cb=None,
                                 cancel_cb=None) -> None:
    """Same as install_cuda_torch() but with the wheel URL already
    resolved by the caller — avoids a second `import torch` (which is
    slow on Windows onefile bundles).  `torch_version` is only used
    to name the temporary .whl file.

    Raises RuntimeError with user-facing wording on any failure.
    """
    sd = sidecar_dir()
    wheel_path = os.path.join(sd, f"torch-{torch_version}+{cuda_tag}.whl")
    extracted = sidecar_extracted_dir()

    # Wipe any half-done previous attempt so we start clean.
    try:
        if os.path.isdir(extracted):
            shutil.rmtree(extracted, ignore_errors=True)
    except Exception:
        pass

    download_wheel(url, wheel_path,
                   progress_cb=download_progress_cb,
                   cancel_cb=cancel_cb)

    if cancel_cb is not None:
        try:
            if cancel_cb():
                try: os.remove(wheel_path)
                except Exception: pass
                raise RuntimeError("Installation cancelled by user.")
        except RuntimeError:
            raise
        except Exception:
            pass

    extract_wheel(wheel_path, extracted,
                  progress_cb=extract_progress_cb)

    try: os.remove(wheel_path)
    except Exception: pass

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
