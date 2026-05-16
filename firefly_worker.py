"""
FIREFLY analysis subprocess worker.

This module deliberately imports NOTHING related to Qt / PySide6 / GUI
toolkits.  Why: when `multiprocessing.spawn` (the macOS-default start
method) creates a child process, it re-imports the module that defines
the target function in order to unpickle and call it.  If the target
lived in `app_qt.py`, the spawned subprocess would re-import
`app_qt.py` → `PySide6` → Qt 6's Metal-backed window compositor —
which on Apple Silicon claims memory from the same unified memory pool
PyTorch's MPS allocator needs.  Two Metal-using processes on a 16 GB
M-series Mac is enough to push PyTorch over the edge with "Insufficient
Memory" command-buffer errors.

By keeping the worker in this Qt-free module, the analysis subprocess
imports only Python stdlib + sptpalm_analysis (numpy / scipy / trackpy /
optionally torch) — no Metal-using framework, full unified-memory pool
available for MPS.

Public entry points (both used by app_qt.py):
    run_analysis(params, msg_queue, cancel_event)
        Single-file analysis.

    run_batch_analysis(params_list, msg_queue, cancel_event)
        Batch mode — same pipeline run sequentially over multiple files
        in a single subprocess (one spawn cost, N analyses).
"""
from __future__ import annotations

import os
import sys
import traceback


# ── MPS allocator tuning — must be set BEFORE torch import anywhere ──────────
# See app_qt.py for the rationale.  Setting here too is cheap
# insurance in case the parent's setting somehow didn't reach the child.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-PROCESS LOG STREAM
# ══════════════════════════════════════════════════════════════════════════════
class QueueLogStream:
    """File-like stream that posts each newline-/carriage-return-terminated
    line to a multiprocessing.Queue as a ('log', line) tuple.

    Used inside the analysis subprocess to forward `print()` calls and
    tqdm progress bars to the parent's Qt log box.  tqdm rewrites a single
    line with '\\r'; we treat both '\\r' and '\\n' as terminators so each
    tqdm update becomes one log entry instead of one giant line.
    """
    def __init__(self, q):
        self._q   = q
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while True:
            idx_n = self._buf.find("\n")
            idx_r = self._buf.find("\r")
            cuts = [i for i in (idx_n, idx_r) if i >= 0]
            if not cuts:
                break
            cut = min(cuts)
            line = self._buf[:cut]
            self._buf = self._buf[cut + 1:]
            if line.strip():
                self._q.put(("log", line.rstrip()))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._q.put(("log", self._buf.rstrip()))
            self._buf = ""

    def isatty(self) -> bool: return False
    def fileno(self):         raise OSError("not a real fd")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PIPELINE (shared by single-file and batch entry points)
# ══════════════════════════════════════════════════════════════════════════════
class _NoTracks(Exception):
    """Raised inside _run_one_analysis when linking produces 0 trajectories.
    The wrapper catches this and emits a sensible 'done' (single-file) or
    'file_done' (batch) payload — no crash report."""


def _run_one_analysis(params: dict, msg_queue, cancel_event,
                      _log, _prog) -> dict:
    """Run the FIREFLY pipeline on one input file.

    Returns the "done"-payload dict (stem, out_dir, figure_path, n_tracks,
    n_locs).  Raises `sptpalm_analysis._Cancelled` if the user stopped via
    `cancel_event`.  Raises `_NoTracks` if linking yielded nothing.  Other
    exceptions propagate so the caller decides whether to abort or continue
    (batch continues to next file; single-file emits a crash report).
    """
    p = params

    from sptpalm_analysis import (
        load_file, preprocess_and_localise_adaptive, link_trajectories,
        compute_msd_and_fit, compute_jdd, compute_turning_angles,
        compute_mobile_fraction_over_time, compute_clusters,
        compute_dwell_times, compute_mss, correct_drift,
        make_figure, save_palmtracer_csvs, apply_roi_mask, _Cancelled,
    )

    # Helper: check stop event at major pipeline boundaries.  Most of the
    # pipeline's interruptibility comes from passing `cancel_event` deep
    # into load_file / preprocess_and_localise_adaptive, but those functions
    # poll only periodically.  Adding explicit checks BETWEEN stages means
    # a Stop click during e.g. the linker's long uninterruptible region
    # will at least halt before the next stage starts.
    def _check_stop():
        if cancel_event.is_set():
            raise _Cancelled()

    fpath = p["file"]
    stem  = os.path.splitext(os.path.basename(fpath))[0]
    out_dir    = p.get("out_dir") or os.path.dirname(os.path.abspath(fpath))
    fig_dir    = os.path.join(out_dir, "figures")
    data_dir   = os.path.join(out_dir, "data")
    extras_dir = os.path.join(out_dir, "firefly_extras")
    for d in (fig_dir, data_dir, extras_dir):
        os.makedirs(d, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    _log(f"\n── Load ──────────────────────────")
    _prog(5, "Loading stack…")
    stack, meta_px, meta_fi = load_file(
        fpath, channel=int(p.get("channel", 0)),
        stop_event=cancel_event)

    # Override file-embedded metadata only when the user explicitly ticked
    # the "Override" checkbox (the GUI sends None otherwise).
    px = p.get("pixel_size") or meta_px or 0.106
    fi = p.get("frame_interval") or meta_fi or 0.02
    n_frames = len(stack)
    _log(f"  Shape: {stack.shape}  (T x Y x X)")
    _log(f"  Frames: {n_frames:,}  |  px={px} µm  fi={fi} s")

    # Sample frames evenly across the stack for the figure-background panel.
    # Doing it before localisation keeps peak RAM down.
    import numpy as _np
    n_proj = min(200, n_frames)
    proj_idx = _np.linspace(0, n_frames - 1, n_proj, dtype=int)
    proj_sample = stack[proj_idx].copy()

    # ── Localisation ──────────────────────────────────────────────────────
    _log(f"\n── Localisation ──────────────────")
    _prog(20, "Localising…")

    # "Auto-detect minmass": pass None and let the pipeline pick a
    # data-dependent threshold from the first chunk's 99th percentile.
    minmass_arg = (None if p.get("auto_minmass", False)
                   else float(p["minmass"]))

    locs, mean_proj, _mm = preprocess_and_localise_adaptive(
        stack,
        diameter=int(p["diameter"]),
        minmass=minmass_arg,
        bg_radius=int(p.get("bg_radius", 10)),
        bg_method=p.get("bg_method", "uniform_filter"),
        workers=int(p["workers"]),
        chunk_size=int(p["chunk_size"]),
        stop_event=cancel_event,
        backend=p["backend"])
    stack_h = stack.shape[1] if stack.ndim >= 3 else 0
    stack_w = stack.shape[2] if stack.ndim >= 3 else 0
    del stack
    _log(f"  → {len(locs):,} localisations")
    _check_stop()

    # ── ROI mask (optional) ───────────────────────────────────────────────
    roi_mode = p.get("roi_mode", "none")
    if roi_mode != "none" and len(locs) > 0:
        _log(f"\n── ROI mask ───────────────────────")
        try:
            from sptpalm_analysis import auto_threshold
            if roi_mode == "auto":
                method = (p.get("roi_auto_method") or "Li").lower()
                thresh, _, _ = auto_threshold(mean_proj, method=method)
            else:  # manual
                thresh = float(p.get("roi_threshold", 0.08))
            roi_mask = (mean_proj > thresh).astype(_np.uint8)
            pct = 100.0 * roi_mask.mean()
            _log(f"  Threshold = {thresh:.4f}  |  {pct:.1f}% of frame")
            n_before = len(locs)
            locs = apply_roi_mask(locs, roi_mask)
            _log(f"  Locs after ROI : {len(locs):,}  "
                 f"(dropped {n_before - len(locs):,})")
        except Exception as roi_exc:
            _log(f"  WARN: ROI mask failed — {roi_exc}.  Continuing without ROI.")

    # ── Drift correction (optional) ───────────────────────────────────────
    if p.get("drift_correct", False) and len(locs) > 0:
        _log(f"\n── Drift correction ───────────────")
        _prog(40, "Correcting drift…")
        try:
            locs, drift_df = correct_drift(
                locs, n_seg_frames=int(p.get("drift_segment", 500)))
            drift_df.to_csv(
                os.path.join(extras_dir, f"{stem}_drift.csv"), index=False)
            _log(f"  Drift correction applied  |  saved {stem}_drift.csv")
        except Exception as exc:
            _log(f"  WARN: drift correction failed — {exc}")

    _check_stop()

    # Belt-and-braces GPU drain before the long CPU-only linking stage.
    try:
        import torch as _torch, gc as _gc
        _gc.collect()
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            if hasattr(_torch.mps, "synchronize"): _torch.mps.synchronize()
            if hasattr(_torch.mps, "empty_cache"): _torch.mps.empty_cache()
        if _torch.cuda.is_available():
            _torch.cuda.synchronize(); _torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Linking ───────────────────────────────────────────────────────────
    _log(f"\n── Linking ───────────────────────")
    _log(f"  Linking {len(locs):,} localisations — single-threaded, "
         f"may take several minutes at high density")
    if len(locs) > 100_000:
        _log(f"  NOTE: very high spot density ({len(locs):,} locs). "
             f"Consider raising minmass to reduce false positives.")
    _prog(50, f"Linking {len(locs):,} localisations…")
    tracks = link_trajectories(
        locs,
        search_range=int(p["search_range"]),
        memory=int(p["memory"]),
        min_len=int(p["min_track_len"]),
        max_len=p.get("max_track_len"))
    n_tracks_found = tracks['particle'].nunique() if len(tracks) else 0
    _log(f"  → {n_tracks_found:,} trajectories")
    _check_stop()

    if n_tracks_found == 0:
        _log("")
        _log("  ⚠  No trajectories were formed.  Likely causes:")
        _log("     • minmass is too LOW → too many noise spots, "
             "linker can't form sensible tracks")
        _log("     • minmass is too HIGH → real spots filtered out, "
             "nothing left to link")
        _log("     • search_range too small for actual particle motion")
        _log("     • If using a GPU backend and only chunk 1 produced "
             "spots, MPS may be in a degraded state on this hw/os "
             "combo — retry with backend='trackpy' to confirm.")
        _log("")
        _log("── Stopping analysis (nothing more to do) ──")
        # Raise a sentinel — caller will turn this into a sensible payload.
        raise _NoTracks({
            "stem": stem, "out_dir": out_dir,
            "figure_path": "", "n_tracks": 0, "n_locs": int(len(locs)),
        })

    # ── MSD + diffusion ───────────────────────────────────────────────────
    _log(f"\n── MSD & diffusion ───────────────")
    _prog(65, "Computing MSD + fits…")
    imsd_df, emsd_df, diff_df = compute_msd_and_fit(
        tracks, px, fi,
        max_lagtime=int(p["max_lagtime"]),
        n_fit=int(p["n_fit"]),
        workers=int(p["workers"]),
        alpha_thresholds=tuple(p.get("alpha_thresholds", (0.5, 0.9, 1.1))))

    # Optional: filter tracks by diffusion coefficient
    if p.get("filter_d_enabled", False) and len(diff_df):
        d_min = float(p.get("filter_d_min", 0.0))
        d_max = float(p.get("filter_d_max", 1.0))
        n_before = len(diff_df)
        mask = diff_df["D"].between(d_min, d_max)
        keep_pids = set(diff_df.loc[mask, "particle"])
        diff_df = diff_df[mask].reset_index(drop=True)
        tracks  = tracks[tracks["particle"].isin(keep_pids)]
        _log(f"  Filter by D [{d_min}, {d_max}]: "
             f"{n_before} → {len(diff_df)} tracks")

    # ── Secondary analyses ────────────────────────────────────────────────
    _log(f"\n── Secondary analyses ────────────")
    _prog(80, "Secondary analyses…")
    jdd = compute_jdd(tracks, px, fi,
                      n_components=int(p.get("jdd_components", 2)))
    ta  = compute_turning_angles(tracks)
    mf  = compute_mobile_fraction_over_time(
        tracks, diff_df, fi,
        d_threshold=float(p.get("mobile_d_threshold", 0.05)))
    _, cluster_stats_df, _, cluster_xy = compute_clusters(
        locs, px,
        eps_um=float(p.get("cluster_eps_nm", 50.0)) / 1000.0,
        min_samples=int(p.get("cluster_min_samples", 10)))
    dwell_df, dwell_tau = compute_dwell_times(tracks, diff_df, fi)
    _check_stop()

    # ── Render figure ─────────────────────────────────────────────────────
    _log(f"\n── Saving ────────────────────────")
    _prog(90, "Rendering figure…")
    fig_data = make_figure(
        proj_sample, tracks, imsd_df, emsd_df, diff_df, px, fi,
        jdd=jdd, turning_angles=ta, mobile_frac_df=mf,
        cluster_locs=cluster_xy, dwell_df=dwell_df, dwell_tau=dwell_tau)
    del proj_sample

    # ── Save outputs ──────────────────────────────────────────────────────
    _prog(95, "Saving outputs…")
    try:
        save_palmtracer_csvs(data_dir, stem, locs, tracks, diff_df, imsd_df,
                             pixel_size_um=float(px),
                             frame_interval_s=float(fi),
                             width=stack_w, height=stack_h,
                             n_frames=int(n_frames))
        _log("  Saved (data/): PALM-Tracer CSVs")
    except Exception as exc:
        _log(f"  WARN: PALM-Tracer export failed: {exc}\n{traceback.format_exc()}")

    locs.to_csv(os.path.join(extras_dir, f"{stem}_localisations.csv"),
                index=False)
    tracks.to_csv(os.path.join(extras_dir, f"{stem}_trajectories.csv"),
                  index=False)
    diff_df.to_csv(os.path.join(extras_dir, f"{stem}_diffusion_summary.csv"),
                   index=False)
    _log(f"  Saved (firefly_extras/): trajectories, locs, diffusion summary")

    figure_path = ""
    try:
        figure_path = os.path.join(fig_dir, f"{stem}_sptpalm_figure.png")
        fig_data["combined"].save(figure_path, dpi=(150, 150))
    except Exception as e:
        _log(f"  WARN: figure save failed: {e}")
        figure_path = ""

    _log(f"\n  Output folder: {out_dir}")
    _prog(100, "Complete!")
    return {
        "stem":        stem,
        "out_dir":     out_dir,
        "figure_path": figure_path,
        "n_tracks":    int(diff_df.shape[0]) if diff_df is not None else 0,
        "n_locs":      int(len(locs)) if locs is not None else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — SINGLE FILE
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(params: dict, msg_queue, cancel_event):
    """Single-file subprocess entry.  Wraps `_run_one_analysis` with the
    stdout/stderr redirect and translates exceptions into terminal queue
    messages ("done" / "stopped" / "error")."""
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)

    def _log(msg: str):       msg_queue.put(("log", msg))
    def _prog(pct, msg):      msg_queue.put(("progress", (int(pct), str(msg))))

    try:
        _log("── Worker subprocess started ──")
        _prog(0, "Importing pipeline…")
        payload = _run_one_analysis(params, msg_queue, cancel_event, _log, _prog)
        msg_queue.put(("done", payload))
    except _NoTracks as nt:
        # Linker produced 0 trajectories — not a crash.  Treat as "done"
        # with the partial payload so the UI resets cleanly.
        msg_queue.put(("done", nt.args[0]))
    except BaseException as exc:
        if type(exc).__name__ in ("_Cancelled", "_Stopped"):
            msg_queue.put(("log", "\n── Stopped by user ──"))
            msg_queue.put(("stopped", None))
        else:
            msg_queue.put(("error", traceback.format_exc()))
    finally:
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — COMPARE  (N-group comparison run in a subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def run_comparison(comparison_params: dict, msg_queue, cancel_event):
    """Run sptpalm_analysis.compare_groups in a subprocess.

    Same rationale for the subprocess as run_analysis: keep matplotlib +
    pandas + scipy import cost out of the Qt main process, and (on
    Apple Silicon) keep Metal contention with Qt to a minimum.

    Messages emitted
    ----------------
      log/progress  — same conventions as single-file mode
      compare_done(payload)
                    — terminal success message.  payload keys:
                        figure_path : str — saved .png
                        output_dir  : str — folder containing all outputs
                        summary_csv : str — per-replicate scalars
                        stats_csv   : str — pairwise tests
                        pdf_report  : str — combined PDF (if requested)
      stopped       — cooperative cancel fired
      error(tb)     — unrecoverable exception
    """
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)

    def _log(msg: str):  msg_queue.put(("log", msg))
    def _prog(pct, msg): msg_queue.put(("progress", (int(pct), str(msg))))

    try:
        _log("── Compare worker subprocess started ──")
        _prog(0, "Importing comparison pipeline…")

        from sptpalm_analysis import compare_groups, _Cancelled

        p = comparison_params

        # compare_groups expects panels as a set; the Qt side ships a list
        # (JSON-friendly).  Normalise here.
        panels = set(p.get("panels") or []) or None

        # Wire progress callback → queue.  compare_groups invokes this
        # periodically during folder loading; we map to percent.
        def _progress_cb(done: int, total: int, msg: str):
            if cancel_event.is_set():
                raise _Cancelled()
            pct = int(100 * done / total) if total else 0
            _prog(pct, msg)

        out_dir   = p.get("output_dir")
        out_stem  = p.get("output_stem", "comparison")
        theme     = p.get("theme", "Dark")
        pdf_report = bool(p.get("pdf_report", True))
        mob_d     = float(p.get("mobile_d_threshold", 0.05))

        _log(f"  Output dir : {out_dir}")
        _log(f"  Output stem: {out_stem}")
        _log(f"  Theme      : {theme}")
        _log(f"  Groups     : {len(p.get('groups', []))}")
        for g in p.get("groups", []):
            _log(f"    {g.get('label', '?'):<20s}"
                 f"({len(g.get('folders', []))} folders)")

        fig, summary_df, stats = compare_groups(
            groups=p["groups"],
            output_dir=out_dir,
            output_stem=out_stem,
            panels=panels,
            theme=theme,
            pdf_report=pdf_report,
            mobile_d_threshold=mob_d,
            progress_cb=_progress_cb)

        # Compose result paths.  compare_groups saves these by convention:
        figure_path = os.path.join(out_dir, f"{out_stem}.png")
        summary_csv = os.path.join(out_dir, f"{out_stem}_summary.csv")
        stats_csv   = os.path.join(out_dir, f"{out_stem}_stats.csv")
        pdf_path    = os.path.join(out_dir, f"{out_stem}_report.pdf")

        _prog(100, "Comparison complete")
        msg_queue.put(("compare_done", {
            "output_dir":  out_dir,
            "figure_path": figure_path if os.path.isfile(figure_path) else "",
            "summary_csv": summary_csv if os.path.isfile(summary_csv) else "",
            "stats_csv":   stats_csv   if os.path.isfile(stats_csv)   else "",
            "pdf_report":  pdf_path    if os.path.isfile(pdf_path)    else "",
            "n_groups":    len(p.get("groups", [])),
        }))

    except BaseException as exc:
        if type(exc).__name__ in ("_Cancelled", "_Stopped"):
            msg_queue.put(("log", "\n── Stopped by user ──"))
            msg_queue.put(("stopped", None))
        else:
            msg_queue.put(("error", traceback.format_exc()))
    finally:
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — BATCH (multiple files in one subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def run_batch_analysis(params_list: list, msg_queue, cancel_event):
    """Run `_run_one_analysis` over each entry in `params_list`.

    One subprocess, N files — amortizes the spawn import cost across the
    whole batch.  Per-file failures don't abort the run: the failing file
    gets a `("file_error", ...)` message and the batch continues.  The
    final summary is `("batch_done", {n_total, n_ok, n_fail, results})`.

    Messages
    --------
    log/progress       — same as single-file mode (forwarded from
                         _run_one_analysis)
    file_done(payload) — emitted after each successful file
    file_error(info)   — emitted after each failed file
    stopped            — user cancelled; whole batch aborts
    batch_done(summary)— terminal message; batch completed normally
    error(tb)          — terminal message; unrecoverable worker error
    """
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)

    def _log(msg: str):  msg_queue.put(("log", msg))
    def _prog(pct, msg): msg_queue.put(("progress", (int(pct), str(msg))))

    try:
        n = len(params_list)
        _log(f"── Batch worker subprocess started — {n} file(s) ──")
        results = []

        for i, params in enumerate(params_list, 1):
            if cancel_event.is_set():
                _log("\n── Batch stopped by user ──")
                msg_queue.put(("stopped", None))
                return

            fname = os.path.basename(params["file"])
            _log("")
            _log("══════════════════════════════════════════════════════════════════")
            _log(f"  [{i}/{n}]  {fname}")
            _log("══════════════════════════════════════════════════════════════════")
            # Overall-batch progress: percent of files completed
            overall_pct = int(100 * (i - 1) / max(1, n))
            _prog(overall_pct, f"[{i}/{n}] {fname}")

            try:
                payload = _run_one_analysis(
                    params, msg_queue, cancel_event, _log, _prog)
                results.append({"index": i, "ok": True, "file": params["file"],
                                **payload})
                msg_queue.put(("file_done", {
                    "index": i, "total": n,
                    "stem":     payload.get("stem"),
                    "out_dir":  payload.get("out_dir"),
                    "n_tracks": payload.get("n_tracks", 0),
                    "n_locs":   payload.get("n_locs", 0),
                }))
            except _NoTracks as nt:
                results.append({"index": i, "ok": True, "file": params["file"],
                                **nt.args[0]})
                msg_queue.put(("file_done", {
                    "index": i, "total": n,
                    "stem":     nt.args[0].get("stem"),
                    "out_dir":  nt.args[0].get("out_dir"),
                    "n_tracks": 0,
                    "n_locs":   nt.args[0].get("n_locs", 0),
                }))
            except BaseException as exc:
                if type(exc).__name__ in ("_Cancelled", "_Stopped"):
                    _log("\n── Batch stopped by user ──")
                    msg_queue.put(("stopped", None))
                    return
                tb = traceback.format_exc()
                _log(f"\n  ⚠ File {i}/{n} ({fname}) FAILED: {exc}")
                _log(tb)
                results.append({"index": i, "ok": False, "file": params["file"],
                                "error": str(exc)})
                msg_queue.put(("file_error", {
                    "index": i, "total": n,
                    "file": params["file"], "tb": tb,
                }))
                # Continue with next file rather than aborting the whole batch

        # All done
        n_ok   = sum(1 for r in results if r.get("ok"))
        n_fail = n - n_ok
        _log("")
        _log("══════════════════════════════════════════════════════════════════")
        _log(f"  Batch complete: {n_ok}/{n} succeeded, {n_fail} failed")
        _log("══════════════════════════════════════════════════════════════════")
        _prog(100, "Batch complete!")
        msg_queue.put(("batch_done", {
            "n_total": n, "n_ok": n_ok, "n_fail": n_fail,
            "results": results,
        }))

    except BaseException:
        msg_queue.put(("error", traceback.format_exc()))
    finally:
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass
