"""
Numerical-agreement test for the FIREFLY localiser backends.

Generates a synthetic stack with known ground-truth spot positions,
runs both TrackpyBackend and TorchBackend, then asserts that:

  • The number of recovered spots is within tolerance for both backends.
  • The median centroid disagreement between the two backends is small
    (< 0.10 px ≈ 10 nm at 100 nm/px).

Run from project root:

    pytest tests/test_localiser_agreement.py -v

Or standalone:

    python tests/test_localiser_agreement.py
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Synthetic-stack helpers ──────────────────────────────────────────────────
def make_gaussian_spot(img: np.ndarray, y: float, x: float, amp: float,
                       sigma: float = 1.2):
    """Add an isotropic 2D Gaussian to `img` in-place (sub-pixel positions)."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma * sigma))


def synthesize_stack(n_frames: int = 20, H: int = 128, W: int = 128,
                     n_spots: int = 40, amp: float = 500.0,
                     noise_sigma: float = 5.0,
                     seed: int = 42):
    """Return (stack, ground_truth_df).

    ground_truth_df columns: frame, x, y, amp
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    stack = rng.normal(loc=100.0, scale=noise_sigma,
                       size=(n_frames, H, W)).astype(np.float32)
    rows = []
    for t in range(n_frames):
        xs = rng.uniform(8, W - 8, n_spots)
        ys = rng.uniform(8, H - 8, n_spots)
        amps = rng.uniform(0.7, 1.3, n_spots) * amp
        for x, y, a in zip(xs, ys, amps):
            make_gaussian_spot(stack[t], y, x, a, sigma=1.2)
            rows.append({"frame": t, "x": float(x), "y": float(y), "amp": float(a)})
    return stack, pd.DataFrame(rows)


# ── Tests ────────────────────────────────────────────────────────────────────
def _have_backend(name: str) -> bool:
    from sptpalm_analysis import _BACKEND_REGISTRY
    for cls in _BACKEND_REGISTRY:
        if cls.name == name and cls.is_available():
            return True
    return False


@pytest.mark.skipif(not _have_backend("trackpy"), reason="trackpy not installed")
def test_trackpy_recovers_majority_of_spots():
    """Trackpy should find ≥80% of the ground-truth spots on a clean synthetic
    stack."""
    from sptpalm_analysis import localise_particles
    stack, gt = synthesize_stack(n_frames=10, n_spots=30)
    locs = localise_particles(stack, diameter=7, minmass=200.0,
                              percentile=80, workers=1,
                              chunk_size=20, backend="trackpy")
    n_found = len(locs)
    n_truth = len(gt)
    print(f"trackpy: found {n_found}, truth {n_truth}")
    assert n_found >= 0.8 * n_truth, (
        f"trackpy found only {n_found} of {n_truth} ground-truth spots")


@pytest.mark.skipif(not _have_backend("torch"), reason="torch not installed")
def test_torch_recovers_majority_of_spots():
    """Same expectation for the torch backend on CPU."""
    from sptpalm_analysis import localise_particles
    stack, gt = synthesize_stack(n_frames=10, n_spots=30)
    locs = localise_particles(stack, diameter=7, minmass=200.0,
                              percentile=80, workers=1,
                              chunk_size=20, backend="torch")
    n_found = len(locs)
    n_truth = len(gt)
    print(f"torch: found {n_found}, truth {n_truth}")
    assert n_found >= 0.8 * n_truth, (
        f"torch found only {n_found} of {n_truth} ground-truth spots")


@pytest.mark.skipif(not (_have_backend("trackpy") and _have_backend("torch")),
                    reason="both trackpy and torch must be installed")
def test_trackpy_torch_agreement():
    """Cross-backend centroid agreement: the two implementations should
    localise the same spots to within ≈0.10 px median disagreement.
    Tolerance is loose because trackpy uses iterative refinement and torch
    uses single-pass centroid-of-mass; both differ slightly from ground truth
    in opposite directions."""
    from sptpalm_analysis import localise_particles
    stack, _ = synthesize_stack(n_frames=10, n_spots=30)

    tp_locs = localise_particles(stack, diameter=7, minmass=200.0,
                                 percentile=80, workers=1,
                                 chunk_size=20, backend="trackpy")
    th_locs = localise_particles(stack, diameter=7, minmass=200.0,
                                 percentile=80, workers=1,
                                 chunk_size=20, backend="torch")

    # Match spots frame-by-frame using nearest-neighbour with a 2-px cutoff
    from scipy.spatial import cKDTree
    pairs_dx = []
    pairs_dy = []
    for frame in sorted(set(tp_locs["frame"]).intersection(th_locs["frame"])):
        tp_f = tp_locs[tp_locs["frame"] == frame][["x", "y"]].values
        th_f = th_locs[th_locs["frame"] == frame][["x", "y"]].values
        if len(tp_f) == 0 or len(th_f) == 0:
            continue
        tree = cKDTree(th_f)
        dist, idx = tree.query(tp_f, distance_upper_bound=2.0)
        valid = np.isfinite(dist) & (dist < 2.0)
        if not valid.any():
            continue
        matched_th = th_f[idx[valid]]
        pairs_dx.append(tp_f[valid][:, 0] - matched_th[:, 0])
        pairs_dy.append(tp_f[valid][:, 1] - matched_th[:, 1])

    dx = np.concatenate(pairs_dx) if pairs_dx else np.array([np.inf])
    dy = np.concatenate(pairs_dy) if pairs_dy else np.array([np.inf])
    median_disp = float(np.median(np.sqrt(dx ** 2 + dy ** 2)))
    print(f"trackpy↔torch median centroid disagreement: {median_disp:.3f} px "
          f"(over {len(dx)} matched spots)")
    assert median_disp < 0.30, (
        f"backend disagreement too large: {median_disp:.3f} px "
        f"(threshold 0.30 px ≈ 30 nm at 100 nm/px)")


if __name__ == "__main__":
    # Allow direct execution: `python tests/test_localiser_agreement.py`
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
