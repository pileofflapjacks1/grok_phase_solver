"""Phase error metrics (circular statistics)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phases to (−π, π]."""
    return (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi


def mean_phase_error(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    weights: Optional[np.ndarray] = None,
    degrees: bool = True,
) -> float:
    """
    Weighted mean absolute phase error (MAPE), accounting for 2π periodicity.

    For centrosymmetric structures phases are 0/π; still valid.

    If overall origin/enantiomorph ambiguity is present, use
    :func:`mean_phase_error_origin_invariant` instead.
    """
    d = wrap_phase(np.asarray(phase_pred) - np.asarray(phase_true))
    ad = np.abs(d)
    if weights is None:
        m = float(np.mean(ad))
    else:
        w = np.asarray(weights, dtype=np.float64)
        m = float(np.sum(w * ad) / (np.sum(w) + 1e-16))
    return float(np.rad2deg(m)) if degrees else m


def mean_phase_error_origin_invariant(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    hkl: np.ndarray,
    weights: Optional[np.ndarray] = None,
    degrees: bool = True,
    n_origin_samples: int = 27,
) -> Tuple[float, np.ndarray]:
    """
    Minimize MAPE over discrete origin shifts t ∈ {0, 1/3, 1/2}^3 samples
    and enantiomorph (φ → −φ).

    Origin shift: φ'(h) = φ(h) − 2π h·t

    Returns (best_error, best_phase_pred_aligned).
    """
    phase_pred = np.asarray(phase_pred, dtype=np.float64)
    phase_true = np.asarray(phase_true, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=np.float64)
    shifts = np.linspace(0, 1, int(round(n_origin_samples ** (1 / 3))), endpoint=False)
    # Use a denser grid along each axis if n is a cube; fallback simple set
    candidates = [0.0, 0.25, 0.5, 0.75]
    best = np.inf
    best_aligned = phase_pred
    for tx in candidates:
        for ty in candidates:
            for tz in candidates:
                t = np.array([tx, ty, tz])
                shift = 2 * np.pi * (hkl @ t)
                for sign in (1.0, -1.0):
                    aligned = sign * phase_pred - shift
                    err = mean_phase_error(aligned, phase_true, weights=weights, degrees=degrees)
                    if err < best:
                        best = err
                        best_aligned = aligned
    return best, best_aligned


def phase_error_histogram(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    bins: int = 36,
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of wrapped phase differences (degrees)."""
    d = np.rad2deg(wrap_phase(np.asarray(phase_pred) - np.asarray(phase_true)))
    counts, edges = np.histogram(d, bins=bins, range=(-180, 180))
    return counts, edges
