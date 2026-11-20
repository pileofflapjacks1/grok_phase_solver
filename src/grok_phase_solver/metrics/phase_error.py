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


def best_origin_shift(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    hkl: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_starts: int = 4,
) -> np.ndarray:
    """
    Continuous origin t maximizing C(t) = Σ_h w_h cos(φp_h − φt_h − 2π h·t).

    C(t) = Re Σ_h w_h e^{iΔ_h} e^{−2πi h·t} is a Fourier sum, so it is
    evaluated on a grid fine enough to resolve its peak (≥ 4 points per
    period of the largest index) with one ``fftn``. The best grid maxima are
    then polished with L-BFGS on t. Returns t in [0, 1)³.
    """
    from scipy.optimize import minimize

    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    dphi = np.asarray(phase_pred, dtype=np.float64) - np.asarray(phase_true, dtype=np.float64)
    w = np.ones(len(dphi)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(dphi) == 0:
        return np.zeros(3)
    hmax = np.abs(hkl).max(axis=0)
    shape = tuple(int(max(8, 4 * m + 4)) for m in hmax)
    grid = np.zeros(shape, dtype=np.complex128)
    np.add.at(
        grid,
        (hkl[:, 0] % shape[0], hkl[:, 1] % shape[1], hkl[:, 2] % shape[2]),
        w * np.exp(1j * dphi),
    )
    C = np.real(np.fft.fftn(grid))  # C[n] = Re Σ w e^{iΔ} e^{−2πi h·n/N}
    flat = np.argsort(C.ravel())[::-1][: max(1, n_starts)]
    hf = hkl.astype(np.float64)

    def neg_c(t):
        arg = dphi - 2 * np.pi * (hf @ t)
        val = -np.sum(w * np.cos(arg))
        grad = -2 * np.pi * (hf.T @ (w * np.sin(arg)))
        return val, grad

    best_t, best_v = np.zeros(3), np.inf
    for f in flat:
        t0 = np.array(np.unravel_index(int(f), shape), dtype=np.float64) / np.array(shape)
        res = minimize(neg_c, t0, jac=True, method="L-BFGS-B")
        if res.fun < best_v:
            best_v, best_t = float(res.fun), np.asarray(res.x)
    return best_t % 1.0


def mean_phase_error_origin_invariant(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    hkl: np.ndarray,
    weights: Optional[np.ndarray] = None,
    degrees: bool = True,
    n_origin_samples: int = 27,
) -> Tuple[float, np.ndarray]:
    """
    Minimize MAPE over origin shifts t and enantiomorph (φ → −φ).

    Origin shift: φ'(h) = φ(h) − 2π h·t

    In P1 the origin is continuous, so a discrete grid alone is not enough:
    perfect phases at a random origin would score like noise. For each hand
    the shift is found by maximizing the phase-agreement function
    C(t) = Σ w cos(Δφ_h − 2π h·t), evaluated on a fine grid with one FFT and
    refined continuously (:func:`best_origin_shift`). The legacy discrete
    set {0, ¼, ½, ¾}³ is also tried, so the result is never worse than before.

    Continuous t is the P1 (most permissive) origin group, so the score is an
    upper bound: fitting t (and the hand) to noise is not free. Measured null
    on the strong-|E| subset of synthetic P1 cells: random phases give
    frac ≤ 20° ≈ 0.17–0.19 (vs 20/180 ≈ 0.11 at a fixed origin). Compare
    predictions against that floor, not against 0.11.

    ``n_origin_samples`` is kept for API compatibility and ignored.

    Returns (best_error, best_phase_pred_aligned).
    """
    phase_pred = np.asarray(phase_pred, dtype=np.float64)
    phase_true = np.asarray(phase_true, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=np.float64)
    best = np.inf
    best_aligned = phase_pred
    if len(phase_pred) == 0:
        return float("nan"), best_aligned
    for sign in (1.0, -1.0):
        t = best_origin_shift(sign * phase_pred, phase_true, hkl, weights=weights)
        aligned = sign * phase_pred - 2 * np.pi * (hkl @ t)
        err = mean_phase_error(aligned, phase_true, weights=weights, degrees=degrees)
        if err < best:
            best = err
            best_aligned = aligned
    candidates = [0.0, 0.25, 0.5, 0.75]
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
