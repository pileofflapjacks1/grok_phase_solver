"""
Strong-seed quality metrics (hard-cliff success bar).

Oracle partial-φ benchmarks showed hard cells strict-solve when ~30% of strong
|E| phases are correct to ≲20° MPE. Full-map OI MPE alone understates whether a
prior is good enough for AI-PhaSeed.

Metrics
-------
- strong_mpe_oi : origin/enantiomorph-invariant MAPE on the strong subset
- frac_within_deg : fraction of strong reflections with |Δφ|_OI ≤ threshold
- would_seed_solve : heuristic True if frac_within_20° ≥ 0.30 (oracle bar)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.metrics.phase_error import (
    mean_phase_error,
    mean_phase_error_origin_invariant,
    wrap_phase,
)
from grok_phase_solver.solvers.direct_methods import normalize_E


def select_strong_indices(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    fraction: float = 0.30,
    n_strong: Optional[int] = None,
    by: str = "E",
    min_n: int = 10,
    max_n: int = 200,
) -> np.ndarray:
    """Indices of strongest reflections (default top 30% by |E|)."""
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl)
    n = len(amp)
    if n_strong is None:
        n_strong = int(np.clip(round(fraction * n), min_n, min(max_n, n)))
    n_strong = int(min(max(n_strong, 1), n))
    if by == "E":
        score = normalize_E(hkl, amp, cell)
    elif by == "F":
        score = amp
    else:
        raise ValueError(by)
    return np.argsort(-score)[:n_strong].astype(int)


def strong_seed_metrics(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    strong_idx: Optional[np.ndarray] = None,
    fraction: float = 0.30,
    within_deg: float = 20.0,
    solve_frac_threshold: float = 0.30,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Evaluate predicted phases on the strong-|E| seed set.

    Returns dict with strong_mpe_oi, frac_within_{within_deg}, would_seed_solve, etc.
    """
    phase_pred = np.asarray(phase_pred, dtype=np.float64)
    phase_true = np.asarray(phase_true, dtype=np.float64)
    hkl = np.asarray(hkl)
    amp = np.asarray(amplitudes, dtype=np.float64)

    if strong_idx is None:
        strong_idx = select_strong_indices(
            hkl, amp, cell, fraction=fraction
        )
    else:
        strong_idx = np.asarray(strong_idx, dtype=int)

    if len(strong_idx) == 0:
        return {
            "n_strong": 0,
            "strong_mpe_oi": None,
            "frac_within_deg": None,
            "within_threshold_deg": within_deg,
            "would_seed_solve": False,
            "strong_fraction": 0.0,
        }

    hp = hkl[strong_idx]
    pp = phase_pred[strong_idx]
    pt = phase_true[strong_idx]
    w = None if weights is None else np.asarray(weights)[strong_idx]
    if w is None:
        w = amp[strong_idx]

    mpe, aligned = mean_phase_error_origin_invariant(
        pp, pt, hp, weights=w, degrees=True
    )
    d = np.abs(wrap_phase(aligned - pt))
    d_deg = np.rad2deg(d)
    # weighted fraction within threshold
    ww = w / (w.sum() + 1e-16)
    frac = float(np.sum(ww * (d_deg <= within_deg)))
    # unweighted also
    frac_uw = float(np.mean(d_deg <= within_deg))

    would = frac >= solve_frac_threshold
    return {
        "n_strong": int(len(strong_idx)),
        "strong_fraction": float(len(strong_idx) / max(len(amp), 1)),
        "strong_mpe_oi": float(mpe),
        "frac_within_deg": frac,
        "frac_within_deg_unweighted": frac_uw,
        "within_threshold_deg": float(within_deg),
        "solve_frac_threshold": float(solve_frac_threshold),
        "would_seed_solve": bool(would),
        "mean_abs_err_strong_deg": float(np.average(d_deg, weights=w)),
        "strong_idx": strong_idx,
    }


def full_and_strong_metrics(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    **kwargs,
) -> Dict:
    """Full-map OI MPE + strong-seed metrics in one dict."""
    mpe_full, _ = mean_phase_error_origin_invariant(
        phase_pred, phase_true, hkl, weights=amplitudes, degrees=True
    )
    strong = strong_seed_metrics(
        phase_pred, phase_true, hkl, amplitudes, cell, **kwargs
    )
    return {
        "full_mpe_oi": float(mpe_full),
        **{k: v for k, v in strong.items() if k != "strong_idx"},
        "n_strong": strong["n_strong"],
    }
