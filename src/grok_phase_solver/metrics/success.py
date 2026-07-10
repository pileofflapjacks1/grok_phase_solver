"""
Strict success criteria for ab initio phasing experiments.

A trial is counted as **solved** only if multiple independent checks pass:

1. **mapCC_OI ≥ τ_cc** — origin/enantiomorph-invariant density correlation
2. **peak recovery ≥ τ_peak** — fraction of true atoms matched by a density peak
   (after best lattice translation of the peak set)
3. **R1 ≤ τ_R** (optional) — residual after placing scatterers at top peaks
   and computing Fcalc (crude model; not SHELXL)

These thresholds are conventions for this project, not universal laws.
Default: τ_cc=0.7, τ_peak=0.5, τ_R=0.45 (lenient R for unrefined peaks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant
from grok_phase_solver.metrics.rfactor import r_factor
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.pipeline.peaks import pick_density_peaks


@dataclass
class SuccessThresholds:
    mapcc_min: float = 0.70
    peak_recovery_min: float = 0.50
    r1_max: float = 0.45
    require_r1: bool = True
    peak_tol: float = 0.15  # fractional min-image distance
    n_peaks_factor: float = 2.0  # pick this many times n_atoms peaks
    min_peak_sigma: float = 1.5


@dataclass
class SuccessReport:
    mapcc_oi: float
    mpe_oi_deg: float
    peak_recovery: float
    r1: Optional[float]
    solved: bool
    details: Dict


def _min_image_frac(f1: np.ndarray, f2: np.ndarray) -> float:
    d = (f1 - f2 + 0.5) % 1.0 - 0.5
    return float(np.linalg.norm(d))


def peak_recovery_score(
    peaks_fract: np.ndarray,
    true_fracs: np.ndarray,
    tol: float = 0.12,
    n_origin_shifts: int = 4,
) -> Tuple[float, np.ndarray]:
    """
    Fraction of true atoms within `tol` of some peak, maximizing over
    discrete origin shifts t ∈ {0, 1/n, ..., (n-1)/n}^3 (coarse).

    Returns (best_fraction, best_shift).
    """
    peaks_fract = np.asarray(peaks_fract, dtype=np.float64).reshape(-1, 3)
    true_fracs = np.asarray(true_fracs, dtype=np.float64).reshape(-1, 3)
    if len(peaks_fract) == 0 or len(true_fracs) == 0:
        return 0.0, np.zeros(3)

    grid = np.linspace(0, 1, n_origin_shifts, endpoint=False)
    best = -1.0
    best_t = np.zeros(3)
    for tx in grid:
        for ty in grid:
            for tz in grid:
                t = np.array([tx, ty, tz])
                matched = 0
                for atom in true_fracs:
                    a = (atom + t) % 1.0
                    ok = any(_min_image_frac(a, p) < tol for p in peaks_fract)
                    if ok:
                        matched += 1
                frac = matched / len(true_fracs)
                if frac > best:
                    best = frac
                    best_t = t
    return float(best), best_t


def r1_from_peaks(
    hkl: np.ndarray,
    F_obs: np.ndarray,
    cell: np.ndarray,
    peak_fracs: np.ndarray,
    n_atoms: int,
    element: str = "C",
) -> float:
    """
    Place up to n_atoms carbon-like scatterers at strongest peaks; R1 vs |F_obs|.
    """
    if len(peak_fracs) == 0:
        return 1.0
    n = min(n_atoms, len(peak_fracs))
    fracs = np.asarray(peak_fracs[:n], dtype=np.float64)
    elements = [element] * n
    b = np.full(n, 5.0)  # mild B
    F_c = compute_structure_factors(hkl, fracs, elements, cell, b_isos=b)
    return r_factor(F_obs, F_c, scale=True)


def evaluate_success(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases_pred: np.ndarray,
    phases_true: np.ndarray,
    cell: np.ndarray,
    true_fracs: np.ndarray,
    density: Optional[np.ndarray] = None,
    thresholds: Optional[SuccessThresholds] = None,
    elements: Optional[Sequence[str]] = None,
) -> SuccessReport:
    """
    Full multi-criterion evaluation of a phasing trial against ground truth.

    Peak recovery is scored on **non-hydrogen** atoms when ``elements`` is given
    (H peaks are weak and not a fair atomicity check at typical resolution).
    """
    thr = thresholds or SuccessThresholds()
    amp = np.asarray(amplitudes, dtype=np.float64)
    F_pred = amp * np.exp(1j * np.asarray(phases_pred))
    F_true = amp * np.exp(1j * np.asarray(phases_true))

    if density is None:
        density = density_from_structure_factors(hkl, F_pred, cell)
    rho_true = density_from_structure_factors(hkl, F_true, cell, shape=density.shape)

    mapcc, shift = map_correlation_origin_invariant(density, rho_true)
    mpe, _ = mean_phase_error_origin_invariant(
        phases_pred, phases_true, hkl, weights=amp
    )

    true_fracs = np.asarray(true_fracs, dtype=np.float64).reshape(-1, 3)
    if elements is not None:
        mask = np.array([e.upper() != "H" for e in elements], dtype=bool)
        if np.any(mask):
            true_fracs = true_fracs[mask]

    n_atoms = len(true_fracs)
    n_pick = max(n_atoms + 5, int(thr.n_peaks_factor * n_atoms))
    peaks = pick_density_peaks(
        density, n_peaks=n_pick, min_sigma=thr.min_peak_sigma
    )
    peak_fracs = np.array([p.fract for p in peaks]) if peaks else np.zeros((0, 3))
    peak_rec, peak_shift = peak_recovery_score(
        peak_fracs, true_fracs, tol=thr.peak_tol, n_origin_shifts=6
    )

    r1 = None
    if thr.require_r1:
        r1 = r1_from_peaks(hkl, amp, cell, peak_fracs, n_atoms=max(n_atoms, 1))

    solved = mapcc >= thr.mapcc_min and peak_rec >= thr.peak_recovery_min
    if thr.require_r1 and r1 is not None:
        solved = solved and (r1 <= thr.r1_max)

    return SuccessReport(
        mapcc_oi=float(mapcc),
        mpe_oi_deg=float(mpe),
        peak_recovery=float(peak_rec),
        r1=r1,
        solved=bool(solved),
        details={
            "map_shift": shift,
            "peak_origin_shift": peak_shift.tolist(),
            "n_peaks": len(peaks),
            "n_atoms_true": n_atoms,
            "n_atoms_all": int(len(elements)) if elements is not None else n_atoms,
            "thresholds": {
                "mapcc_min": thr.mapcc_min,
                "peak_recovery_min": thr.peak_recovery_min,
                "r1_max": thr.r1_max,
                "require_r1": thr.require_r1,
            },
        },
    )
