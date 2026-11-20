"""
Strict success criteria for ab initio phasing experiments.

A trial is counted as **solved** only if multiple independent checks pass:

1. **mapCC_OI ≥ τ_cc** — origin/enantiomorph-invariant density correlation
2. **peak recovery ≥ τ_peak** — fraction of true atoms matched one-to-one by a
   density peak within ``peak_tol_angstrom`` (Å), maximized over a continuous
   (P1) origin translation and both hands (x → −x)
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
    # Legacy fractional min-image tolerance. Only the deprecated cell-free
    # peak_recovery_score path and the typed-R1 diagnostic still use it.
    peak_tol: float = 0.15
    # Peak ↔ atom match distance for peak recovery (Å). ~half a C–C bond:
    # tight enough that random peaks fail, loose enough for grid-quantized peaks.
    peak_tol_angstrom: float = 0.5
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


def _min_image_dist_angstrom(
    a_fract: np.ndarray, p_fract: np.ndarray, G: np.ndarray
) -> np.ndarray:
    """(n_a, n_p) minimum-image distances in Å between two fractional sets."""
    d = a_fract[:, None, :] - p_fract[None, :, :]
    d = d - np.round(d)
    return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, G, d), 0.0))


def match_peaks_to_atoms(
    peaks_fract: np.ndarray,
    true_fracs: np.ndarray,
    cell: np.ndarray,
    tol_angstrom: float = 0.5,
    allow_inversion: bool = True,
    max_anchor_peaks: int = 20,
) -> Dict:
    """
    One-to-one peak ↔ atom matching under the P1 origin group.

    The peak set may differ from the truth by any lattice translation t and
    (for a non-centrosymmetric solution) by inversion, so atoms are mapped as
    ``hand · x + t`` before matching. Candidate translations come from
    anchoring each true atom on each of the strongest peaks
    (``t = p_j − hand · x_i``), so a correct solution always has an exact
    candidate. The best candidates are refined by least squares on the matched
    pairs and scored with a Hungarian assignment (each peak used at most once).

    Distances are in Å through the direct metric tensor, so the tolerance does
    not depend on cell edge lengths.

    For space groups above P1 this is the most permissive origin choice
    (an upper bound on recovery); peaks must be the full-cell list matching
    the atoms passed in.

    Returns dict with ``recovery``, ``n_matched``, ``translation``, ``hand``
    (+1 / −1), ``rms_angstrom`` (over matched pairs) and ``pairs``.
    """
    from scipy.optimize import linear_sum_assignment

    from grok_phase_solver.physics.reciprocal import metric_tensor

    peaks = np.asarray(peaks_fract, dtype=np.float64).reshape(-1, 3)
    atoms = np.asarray(true_fracs, dtype=np.float64).reshape(-1, 3)
    empty = {
        "recovery": 0.0,
        "n_matched": 0,
        "translation": np.zeros(3),
        "hand": 1,
        "rms_angstrom": None,
        "pairs": [],
    }
    if len(peaks) == 0 or len(atoms) == 0:
        return empty
    G = metric_tensor(np.asarray(cell, dtype=np.float64))
    anchors = peaks[: max(1, int(max_anchor_peaks))]

    def _score(x: np.ndarray, t: np.ndarray) -> Tuple[int, np.ndarray, list]:
        D = _min_image_dist_angstrom((x + t) % 1.0, peaks, G)
        cost = np.where(D <= tol_angstrom, D, 1e6)
        ri, ci = linear_sum_assignment(cost)
        ok = cost[ri, ci] <= tol_angstrom
        return int(ok.sum()), D[ri[ok], ci[ok]], list(zip(ri[ok].tolist(), ci[ok].tolist()))

    best = dict(empty)
    best_key = (-1, -np.inf)
    hands = (1, -1) if allow_inversion else (1,)
    for hand in hands:
        x = hand * atoms
        # All anchor translations at once: (n_atoms * n_anchor, 3)
        T = (anchors[None, :, :] - x[:, None, :]).reshape(-1, 3)
        T = T - np.floor(T)
        # Cheap many-to-one count per candidate to shortlist
        counts = np.empty(len(T), dtype=int)
        for s in range(0, len(T), 256):
            Tc = T[s : s + 256]
            d = (x[None, :, None, :] + Tc[:, None, None, :]) - peaks[None, None, :, :]
            d = d - np.round(d)
            dist2 = np.einsum("...i,ij,...j->...", d, G, d)
            counts[s : s + 256] = (dist2.min(axis=2) <= tol_angstrom**2).sum(axis=1)
        for idx in np.argsort(-counts, kind="stable")[:8]:
            t = T[idx]
            for _ in range(3):  # refine t on matched pairs
                n, _, pairs = _score(x, t)
                if n == 0:
                    break
                ai = np.array([p[0] for p in pairs])
                pj = np.array([p[1] for p in pairs])
                r = peaks[pj] - (x[ai] + t)
                r = r - np.round(r)
                t = (t + r.mean(axis=0)) % 1.0
            n, dists, pairs = _score(x, t)
            rms = float(np.sqrt(np.mean(dists**2))) if n else np.inf
            key = (n, -rms)
            if key > best_key:
                best_key = key
                best = {
                    "recovery": n / len(atoms),
                    "n_matched": n,
                    "translation": t.copy(),
                    "hand": hand,
                    "rms_angstrom": rms if n else None,
                    "pairs": pairs,
                }
    return best


def peak_recovery_score(
    peaks_fract: np.ndarray,
    true_fracs: np.ndarray,
    tol: float = 0.12,
    n_origin_shifts: int = 4,
    *,
    cell: Optional[np.ndarray] = None,
    tol_angstrom: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    Fraction of true atoms matched by a peak, and the best translation.

    With ``cell`` (recommended) this is :func:`match_peaks_to_atoms`: Å
    tolerance, one-to-one matching, continuous origin, both hands. The
    returned translation applies to ``hand · x``; read ``hand`` from
    :func:`match_peaks_to_atoms` when the solution may be inverted.

    Without ``cell`` the legacy score is kept for old scripts: a *fractional*
    ``tol`` on a coarse {0, 1/n, …}³ origin grid, many-to-one, one hand.
    Random peak sets pass that legacy score (≈0.6–0.9 recovery at tol 0.15),
    so do not use it as a success gate.
    """
    if cell is not None:
        m = match_peaks_to_atoms(peaks_fract, true_fracs, cell, tol_angstrom=tol_angstrom)
        return float(m["recovery"]), np.asarray(m["translation"])
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
    match = match_peaks_to_atoms(
        peak_fracs, true_fracs, cell, tol_angstrom=thr.peak_tol_angstrom
    )
    peak_rec = float(match["recovery"])
    peak_shift = np.asarray(match["translation"])

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
            "peak_hand": int(match["hand"]),
            "peak_rms_angstrom": match["rms_angstrom"],
            "peak_tol_angstrom": thr.peak_tol_angstrom,
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
