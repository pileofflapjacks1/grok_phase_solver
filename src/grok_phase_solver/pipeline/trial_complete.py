"""
Research-only trial structure completion after phasing + peak picking (v0.8).

Given a density map and an initial peak list, attempt a light iterative
completion: assign carbon placeholders to residual peaks, recompute Fcalc
phases, and blend with observed |F| for a slightly denser trial model.

**Not production.** Does not claim automatic atom typing, chain tracing, or
low-resolution protein model building. Physics peak-picking remains the
default; this is an optional post-step for small-molecule / hard-path demos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.pipeline.peaks import DensityPeak, pick_density_peaks
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.structure_factors import compute_structure_factors


def complete_trial_from_density(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    density: np.ndarray,
    phases: np.ndarray,
    *,
    n_peaks: int = 40,
    n_cycles: int = 2,
    element: str = "C",
    b_iso: float = 10.0,
    min_sigma: float = 2.2,
    d_min: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
    """
    Iterate peak pick → Fcalc → optional density rebuild → re-pick.

    Returns
    -------
    fracs : (N,3) fractional coords of trial atoms
    elements : element symbols (placeholders)
    phases_out : updated phases (radians)
    meta : research diagnostics
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64).copy()
    rho = np.asarray(density, dtype=np.float64)
    peaks: List[DensityPeak] = []
    fracs = np.zeros((0, 3), dtype=np.float64)
    els: List[str] = []

    for c in range(max(1, int(n_cycles))):
        peaks = pick_density_peaks(
            rho, n_peaks=n_peaks, min_sigma=min_sigma
        )
        if not peaks:
            break
        fracs = np.vstack([p.fract for p in peaks])
        els = [element] * len(fracs)
        b = np.full(len(fracs), float(b_iso), dtype=np.float64)
        F = compute_structure_factors(hkl, fracs, els, cell, b_isos=b)
        # blend model phases into current with soft weight
        ph_fc = np.angle(F)
        w = 0.35 + 0.15 * c
        ph = np.angle(
            (1.0 - w) * np.exp(1j * ph) + w * np.exp(1j * ph_fc)
        )
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, shape=rho.shape, d_min=d_min
        )

    meta: Dict[str, Any] = {
        "algorithm": "trial_complete_research",
        "research_only": True,
        "n_cycles": int(n_cycles),
        "n_atoms": int(len(els)),
        "element_placeholder": element,
        "n_peaks_last": len(peaks),
        "note": (
            "Research-only residual peak completion. Not auto atom typing "
            "or protein model building. Prefer trial.res + SHELXL for real work."
        ),
    }
    return fracs, els, ph, meta
