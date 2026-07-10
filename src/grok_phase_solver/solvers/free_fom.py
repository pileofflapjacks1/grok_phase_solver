"""
Truth-free figures of merit for ranking phase sets / hybrids.

Used when ground-truth phases are unavailable (real experiments) or for
conditional hybrid decisions (apply CF polish only if FOM improves).

FOMs (higher is better after sign flip where noted):
  - positivity: fraction of non-negative density (higher better)
  - skewness: atomic maps are positively skewed (higher better)
  - R_mod: residual of density-derived |F| vs |F_obs| after one ER step (lower better)
  - composite: weighted combination → higher better
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.projectors import (
    density_to_F,
    project_modulus,
    project_positivity,
    r_factor_moduli,
)


def free_fom(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    density: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute truth-free diagnostics for a phase set."""
    amp = np.asarray(amplitudes, dtype=np.float64)
    if density is None:
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * np.asarray(phases)), cell
        )
    rho = np.asarray(density, dtype=np.float64)
    pos_frac = float((rho >= 0).mean())
    # Fisher skewness
    m = rho.mean()
    s = rho.std() + 1e-16
    skew = float(np.mean(((rho - m) / s) ** 3))
    # One ER step residual
    rho_p = project_positivity(rho)
    F = density_to_F(rho_p, hkl, cell)
    F_m = project_modulus(F, amp)
    R = r_factor_moduli(F_m, amp)
    # Composite: prefer high positivity, high skew, low R
    # Map R∈[0,∞) → score; clamp
    R_score = 1.0 / (1.0 + R)
    skew_score = 1.0 / (1.0 + np.exp(-skew))  # logistic of skew
    composite = 0.4 * pos_frac + 0.3 * skew_score + 0.3 * R_score
    return {
        "pos_frac": pos_frac,
        "skewness": skew,
        "R_after_ER": R,
        "composite": float(composite),
    }


def should_accept_polish(
    fom_before: Dict[str, float],
    fom_after: Dict[str, float],
    min_delta: float = 0.01,
) -> bool:
    """Accept polish if composite FOM improves by at least min_delta."""
    return fom_after["composite"] >= fom_before["composite"] + min_delta
