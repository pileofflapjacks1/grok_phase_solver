"""Crystallographic R-factors."""

from __future__ import annotations

from typing import Optional

import numpy as np

from grok_phase_solver.physics.structure_factors import scale_to_observed


def r_factor(
    F_obs: np.ndarray,
    F_calc: np.ndarray,
    weights: Optional[np.ndarray] = None,
    scale: bool = True,
) -> float:
    """
    R = Σ ||F_obs| − k|F_calc|| / Σ |F_obs|

    Classic residual between observed and calculated amplitudes.
    """
    Fo = np.abs(np.asarray(F_obs, dtype=np.float64))
    Fc = np.abs(np.asarray(F_calc, dtype=np.complex128))
    if scale:
        _, k = scale_to_observed(Fc.astype(np.complex128), Fo, weights=weights)
        Fc = k * Fc
    if weights is None:
        return float(np.sum(np.abs(Fo - Fc)) / (np.sum(Fo) + 1e-16))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * np.abs(Fo - Fc)) / (np.sum(w * Fo) + 1e-16))


def r_free(
    F_obs: np.ndarray,
    F_calc: np.ndarray,
    free_flag: np.ndarray,
    free_value: int = 1,
) -> float:
    """R-factor on free-set reflections only."""
    mask = np.asarray(free_flag) == free_value
    if not np.any(mask):
        raise ValueError("No free-set reflections")
    return r_factor(F_obs[mask], F_calc[mask])
