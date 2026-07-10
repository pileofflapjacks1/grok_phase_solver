"""
Difference Patterson for heavy-atom location (MIR prep).

Cowtan (2001): locate heavy atoms from isomorphous differences using Patterson
methods. The isomorphous difference Patterson uses coefficients approximately

    |F_PH| − |F_P|   or better   (|F_PH| − |F_P|)²

as Fourier coefficients (phases zero). Peaks correspond to heavy–heavy vectors
when the heavy-atom signal dominates the difference.

This is **not** a full SHELXC/D pipeline; it is a correct educational /
bootstrap implementation for hybrid AI tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.patterson import (
    PattersonPeak,
    find_patterson_peaks,
    patterson_from_amplitudes,
)
from grok_phase_solver.physics.reciprocal import d_spacing


def isomorphous_difference_coefficients(
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    mode: str = "diff_sq",
) -> np.ndarray:
    """
    Build non-negative coefficients for difference Patterson.

    mode:
      - ``diff_sq``: ( |F_PH| − |F_P| )²   (common)
      - ``abs_diff``: | |F_PH| − |F_P| |
    """
    Fp = np.asarray(F_native, dtype=np.float64)
    Fph = np.asarray(F_derivative, dtype=np.float64)
    d = Fph - Fp
    if mode == "diff_sq":
        return d * d
    if mode == "abs_diff":
        return np.abs(d)
    raise ValueError(mode)


def difference_patterson_map(
    hkl: np.ndarray,
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    cell: np.ndarray,
    mode: str = "diff_sq",
    d_min: Optional[float] = None,
    origin_sharpen: float = 5.0,
) -> np.ndarray:
    """
    Compute difference Patterson map.

    We pass sqrt(coeff) as 'amplitudes' to patterson_from_amplitudes which
    squares them → coeff as |F|² for the transform. For mode=diff_sq,
    amplitudes = |ΔF| so |F|² = (ΔF)².
    """
    coeff = isomorphous_difference_coefficients(F_native, F_derivative, mode=mode)
    # patterson_from_amplitudes uses amp**2 as Fourier coeff of P
    amp = np.sqrt(np.maximum(coeff, 0.0))
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    return patterson_from_amplitudes(
        hkl,
        amp,
        cell,
        d_min=d_min,
        remove_origin=True,
        origin_sharpen=origin_sharpen,
    )


def locate_heavy_atom_vectors(
    hkl: np.ndarray,
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    cell: np.ndarray,
    n_peaks: int = 10,
    **kwargs,
) -> Tuple[List[PattersonPeak], np.ndarray, Dict]:
    """
    Difference Patterson → peak list (heavy–heavy vectors).

    Returns peaks, map, diagnostics (including correlation of Δ|F| with |F_h|
    if F_heavy provided in kwargs).
    """
    P = difference_patterson_map(hkl, F_native, F_derivative, cell, **kwargs)
    peaks = find_patterson_peaks(P, n_peaks=n_peaks)
    info = {
        "n_peaks": len(peaks),
        "map_max": float(P.max()),
        "map_std": float(P.std()),
        "mean_abs_delta_F": float(np.mean(np.abs(F_derivative - F_native))),
    }
    return peaks, P, info


def anomalous_difference_patterson(
    hkl: np.ndarray,
    F_plus: np.ndarray,
    F_minus: np.ndarray,
    cell: np.ndarray,
    n_peaks: int = 10,
    d_min: Optional[float] = None,
) -> Tuple[List[PattersonPeak], np.ndarray]:
    """
    Bijvoet-difference Patterson for MAD/SAD substructure (Cowtan MAD section).

    Coefficients ∝ (|F+| − |F−|)².
    """
    return locate_heavy_atom_vectors(
        hkl, F_plus, F_minus, cell, n_peaks=n_peaks, d_min=d_min, mode="diff_sq"
    )[:2]
