"""
Patterson-method baseline solver API.

Computes Patterson map from |F|, picks peaks, and (for tiny structures)
attempts a naive heavy-atom / few-atom bootstrap for comparison metrics.

Full automated Patterson solving (image-seeking, symmetry minimum functions)
is deferred; this module provides the classical baseline Cowtan describes
for small molecules and heavy-atom location (MIR/MAD prep).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.patterson import (
    find_patterson_peaks,
    patterson_from_amplitudes,
    patterson_peak_recovery_score,
)
from grok_phase_solver.physics.reciprocal import d_spacing


def patterson_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_peaks: int = 15,
    d_min: Optional[float] = None,
    origin_sharpen: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Patterson baseline: map + peak list; phases remain undetermined in general.

    Returns
    -------
    phases : random phases (Patterson alone does not yield general phases;
        for centrosymmetric heavy-atom cases user should call specialized path)
    patterson_map : real-space P(u)
    info : peaks, diagnostics
    """
    rng = np.random.default_rng(seed)
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    P = patterson_from_amplitudes(
        hkl,
        amplitudes,
        cell,
        d_min=d_min,
        remove_origin=True,
        origin_sharpen=origin_sharpen,
    )
    peaks = find_patterson_peaks(P, n_peaks=n_peaks)
    if verbose:
        print(f"  Patterson: shape={P.shape}, n_peaks={len(peaks)}")
        for p in peaks[:5]:
            print(f"    peak {p.rank}: fract={p.fract}, height={p.height:.3g}")

    # Patterson does not assign general phases — return random as null phase
    # but attach map for hybrid pipelines (heavy-atom search → MIR)
    phases = rng.uniform(-np.pi, np.pi, size=len(amplitudes))
    info = {
        "peaks": peaks,
        "map": P,
        "method": "patterson",
        "note": (
            "Patterson yields interatomic vectors, not general phases. "
            "Use peaks for heavy-atom location (MIR/MAD) or few-atom solve."
        ),
    }
    return phases, P, info


def patterson_density_check(
    hkl: np.ndarray,
    F_complex: np.ndarray,
    cell: np.ndarray,
) -> Dict[str, float]:
    """
    Verify P ≈ autocorrelation(ρ) for phased F (Parseval consistency check).
    """
    from grok_phase_solver.physics.patterson import autocorrelation_density

    amp = np.abs(F_complex)
    P_F = patterson_from_amplitudes(hkl, amp, cell, remove_origin=False)
    rho = density_from_structure_factors(hkl, F_complex, cell, shape=P_F.shape)
    P_rho = autocorrelation_density(rho)
    # Correlate (origin-aligned)
    a = P_F.ravel() - P_F.mean()
    b = P_rho.ravel() - P_rho.mean()
    cc = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-16))
    return {"patterson_vs_autocorr_cc": cc, "P_max": float(P_F.max()), "rho_max": float(rho.max())}
