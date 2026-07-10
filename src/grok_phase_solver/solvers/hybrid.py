"""
Hybrid solvers: classical iterative methods seeded by phases from another source.

Sources of seed phases:
- Neural network predictions (Phase 2 MLP / PhAI)
- MIR centroid phases
- MR model phases
- Direct methods

Then polish with charge flipping, HIO, or density modification while
always reimposing observed |F| (Fourier consistency — non-negotiable physics).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.density_modification import density_modification_cycle
from grok_phase_solver.solvers.hio import hio_solve


def hybrid_phase_retrieval(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phase_seed: np.ndarray,
    polish: str = "charge_flipping",
    n_iter: int = 80,
    seed: int = 0,
    solvent_fraction: float = 0.45,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Seed iterative phasing with ``phase_seed`` (radians).

    polish:
      - ``charge_flipping``
      - ``hio``
      - ``density_modification``
      - ``none`` (return seed only, density from |F|e^{iφ_seed})
    """
    phase_seed = np.asarray(phase_seed, dtype=np.float64)
    if len(phase_seed) != len(amplitudes):
        raise ValueError("phase_seed length must match amplitudes")

    history: Dict = {"seed": "external", "polish": polish}

    if polish == "none":
        from grok_phase_solver.physics.density import density_from_structure_factors

        F = amplitudes * np.exp(1j * phase_seed)
        rho = density_from_structure_factors(hkl, F, cell)
        return phase_seed, rho, history

    if polish == "charge_flipping":
        phases, rho, h = charge_flipping_solve(
            hkl,
            amplitudes,
            cell,
            n_iter=n_iter,
            seed=seed,
            phase_init=phase_seed,
            verbose=verbose,
        )
        history.update(h)
        return phases, rho, history

    if polish == "hio":
        phases, rho, h = hio_solve(
            hkl,
            amplitudes,
            cell,
            n_iter=n_iter,
            seed=seed,
            phase_init=phase_seed,
            verbose=verbose,
        )
        history.update(h)
        return phases, rho, history

    if polish == "density_modification":
        phases, rho, h = density_modification_cycle(
            hkl,
            amplitudes,
            phase_seed,
            cell,
            n_iter=max(5, n_iter // 5),
            solvent_fraction=solvent_fraction,
            verbose=verbose,
        )
        history.update(h)
        return phases, rho, history

    raise ValueError(f"Unknown polish method: {polish}")


def blend_phases(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    weight_a: np.ndarray,
) -> np.ndarray:
    """
    Combine two phase sets with per-reflection weights using complex vectors.

    φ = arg( w e^{iφ_a} + (1−w) e^{iφ_b} )
    """
    w = np.clip(np.asarray(weight_a, dtype=np.float64), 0, 1)
    z = w * np.exp(1j * phase_a) + (1 - w) * np.exp(1j * phase_b)
    return np.angle(z)
