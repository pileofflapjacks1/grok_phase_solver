"""
Conditional hybrid phasing: seed → optional classical polish if free FOM improves.

Avoids the observed pathology where CF after PhAI *destroys* a good neural prior
at low resolution (see fair_phai_benchmark).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.free_fom import (
    DEFAULT_MAX_R_POS_REGRESSION,
    DEFAULT_MIN_DELTA,
    DEFAULT_REWRITE_DISP,
    DEFAULT_REWRITE_MIN_R_IMPROVE,
    compare_fom,
    free_fom,
    phase_displacement,
    should_accept_polish,
)
from grok_phase_solver.solvers.iterative_retrieval import difference_map_solve, raar_solve
from grok_phase_solver.solvers.phase_recycle import phase_recycle


def conditional_polish(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases_seed: np.ndarray,
    polish: str = "charge_flipping",
    n_iter: int = 80,
    seed: int = 0,
    d_min: Optional[float] = None,
    min_delta: float = DEFAULT_MIN_DELTA,
    max_R_pos_regression: float = DEFAULT_MAX_R_POS_REGRESSION,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Start from phases_seed; run polish; keep polished result only if free FOM rises
    without a serious R₊ regression (see ``should_accept_polish``).

    Returns (phases, density, info).
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    phases_seed = np.asarray(phases_seed, dtype=np.float64)
    rho_seed = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases_seed), cell, d_min=d_min
    )
    fom0 = free_fom(hkl, amp, phases_seed, cell, density=rho_seed)

    if polish == "charge_flipping":
        ph, rho, hist = charge_flipping_solve(
            hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min,
            phase_init=phases_seed, verbose=verbose,
        )
    elif polish == "raar":
        ph, rho, hist = raar_solve(
            hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min,
            phase_init=phases_seed, verbose=verbose,
        )
    elif polish == "difference_map":
        ph, rho, hist = difference_map_solve(
            hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min,
            phase_init=phases_seed, verbose=verbose,
        )
    elif polish == "recycle":
        ph, rho, hist = phase_recycle(
            hkl, amp, cell, n_cycles=max(5, n_iter // 10), seed=seed, d_min=d_min,
            phase_init=phases_seed, verbose=verbose,
        )
    elif polish == "none":
        return phases_seed, rho_seed, {
            "accepted_polish": False,
            "polish": polish,
            "fom_seed": fom0,
            "fom_final": fom0,
        }
    else:
        raise ValueError(polish)

    fom1 = free_fom(hkl, amp, ph, cell, density=rho)
    disp = phase_displacement(phases_seed, ph, weights=amp)
    accept = should_accept_polish(
        fom0,
        fom1,
        min_delta=min_delta,
        max_R_pos_regression=max_R_pos_regression,
        phase_disp=disp,
        rewrite_disp_threshold=DEFAULT_REWRITE_DISP,
        rewrite_min_R_improve=DEFAULT_REWRITE_MIN_R_IMPROVE,
    )
    info = {
        "accepted_polish": accept,
        "polish": polish,
        "fom_seed": fom0,
        "fom_polished": fom1,
        "fom_final": fom1 if accept else fom0,
        "fom_delta": compare_fom(fom0, fom1),
        "phase_displacement": disp,
        "history": hist,
        "gate": {
            "min_delta": min_delta,
            "max_R_pos_regression": max_R_pos_regression,
            "rewrite_disp_threshold": DEFAULT_REWRITE_DISP,
            "rewrite_min_R_improve": DEFAULT_REWRITE_MIN_R_IMPROVE,
            "fom_version": fom0.get("fom_version"),
        },
    }
    if verbose:
        print(
            f"  conditional {polish}: composite {fom0['composite']:.3f}→{fom1['composite']:.3f} "
            f"R₊ {fom0['R_pos']:.3f}→{fom1['R_pos']:.3f} disp={disp:.3f} accept={accept}"
        )
    if accept:
        return ph, rho, info
    return phases_seed, rho_seed, info


def phai_conditional_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    polish: str = "raar",
    n_iter: int = 80,
    n_phai_cycles: int = 5,
    seed: int = 0,
    d_min: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """PhAI fair seed + conditional polish (if PhAI unavailable, CF alone)."""
    try:
        from grok_phase_solver.models.phai_fair import run_phai_fair
        from grok_phase_solver.models.phai_runner import phai_available

        if phai_available():
            ph0, meta = run_phai_fair(
                hkl, amplitudes, n_cycles=n_phai_cycles, seed=seed
            )
            ph, rho, info = conditional_polish(
                hkl, amplitudes, cell, ph0, polish=polish, n_iter=n_iter,
                seed=seed, d_min=d_min, verbose=verbose,
            )
            info["phai_meta"] = meta
            info["seed"] = "phai_fair"
            return ph, rho, info
    except Exception as e:
        if verbose:
            print(f"  PhAI unavailable ({e}); CF fallback")

    ph, rho, hist = charge_flipping_solve(
        hkl, amplitudes, cell, n_iter=n_iter, seed=seed, d_min=d_min, verbose=verbose
    )
    return ph, rho, {"seed": "cf_fallback", "history": hist, "accepted_polish": True}
