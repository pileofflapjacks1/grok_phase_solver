"""
Multistart ensemble phase retrieval with free-FOM selection.

Runs several independent starts of charge flipping and/or RAAR (and optionally
DiffMap), ranks solutions by truth-free composite FOM, and returns the best.

This is the classical analogue of multistart direct methods: different random
phase seeds explore different basins; free FOMs pick without ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.iterative_retrieval import (
    difference_map_solve,
    raar_solve,
)


def _run_one(
    method: str,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int,
    seed: int,
    d_min: Optional[float],
    phase_init: Optional[np.ndarray],
    **method_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if method == "cf":
        return charge_flipping_solve(
            hkl,
            amplitudes,
            cell,
            n_iter=n_iter,
            seed=seed,
            d_min=d_min,
            phase_init=phase_init,
            **{k: v for k, v in method_kwargs.items() if k in (
                "delta_sigma", "weak_fraction", "centrosymmetric", "sampling", "verbose",
            )},
        )
    if method == "raar":
        return raar_solve(
            hkl,
            amplitudes,
            cell,
            n_iter=n_iter,
            seed=seed,
            d_min=d_min,
            phase_init=phase_init,
            **{k: v for k, v in method_kwargs.items() if k in (
                "beta", "real_proj", "sampling", "verbose",
            )},
        )
    if method == "diffmap":
        return difference_map_solve(
            hkl,
            amplitudes,
            cell,
            n_iter=n_iter,
            seed=seed,
            d_min=d_min,
            phase_init=phase_init,
            **{k: v for k, v in method_kwargs.items() if k in (
                "beta", "real_proj", "delta_sigma", "sampling", "verbose",
            )},
        )
    raise ValueError(f"Unknown ensemble method: {method}")


def ensemble_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    methods: Sequence[str] = ("cf", "raar"),
    n_starts: int = 5,
    n_iter: int = 120,
    base_seed: int = 0,
    d_min: Optional[float] = None,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
    **method_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multistart ensemble: run ``n_starts`` trials per method, pick best free FOM.

    Parameters
    ----------
    methods : sequence of ``"cf"``, ``"raar"``, ``"diffmap"``
    n_starts : independent random seeds per method
    n_iter : iterations per trial
    base_seed : first seed; trial *i* uses ``base_seed + i``
    phase_init : optional shared seed phases (still varies CF weak-phase RNG)

    Returns
    -------
    phases, density, info
        ``info`` includes ``best_fom``, ``all_foms``, ``best_method``, ``n_trials``.
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    trials: List[Dict] = []
    best = None  # (composite, phases, rho, trial_meta)

    trial_id = 0
    for method in methods:
        for s in range(n_starts):
            seed = base_seed + trial_id
            trial_id += 1
            try:
                ph, rho, hist = _run_one(
                    method,
                    hkl,
                    amp,
                    cell,
                    n_iter=n_iter,
                    seed=seed,
                    d_min=d_min,
                    phase_init=phase_init,
                    verbose=False,
                    **method_kwargs,
                )
                fom = free_fom(hkl, amp, ph, cell, density=rho)
                meta = {
                    "method": method,
                    "seed": seed,
                    "composite": fom["composite"],
                    "pos_frac": fom["pos_frac"],
                    "skewness": fom["skewness"],
                    "R_after_ER": fom["R_after_ER"],
                    "final_R": hist.get("final_R", hist.get("R", [None])[-1] if hist.get("R") else None),
                }
                trials.append(meta)
                if verbose:
                    print(
                        f"  ensemble [{method}] seed={seed} "
                        f"composite={fom['composite']:.3f} R={meta['final_R']}"
                    )
                if best is None or fom["composite"] > best[0]:
                    best = (fom["composite"], ph, rho, meta, fom, hist)
            except Exception as e:
                trials.append({
                    "method": method,
                    "seed": seed,
                    "error": str(e),
                    "composite": -1.0,
                })
                if verbose:
                    print(f"  ensemble [{method}] seed={seed} ERROR: {e}")

    if best is None:
        raise RuntimeError("All ensemble trials failed")

    _, ph, rho, meta, fom, hist = best
    info = {
        "algorithm": "ensemble",
        "methods": list(methods),
        "n_starts_per_method": n_starts,
        "n_trials": len(trials),
        "best_method": meta["method"],
        "best_seed": meta["seed"],
        "best_fom": fom,
        "all_foms": trials,
        "history": hist,
    }
    if verbose:
        print(
            f"  ensemble BEST: {meta['method']} seed={meta['seed']} "
            f"composite={fom['composite']:.3f}"
        )
    return ph, rho, info


def ensemble_cf_raar(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_starts: int = 5,
    n_iter: int = 120,
    base_seed: int = 0,
    d_min: Optional[float] = None,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Convenience: multistart CF + RAAR, free-FOM pick."""
    return ensemble_solve(
        hkl,
        amplitudes,
        cell,
        methods=("cf", "raar"),
        n_starts=n_starts,
        n_iter=n_iter,
        base_seed=base_seed,
        d_min=d_min,
        verbose=verbose,
        **kwargs,
    )
