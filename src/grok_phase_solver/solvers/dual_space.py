"""
Dual-space recycling (educational, SHELXD-inspired).

Classic Sheldrick dual-space idea:
  reciprocal space  →  real-space peak list (atomicity)
  atom list         →  Fcalc phases → re-impose |Fobs|

This is **not** SHELXD. It is a transparent in-repo baseline for head-to-head
tables when the proprietary ``shelxd`` binary is unavailable. Use
``solvers/shelxd_runner.py`` for the real external tool.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.pipeline.peaks import pick_density_peaks
from grok_phase_solver.solvers.free_fom import free_fom


def _phases_from_atoms(
    hkl: np.ndarray,
    fracs: np.ndarray,
    elements: List[str],
    cell: np.ndarray,
) -> np.ndarray:
    if len(fracs) == 0:
        return np.zeros(len(hkl))
    F = compute_structure_factors(hkl, fracs, elements, cell)
    return np.angle(F)


def _r_factor(amp_obs: np.ndarray, amp_calc: np.ndarray) -> float:
    o = np.asarray(amp_obs, dtype=np.float64)
    c = np.asarray(amp_calc, dtype=np.float64)
    scale = float(np.dot(o, c) / (np.dot(c, c) + 1e-16))
    return float(np.sum(np.abs(o - scale * c)) / (np.sum(o) + 1e-16))


def dual_space_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_atoms: int = 12,
    n_cycles: int = 40,
    n_starts: int = 8,
    seed: int = 0,
    d_min: Optional[float] = None,
    element: str = "C",
    peak_sigma: float = 2.0,
    prior_phases: Optional[np.ndarray] = None,
    polish_cf_iters: int = 20,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multi-start dual-space atom recycling.

    Parameters
    ----------
    n_atoms
        Number of peaks kept each cycle (FIND n analogue).
    n_cycles
        Dual-space iterations per start.
    n_starts
        Random phase starts (plus optional prior start).
    """
    hkl = np.asarray(hkl, dtype=np.float64)
    amp = np.asarray(amplitudes, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    rng = np.random.default_rng(seed)

    best = {
        "score": -1e99,
        "phases": None,
        "rho": None,
        "R": 1.0,
        "fom": None,
        "start": -1,
        "fracs": None,
    }

    starts: List[np.ndarray] = []
    if prior_phases is not None:
        starts.append(np.asarray(prior_phases, dtype=np.float64).copy())
    for s in range(n_starts):
        starts.append(rng.uniform(-np.pi, np.pi, size=len(amp)))

    for si, ph0 in enumerate(starts):
        ph = ph0.copy()
        fracs_best = None
        for cy in range(n_cycles):
            rho = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph), cell, d_min=d_min
            )
            peaks = pick_density_peaks(
                rho, n_peaks=n_atoms, min_sigma=peak_sigma, min_fract_dist=0.06
            )
            if not peaks:
                # random restart mid-run
                ph = rng.uniform(-np.pi, np.pi, size=len(amp))
                continue
            fracs = np.array([p.fract for p in peaks], dtype=np.float64)
            elements = [element] * len(fracs)
            # optional: weight occupancy by peak height
            Fcalc = compute_structure_factors(hkl, fracs, elements, cell)
            ph_calc = np.angle(Fcalc)
            # gradual blend early → later trust atoms more
            alpha = 0.35 + 0.55 * (cy / max(n_cycles - 1, 1))
            # complex average then re-take angle
            z = (1 - alpha) * np.exp(1j * ph) + alpha * np.exp(1j * ph_calc)
            ph = np.angle(z)
            fracs_best = fracs

        # score start
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, d_min=d_min
        )
        fom = free_fom(hkl, amp, ph, cell, density=rho, include_shells=False)
        R = 1.0
        if fracs_best is not None and len(fracs_best):
            Fcalc = compute_structure_factors(
                hkl, fracs_best, [element] * len(fracs_best), cell
            )
            R = _r_factor(amp, np.abs(Fcalc))
        # composite: free FOM up, R down
        score = float(fom["composite"]) - 0.5 * R
        if score > best["score"]:
            best.update(
                {
                    "score": score,
                    "phases": ph.copy(),
                    "rho": rho,
                    "R": R,
                    "fom": fom,
                    "start": si,
                    "fracs": fracs_best,
                }
            )
        if verbose:
            print(
                f"  dual_space start {si}: FOM={fom['composite']:.3f} R≈{R:.3f} "
                f"score={score:.3f}"
            )

    ph = best["phases"]
    rho = best["rho"]
    if ph is None:
        ph = rng.uniform(-np.pi, np.pi, size=len(amp))
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, d_min=d_min
        )

    # light CF polish
    if polish_cf_iters > 0:
        from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

        ph_p, rho_p, _ = charge_flipping_solve(
            hkl, amp, cell, n_iter=polish_cf_iters, seed=seed + 7, d_min=d_min,
            phase_init=ph,
        )
        fom_p = free_fom(hkl, amp, ph_p, cell, density=rho_p, include_shells=False)
        fom0 = best["fom"] or free_fom(hkl, amp, ph, cell, density=rho, include_shells=False)
        if fom_p["composite"] >= fom0["composite"] - 0.02:
            ph, rho = ph_p, rho_p
            best["fom"] = fom_p
            best["polished"] = True
        else:
            best["polished"] = False
    else:
        best["polished"] = False

    info = {
        "method": "dual_space",
        "note": (
            "Educational dual-space recycling (SHELXD-inspired). "
            "Not SHELXD — use shelxd_runner for the external binary."
        ),
        "n_atoms": n_atoms,
        "n_cycles": n_cycles,
        "n_starts": len(starts),
        "best_start": best["start"],
        "R_partial": best["R"],
        "score": best["score"],
        "free_fom": best["fom"],
        "polished_cf": best.get("polished", False),
        "n_peaks_model": 0 if best["fracs"] is None else len(best["fracs"]),
    }
    return ph, rho, info
