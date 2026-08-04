"""
Research generative structure / density proposal (v0.12).

Inspired by end-to-end generative crystallography (XDXD-style) and
powder/structure diffusion (PXRDGen / XRDSol conceptual lineage), this module
offers a **lightweight, weight-free** proposal path:

1. Short classical density (CF) → peak pick → trial atom list (composition-aware).
2. Soft Fcalc phase seed from those atoms (MR-lite style).
3. Optional Langevin / CF polish with physics fallback.

Honest non-claims
-----------------
- Not a trained generative model; no external weights redistributed.
- Not auto default; research-flagged only.
- Hard ab initio remains unsolved; prefer partial_phaseed for hard cells.
- Full SE(3) equivariant atomic diffusion remains a future external research path
  (see ``diffusion_se3_stub``).

Physics fallback: classical ensemble / charge flipping if proposal fails.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.projectors import unit_cell_volume


def generative_structure_available() -> bool:
    """Always available as a physics/heuristic path (no learned weights required)."""
    return True


def estimate_composition_from_volume(
    cell: np.ndarray,
    *,
    n_atoms_user: Optional[int] = None,
    default_element: str = "C",
    ha_element: Optional[str] = None,
    vol_per_atom: float = 18.0,
) -> Tuple[List[str], int]:
    """
    Rough ASU composition guess from cell volume (or user n_atoms).

    Returns (elements list for trial model, n_atoms).
    """
    if n_atoms_user is not None and n_atoms_user > 0:
        n = int(n_atoms_user)
    else:
        vol = float(unit_cell_volume(np.asarray(cell, dtype=np.float64)))
        n = int(np.clip(round(vol / max(vol_per_atom, 1.0)), 4, 80))
    els = [default_element] * n
    if ha_element is not None and n >= 1:
        els[0] = ha_element
        if n >= 4:
            els[1] = "O"
            els[2] = "N"
    else:
        for i in range(min(n // 4, 3)):
            els[i] = "O"
        for i in range(n // 4, min(n // 4 + 2, n)):
            els[i] = "N"
    return els, n


def propose_trial_atoms_from_density(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_atoms: int = 12,
    elements: Optional[Sequence[str]] = None,
    d_min: Optional[float] = None,
    seed: int = 0,
    n_cf_iter: int = 25,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Classical CF density → peak pick → trial fractional coordinates.

    Research heuristic for generative proposal seeds (not a trained denoise model).
    """
    from grok_phase_solver.pipeline.peaks import pick_density_peaks
    from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    meta: Dict = {
        "algorithm": "cf_peak_proposal",
        "n_atoms": int(n_atoms),
        "research_only": True,
    }
    try:
        ph_cf, rho, _ = charge_flipping_solve(
            hkl, amp, cell, n_iter=int(n_cf_iter), seed=seed, d_min=d_min
        )
        peaks = pick_density_peaks(rho, n_peaks=max(n_atoms * 2, 8), min_sigma=1.5)
        fracs_list = [np.asarray(p.fract, dtype=np.float64) for p in peaks]
        meta["n_peaks_found"] = len(fracs_list)
        meta["cf_ok"] = True
    except Exception as e:
        fracs_list = []
        meta["cf_ok"] = False
        meta["cf_error"] = str(e)

    if not fracs_list:
        rng = np.random.default_rng(seed)
        fracs = rng.random((n_atoms, 3))
        meta["algorithm"] = "random_fallback"
    else:
        fracs = np.vstack(fracs_list)[:n_atoms]
        if len(fracs) < n_atoms:
            rng = np.random.default_rng(seed)
            pad = rng.random((n_atoms - len(fracs), 3))
            fracs = np.vstack([fracs, pad])

    els = list(elements) if elements is not None else ["C"] * n_atoms
    if len(els) < n_atoms:
        els = (els + ["C"] * n_atoms)[:n_atoms]
    return np.asarray(fracs, dtype=np.float64), els[:n_atoms], meta


def propose_phases_from_trial(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    *,
    blend: float = 0.35,
    seed_phases: Optional[np.ndarray] = None,
    d_min: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Soft Fcalc phase seed from trial atoms (generative proposal → phase prior).

    Returns (phases, density, meta). Physics |F| projection retained.
    """
    from grok_phase_solver.physics.structure_factors import compute_structure_factors

    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    F = compute_structure_factors(hkl, fracs, list(elements), cell)
    ph_fc = np.angle(F)
    if seed_phases is not None:
        ph0 = np.asarray(seed_phases, dtype=np.float64)
        w = float(np.clip(blend, 0.0, 1.0))
        ph = np.angle((1.0 - w) * np.exp(1j * ph0) + w * np.exp(1j * ph_fc))
    else:
        ph = ph_fc
    Fobs = amp * np.exp(1j * ph)
    rho = density_from_structure_factors(hkl, Fobs, cell, d_min=d_min)
    meta = {
        "algorithm": "generative_fcalc_seed",
        "blend": float(blend),
        "n_trial_atoms": int(len(fracs)),
        "research_only": True,
        "note": (
            "Heuristic CF-peak→Fcalc seed; not a trained generative model. "
            "Prefer partial_phaseed for hard experimental data."
        ),
    }
    return ph, rho, meta


def generative_structure_propose(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_atoms: Optional[int] = None,
    elements: Optional[Sequence[str]] = None,
    ha_element: Optional[str] = None,
    d_min: Optional[float] = None,
    blend: float = 0.35,
    polish: str = "none",
    n_polish: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    End-to-end research proposal: composition guess → CF peaks → Fcalc seed
    → optional classical / Langevin polish.

    ``polish``: ``none`` | ``cf`` | ``diffusion`` (physics Langevin fallback).
    """
    if elements is None:
        els, n = estimate_composition_from_volume(
            cell, n_atoms_user=n_atoms, ha_element=ha_element
        )
    else:
        els = list(elements)
        n = int(n_atoms or len(els))

    fracs, els2, pmeta = propose_trial_atoms_from_density(
        hkl, amplitudes, cell, n_atoms=n, elements=els, d_min=d_min, seed=seed
    )
    ph, rho, fmeta = propose_phases_from_trial(
        hkl, amplitudes, cell, fracs, els2, blend=blend, d_min=d_min
    )
    meta: Dict = {
        "algorithm": "generative_structure_propose_v1",
        "research_only": True,
        "density_proposal": pmeta,
        "fcalc_seed": fmeta,
        "trial_fracs": fracs.tolist(),
        "trial_elements": list(els2),
        "available": generative_structure_available(),
        "fallback": "ensemble / charge_flipping / partial_phaseed",
        "note": (
            "Research generative path (no trained weights). "
            "Not used by auto; does not claim end-to-end phase solution."
        ),
    }

    if polish == "cf":
        from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

        ph2, rho2, ch = charge_flipping_solve(
            hkl, amplitudes, cell, n_iter=n_polish, seed=seed, d_min=d_min
        )
        # keep softer blend with proposal
        ph = np.angle(0.4 * np.exp(1j * ph) + 0.6 * np.exp(1j * ph2))
        rho = density_from_structure_factors(
            hkl, amplitudes * np.exp(1j * ph), cell, d_min=d_min
        )
        meta["polish"] = {"method": "cf", "n_iter": n_polish}
    elif polish in ("diffusion", "langevin"):
        from grok_phase_solver.models.diffusion_phase import reverse_diffusion_phases

        ph, rho, hist = reverse_diffusion_phases(
            hkl,
            amplitudes,
            cell,
            n_steps=max(5, n_polish // 2),
            seed_phases=ph,
            seed=seed,
            d_min=d_min,
            use_learned_score=False,
        )
        meta["polish"] = {"method": "diffusion_langevin", "hist_keys": list(hist.keys()) if isinstance(hist, dict) else []}
    else:
        meta["polish"] = {"method": "none"}

    return ph, rho, meta
