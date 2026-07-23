"""
Conditional diffusion-style phase completion (experimental, NumPy-first).

Research-aligned hybrid for density / phase refinement conditioned on |F(hkl)|
and optional partial phase seeds. Inspired by score-based / iterative denoising
ideas in powder and single-crystal ML phasing literature (e.g. PXRDnet-style
map completion, equivariant diffusion phasing concepts) — **not** a reimplementation
of those systems and **not** claimed to match their published metrics.

Algorithm (physics-first Langevin / annealed noise on the phase circle)
----------------------------------------------------------------------
1. Initialize φ from seed (or random on weak reflections).
2. For t = T … 1 (annealing schedule):
   a. Build F = |F_obs| exp(iφ); ρ = IFFT(F)
   b. Project positivity (and optional light solvent flatten)
   c. F' = FFT(ρ); reimpose |F_obs| → φ_data
   d. Score / drift: blend φ toward φ_data and seed prior
   e. Add annealed Gaussian noise on the circle (diffusion reverse step)
3. Optional free-FOM–gated classical polish (CF)

Physics checks
--------------
- Modulus projection each step (Parseval-friendly embedding via existing FFT)
- Positivity projection (atomicity proxy)
- Seed re-imposition weight (partial-φ consistency)

Honest status
-------------
- Default path is **inference-only** with no external neural weights.
- Optional torch score network hook if a checkpoint is provided later.
- Prefer ensemble / AI-PhaSeed / partial_phaseed for production hard-data solves.
- Hard ab initio seed bar is unchanged; this module is a hybrid **basin polish**.

References (inspired by / related)
----------------------------------
- Score-based generative models for inverse problems (Song et al. lineage)
- PXRDnet (Nature Materials 2025) — powder / map completion (conceptual)
- XRDSol (Nature Comm. 2026) — equivariant diffusion phasing (conceptual)
- Project: AI-PhaSeed, free FOM v2.1, projectors
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grok_phase_solver.physics.density import (
    density_from_structure_factors,
    grid_shape_from_resolution,
)
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.hybrid import blend_phases
from grok_phase_solver.solvers.phase_recycle import fourier_modulus_projection
from grok_phase_solver.solvers.projectors import (
    density_to_F,
    project_positivity,
    r_factor_moduli,
    unit_cell_volume,
)

PathLike = Union[str, Path]


def diffusion_phase_available(weights_path: Optional[PathLike] = None) -> bool:
    """
    True if a trained score-network checkpoint is present.

    The physics Langevin path always runs; this flag is for optional NN scoring.
    """
    if weights_path is not None and Path(weights_path).is_file():
        return True
    cand = Path(__file__).resolve().parents[3] / "models" / "diffusion_phase.pt"
    return cand.is_file()


def _noise_schedule(n_steps: int, sigma_max: float = 0.8, sigma_min: float = 0.02) -> np.ndarray:
    """Log-spaced noise levels (high → low), radians scale for phase circle."""
    if n_steps <= 1:
        return np.array([sigma_min], dtype=np.float64)
    return np.geomspace(sigma_max, sigma_min, n_steps)


def _wrap(ph: np.ndarray) -> np.ndarray:
    return (np.asarray(ph, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def physics_score_step(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: np.ndarray,
    *,
    seed_phases: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    seed_weight: float = 0.5,
    d_min: Optional[float] = None,
    solvent_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    One physics-consistent denoising step: positivity + modulus → data phase.

    Returns (phases_data, R_pos_proxy, density).
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64).copy()

    def density_op(rho: np.ndarray) -> np.ndarray:
        out = project_positivity(rho)
        if solvent_fraction is not None and solvent_fraction > 0:
            from grok_phase_solver.solvers.density_modification import solvent_flatten

            out = solvent_flatten(out, solvent_fraction=solvent_fraction)
        return out

    ph2, rho = fourier_modulus_projection(
        hkl, amp, ph, cell, density_op=density_op, d_min=d_min
    )
    if seed_phases is not None and seed_weight > 0:
        sp = np.asarray(seed_phases, dtype=np.float64)
        if seed_mask is not None:
            m = np.asarray(seed_mask, dtype=bool)
            w = np.zeros(len(ph2), dtype=np.float64)
            w[m] = float(seed_weight)
            ph2 = blend_phases(sp, ph2, w)
        else:
            ph2 = blend_phases(sp, ph2, np.full(len(ph2), float(seed_weight)))
    F = density_to_F(project_positivity(rho), hkl, cell)
    R = r_factor_moduli(F, amp)
    return _wrap(ph2), float(R), rho


def reverse_diffusion_phases(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    seed_phases: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    n_steps: int = 25,
    seed: int = 0,
    d_min: Optional[float] = None,
    seed_weight_start: float = 0.85,
    seed_weight_end: float = 0.35,
    data_weight: float = 0.55,
    sigma_max: float = 0.75,
    sigma_min: float = 0.03,
    solvent_fraction: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Annealed Langevin-style reverse process on the phase torus.

    φ ← (1−α) φ + α φ_data + √(2η) ξ   with ξ ~ N(0,I) on R, then wrap.

    Returns phases, density, history.
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    n = len(amp)
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))

    if seed_phases is not None:
        ph = np.asarray(seed_phases, dtype=np.float64).copy()
        if seed_mask is not None:
            m = np.asarray(seed_mask, dtype=bool)
            ph = ph.copy()
            # randomize unknown
            ph[~m] = rng.uniform(-np.pi, np.pi, size=int((~m).sum()))
    else:
        ph = rng.uniform(-np.pi, np.pi, size=n)

    sigmas = _noise_schedule(n_steps, sigma_max=sigma_max, sigma_min=sigma_min)
    history: Dict = {
        "R": [],
        "sigma": [],
        "seed_weight": [],
        "algorithm": "diffusion_phase_langevin",
    }
    rho = None
    for t, sigma in enumerate(sigmas):
        # anneal seed weight high → low
        if n_steps <= 1:
            sw = seed_weight_end
        else:
            u = t / (n_steps - 1)
            sw = (1 - u) * seed_weight_start + u * seed_weight_end
        # data weight grows as noise falls
        dw = float(data_weight) * (1.0 - 0.4 * (sigma / max(sigma_max, 1e-6)))

        ph_data, R, rho = physics_score_step(
            hkl,
            amp,
            cell,
            ph,
            seed_phases=seed_phases,
            seed_mask=seed_mask,
            seed_weight=sw,
            d_min=d_min,
            solvent_fraction=solvent_fraction,
        )
        # reverse step: pull to data + noise
        # complex average for circular blend
        z = (1.0 - dw) * np.exp(1j * ph) + dw * np.exp(1j * ph_data)
        ph = np.angle(z)
        if sigma > 1e-8:
            ph = _wrap(ph + rng.normal(0.0, sigma, size=n))
        # hard lock seeds at early steps
        if seed_phases is not None and seed_mask is not None and sw > 0.5:
            m = np.asarray(seed_mask, dtype=bool)
            sp = np.asarray(seed_phases, dtype=np.float64)
            ph[m] = blend_phases(sp[m], ph[m], np.full(int(m.sum()), min(sw, 1.0)))

        history["R"].append(R)
        history["sigma"].append(float(sigma))
        history["seed_weight"].append(float(sw))
        if verbose and (t % 5 == 0 or t == n_steps - 1):
            print(f"  diffusion {t+1}/{n_steps}  R≈{R:.4f}  σ={sigma:.3f}  w_seed={sw:.2f}")

    # final clean physics step (no noise)
    ph, R_final, rho = physics_score_step(
        hkl,
        amp,
        cell,
        ph,
        seed_phases=seed_phases,
        seed_mask=seed_mask,
        seed_weight=seed_weight_end,
        d_min=d_min,
        solvent_fraction=solvent_fraction,
    )
    history["final_R"] = R_final
    history["n_steps"] = n_steps
    if rho is None:
        F = amp * np.exp(1j * ph)
        shape = grid_shape_from_resolution(cell, d_min, sampling=3.0)
        rho = density_from_structure_factors(hkl, F, cell, shape=shape, d_min=d_min)
    return ph, rho, history


def diffusion_hybrid_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    seed_phases: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    n_steps: int = 20,
    n_starts: int = 2,
    seed: int = 0,
    d_min: Optional[float] = None,
    polish: str = "charge_flipping",
    n_polish: int = 40,
    use_free_fom_gate: bool = True,
    solvent_fraction: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multistart diffusion reverse process + optional free-FOM gated polish.

    If no seed is given, random phases are used (weak ab initio baseline).
    With partial seeds, behaves as a **diffusion hybrid** extension path.
    """
    from grok_phase_solver.solvers.free_fom import free_fom
    from grok_phase_solver.solvers.conditional_hybrid import conditional_polish

    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    best = None
    trials: List[Dict] = []

    for s in range(max(1, n_starts)):
        ph, rho, hist = reverse_diffusion_phases(
            hkl,
            amp,
            cell,
            seed_phases=seed_phases,
            seed_mask=seed_mask,
            n_steps=n_steps,
            seed=seed + s,
            d_min=d_min,
            solvent_fraction=solvent_fraction,
            verbose=verbose and s == 0,
        )
        fom = free_fom(hkl, amp, ph, cell, density=rho)
        polish_info: Dict = {"accepted_polish": False, "polish": polish}
        if polish and polish != "none" and use_free_fom_gate:
            ph2, rho2, polish_info = conditional_polish(
                hkl,
                amp,
                cell,
                ph,
                polish=polish,
                n_iter=n_polish,
                seed=seed + s,
                d_min=d_min,
                verbose=verbose and s == 0,
            )
            ph, rho = ph2, rho2
            fom = free_fom(hkl, amp, ph, cell, density=rho)
        trial = {
            "start": s,
            "composite": fom["composite"],
            "R_pos": fom["R_pos"],
            "fom": fom,
            "history": hist,
            "polish": polish_info,
        }
        trials.append(trial)
        if best is None or fom["composite"] > best[0]:
            best = (fom["composite"], ph, rho, trial)

    assert best is not None
    _, ph, rho, trial = best
    info = {
        "algorithm": "diffusion_hybrid",
        "status": "experimental",
        "trained_score_net": diffusion_phase_available(),
        "n_starts": n_starts,
        "n_steps": n_steps,
        "best_trial": trial,
        "all_trials": trials,
        "fom_final": trial["fom"],
        "note": (
            "Physics Langevin diffusion hybrid; no claim of PXRDnet/XRDSol "
            "parity. Prefer partial-φ / AI-PhaSeed when seeds are strong."
        ),
    }
    return ph, rho, info


# Back-compat alias used by the v0.4 stub API
def conditional_diffusion_complete(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    *,
    n_steps: int = 20,
    seed: int = 0,
    d_min: Optional[float] = None,
    seed_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Complete / refine phases given a seed vector (full length).

    Wrapper around ``reverse_diffusion_phases`` for the old stub signature.
    """
    ph, rho, hist = reverse_diffusion_phases(
        hkl,
        amplitudes,
        cell,
        seed_phases=seed_phases,
        seed_mask=seed_mask,
        n_steps=n_steps,
        seed=seed,
        d_min=d_min,
    )
    hist = dict(hist)
    hist["status"] = "experimental_physics_langevin"
    hist["trained"] = diffusion_phase_available()
    hist["density_shape"] = list(rho.shape)
    return ph, hist
