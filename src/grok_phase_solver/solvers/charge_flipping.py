"""
Charge-flipping algorithm (Oszlányi & Sütő, 2004/2005/2008).

Iterative scheme:
1. Start with random phases (or seeds) + observed amplitudes → F
2. ρ ← F⁻¹(F), with F(000) carried from the previous cycle
3. Flip the sign of density below δ = k·σ(ρ)
4. G ← F(ρ_flipped); keep G(000); strong reflections get |F_obs| with the
   phases of G, weak ones keep G with a π/2 phase shift
5. Repeat until the R-factor phase transition (or ``n_iter``)

This is a physics baseline for ab initio phasing of small structures.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    charge_flip,
    density_from_grid,
    density_from_structure_factors,
    extract_reflections_from_grid,
    grid_from_density,
    grid_shape_from_resolution,
    place_reflections_on_grid,
)
from grok_phase_solver.physics.reciprocal import d_spacing


def _extract_F_from_grid(F_grid: np.ndarray, hkl: np.ndarray) -> np.ndarray:
    """Kept for importers; see ``physics.density.extract_reflections_from_grid``."""
    return extract_reflections_from_grid(F_grid, hkl)


def charge_flipping_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 200,
    delta_sigma: float = 0.8,
    weak_fraction: float = 0.2,
    seed: int = 0,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    phase_init: Optional[np.ndarray] = None,
    centrosymmetric: bool = False,
    verbose: bool = False,
    include_f000: bool = True,
    stop_after_transition: bool = True,
    polish_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run charge flipping.

    Parameters
    ----------
    hkl : (M, 3)
    amplitudes : (M,) observed |F|
    cell : (6,)
    n_iter : maximum iterations. Charge flipping needs hundreds to thousands
        of cycles before its phase transition; very small budgets mostly
        return the plateau.
    delta_sigma : flip threshold δ = delta_sigma · σ(ρ), recomputed each cycle.
        Default 0.8: on 16 held-out synthetic P1 cells (n = 12–20,
        d = 0.9–1.2 Å, 1500 cycles, one start) 0.8 solved 15/16 with
        weak_fraction 0.2, vs 12/16 at 1.0, 7/16 at 1.1 and 8/16 at 0.65;
        0.5 collapsed to false single-peak solutions
    weak_fraction : fraction of weakest reflections with free amplitude and a
        π/2 phase shift each cycle (Oszlányi & Sütő 2005); 0 disables it
    seed : RNG seed
    phase_init : optional initial phases (radians), e.g. a partial seed
    centrosymmetric : snap phases to {0, π}
    include_f000 : carry F(000) from the flipped map into the next cycle
        (left free, as in Oszlányi & Sütő 2004) instead of resetting it to 0
    stop_after_transition : stop ``polish_iter`` cycles after the phase
        transition (R below 0.4× the random-phase reference R); run all ``n_iter``
        otherwise
    polish_iter : cycles kept after the transition

    Returns
    -------
    phases : (M,) recovered phases (radians)
    rho : real-space density grid
    history : diagnostics dict. ``converged`` / ``transition_iter`` report the
        R-factor phase transition, a truth-free *hint* of a solution. It is
        calibrated at the default ``delta_sigma``. With a much smaller δ
        (≈0.5σ) CF collapses into its known false fixed point (density piled
        into a few "uranium atom" peaks), R still drops, and the flag fires
        without a solution. Treat it as a hint, not a proof; check the map.
        On real data it is conservative: on COD 2200239 (P1-expanded Fobs,
        0.84 Å) three runs solved (map CC 0.83–0.86 vs the deposited model)
        but only one tripped the flag. A missed flag costs run time (no
        early stop), not correctness.

    Space groups above P1: pass the full P1 reflection sphere (symmetry
    equivalents expanded). Running on the unique wedge alone leaves most of
    reciprocal space empty and does not converge.
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    amp = np.asarray(amplitudes, dtype=np.float64)

    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)
    V = _cell_volume(cell)

    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()
    if centrosymmetric:
        phases = np.where(np.cos(phases) >= 0.0, 0.0, np.pi)

    # Weak reflections: amplitude left free, phase shifted by π/2 each cycle
    # (Oszlányi & Sütő 2005). Strong ones carry the modulus constraint.
    n_weak = int(round(weak_fraction * len(amp))) if weak_fraction > 0 else 0
    weak_mask = np.zeros(len(amp), dtype=bool)
    if n_weak:
        weak_mask[np.argsort(amp)[:n_weak]] = True
    strong = ~weak_mask
    # The π/2 shift must respect Friedel symmetry: +π/2 on one hemisphere and
    # −π/2 on the mates, so F(−h) = F(h)* still holds. A uniform +π/2 on a
    # list that holds both h and −h makes the weak terms cancel in Re ρ.
    first_nonzero = np.where(hkl[:, 0] != 0, hkl[:, 0], np.where(hkl[:, 1] != 0, hkl[:, 1], hkl[:, 2]))
    weak_shift = np.where(first_nonzero >= 0, 1j, -1j)
    # Weak reflections seed at their observed amplitude, then evolve freely.
    F = amp * np.exp(1j * phases)
    F000 = 0.0

    history: Dict = {"R": [], "neg_frac": [], "delta": []}
    transition_at: Optional[int] = None
    G_hkl = F

    def _flip_cycle(F_in: np.ndarray, f000: float):
        F_grid = place_reflections_on_grid(hkl, F_in, shape, friedel_complete=True)
        if include_f000:
            F_grid[0, 0, 0] = f000
        rho_in = density_from_grid(F_grid, V)
        d = float(delta_sigma) * (float(rho_in.std()) + 1e-16)
        G_grid = grid_from_density(charge_flip(rho_in, delta=d), V)
        return rho_in, d, G_grid

    def _r_strong(G: np.ndarray) -> float:
        Fc = np.abs(G[strong])
        Fo = amp[strong]
        k = np.sum(Fo * Fc) / (np.sum(Fc * Fc) + 1e-16)
        return float(np.sum(np.abs(Fo - k * Fc)) / (np.sum(Fo) + 1e-16))

    # Reference R: one flip cycle from random phases (a separate stream, so
    # the solve itself is unchanged). A run is "converged" once its R falls
    # below _TRANSITION_RATIO × this reference — the CF phase transition.
    ref_rng = np.random.default_rng(seed + 7919)
    _, _, G_ref = _flip_cycle(amp * np.exp(1j * ref_rng.uniform(-np.pi, np.pi, len(amp))), 0.0)
    R_ref = _r_strong(extract_reflections_from_grid(G_ref, hkl))

    for it in range(n_iter):
        rho, delta, G_grid = _flip_cycle(F, F000)
        if include_f000:
            F000 = float(G_grid[0, 0, 0].real)
        G_hkl = extract_reflections_from_grid(G_grid, hkl)

        phases = np.angle(G_hkl)
        if centrosymmetric:
            phases = np.where(np.cos(phases) >= 0.0, 0.0, np.pi)
        F = amp * np.exp(1j * phases)
        if n_weak:
            if centrosymmetric:
                # A π/2 shift would leave the real axis; keep weak terms free only.
                F[weak_mask] = np.abs(G_hkl[weak_mask]) * np.exp(1j * phases[weak_mask])
            else:
                F[weak_mask] = G_hkl[weak_mask] * weak_shift[weak_mask]

        # R on the modulus-constrained (strong) reflections, fitted scale
        R = _r_strong(G_hkl)
        history["R"].append(R)
        history["neg_frac"].append(float((rho < 0).mean()))
        history["delta"].append(delta)

        # Phase transition; after it run ``polish_iter`` more cycles and stop
        # (if early stopping is on).
        if transition_at is None and R < _TRANSITION_RATIO * R_ref:
            transition_at = it
        if verbose and (it % 100 == 0 or it == n_iter - 1):
            print(f"  CF iter {it:5d}  R={R:.4f}  neg={history['neg_frac'][-1]:.3f}")
        if (
            stop_after_transition
            and transition_at is not None
            and it - transition_at >= polish_iter
        ):
            break

    # Final map: observed moduli with the last flipped-density phases
    phases = np.angle(G_hkl)
    if centrosymmetric:
        phases = np.where(np.cos(phases) >= 0.0, 0.0, np.pi)
    F_final = amp * np.exp(1j * phases)
    rho_final = density_from_structure_factors(hkl, F_final, cell, shape=shape)
    history["shape"] = shape
    history["n_iter"] = len(history["R"])
    history["n_iter_max"] = n_iter
    history["transition_iter"] = transition_at
    history["R_random_ref"] = R_ref
    history["converged"] = transition_at is not None
    return phases, rho_final, history


# Drop below the random-phase reference R (one flip cycle from random
# phases, R_ref ≈ 0.51–0.58) that counts as the phase transition. On 16
# synthetic P1 cells (n = 12–30, d = 0.9–1.5 Å) unsolved runs plateaued at
# 0.54–0.60 × R_ref and solved runs at 0.24–0.25 × R_ref; 0.4 sits in the gap.
_TRANSITION_RATIO = 0.4


def _cell_volume(cell: np.ndarray) -> float:
    a, b, c, al, be, ga = cell
    cos_al, cos_be, cos_ga = np.cos(np.deg2rad([al, be, ga]))
    v2 = 1 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga
    return float(a * b * c * np.sqrt(max(v2, 0.0)))
