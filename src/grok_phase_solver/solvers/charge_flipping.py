"""
Charge-flipping algorithm (Oszlányi & Sütő, 2004/2008).

Iterative scheme:
1. Start with random phases (or seeds) + observed amplitudes → F
2. ρ ← IFFT(F)
3. Flip sign of density below threshold δ (dynamic fraction of σ)
4. F' ← FFT(ρ_flipped); replace |F'| with |F_obs| (Fourier projection)
5. Repeat

This is a physics baseline for ab initio phasing of small structures.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    charge_flip,
    density_from_structure_factors,
    grid_shape_from_resolution,
    place_reflections_on_grid,
)
from grok_phase_solver.physics.reciprocal import d_spacing


def _extract_F_from_grid(
    F_grid: np.ndarray,
    hkl: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = F_grid.shape

    def idx(h, n):
        h = int(h)
        if h < 0:
            h = n + h
        if 0 <= h < n:
            return h
        return None

    out = np.zeros(len(hkl), dtype=np.complex128)
    for i, (h, k, l) in enumerate(hkl):
        ih, ik, il = idx(h, nx), idx(k, ny), idx(l, nz)
        if ih is None or ik is None or il is None:
            continue
        out[i] = F_grid[ih, ik, il]
    return out


def charge_flipping_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 200,
    delta_sigma: float = 1.0,
    weak_fraction: float = 0.2,
    seed: int = 0,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run charge flipping.

    Parameters
    ----------
    hkl : (M, 3)
    amplitudes : (M,) observed |F|
    cell : (6,)
    n_iter : iterations
    delta_sigma : flip threshold in units of density σ (dynamic)
    weak_fraction : fraction of weakest reflections whose phases are randomized
        each iteration (classic CF improvement)
    seed : RNG seed
    phase_init : optional initial phases (radians)

    Returns
    -------
    phases : (M,) recovered phases (radians)
    rho : real-space density grid
    history : diagnostics dict
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)

    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)

    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()

    # Weak reflections mask (by amplitude)
    order = np.argsort(amp)
    n_weak = max(1, int(weak_fraction * len(amp)))
    weak_mask = np.zeros(len(amp), dtype=bool)
    weak_mask[order[:n_weak]] = True

    history = {"R": [], "neg_frac": [], "delta": []}

    # Volume for scaling
    a, b, c, al, be, ga = cell
    alr, ber, gar = np.deg2rad([al, be, ga])
    cos_al, cos_be, cos_ga = np.cos(alr), np.cos(ber), np.cos(gar)
    v2 = 1 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga
    V = a * b * c * np.sqrt(max(v2, 0.0))

    for it in range(n_iter):
        F = amp * np.exp(1j * phases)
        F_grid = place_reflections_on_grid(hkl, F, shape, friedel_complete=True)
        rho = np.real(np.fft.ifftn(F_grid) * (np.prod(shape) / V))

        sigma = float(rho.std()) + 1e-16
        delta = delta_sigma * 0.0  # classic CF uses δ near 0; use small floor
        # Dynamic threshold: slightly positive fraction of sigma for noise
        delta = 0.1 * delta_sigma * sigma
        rho_flip = charge_flip(rho, delta=delta)

        # FFT back
        F_new_grid = np.fft.fftn(rho_flip) * (V / np.prod(shape))
        F_new = _extract_F_from_grid(F_new_grid, hkl)

        # Fourier modulus projection
        phases = np.angle(F_new)
        # Randomize weak phases only in early iterations (escape local minima),
        # then freeze to allow refinement — classic CF schedule.
        if weak_fraction > 0 and it < max(10, n_iter // 3):
            phases[weak_mask] = rng.uniform(-np.pi, np.pi, size=n_weak)

        # R-factor vs observed amplitudes
        Fc = np.abs(F_new)
        k = np.sum(amp * Fc) / (np.sum(Fc * Fc) + 1e-16)
        R = float(np.sum(np.abs(amp - k * Fc)) / (np.sum(amp) + 1e-16))
        history["R"].append(R)
        history["neg_frac"].append(float((rho < 0).mean()))
        history["delta"].append(delta)

        if verbose and (it % 20 == 0 or it == n_iter - 1):
            print(f"  CF iter {it:4d}  R={R:.4f}  neg={history['neg_frac'][-1]:.3f}")

    # Final density with constrained moduli
    F_final = amp * np.exp(1j * phases)
    rho_final = density_from_structure_factors(hkl, F_final, cell, shape=shape)
    history["shape"] = shape
    history["n_iter"] = n_iter
    return phases, rho_final, history
