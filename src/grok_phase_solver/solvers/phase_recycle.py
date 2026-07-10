"""
Phase recycling — physics core of PhAI-style iteration.

Algorithm (correct Fourier consistency at each step):

  1. Start with phases φ⁰ (random, zero, or network seed)
  2. Form F = |F_obs| exp(iφ)
  3. ρ = IFFT(F)
  4. Optional density modification (positivity / solvent flatten)
  5. F' = FFT(ρ)
  6. φ ← arg(F')   (keep observed moduli: project onto |F_obs|)
  7. Repeat for n cycles

This is **not** the PhAI neural network; it is the modular projection loop
that any NN phase predictor should wrap. PhAI injects a learned map
from (|F|, φ_in) → φ_out inside the cycle; we provide:

  - pure physics recycle (DM / positivity)
  - recycle with an external phase_fn(|F|, φ) callback (NN hook)

References: Fienup HIO/ER; PhAI phase recycling (Larsen et al., Science 2024).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    density_from_structure_factors,
    grid_shape_from_resolution,
    place_reflections_on_grid,
)
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.density_modification import solvent_flatten


PhaseFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# phase_fn(hkl, amplitudes, phases_in) -> phases_out


def _volume(cell: np.ndarray) -> float:
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
    v2 = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    return float(a * b * c * np.sqrt(max(v2, 0.0)))


def _extract_F(F_grid: np.ndarray, hkl: np.ndarray) -> np.ndarray:
    nx, ny, nz = F_grid.shape

    def idx(h, n):
        h = int(h)
        if h < 0:
            h = n + h
        return h if 0 <= h < n else None

    out = np.zeros(len(hkl), dtype=np.complex128)
    for i, (h, k, l) in enumerate(hkl):
        ih, ik, il = idx(h, nx), idx(k, ny), idx(l, nz)
        if None not in (ih, ik, il):
            out[i] = F_grid[ih, ik, il]
    return out


def fourier_modulus_projection(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    density_op: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One ER-style cycle: |F|e^{iφ} → ρ → density_op(ρ) → FFT → new φ.

    Returns (new_phases, rho_after_mod).
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)
    V = _volume(cell)
    N = float(np.prod(shape))

    F = amp * np.exp(1j * phases)
    F_grid = place_reflections_on_grid(hkl, F, shape, friedel_complete=True)
    rho = np.real(np.fft.ifftn(F_grid) * (N / V))
    if density_op is not None:
        rho = density_op(rho)
    F_new_grid = np.fft.fftn(rho) * (V / N)
    F_new = _extract_F(F_new_grid, hkl)
    return np.angle(F_new), rho


def phase_recycle(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_cycles: int = 5,
    phase_init: Optional[np.ndarray] = None,
    phase_fn: Optional[PhaseFn] = None,
    use_positivity: bool = True,
    solvent_fraction: Optional[float] = None,
    seed: int = 0,
    d_min: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multi-cycle phase recycling.

    Parameters
    ----------
    phase_fn : optional neural / classical map
        Called as phase_fn(hkl, amplitudes, phases) → phases before each
        Fourier modulus projection. If None, only physics projection is used.
    use_positivity : set ρ<0 to 0 each cycle
    solvent_fraction : if set, Wang-style solvent flatten with this fraction
    """
    rng = np.random.default_rng(seed)
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()

    def density_op(rho: np.ndarray) -> np.ndarray:
        out = rho
        if use_positivity:
            out = np.maximum(out, 0.0)
        if solvent_fraction is not None:
            out = solvent_flatten(out, solvent_fraction=solvent_fraction)
        return out

    history = {"R": [], "cycle": []}
    rho = None
    for c in range(n_cycles):
        if phase_fn is not None:
            phases = phase_fn(hkl, amp, phases)
        phases, rho = fourier_modulus_projection(
            hkl, amp, phases, cell, density_op=density_op, d_min=d_min
        )
        # residual R of current density vs |F|
        F_check = amp * np.exp(1j * phases)
        # recompute R from last projection's intermediate is approximate
        history["R"].append(float("nan"))  # filled below if needed
        history["cycle"].append(c)
        if verbose:
            print(f"  recycle cycle {c+1}/{n_cycles}")

    # Final density with constrained moduli
    F_final = amp * np.exp(1j * phases)
    rho = density_from_structure_factors(hkl, F_final, cell, d_min=d_min)
    # True R of final
    # (moduli forced → R defined via density-derived |F_c| before projection)
    phases2, _ = fourier_modulus_projection(
        hkl, amp, phases, cell, density_op=density_op, d_min=d_min
    )
    # R between amp and |FFT(ρ_mod)| before reimpose — use one extra FFT
    shape = rho.shape
    V = _volume(cell)
    N = float(np.prod(shape))
    F_grid = np.fft.fftn(rho) * (V / N)
    Fc = np.abs(_extract_F(F_grid, hkl))
    k = np.sum(amp * Fc) / (np.sum(Fc * Fc) + 1e-16)
    R = float(np.sum(np.abs(amp - k * Fc)) / (np.sum(amp) + 1e-16))
    history["R"] = [R]
    history["final_R"] = R
    history["n_cycles"] = n_cycles
    return phases, rho, history
