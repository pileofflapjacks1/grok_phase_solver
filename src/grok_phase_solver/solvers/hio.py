"""
Hybrid Input–Output (HIO) algorithm (Fienup) for phase retrieval.

Alternates Fourier-domain modulus constraint with real-space support /
positivity constraints. Classic algorithm for coherent diffraction imaging
and crystallographic density modification when a support is known.

For crystals without known support we use positivity + optional solvent mask.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    density_from_structure_factors,
    grid_shape_from_resolution,
    place_reflections_on_grid,
)
from grok_phase_solver.physics.reciprocal import d_spacing


def _extract_F_from_grid(F_grid: np.ndarray, hkl: np.ndarray) -> np.ndarray:
    nx, ny, nz = F_grid.shape

    def idx(h, n):
        h = int(h)
        if h < 0:
            h = n + h
        return h if 0 <= h < n else None

    out = np.zeros(len(hkl), dtype=np.complex128)
    for i, (h, k, l) in enumerate(hkl):
        ih, ik, il = idx(h, nx), idx(k, ny), idx(l, nz)
        if None in (ih, ik, il):
            continue
        out[i] = F_grid[ih, ik, il]
    return out


def hio_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 200,
    beta: float = 0.9,
    positivity: bool = True,
    support_fraction: Optional[float] = None,
    seed: int = 0,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Hybrid Input–Output phase retrieval.

    Parameters
    ----------
    beta : HIO feedback parameter (typically 0.7–0.9)
    positivity : enforce ρ ≥ 0 outside violation set
    support_fraction : if set, keep only top fraction of voxels by |ρ| as support
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)

    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)

    a, b, c, al, be, ga = cell
    alr, ber, gar = np.deg2rad([al, be, ga])
    cos_al, cos_be, cos_ga = np.cos(alr), np.cos(ber), np.cos(gar)
    v2 = 1 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga
    V = a * b * c * np.sqrt(max(v2, 0.0))
    N = np.prod(shape)

    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()

    F = amp * np.exp(1j * phases)
    F_grid = place_reflections_on_grid(hkl, F, shape, friedel_complete=True)
    rho = np.real(np.fft.ifftn(F_grid) * (N / V))
    rho_in = rho.copy()

    history = {"R": [], "neg_frac": []}

    for it in range(n_iter):
        # Fourier projection of current estimate
        F_grid = np.fft.fftn(rho) * (V / N)
        F_ext = _extract_F_from_grid(F_grid, hkl)
        phases = np.angle(F_ext)
        # Enforce observed moduli
        F_proj = amp * np.exp(1j * phases)
        F_grid_proj = place_reflections_on_grid(hkl, F_proj, shape, friedel_complete=True)
        rho_F = np.real(np.fft.ifftn(F_grid_proj) * (N / V))

        # Support / positivity constraint
        if support_fraction is not None:
            thresh = np.quantile(np.abs(rho_F), 1.0 - support_fraction)
            support = np.abs(rho_F) >= thresh
        else:
            support = np.ones_like(rho_F, dtype=bool)

        violate = ~support
        if positivity:
            violate |= rho_F < 0

        # HIO update
        rho_out = rho_F.copy()
        rho_out[violate] = rho_in[violate] - beta * rho_F[violate]
        rho_in = rho_out
        rho = rho_out

        # R-factor
        Fc = np.abs(F_ext)
        k = np.sum(amp * Fc) / (np.sum(Fc * Fc) + 1e-16)
        R = float(np.sum(np.abs(amp - k * Fc)) / (np.sum(amp) + 1e-16))
        history["R"].append(R)
        history["neg_frac"].append(float((rho_F < 0).mean()))

        if verbose and (it % 20 == 0 or it == n_iter - 1):
            print(f"  HIO iter {it:4d}  R={R:.4f}  neg={history['neg_frac'][-1]:.3f}")

    F_final = amp * np.exp(1j * phases)
    rho_final = density_from_structure_factors(hkl, F_final, cell, shape=shape)
    history["shape"] = shape
    history["n_iter"] = n_iter
    return phases, rho_final, history
