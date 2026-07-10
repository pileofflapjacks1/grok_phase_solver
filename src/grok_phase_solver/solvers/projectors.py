"""
Real- and reciprocal-space projectors for iterative phase retrieval.

Shared Fourier conventions with charge_flipping / phase_recycle:

  F_grid = place(hkl, |F| e^{iφ})
  ρ = Re(ifftn(F_grid)) * (N / V)
  F' = fftn(ρ) * (V / N)
  modulus projection: F ← |F_obs| * F' / |F'|

References
----------
- Fienup, J. R. (1982). Phase retrieval algorithms. Appl. Opt. 21, 2758.
- Elser, V. (2003). Phase retrieval by iterated projections. JOSA A 20, 40.
- Luke, D. R. (2005). Relaxed averaged alternating reflections (RAAR). Inverse Problems 21, 37.
- Oszlányi & Sütő (2004). Ab initio structure solution by charge flipping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    grid_shape_from_resolution,
    place_reflections_on_grid,
)
from grok_phase_solver.physics.reciprocal import d_spacing


def unit_cell_volume(cell: np.ndarray) -> float:
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
    v2 = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    return float(a * b * c * np.sqrt(max(v2, 0.0)))


def extract_F_from_grid(F_grid: np.ndarray, hkl: np.ndarray) -> np.ndarray:
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


def density_from_phases(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    shape: Tuple[int, int, int],
) -> np.ndarray:
    V = unit_cell_volume(cell)
    N = float(np.prod(shape))
    F = np.asarray(amplitudes) * np.exp(1j * np.asarray(phases))
    F_grid = place_reflections_on_grid(hkl, F, shape, friedel_complete=True)
    return np.real(np.fft.ifftn(F_grid) * (N / V))


def density_to_F(
    rho: np.ndarray,
    hkl: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    V = unit_cell_volume(cell)
    N = float(np.prod(rho.shape))
    F_grid = np.fft.fftn(rho) * (V / N)
    return extract_F_from_grid(F_grid, hkl)


def project_modulus(
    F: np.ndarray,
    amplitudes: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """P_M: replace |F| with observed amplitudes, keep phase of F."""
    mag = np.abs(F)
    scale = np.asarray(amplitudes, dtype=np.float64) / np.maximum(mag, eps)
    return F * scale


def project_positivity(rho: np.ndarray) -> np.ndarray:
    """P_+: ρ ← max(ρ, 0)."""
    return np.maximum(rho, 0.0)


def project_charge_flip(rho: np.ndarray, delta: float = 0.0) -> np.ndarray:
    """P_CF: flip sign where ρ < δ."""
    out = rho.copy()
    mask = out < delta
    out[mask] = -out[mask]
    return out


def project_support(rho: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Zero density outside boolean support mask."""
    out = rho.copy()
    out[~support] = 0.0
    return out


def r_factor_moduli(F: np.ndarray, amplitudes: np.ndarray) -> float:
    Fc = np.abs(F)
    Fo = np.asarray(amplitudes, dtype=np.float64)
    k = np.sum(Fo * Fc) / (np.sum(Fc * Fc) + 1e-16)
    return float(np.sum(np.abs(Fo - k * Fc)) / (np.sum(Fo) + 1e-16))


def setup_grid(
    hkl: np.ndarray,
    cell: np.ndarray,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
) -> Tuple[Tuple[int, int, int], float]:
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)
    return shape, d_min
