"""
Classical density modification (phase improvement).

Cowtan / Wang solvent flattening: protein crystals contain disordered solvent
regions where density should be flat. After an initial phase estimate:

1. Identify solvent mask (low variance / low density regions)
2. Flatten solvent toward mean solvent density
3. Optionally sharpen protein region
4. FFT → new phases; recombine with observed |F| (Fourier projection)
5. Iterate

Also: positivity projection and simple histogram matching toward expected
density distribution (gamma-like for proteins is deferred; we use positivity
+ optional Gaussian target moments).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    density_from_structure_factors,
    place_reflections_on_grid,
    grid_shape_from_resolution,
)
from grok_phase_solver.physics.reciprocal import d_spacing


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


def solvent_mask(
    rho: np.ndarray,
    solvent_fraction: float = 0.5,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Binary solvent mask: True = solvent.

    Uses local mean of ρ (box smooth) and thresholds the lowest
    solvent_fraction of voxels as solvent (Wang-style simplification).
    """
    from scipy.ndimage import uniform_filter

    r = np.asarray(rho, dtype=np.float64)
    # local average as "envelope" proxy
    size = max(3, int(2 * smooth_sigma) | 1)
    local = uniform_filter(r, size=size, mode="wrap")
    thresh = np.quantile(local, solvent_fraction)
    return local <= thresh


def estimate_solvent_fraction(
    rho: np.ndarray,
    *,
    default: float = 0.45,
    protein_mode: bool = False,
) -> float:
    """
    Heuristic solvent fraction from density histogram.

    ``protein_mode``: bias toward higher solvent (0.45–0.65) typical of
    macromolecular crystals. Small-molecule default stays ~0.3–0.5.
    """
    r = np.asarray(rho, dtype=np.float64).ravel()
    if r.size < 8:
        return float(default)
    # fraction of voxels below mean as crude solvent proxy
    below = float(np.mean(r < np.mean(r)))
    if protein_mode:
        frac = float(np.clip(0.35 + 0.4 * below, 0.40, 0.70))
    else:
        frac = float(np.clip(0.15 + 0.45 * below, 0.20, 0.55))
    return frac


def solvent_flatten(
    rho: np.ndarray,
    solvent_fraction: float = 0.5,
    solvent_level: Optional[float] = None,
    *,
    auto_fraction: bool = False,
    protein_mode: bool = False,
) -> np.ndarray:
    """
    Set solvent voxels to constant level (default: mean of solvent).

    ``auto_fraction``: estimate solvent fraction from the map (optional).
    ``protein_mode``: prefer higher solvent fractions and softer protein floor.
    """
    if auto_fraction or solvent_fraction is None:
        solvent_fraction = estimate_solvent_fraction(
            rho, default=0.50 if protein_mode else 0.45, protein_mode=protein_mode
        )
    mask = solvent_mask(rho, solvent_fraction=float(solvent_fraction))
    out = rho.copy()
    if solvent_level is None:
        solvent_level = float(rho[mask].mean()) if np.any(mask) else 0.0
    out[mask] = solvent_level
    # mild positivity in protein / ordered region
    protein = ~mask
    floor = 0.0 if not protein_mode else float(np.percentile(out[protein], 5)) if np.any(protein) else 0.0
    out[protein] = np.maximum(out[protein], floor if protein_mode else 0.0)
    return out


def density_modification_cycle(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 10,
    solvent_fraction: float = 0.45,
    d_min: Optional[float] = None,
    verbose: bool = False,
    protein_mode: bool = False,
    auto_solvent: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Iterate: map → solvent flatten → FFT → impose |F_obs|.

    ``protein_mode`` / ``auto_solvent``: higher-solvent envelope heuristics
    for macromolecular-like maps (still useful for small-molecule wet cells).

    Returns final phases, final ρ, history.
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64).copy()
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=3.0)
    V = _volume(cell)
    N = np.prod(shape)
    history: Dict = {"R": [], "solvent_fraction": [], "protein_mode": protein_mode}

    for it in range(n_iter):
        F = amp * np.exp(1j * phases)
        rho = density_from_structure_factors(hkl, F, cell, shape=shape)
        sf = solvent_fraction
        if auto_solvent or protein_mode:
            sf = estimate_solvent_fraction(
                rho, default=solvent_fraction, protein_mode=protein_mode
            )
        rho_mod = solvent_flatten(
            rho,
            solvent_fraction=sf,
            protein_mode=protein_mode,
        )
        F_grid = np.fft.fftn(rho_mod) * (V / N)
        F_new = _extract_F(F_grid, hkl)
        phases = np.angle(F_new)
        # modulus projection
        Fc = np.abs(F_new)
        k = np.sum(amp * Fc) / (np.sum(Fc * Fc) + 1e-16)
        R = float(np.sum(np.abs(amp - k * Fc)) / (np.sum(amp) + 1e-16))
        history["R"].append(R)
        history["solvent_fraction"].append(float(sf))
        if verbose and it % 2 == 0:
            print(f"  DM iter {it}: R={R:.4f} solvent_f={sf:.2f}")

    F_final = amp * np.exp(1j * phases)
    rho_final = density_from_structure_factors(hkl, F_final, cell, shape=shape)
    return phases, rho_final, history
