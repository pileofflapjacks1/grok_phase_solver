"""
Electron density maps from structure factors (inverse Fourier transform).

ρ(r) = (1/V) Σ_h F(h) exp(−2π i h·r)

On a discrete grid this is an inverse FFT with appropriate indexing.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def grid_shape_from_resolution(
    cell: np.ndarray,
    d_min: float,
    sampling: float = 3.0,
) -> Tuple[int, int, int]:
    """
    Choose grid dimensions so that voxel size ≈ d_min / sampling.

    sampling=3 is a common crystallographic rule of thumb (~0.33 Å at 1 Å data).
    """
    a, b, c = cell[:3]
    nx = max(8, int(np.ceil(sampling * a / d_min)))
    ny = max(8, int(np.ceil(sampling * b / d_min)))
    nz = max(8, int(np.ceil(sampling * c / d_min)))
    # Prefer even dimensions for FFT
    nx += nx % 2
    ny += ny % 2
    nz += nz % 2
    return nx, ny, nz


def density_from_grid(F_grid: np.ndarray, volume: float, real: bool = True) -> np.ndarray:
    """
    ρ(r) = (1/V) Σ_h F(h) exp(−2πi h·r) on the grid r = (i/nx, j/ny, k/nz).

    numpy ``fftn`` carries the exp(−2πi …) kernel, so this is ``fftn(F)/V``.
    This is the single source of truth for the F → ρ sign convention; it must
    stay the exact inverse of :func:`grid_from_density`.
    """
    rho = np.fft.fftn(F_grid) / volume
    return np.real(rho) if real else rho


def grid_from_density(rho: np.ndarray, volume: float) -> np.ndarray:
    """
    F(h) = (V/N) Σ_r ρ(r) exp(+2πi h·r) on the FFT grid — i.e. ``V · ifftn(ρ)``.

    Exact inverse of :func:`density_from_grid`; matches the
    ``compute_structure_factors`` convention F = Σ f exp(+2πi h·x).
    """
    return np.fft.ifftn(rho) * volume


def place_reflections_on_grid(
    hkl: np.ndarray,
    F: np.ndarray,
    shape: Tuple[int, int, int],
    friedel_complete: bool = True,
) -> np.ndarray:
    """
    Embed sparse F(hkl) into a full FFT array (complex).

    Uses numpy FFT frequency order: index h → h if h>=0 else nx+h.
    If friedel_complete, enforces F(−h) = F(h)* for real density.
    """
    grid = np.zeros(shape, dtype=np.complex128)
    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.complex128).reshape(-1)
    if len(hkl) == 0:
        return grid
    if friedel_complete:
        # Mates first, so a reflection given explicitly in the list wins over
        # the conjugate implied by its partner.
        mates = -hkl
        ok = _in_grid(mates, shape) & np.any(hkl != 0, axis=1)
        grid[_grid_index(mates[ok], shape)] = np.conj(F[ok])
    ok = _in_grid(hkl, shape)
    grid[_grid_index(hkl[ok], shape)] = F[ok]
    return grid


def _in_grid(hkl: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Indices representable on an FFT grid of this shape: −n ≤ h < n."""
    n = np.asarray(shape, dtype=np.int64)
    return np.all((hkl >= -n) & (hkl < n), axis=1)


def _grid_index(hkl: np.ndarray, shape: Tuple[int, int, int]):
    """numpy FFT frequency order: h → h mod n, as a fancy-index tuple."""
    return (hkl[:, 0] % shape[0], hkl[:, 1] % shape[1], hkl[:, 2] % shape[2])


def extract_reflections_from_grid(F_grid: np.ndarray, hkl: np.ndarray) -> np.ndarray:
    """
    Read F(hkl) back from an FFT-ordered grid (inverse of placement).

    Reflections outside the grid return 0, as the loop versions in the
    solvers do.
    """
    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    out = np.zeros(len(hkl), dtype=np.complex128)
    if len(hkl) == 0:
        return out
    ok = _in_grid(hkl, F_grid.shape)
    out[ok] = F_grid[_grid_index(hkl[ok], F_grid.shape)]
    return out


def density_from_structure_factors(
    hkl: np.ndarray,
    F: np.ndarray,
    cell: np.ndarray,
    shape: Optional[Tuple[int, int, int]] = None,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    real_part_only: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute ρ(r) on a real-space grid (see :func:`density_from_grid`).

    Grid voxel (i, j, k) is fractional position (i/nx, j/ny, k/nz), with the
    crystallographic sign convention matching ``compute_structure_factors``
    (F = Σ f exp(+2πi h·x)), so peaks of a map from true phases sit at +x.
    """
    from .reciprocal import d_spacing

    if shape is None:
        if d_min is None:
            d = d_spacing(hkl, cell)
            d_min = float(np.min(d))
        shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)

    # Unit cell volume
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    cos_al, cos_be, cos_ga = np.cos(al), np.cos(be), np.cos(ga)
    v2 = 1 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga
    V = a * b * c * np.sqrt(max(v2, 0.0))

    F_grid = place_reflections_on_grid(hkl, F, shape, friedel_complete=True)
    # Optional torch FFT on cuda/mps (same formula; falls back to NumPy)
    if device and device not in ("cpu", "numpy", ""):
        try:
            from grok_phase_solver.physics.device import fftn as _fftn, resolve_device

            rho = _fftn(F_grid, device=resolve_device(device)) / V
        except Exception:
            rho = density_from_grid(F_grid, V, real=False)
    else:
        rho = density_from_grid(F_grid, V, real=False)
    if real_part_only:
        return np.real(rho)
    return rho


def grid_coordinates(
    cell: np.ndarray,
    shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fractional grid coordinates along each axis."""
    nx, ny, nz = shape
    x = np.arange(nx) / nx
    y = np.arange(ny) / ny
    z = np.arange(nz) / nz
    return x, y, z


def positivity_projection(rho: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """Set negative density to zero (optionally scaled)."""
    out = rho.copy()
    out[out < 0] = 0.0
    if fraction < 1.0:
        out = fraction * out + (1.0 - fraction) * rho
    return out


def charge_flip(rho: np.ndarray, delta: float = 0.0) -> np.ndarray:
    """
    Oszlányi–Sütő charge flipping: flip sign of density below threshold δ.

    ρ'(r) = −ρ(r)  if ρ(r) < δ, else ρ(r).
    """
    out = rho.copy()
    mask = out < delta
    out[mask] = -out[mask]
    return out


def map_statistics(rho: np.ndarray) -> dict:
    """Basic density statistics for diagnostics."""
    r = np.asarray(rho, dtype=np.float64)
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "frac_negative": float((r < 0).mean()),
        "skewness": float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-16)),
    }
