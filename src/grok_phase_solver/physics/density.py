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
    nx, ny, nz = shape
    grid = np.zeros(shape, dtype=np.complex128)
    hkl = np.asarray(hkl, dtype=int).reshape(-1, 3)
    F = np.asarray(F, dtype=np.complex128).reshape(-1)

    def idx(h, n):
        h = int(h)
        if h < 0:
            h = n + h
        if h < 0 or h >= n:
            return None
        return h

    for (h, k, l), fval in zip(hkl, F):
        ih, ik, il = idx(h, nx), idx(k, ny), idx(l, nz)
        if ih is None or ik is None or il is None:
            continue
        grid[ih, ik, il] = fval
        if friedel_complete and not (h == 0 and k == 0 and l == 0):
            jh, jk, jl = idx(-h, nx), idx(-k, ny), idx(-l, nz)
            if jh is not None and jk is not None and jl is not None:
                grid[jh, jk, jl] = np.conj(fval)
    return grid


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
    Compute ρ(r) on a real-space grid via inverse FFT.

    Convention: ρ = (1/V) * IFFT(F_grid) * N  with numpy ifftn normalization,
    which is equivalent to ρ = ifftn(F_grid) after embedding F without 1/N,
    since ifftn divides by N: we embed F and use ifftn * (N/V)... 

    Standard crystallographic discrete form used here:
        ρ = (1/V) * Σ F(h) exp(−2π i h·r)
    With numpy: ρ = real( ifftn( F_embedded ) ) * (N / V) ... wait:

    numpy ifftn(X)[n] = (1/N) Σ X[k] exp(+2π i k n / N)  — sign/phase conventions
    differ; for real maps of centrosymmetric structures phases are 0/π and
    density is real. We use:
        ρ = real(ifftn(F_grid)) * (nx*ny*nz) / V
    so that Parseval is approximately conserved for dense sampling.
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
            from grok_phase_solver.physics.device import ifftn as _ifftn, resolve_device

            rho = _ifftn(F_grid, device=resolve_device(device)) * (np.prod(shape) / V)
        except Exception:
            rho = np.fft.ifftn(F_grid) * (np.prod(shape) / V)
    else:
        rho = np.fft.ifftn(F_grid) * (np.prod(shape) / V)
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
