"""
Patterson function — phase-free map of interatomic vectors.

Following Patterson (1934) and the pedagogical treatment of Cowtan (2001):

    P(u) = (1/V) Σ_h |F(h)|² exp(−2π i h·u)

Equivalently, P is the autocorrelation of the electron density:

    P(u) = ∫_V ρ(r) ρ(r+u) dr

Peaks of P lie at interatomic vectors r_i − r_j (plus origin peak).
For N atoms there are N(N−1) non-origin vectors (many overlapped),
so classical Patterson solving is limited to ~20–50 atoms unless a
heavy-atom subset dominates (Cowtan, 2001).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    density_from_structure_factors,
    grid_shape_from_resolution,
    place_reflections_on_grid,
)
from grok_phase_solver.physics.reciprocal import d_spacing


@dataclass
class PattersonPeak:
    """A local maximum in the Patterson map (fractional coordinates)."""

    fract: np.ndarray  # (3,) in [0,1)
    height: float
    rank: int


def patterson_from_amplitudes(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    shape: Optional[Tuple[int, int, int]] = None,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    remove_origin: bool = False,
    origin_sharpen: float = 0.0,
) -> np.ndarray:
    """
    Compute the Patterson map P(u) from |F| only (phases set to zero).

    Parameters
    ----------
    hkl, amplitudes : observed Miller indices and |F|
    cell : unit cell (6,)
    remove_origin : if True, subtract mean of |F|² so origin peak is reduced
        (equivalent to removing F(000)-like constant term contribution)
    origin_sharpen : if >0, multiply |F|² by exp(B s²) with B=origin_sharpen
        (sharpening for heavy-atom vector search)

    Returns
    -------
    P : real 3D array on the unit-cell grid
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    I = amp * amp  # |F|²

    if origin_sharpen > 0:
        d = d_spacing(hkl, cell)
        s = 1.0 / (2.0 * np.maximum(d, 1e-8))
        I = I * np.exp(origin_sharpen * s * s)

    if remove_origin:
        I = I - np.mean(I)

    # Phases all zero → F_P = |F|² (real, positive)
    F_pat = I.astype(np.complex128)
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    if shape is None:
        shape = grid_shape_from_resolution(cell, d_min, sampling=sampling)

    return density_from_structure_factors(
        hkl, F_pat, cell, shape=shape, real_part_only=True
    )


def autocorrelation_density(rho: np.ndarray) -> np.ndarray:
    """
    Real-space autocorrelation via FFT (Parseval form of Patterson).

    P = IFFT(|FFT(ρ)|²)  (circular, unit-cell periodic)
    """
    R = np.fft.fftn(rho)
    return np.real(np.fft.ifftn(np.abs(R) ** 2))


def find_patterson_peaks(
    P: np.ndarray,
    n_peaks: int = 20,
    min_fract_dist: float = 0.08,
    exclude_origin: bool = True,
    origin_radius: float = 0.12,
) -> List[PattersonPeak]:
    """
    Greedy peak picking on a Patterson map with fractional-distance NMS.

    Does not use sophisticated symmetry-peak elimination; intended for
    teaching and heavy-atom vector bootstrap, not production SHELXS.
    """
    P = np.asarray(P, dtype=np.float64)
    nx, ny, nz = P.shape
    work = P.copy()
    # Suppress origin peak neighborhood
    if exclude_origin:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    fx, fy, fz = i / nx, j / ny, k / nz
                    # min-image distance to origin
                    dx = min(fx, 1 - fx)
                    dy = min(fy, 1 - fy)
                    dz = min(fz, 1 - fz)
                    if np.sqrt(dx * dx + dy * dy + dz * dz) < origin_radius:
                        work[i, j, k] = work.min()

    peaks: List[PattersonPeak] = []
    flat = work.ravel()
    for rank in range(n_peaks):
        idx = int(np.argmax(flat))
        height = float(flat[idx])
        if height <= 0:
            break
        i, j, k = np.unravel_index(idx, work.shape)
        fract = np.array([i / nx, j / ny, k / nz], dtype=np.float64)
        # NMS: zero neighborhood
        peaks.append(PattersonPeak(fract=fract, height=height, rank=rank))
        # zero ball in index space
        ri = max(1, int(min_fract_dist * nx))
        rj = max(1, int(min_fract_dist * ny))
        rk = max(1, int(min_fract_dist * nz))
        for di in range(-ri, ri + 1):
            for dj in range(-rj, rj + 1):
                for dk in range(-rk, rk + 1):
                    ii = (i + di) % nx
                    jj = (j + dj) % ny
                    kk = (k + dk) % nz
                    work[ii, jj, kk] = work.min()
        flat = work.ravel()
    return peaks


def interatomic_vectors_from_atoms(
    fracs: np.ndarray,
    include_self: bool = False,
) -> np.ndarray:
    """All fractional interatomic vectors r_i − r_j (mod 1)."""
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    vecs = []
    n = len(fracs)
    for i in range(n):
        for j in range(n):
            if not include_self and i == j:
                continue
            v = (fracs[i] - fracs[j]) % 1.0
            vecs.append(v)
    return np.array(vecs) if vecs else np.zeros((0, 3))


def patterson_peak_recovery_score(
    peaks: Sequence[PattersonPeak],
    true_fracs: np.ndarray,
    tol: float = 0.08,
) -> float:
    """
    Fraction of non-origin true interatomic vectors matched by a peak
    within fractional tolerance (min-image). Diagnostic for tests.
    """
    true_vecs = interatomic_vectors_from_atoms(true_fracs, include_self=False)
    if len(true_vecs) == 0 or len(peaks) == 0:
        return 0.0
    matched = 0
    for v in true_vecs:
        for p in peaks:
            d = (v - p.fract + 0.5) % 1.0 - 0.5
            if np.linalg.norm(d) < tol:
                matched += 1
                break
    return matched / len(true_vecs)
