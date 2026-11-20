"""
Simple peak picking on electron density for a first atom list.

Not a SHELXS peak-search replacement: fractional coords of strongest
positive density maxima, with basic NMS. Exported as Q peaks for Olex2
hand-build / peak picking — not a SHELXL starting model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DensityPeak:
    rank: int
    fract: np.ndarray  # (3,)
    height: float
    height_sigma: float  # height / map σ


def pick_density_peaks(
    rho: np.ndarray,
    n_peaks: int = 30,
    min_fract_dist: float = 0.08,
    min_sigma: float = 2.5,
    subvoxel: bool = True,
) -> List[DensityPeak]:
    """
    Greedy NMS peak pick on a periodic density grid.

    With ``subvoxel`` (default), each peak position is refined per axis by a
    3-point parabola through the neighbouring voxels (periodic). Grid-point
    positions alone carry up to half a voxel (~0.15 Å at sampling 3, 0.9 Å)
    of error per axis, which on its own inflates peak-model R1 by ~0.1.
    """
    rho = np.asarray(rho, dtype=np.float64)
    sigma = float(rho.std()) + 1e-16
    mean = float(rho.mean())
    work = rho.copy()
    # suppress negative / near-mean
    work[work < mean + min_sigma * sigma] = mean

    nx, ny, nz = rho.shape
    peaks: List[DensityPeak] = []
    ri = max(1, int(min_fract_dist * nx))
    rj = max(1, int(min_fract_dist * ny))
    rk = max(1, int(min_fract_dist * nz))

    for rank in range(n_peaks):
        idx = int(np.argmax(work))
        h = float(work.ravel()[idx])
        if h <= mean + min_sigma * sigma:
            break
        i, j, k = np.unravel_index(idx, work.shape)
        pos = np.array([i, j, k], dtype=np.float64)
        if subvoxel:
            c = rho[i, j, k]
            base = (i, j, k)
            for ax, n_ax in enumerate((nx, ny, nz)):
                lo = list(base)
                hi = list(base)
                lo[ax] = (base[ax] - 1) % n_ax
                hi[ax] = (base[ax] + 1) % n_ax
                m, p = rho[tuple(lo)], rho[tuple(hi)]
                curv = m - 2.0 * c + p
                if curv < 0:  # proper local maximum along this axis
                    pos[ax] += float(np.clip(0.5 * (m - p) / curv, -0.5, 0.5))
        fract = (pos / np.array([nx, ny, nz], dtype=np.float64)) % 1.0
        peaks.append(
            DensityPeak(
                rank=rank,
                fract=fract,
                height=h,
                height_sigma=(h - mean) / sigma,
            )
        )
        for di in range(-ri, ri + 1):
            for dj in range(-rj, rj + 1):
                for dk in range(-rk, rk + 1):
                    work[(i + di) % nx, (j + dj) % ny, (k + dk) % nz] = mean
    return peaks


def peaks_to_xyz_lines(
    peaks: List[DensityPeak],
    cell: np.ndarray,
    element: str = "Q",
) -> List[str]:
    """Cartesian approx. XYZ for visualization (orthogonalization)."""
    from grok_phase_solver.io.cif import CrystalStructure

    M = CrystalStructure("t", cell, "P1").orth_matrix
    lines = [f"{len(peaks)}", "gps-solve density peaks"]
    for p in peaks:
        xyz = M @ p.fract
        lines.append(f"{element:2s} {xyz[0]:12.5f} {xyz[1]:12.5f} {xyz[2]:12.5f}")
    return lines
