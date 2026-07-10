"""
Simple peak picking on electron density for a first atom list.

Not a SHELXS peak-search replacement: fractional coords of strongest
positive density maxima, with basic NMS. Useful as a starting model for
Olex2/SHELXL refinement after phasing.
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
) -> List[DensityPeak]:
    """
    Greedy NMS peak pick on a periodic density grid.
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
        fract = np.array([i / nx, j / ny, k / nz], dtype=np.float64)
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
