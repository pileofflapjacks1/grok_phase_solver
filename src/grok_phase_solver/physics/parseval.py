"""
Parseval / Plancherel identities for crystallographic Fourier pairs.

Continuous form (unit cell volume V):

    (1/V) ∫_V |ρ(r)|² dr  =  (1/V²) Σ_h |F(h)|²

Discrete FFT conventions in this package:
    F_grid = FFT(ρ) * (V / N)     roughly  (forward embedding)
    ρ      = IFFT(F_grid) * (N / V)

So Σ |F_embedded|² / N  ≈  V * ⟨ρ²⟩  under dense sampling of reflections.
We report relative error for diagnostics — perfect equality requires:
all reflections on the grid, consistent centering, and no missing data.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .density import density_from_structure_factors, place_reflections_on_grid
from .reciprocal import d_spacing


def unit_cell_volume(cell: np.ndarray) -> float:
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    cos_al, cos_be, cos_ga = np.cos(al), np.cos(be), np.cos(ga)
    v2 = 1 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga
    return float(a * b * c * np.sqrt(max(v2, 0.0)))


def parseval_check(
    hkl: np.ndarray,
    F: np.ndarray,
    cell: np.ndarray,
    shape: Optional[Tuple[int, int, int]] = None,
    d_min: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compare reciprocal-space energy to real-space density energy.

    Returns dict with E_recip, E_real, relative_error.
    """
    F = np.asarray(F, dtype=np.complex128)
    V = unit_cell_volume(cell)
    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))

    rho = density_from_structure_factors(
        hkl, F, cell, shape=shape, d_min=d_min, sampling=4.0
    )
    N = rho.size
    # Discrete Plancherel for our embedding:
    # ρ = ifftn(F_grid) * (N/V) with F_grid carrying structure factors
    # ⇒ ⟨|ρ|²⟩ * V  ≈  (1/V) Σ_h |F(h)|²   in continuum limit
    E_real = float(np.mean(np.abs(rho) ** 2) * V)
    # Sum only unique observed F (with Friedel already in list if expand)
    E_recip = float(np.sum(np.abs(F) ** 2) / V)

    rel = abs(E_real - E_recip) / (abs(E_recip) + 1e-16)
    return {
        "E_real": E_real,
        "E_recip": E_recip,
        "relative_error": rel,
        "volume": V,
        "N_grid": float(N),
        "N_refl": float(len(F)),
    }


def friedel_check(hkl: np.ndarray, F: np.ndarray, tol: float = 1e-4) -> Dict[str, float]:
    """
    For real density (no anomalous): F(−h) = F(h)*.

    Returns fraction of Friedel pairs satisfying the relation and max error.
    """
    lookup = {}
    for i, (h, k, l) in enumerate(np.asarray(hkl, dtype=int)):
        lookup[(int(h), int(k), int(l))] = i
    errs = []
    n_pairs = 0
    for (h, k, l), i in lookup.items():
        j = lookup.get((-h, -k, -l))
        if j is None or j <= i:
            continue
        n_pairs += 1
        errs.append(abs(F[j] - np.conj(F[i])))
    if not errs:
        return {"n_pairs": 0, "frac_ok": 1.0, "max_err": 0.0}
    errs = np.array(errs)
    ok = np.mean(errs < tol * (np.maximum(np.abs(F).mean(), 1e-6)))
    return {"n_pairs": n_pairs, "frac_ok": float(ok), "max_err": float(errs.max())}
