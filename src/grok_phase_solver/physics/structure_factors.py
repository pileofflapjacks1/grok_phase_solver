"""
Structure factor calculation.

F(h) = Σ_j occ_j f_j(s) T_j(s) exp(2π i h·r_j)

Direct summation is exact for small unit cells; FFT path available for grids.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .form_factors import atomic_form_factor, debye_waller
from .reciprocal import d_spacing


def compute_structure_factors(
    hkl: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    cell: np.ndarray,
    b_isos: Optional[np.ndarray] = None,
    occs: Optional[np.ndarray] = None,
    anomalous: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Direct-summation structure factors.

    Parameters
    ----------
    hkl : (M, 3) Miller indices
    fracs : (N, 3) fractional coordinates (full unit cell contents)
    elements : length-N element symbols
    cell : (6,) unit cell
    b_isos : (N,) B-factors (Å²); default 0
    occs : (N,) occupancies; default 1
    anomalous : (N,) complex f' + i f''; optional

    Returns
    -------
    F : (M,) complex structure factors
    """
    hkl = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    n_atoms = fracs.shape[0]
    n_refl = hkl.shape[0]

    if b_isos is None:
        b_isos = np.zeros(n_atoms)
    else:
        b_isos = np.asarray(b_isos, dtype=np.float64).reshape(n_atoms)
    if occs is None:
        occs = np.ones(n_atoms)
    else:
        occs = np.asarray(occs, dtype=np.float64).reshape(n_atoms)

    # s = sinθ/λ = 1/(2d)
    d = d_spacing(hkl, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-8))

    # Phase factors: exp(2π i h·r) — (M, N)
    # hkl @ fracs.T
    phase = 2.0 * np.pi * (hkl @ fracs.T)
    exp_term = np.exp(1j * phase)  # (M, N)

    F = np.zeros(n_refl, dtype=np.complex128)
    # Group by element for form-factor efficiency
    unique_els = {}
    for i, el in enumerate(elements):
        unique_els.setdefault(el, []).append(i)

    for el, idxs in unique_els.items():
        idxs = np.array(idxs, dtype=int)
        fj = atomic_form_factor(el, s)  # (M,)
        for j in idxs:
            T = debye_waller(b_isos[j], s)
            fjT = fj * T * occs[j]
            if anomalous is not None:
                fjT = fjT + anomalous[j]
            F += fjT * exp_term[:, j]

    return F


def structure_factors_from_density(
    density: np.ndarray,
    volume: float,
) -> np.ndarray:
    """
    Discrete Fourier transform: F(h) ≈ V/N * FFT(ρ).

    Convention: numpy FFT with reciprocal-lattice indexing.
    ``density`` is on a periodic real-space grid covering the unit cell.
    """
    n = density.size
    return (volume / n) * np.fft.fftn(density)


def scale_to_observed(
    F_calc: np.ndarray,
    F_obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Least-squares scale k minimizing ||k|F_calc| − |F_obs|||².

    Returns scaled F_calc (complex, phases preserved) and k.
    """
    ac = np.abs(F_calc)
    ao = np.abs(F_obs)
    if weights is None:
        weights = np.ones_like(ao)
    num = np.sum(weights * ac * ao)
    den = np.sum(weights * ac * ac) + 1e-16
    k = float(num / den)
    return k * F_calc, k


def apply_resolution_falloff(
    F: np.ndarray,
    hkl: np.ndarray,
    cell: np.ndarray,
    B_overall: float,
) -> np.ndarray:
    """Multiply F by overall Debye–Waller exp(−B s²)."""
    d = d_spacing(hkl, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-8))
    return F * np.exp(-B_overall * s * s)
