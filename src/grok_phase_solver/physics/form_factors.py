"""
X-ray atomic form factors (Waasmaier–Kirfel 5-Gaussian expansion).

f(s) = Σ_{i=1}^{5} a_i exp(−b_i s²) + c
where s = sin(θ)/λ in Å⁻¹.

Coefficients from International Tables / common crystallographic tables
for light atoms commonly found in organics.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# (a1..a5, b1..b5, c) Waasmaier-Kirfel style simplified ITC coefficients
# Values approximate ITC Vol. C for neutral atoms; sufficient for Phase-1 demos.
_FORM_FACTORS: Dict[str, Tuple[float, ...]] = {
    # a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c
    "H": (0.493002, 0.322912, 0.140191, 0.040810, 0.0, 10.5109, 26.1257, 3.14236, 57.7997, 0.0, 0.003038),
    "C": (2.31000, 1.02000, 1.58860, 0.865000, 0.0, 20.8439, 10.2075, 0.568700, 51.6512, 0.0, 0.215600),
    "N": (12.2126, 3.13220, 2.01250, 1.16630, 0.0, 0.005700, 9.89330, 28.9975, 0.582600, 0.0, -11.529),
    "O": (3.04850, 2.28680, 1.54630, 0.867000, 0.0, 13.2771, 5.70110, 0.323900, 32.9089, 0.0, 0.250800),
    "F": (3.53920, 2.64120, 1.51700, 1.02430, 0.0, 10.2825, 4.29440, 0.261500, 26.1476, 0.0, 0.277600),
    "P": (6.43450, 4.17910, 1.78000, 1.49080, 0.0, 1.90670, 27.1570, 0.526000, 68.1645, 0.0, 1.11490),
    "S": (6.90530, 5.20340, 1.43790, 1.58630, 0.0, 1.46790, 22.2151, 0.253600, 56.1720, 0.0, 0.866900),
    "CL": (11.4604, 7.19640, 6.25560, 1.64550, 0.0, 0.010400, 1.16620, 18.5194, 47.7784, 0.0, -9.55740),
    "Cl": (11.4604, 7.19640, 6.25560, 1.64550, 0.0, 0.010400, 1.16620, 18.5194, 47.7784, 0.0, -9.55740),
    "BR": (17.1789, 5.23580, 5.63770, 3.98510, 0.0, 2.17230, 16.5796, 0.260900, 41.4328, 0.0, 2.95570),
    "Br": (17.1789, 5.23580, 5.63770, 3.98510, 0.0, 2.17230, 16.5796, 0.260900, 41.4328, 0.0, 2.95570),
    "I": (20.1472, 18.9949, 7.51380, 2.27350, 0.0, 4.34700, 0.381400, 27.7660, 66.8776, 0.0, 4.07120),
    "NA": (4.76260, 3.17360, 1.26740, 1.11280, 0.0, 3.28500, 8.84220, 0.313600, 129.424, 0.0, 0.676000),
    "Na": (4.76260, 3.17360, 1.26740, 1.11280, 0.0, 3.28500, 8.84220, 0.313600, 129.424, 0.0, 0.676000),
    "MG": (5.42040, 2.17350, 1.22690, 2.30730, 0.0, 2.82750, 79.2611, 0.380800, 7.19370, 0.0, 0.858400),
    "Mg": (5.42040, 2.17350, 1.22690, 2.30730, 0.0, 2.82750, 79.2611, 0.380800, 7.19370, 0.0, 0.858400),
    "CA": (15.6348, 7.95180, 8.43720, 0.853700, 0.0, 1.52560, 8.50960, 0.116400, 38.4030, 0.0, -14.875),
    "Ca": (15.6348, 7.95180, 8.43720, 0.853700, 0.0, 1.52560, 8.50960, 0.116400, 38.4030, 0.0, -14.875),
    "FE": (11.7695, 7.35730, 3.52220, 2.30450, 0.0, 4.76110, 0.307200, 15.3535, 76.8805, 0.0, 1.03690),
    "Fe": (11.7695, 7.35730, 3.52220, 2.30450, 0.0, 4.76110, 0.307200, 15.3535, 76.8805, 0.0, 1.03690),
}

# Point-atom Z defaults for unknown elements
_Z = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17,
    "AR": 18, "K": 19, "CA": 20, "FE": 26, "BR": 35, "I": 53,
}


def form_factor_table() -> Dict[str, Tuple[float, ...]]:
    """Return a copy of the internal form-factor coefficient table."""
    return dict(_FORM_FACTORS)


def atomic_form_factor(element: str, s: np.ndarray) -> np.ndarray:
    """
    Compute f(s) for element at scattering vectors s = sin(θ)/λ (Å⁻¹).

    Parameters
    ----------
    element : chemical symbol (e.g. 'C', 'O', 'Cl')
    s : array of s values

    Returns
    -------
    f : same shape as s
    """
    s = np.asarray(s, dtype=np.float64)
    s2 = s * s
    el = element.strip()
    # normalize e.g. 'cl' -> try Cl, CL
    key = el if el in _FORM_FACTORS else el.upper() if el.upper() in _FORM_FACTORS else el.capitalize()

    if key in _FORM_FACTORS:
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c = _FORM_FACTORS[key]
        f = (
            a1 * np.exp(-b1 * s2)
            + a2 * np.exp(-b2 * s2)
            + a3 * np.exp(-b3 * s2)
            + a4 * np.exp(-b4 * s2)
            + a5 * np.exp(-b5 * s2)
            + c
        )
        return f

    # Fallback: point atom with Z electrons (constant form factor)
    z = _Z.get(el.upper(), 6)
    return np.full_like(s, float(z), dtype=np.float64)


def debye_waller(b_iso: float, s: np.ndarray) -> np.ndarray:
    """Temperature factor T = exp(−B s²) with s = sinθ/λ."""
    s = np.asarray(s, dtype=np.float64)
    return np.exp(-float(b_iso) * s * s)
