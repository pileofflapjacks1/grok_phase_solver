"""Side R1 for the frozen strict bar. Not an input to ``solved``.

The strict gate remains ``r1_from_peaks``: the strongest peaks as carbon,
B = 5. This helper places the same peaks, assigns a deposited element when
a peak matches a non-H site, and leaves every other peak as carbon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from grok_phase_solver.metrics.rfactor import r_factor
from grok_phase_solver.metrics.success import _min_image_frac
from grok_phase_solver.physics.structure_factors import compute_structure_factors


@dataclass
class TypedPeakR1:
    """Diagnostic residual. ``solved`` does not read this value."""

    r1: float
    n_peaks_used: int
    n_matched: int
    elements: List[str]


def r1_from_typed_peaks(
    hkl: np.ndarray,
    F_obs: np.ndarray,
    cell: np.ndarray,
    peak_fracs: np.ndarray,
    true_fracs: np.ndarray,
    true_elements: Sequence[str],
    *,
    n_atoms: int,
    origin_shift: np.ndarray,
    tol: float = 0.15,
    b_iso: float = 5.0,
) -> TypedPeakR1:
    """
    R1 of the strongest ``n_atoms`` peaks with deposited element types.

    Peaks are the same list ``r1_from_peaks`` uses (strongest first). Each
    peak is matched, within ``tol`` fractional min-image distance, to one
    unused non-H site after ``origin_shift`` (the shift ``peak_recovery_score``
    applies to the true sites). A matched peak takes that site's element.
    An unmatched peak stays carbon. B and the scaled amplitude residual match
    ``r1_from_peaks``.
    """
    peaks = np.asarray(peak_fracs, dtype=np.float64).reshape(-1, 3)
    atoms = np.asarray(true_fracs, dtype=np.float64).reshape(-1, 3)
    elements = [str(e) for e in true_elements]
    if len(elements) != len(atoms):
        raise ValueError(
            f"true_elements length {len(elements)} != true_fracs {len(atoms)}"
        )
    n = min(int(n_atoms), len(peaks))
    if n <= 0:
        return TypedPeakR1(r1=1.0, n_peaks_used=0, n_matched=0, elements=[])

    shift = np.asarray(origin_shift, dtype=np.float64).reshape(3)
    aligned = (atoms + shift) % 1.0
    used = np.zeros(len(aligned), dtype=bool)
    assigned: List[str] = []
    n_matched = 0
    for peak in peaks[:n]:
        best_j = -1
        best_d = float(tol)
        for j, atom in enumerate(aligned):
            if used[j]:
                continue
            dist = _min_image_frac(peak, atom)
            if dist < best_d:
                best_d = dist
                best_j = j
        if best_j < 0:
            assigned.append("C")
            continue
        used[best_j] = True
        assigned.append(elements[best_j])
        n_matched += 1

    b = np.full(n, float(b_iso), dtype=np.float64)
    F_c = compute_structure_factors(
        hkl, peaks[:n], assigned, cell, b_isos=b
    )
    return TypedPeakR1(
        r1=float(r_factor(F_obs, F_c, scale=True)),
        n_peaks_used=n,
        n_matched=n_matched,
        elements=assigned,
    )
