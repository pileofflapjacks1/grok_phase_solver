"""
Wilson plot and domain-gap diagnostics.

Wilson (1942): for random atoms, ⟨|F|²⟩_shell ≈ Σ_j f_j(s)² exp(−2 B s²)
so ln( ⟨|F|²⟩ / Σ f² ) ≈ −2 B s²  (linear in s²).

Domain gap: compare synthetic vs experimental Wilson slopes / χ².
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.form_factors import atomic_form_factor
from grok_phase_solver.physics.reciprocal import d_spacing, resolution_shells


def wilson_plot(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    elements: Optional[Sequence[str]] = None,
    n_shells: int = 12,
) -> Dict[str, np.ndarray]:
    """
    Compute Wilson plot points.

    If elements is None, uses a single carbon-equivalent scale (Σ f² ∝ f_C²).
    Returns s2 centers, ln_mean_I, and linear fit (B_overall, intercept).
    """
    hkl = np.asarray(hkl)
    amp = np.asarray(amplitudes, dtype=np.float64)
    d = d_spacing(hkl, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-8))
    s2 = s * s
    I = amp * amp

    shell_id, _, _ = resolution_shells(d, n_shells=n_shells)
    s2_c = []
    lnI = []
    for k in range(n_shells):
        m = shell_id == k
        if np.sum(m) < 5:
            continue
        mean_I = np.mean(I[m])
        s2m = np.mean(s2[m])
        if elements is not None:
            # Σ f_j² at this s
            sm = np.sqrt(s2m)
            sum_f2 = sum(float(atomic_form_factor(el, np.array([sm]))[0]) ** 2 for el in elements)
            y = np.log(mean_I / (sum_f2 + 1e-16) + 1e-16)
        else:
            # relative Wilson: ln ⟨I⟩ only
            y = np.log(mean_I + 1e-16)
        s2_c.append(s2m)
        lnI.append(y)

    s2_c = np.array(s2_c)
    lnI = np.array(lnI)
    # linear fit lnI = a + b s², B = −b/2 for absolute Wilson with Σf²
    if len(s2_c) >= 2:
        b, a = np.polyfit(s2_c, lnI, 1)
        B = -0.5 * b if elements is not None else -b  # interpretive
    else:
        a, b, B = 0.0, 0.0, 0.0
    return {
        "s2": s2_c,
        "ln_mean": lnI,
        "intercept": float(a),
        "slope": float(b),
        "B_overall": float(B),
    }


def domain_gap_wilson(
    hkl_a: np.ndarray,
    amp_a: np.ndarray,
    cell_a: np.ndarray,
    hkl_b: np.ndarray,
    amp_b: np.ndarray,
    cell_b: np.ndarray,
    n_shells: int = 10,
) -> Dict[str, float]:
    """
    Compare two datasets' Wilson slopes and normalized intensity histograms.

    Returns slope difference and KS-like L1 distance between I distributions
    on common s² bins (rough domain-gap score; lower is more similar).
    """
    wa = wilson_plot(hkl_a, amp_a, cell_a, n_shells=n_shells)
    wb = wilson_plot(hkl_b, amp_b, cell_b, n_shells=n_shells)
    slope_diff = abs(wa["slope"] - wb["slope"])
    # intensity CDF on log scale
    Ia = np.sort(amp_a**2)
    Ib = np.sort(amp_b**2)
    Ia = Ia / (Ia.mean() + 1e-16)
    Ib = Ib / (Ib.mean() + 1e-16)
    # sample quantiles
    q = np.linspace(0.1, 0.9, 17)
    qa = np.quantile(Ia, q)
    qb = np.quantile(Ib, q)
    l1 = float(np.mean(np.abs(qa - qb)))
    return {
        "slope_diff": float(slope_diff),
        "B_a": wa["B_overall"],
        "B_b": wb["B_overall"],
        "intensity_quantile_L1": l1,
        "domain_gap_score": float(slope_diff + l1),
    }
