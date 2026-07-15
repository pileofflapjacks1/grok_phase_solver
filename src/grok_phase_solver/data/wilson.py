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


def amplitude_moments(amplitudes: np.ndarray) -> Dict[str, float]:
    """Normalized amplitude distribution moments (resolution-agnostic)."""
    a = np.asarray(amplitudes, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) < 5:
        return {"n": float(len(a)), "mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    x = a / (a.mean() + 1e-16)
    m = float(x.mean())
    s = float(x.std())
    z = (x - m) / (s + 1e-16)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return {
        "n": float(len(a)),
        "mean": m,
        "std": s,
        "skew": skew,
        "kurt": kurt,
        "frac_weak_0.5": float(np.mean(x < 0.5)),
        "frac_strong_2": float(np.mean(x > 2.0)),
    }


def resolution_coverage(
    hkl: np.ndarray,
    cell: np.ndarray,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
) -> Dict[str, float]:
    """d-min/max and shell completeness proxy (observed count only)."""
    d = d_spacing(hkl, cell)
    d = d[np.isfinite(d)]
    out = {
        "d_min_obs": float(d.min()) if len(d) else None,
        "d_max_obs": float(d.max()) if len(d) else None,
        "n_refl": int(len(d)),
        "median_d": float(np.median(d)) if len(d) else None,
    }
    if d_min is not None:
        out["frac_above_d_min"] = float(np.mean(d >= d_min))
    if d_max is not None:
        out["frac_below_d_max"] = float(np.mean(d <= d_max))
    return out


def domain_gap_report(
    synth: Dict,
    exp: Dict,
    *,
    n_shells: int = 10,
    label_a: str = "synthetic",
    label_b: str = "experimental",
) -> Dict:
    """
    Rich domain-gap summary between two |F| datasets.

    Each of synth/exp: keys hkl, amplitudes, cell; optional elements, name.
    """
    gap = domain_gap_wilson(
        synth["hkl"], synth["amplitudes"], synth["cell"],
        exp["hkl"], exp["amplitudes"], exp["cell"],
        n_shells=n_shells,
    )
    ma = amplitude_moments(synth["amplitudes"])
    mb = amplitude_moments(exp["amplitudes"])
    ra = resolution_coverage(synth["hkl"], synth["cell"])
    rb = resolution_coverage(exp["hkl"], exp["cell"])
    wa = wilson_plot(synth["hkl"], synth["amplitudes"], synth["cell"], n_shells=n_shells)
    wb = wilson_plot(exp["hkl"], exp["amplitudes"], exp["cell"], n_shells=n_shells)

    moment_l1 = float(
        abs(ma["skew"] - mb["skew"])
        + abs(ma["kurt"] - mb["kurt"])
        + abs(ma["frac_strong_2"] - mb["frac_strong_2"])
    )
    score = float(gap["domain_gap_score"] + 0.25 * moment_l1)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "wilson": gap,
        "wilson_a": {
            "B_overall": wa["B_overall"],
            "slope": wa["slope"],
            "n_shells": int(len(wa["s2"])),
        },
        "wilson_b": {
            "B_overall": wb["B_overall"],
            "slope": wb["slope"],
            "n_shells": int(len(wb["s2"])),
        },
        "moments_a": ma,
        "moments_b": mb,
        "resolution_a": ra,
        "resolution_b": rb,
        "moment_gap": moment_l1,
        "domain_gap_score": score,
        "interpretation": (
            "Lower domain_gap_score ⇒ more similar |F| statistics. "
            "Large score suggests synthetic training may not transfer to this experiment."
        ),
    }


def mean_domain_gap_vs_experiment(
    synthetic_list: Sequence[Dict],
    exp: Dict,
    n_shells: int = 10,
) -> Dict:
    """
    Average domain-gap score of many synthetic cells vs one experimental set.
    """
    rows = []
    for i, s in enumerate(synthetic_list):
        r = domain_gap_report(
            s, exp, n_shells=n_shells,
            label_a=s.get("name", f"synth_{i}"),
            label_b=exp.get("name", "exp"),
        )
        rows.append(r)
    scores = [r["domain_gap_score"] for r in rows]
    return {
        "n_synthetic": len(rows),
        "mean_domain_gap_score": float(np.mean(scores)) if scores else None,
        "std_domain_gap_score": float(np.std(scores)) if scores else None,
        "min_domain_gap_score": float(np.min(scores)) if scores else None,
        "max_domain_gap_score": float(np.max(scores)) if scores else None,
        "per_structure": rows,
    }
