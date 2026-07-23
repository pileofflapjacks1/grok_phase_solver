"""
Uncertainty quantification for phases and maps (truth-free).

Provides circular statistics across multistart / hybrid phase sets and
simple per-reflection phase probabilities for report.md and GUI.

Methods
-------
1. **Multistart circular variance** — given several phase vectors (same |F|),
   compute mean resultant length R̄ and circular std per reflection.
2. **Bootstrap modulus residual proxy** — light amplitude bootstrap of free FOM.
3. **Map uncertainty** — voxel-wise std of densities from multiple phase sets
   (optional, memory-heavy; shape-matched).

Honest limits
-------------
- Multistart agreement ≠ correctness (all starts may share a wrong basin).
- Use with free FOM and seed-quality Class; not a substitute for R_free in LS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _wrap(ph: np.ndarray) -> np.ndarray:
    return (np.asarray(ph, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def circular_mean_resultant(
    phase_sets: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-reflection circular mean, mean resultant length R̄, and circular std.

    R̄ ∈ [0, 1]: 1 = perfect agreement across starts; 0 = uniform on circle.
    Circular std ≈ √(−2 log R̄) (radians), capped.
    """
    if not phase_sets:
        raise ValueError("phase_sets empty")
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in phase_sets], axis=0)
    n_s, n_r = arr.shape
    if weights is None:
        w = np.ones(n_s, dtype=np.float64) / n_s
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-16)
    z = np.exp(1j * arr)  # (S, R)
    mean_z = np.tensordot(w, z, axes=(0, 0))  # (R,)
    Rbar = np.abs(mean_z)
    mu = np.angle(mean_z)
    # circular std
    with np.errstate(divide="ignore", invalid="ignore"):
        cstd = np.sqrt(np.maximum(-2.0 * np.log(np.clip(Rbar, 1e-12, 1.0)), 0.0))
    cstd = np.where(Rbar < 1e-8, np.pi, cstd)
    return mu, Rbar.astype(np.float64), cstd.astype(np.float64)


def phase_probability_from_resultant(
    Rbar: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Map mean resultant length to a pseudo phase probability ∈ (0, 1].

    p ≈ R̄^(1/T); T=1 default. Not a calibrated Bayesian posterior.
    """
    R = np.asarray(Rbar, dtype=np.float64)
    t = max(float(temperature), 1e-6)
    return np.clip(R ** (1.0 / t), 0.0, 1.0)


def multistart_phase_uncertainty(
    phase_sets: Sequence[np.ndarray],
    amplitudes: Optional[np.ndarray] = None,
    *,
    weights: Optional[Sequence[float]] = None,
    free_fom_composites: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Full UQ summary from multiple phase solutions.

    If ``free_fom_composites`` given, use them as start weights (softmax).
    """
    if free_fom_composites is not None and weights is None:
        fc = np.asarray(free_fom_composites, dtype=np.float64)
        # softmax weights
        x = fc - np.max(fc)
        ex = np.exp(np.clip(x * 4.0, -20, 20))
        weights = (ex / (np.sum(ex) + 1e-16)).tolist()

    mu, Rbar, cstd = circular_mean_resultant(phase_sets, weights=weights)
    p_phi = phase_probability_from_resultant(Rbar)
    amp = None if amplitudes is None else np.asarray(amplitudes, dtype=np.float64)
    if amp is not None and len(amp) == len(Rbar):
        wamp = amp / (np.sum(amp) + 1e-16)
        mean_R = float(np.sum(wamp * Rbar))
        mean_p = float(np.sum(wamp * p_phi))
        mean_cstd_deg = float(np.sum(wamp * np.rad2deg(cstd)))
    else:
        mean_R = float(np.mean(Rbar))
        mean_p = float(np.mean(p_phi))
        mean_cstd_deg = float(np.mean(np.rad2deg(cstd)))

    # strong reflections if amplitudes given
    strong_frac_confident = None
    if amp is not None and len(amp) == len(Rbar):
        n_strong = max(int(0.3 * len(amp)), 5)
        idx = np.argsort(-amp)[:n_strong]
        strong_frac_confident = float(np.mean(Rbar[idx] >= 0.7))

    return {
        "n_starts": len(phase_sets),
        "mean_resultant_length": mean_R,
        "mean_phase_probability": mean_p,
        "mean_circular_std_deg": mean_cstd_deg,
        "frac_high_confidence": float(np.mean(Rbar >= 0.7)),
        "frac_low_confidence": float(np.mean(Rbar < 0.4)),
        "strong_frac_confident": strong_frac_confident,
        "phase_mean": mu,
        "resultant_length": Rbar,
        "circular_std_rad": cstd,
        "phase_probability": p_phi,
        "method": "multistart_circular",
        "note": (
            "Agreement across starts; may be jointly wrong. "
            "Use with free FOM / seed quality."
        ),
    }


def density_uncertainty(
    densities: Sequence[np.ndarray],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Voxel-wise mean and std of density maps (same shape required)."""
    arr = np.stack([np.asarray(d, dtype=np.float64) for d in densities], axis=0)
    if weights is None:
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-16)
        mean = np.tensordot(w, arr, axes=(0, 0))
        var = np.tensordot(w, (arr - mean) ** 2, axes=(0, 0))
        std = np.sqrt(np.maximum(var, 0.0))
    return {
        "mean_density": mean,
        "std_density": std,
        "mean_std": float(np.mean(std)),
        "max_std": float(np.max(std)),
        "relative_uncertainty": float(np.mean(std) / (np.std(mean) + 1e-16)),
    }


def bootstrap_free_fom_spread(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    *,
    n_boot: int = 12,
    seed: int = 0,
    frac: float = 0.85,
) -> Dict[str, float]:
    """
    Bootstrap free-FOM composite by random reflection subsets (truth-free UQ).

    Cheap stability diagnostic for report.md.
    """
    from grok_phase_solver.solvers.free_fom import free_fom

    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64)
    n = len(amp)
    m = max(int(frac * n), 10)
    comps = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        try:
            f = free_fom(hkl[idx], amp[idx], ph[idx], cell)
            comps.append(float(f["composite"]))
        except Exception:
            continue
    if not comps:
        return {"n_boot": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n_boot": len(comps),
        "mean": float(np.mean(comps)),
        "std": float(np.std(comps)),
        "min": float(np.min(comps)),
        "max": float(np.max(comps)),
    }
