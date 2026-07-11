"""
Truth-free figures of merit for ranking phase sets / hybrids.

Scientific motivation
---------------------
When phases are attached to observed moduli, ρ = IFFT(|F_obs| e^{iφ})
**already satisfies** the Fourier modulus constraint (up to numerics).
Therefore a residual computed *after* modulus projection is vacuous
(always ≈ 0). Informative free FOMs must probe **real-space structure**
and **self-consistency of physical constraints** not already hard-enforced.

Primary diagnostics (all truth-free):

1. **R₊ (positivity residual)** — project ρ → max(ρ,0), FFT, compare |F|
   to |F_obs|. Atomic positive densities keep low R₊; random phases yield
   large negative density whose clipping wrecks moduli (Fienup / CF lore).

2. **Atomicity / peakiness** — correct small-molecule maps are sparse and
   peaked: high kurtosis, high max/σ, density concentrated in few voxels,
   strong NMS peaks.

3. **Skewness** — atomic ρ is positively skewed (heavy positive tail).

4. **Optional shell R₊** — same as R₊ in resolution shells (detects
   high-res over-randomization).

5. **Optional Sayre residual** — equal-atom model: F ≈ α (F ⋆ F); useful
   at atomic resolution.

Composite
---------
Scores in [0,1] are combined with calibrated weights (see
``DEFAULT_WEIGHTS`` and ``scripts/calibrate_free_fom.py``). Higher
``composite`` is better. ``should_accept_polish`` is deliberately
**conservative**: requires composite gain *and* no serious R₊ regression.

References
----------
- Fienup, J. R. (1982). Phase retrieval algorithms. Appl. Opt. 21, 2758.
- Oszlányi & Sütő (2004/2008). Charge flipping.
- Sayre, D. (1952). The squaring method. Acta Cryst. 5, 60.
- Historical FOMs (ABSFOM, etc.) in direct methods — density-side analogue.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.projectors import (
    density_to_F,
    project_positivity,
    r_factor_moduli,
)


# Calibrated on synthetic + COD Fcalc pairs (see calibrate_free_fom.py).
# Weights sum ≈ 1; higher weight = more influence on ranking.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "R_pos": 0.35,       # positivity residual (higher score = lower R)
    "kurtosis": 0.20,    # peakiness / heavy tails
    "peakiness": 0.20,   # max/σ and top-voxel concentration
    "skew": 0.15,        # positive skew of atomic maps
    "pos_frac": 0.10,    # weak: CF maps can be less positive than Fcalc
}

# Conservative polish gate (calibrated to reduce false accepts of bad CF)
DEFAULT_MIN_DELTA = 0.02
DEFAULT_MAX_R_POS_REGRESSION = 0.03  # absolute R₊ may not worsen by more than this
# Large phase rewrites (e.g. CF after PhAI) must improve R₊ by this much.
# Blocks COD 2016452 @ 1.2–2.0 Å false accepts while keeping @ 0.9 Å true accept.
DEFAULT_REWRITE_DISP = 0.50          # amplitude-weighted mean(1−cos Δφ)
DEFAULT_REWRITE_MIN_R_IMPROVE = 0.08  # required R₊ decrease if rewrite


def _logistic(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    z = (x - center) / (scale + 1e-16)
    # numerically stable
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def _score_R(R: float) -> float:
    """Map R ∈ [0, ∞) → (0, 1]; R=0 → 1, R=0.3 → ~0.77, R=1 → 0.5."""
    return float(1.0 / (1.0 + max(R, 0.0)))


def _score_kurtosis(k: float) -> float:
    """
    Excess kurtosis of atomic maps is large and positive.
    Random Gaussian field: excess kurtosis ≈ 0.
    Map k≥0 through soft saturating transform.
    """
    return _logistic(k, center=3.0, scale=3.0)


def _score_skew(skew: float) -> float:
    """Positive skew preferred; random ≈ 0."""
    return _logistic(skew, center=1.0, scale=1.5)


def _score_peakiness(max_over_sigma: float, top_frac_mass: float) -> float:
    """
    max_over_sigma: max(ρ)/σ(ρ) — large for sharp peaks.
    top_frac_mass: fraction of positive density in top 5% voxels.
    """
    s_max = _logistic(max_over_sigma, center=6.0, scale=3.0)
    s_mass = float(np.clip(top_frac_mass, 0.0, 1.0))
    return 0.6 * s_max + 0.4 * s_mass


def positivity_residual(
    rho: np.ndarray,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
) -> float:
    """
    R₊ = R-factor between |F_obs| and |FFT(max(ρ,0))|.

    This is the free residual that was *intended* by the old R_after_ER
    (which incorrectly re-imposed moduli before measuring R → always 0).
    """
    rho_p = project_positivity(np.asarray(rho, dtype=np.float64))
    F = density_to_F(rho_p, hkl, cell)
    return r_factor_moduli(F, amplitudes)


def shell_positivity_residuals(
    rho: np.ndarray,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_shells: int = 4,
) -> Dict[str, float]:
    """R₊ in equal-count resolution shells (low → high resolution)."""
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    rho_p = project_positivity(np.asarray(rho, dtype=np.float64))
    F = density_to_F(rho_p, hkl, cell)
    Fc = np.abs(F)
    d = d_spacing(hkl, cell)
    # sort by d descending (low res first)
    order = np.argsort(-d)
    n = len(order)
    if n < n_shells * 4:
        n_shells = max(1, n // 8)
    edges = np.linspace(0, n, n_shells + 1, dtype=int)
    out: Dict[str, float] = {}
    for i in range(n_shells):
        idx = order[edges[i] : edges[i + 1]]
        if len(idx) == 0:
            continue
        Fo, Fc_s = amp[idx], Fc[idx]
        k = np.sum(Fo * Fc_s) / (np.sum(Fc_s * Fc_s) + 1e-16)
        R = float(np.sum(np.abs(Fo - k * Fc_s)) / (np.sum(Fo) + 1e-16))
        out[f"R_pos_shell_{i}"] = R
    if out:
        out["R_pos_shell_mean"] = float(np.mean(list(out.values())))
        # high-res shell is last
        out["R_pos_shell_hires"] = out[f"R_pos_shell_{n_shells - 1}"]
    return out


def density_atomicity_stats(rho: np.ndarray, top_pct: float = 5.0) -> Dict[str, float]:
    """Truth-free real-space atomicity diagnostics."""
    rho = np.asarray(rho, dtype=np.float64)
    flat = rho.ravel()
    m = float(flat.mean())
    s = float(flat.std()) + 1e-16
    z = (flat - m) / s
    skew = float(np.mean(z ** 3))
    # excess kurtosis (Fisher): normal → 0
    kurt = float(np.mean(z ** 4) - 3.0)
    pos_frac = float((flat >= 0).mean())
    max_over_sigma = float((flat.max() - m) / s)

    # concentration of positive mass in top voxels
    pos = np.maximum(flat, 0.0)
    total_pos = float(pos.sum()) + 1e-16
    n_top = max(1, int(len(flat) * top_pct / 100.0))
    top = np.partition(pos, -n_top)[-n_top:]
    top_frac_mass = float(top.sum() / total_pos)

    # crude peak count proxy: local maxima above mean+2σ (6-neighbour)
    thr = m + 2.0 * s
    # pad periodic
    r = rho
    peaks = (
        (r >= thr)
        & (r >= np.roll(r, 1, 0))
        & (r >= np.roll(r, -1, 0))
        & (r >= np.roll(r, 1, 1))
        & (r >= np.roll(r, -1, 1))
        & (r >= np.roll(r, 1, 2))
        & (r >= np.roll(r, -1, 2))
    )
    n_peaks = int(peaks.sum())
    # peakiness from strongest peaks
    if n_peaks > 0:
        peak_heights = r[peaks]
        mean_peak_sigma = float((peak_heights.mean() - m) / s)
    else:
        mean_peak_sigma = 0.0

    return {
        "pos_frac": pos_frac,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "max_over_sigma": max_over_sigma,
        "top_frac_mass": top_frac_mass,
        "n_local_maxima": float(n_peaks),
        "mean_peak_sigma": mean_peak_sigma,
    }


def sayre_residual(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    n_strong: int = 200,
) -> float:
    """
    Cheap Sayre-like residual on strong reflections.

    For equal atoms, phases of strong E's satisfy approximate convolution
    consistency. We compare unit vectors of F with a triple-product proxy:
    use self-consistency of strongest |F| phases under random-pair
    products is expensive; here we use a simplified density-free form:

      For sorted strong reflections, residual of phase-invariant
      cos(φ_h + φ_k − φ_{h+k}) when h+k is observed — average
      1 − ⟨κ-weighted cos⟩ style without E-normalization.

    Returns value in [0, 2] roughly (lower better); 1.0 if insufficient data.
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    n = len(amp)
    if n < 30:
        return 1.0
    # index map
    key = {tuple(h.tolist()): i for i, h in enumerate(hkl)}
    order = np.argsort(-amp)[: min(n_strong, n)]
    # sample O(M) pairs among strong
    rng = np.random.default_rng(0)
    m = min(len(order), 80)
    idxs = order[:m]
    cos_sum = 0.0
    w_sum = 0.0
    for _ in range(min(400, m * m // 2)):
        i, j = rng.choice(idxs, size=2, replace=False)
        h = hkl[i] + hkl[j]
        t = tuple(h.tolist())
        if t not in key:
            continue
        k = key[t]
        w = amp[i] * amp[j] * amp[k]
        c = np.cos(ph[i] + ph[j] - ph[k])
        cos_sum += w * c
        w_sum += w
    if w_sum < 1e-16:
        return 1.0
    mean_cos = cos_sum / w_sum  # in [-1, 1]; true atomic → often > 0
    return float(1.0 - mean_cos)  # lower better; random ~ 1


def free_fom(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    density: np.ndarray | None = None,
    weights: Optional[Dict[str, float]] = None,
    include_shells: bool = True,
    include_sayre: bool = True,
) -> Dict[str, float]:
    """
    Truth-free diagnostics for a phase set.

    Returns a dict with raw metrics, per-component scores in ~[0,1],
    and ``composite`` (higher better). Backward-compatible keys:
    ``pos_frac``, ``skewness``, ``R_after_ER`` (now correctly = R₊),
    ``composite``.
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    if density is None:
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases), cell
        )
    rho = np.asarray(density, dtype=np.float64)

    # --- core residual (FIXED) ---
    R_pos = positivity_residual(rho, hkl, amp, cell)

    # --- atomicity ---
    atom = density_atomicity_stats(rho)

    # --- scores ---
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    # normalize weights
    wsum = sum(w.values()) + 1e-16
    w = {k: v / wsum for k, v in w.items()}

    s_R = _score_R(R_pos)
    s_kurt = _score_kurtosis(atom["excess_kurtosis"])
    s_peak = _score_peakiness(atom["max_over_sigma"], atom["top_frac_mass"])
    s_skew = _score_skew(atom["skewness"])
    s_pos = float(np.clip(atom["pos_frac"], 0.0, 1.0))

    composite = (
        w.get("R_pos", 0) * s_R
        + w.get("kurtosis", 0) * s_kurt
        + w.get("peakiness", 0) * s_peak
        + w.get("skew", 0) * s_skew
        + w.get("pos_frac", 0) * s_pos
    )

    out: Dict[str, float] = {
        # raw
        "R_pos": R_pos,
        "R_after_ER": R_pos,  # correct semantics; name kept for API compat
        "pos_frac": atom["pos_frac"],
        "skewness": atom["skewness"],
        "excess_kurtosis": atom["excess_kurtosis"],
        "max_over_sigma": atom["max_over_sigma"],
        "top_frac_mass": atom["top_frac_mass"],
        "n_local_maxima": atom["n_local_maxima"],
        "mean_peak_sigma": atom["mean_peak_sigma"],
        # scores
        "score_R_pos": s_R,
        "score_kurtosis": s_kurt,
        "score_peakiness": s_peak,
        "score_skew": s_skew,
        "score_pos_frac": s_pos,
        "composite": float(composite),
        "fom_version": 2.0,
    }

    if include_shells:
        shells = shell_positivity_residuals(rho, hkl, amp, cell)
        out.update(shells)
        if "R_pos_shell_hires" in shells:
            # mild bonus/penalty: high-res R₊ should not explode
            out["score_shell_hires"] = _score_R(shells["R_pos_shell_hires"])
            # blend 5% into composite without changing weights API
            out["composite"] = float(0.95 * out["composite"] + 0.05 * out["score_shell_hires"])

    if include_sayre:
        R_say = sayre_residual(hkl, amp, phases)
        out["R_sayre"] = R_say
        out["score_sayre"] = _score_R(R_say)  # R_sayre~0 good → score~1; ~1 random → 0.5
        out["composite"] = float(0.92 * out["composite"] + 0.08 * out["score_sayre"])

    return out


def phase_displacement(
    phases_before: np.ndarray,
    phases_after: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Amplitude-weighted mean of (1 − cos Δφ) ∈ [0, 2].

    0 = identical phases; ~1 ≈ random relative phases; large values mean the
    polish is a *rewrite*, not a local refinement.
    """
    p0 = np.asarray(phases_before, dtype=np.float64)
    p1 = np.asarray(phases_after, dtype=np.float64)
    c = 1.0 - np.cos(p1 - p0)
    if weights is None:
        return float(np.mean(c))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.average(c, weights=w))


def should_accept_polish(
    fom_before: Dict[str, float],
    fom_after: Dict[str, float],
    min_delta: float = DEFAULT_MIN_DELTA,
    max_R_pos_regression: float = DEFAULT_MAX_R_POS_REGRESSION,
    require_R_pos_not_worse: bool = True,
    phase_disp: Optional[float] = None,
    rewrite_disp_threshold: float = DEFAULT_REWRITE_DISP,
    rewrite_min_R_improve: float = DEFAULT_REWRITE_MIN_R_IMPROVE,
) -> bool:
    """
    Accept polish if free FOM improves without wrecking positivity residual.

    Rules:
      1. composite_after ≥ composite_before + min_delta
      2. R_pos_after ≤ R_pos_before + max_R_pos_regression  (if require_R_pos_not_worse)
      3. **Rewrite trust-region** (if phase_disp is given and ≥ rewrite_disp_threshold):
         require R_pos_before − R_pos_after ≥ rewrite_min_R_improve

    Rule 3 addresses the COD 2016452 pathology: CF can raise composite / lower
    R₊ slightly while *destroying* a good PhAI seed at low resolution. Helpful
    polishes (e.g. PhAI+CF @ 0.9 Å) improve R₊ substantially (~0.12); harmful
    ones improve R₊ only modestly (~0.01–0.06) while rewriting phases.
    """
    if fom_after.get("composite", -1) < fom_before.get("composite", 0) + min_delta:
        return False
    R0 = fom_before.get("R_pos", fom_before.get("R_after_ER", 0.0))
    R1 = fom_after.get("R_pos", fom_after.get("R_after_ER", 0.0))
    if require_R_pos_not_worse:
        if R1 > R0 + max_R_pos_regression:
            return False
    if phase_disp is not None and phase_disp >= rewrite_disp_threshold:
        if (R0 - R1) < rewrite_min_R_improve:
            return False
    return True


def rank_phase_sets(
    foms: Sequence[Dict[str, float]],
) -> np.ndarray:
    """Return indices that sort phase sets best→worst by composite."""
    comps = np.array([f.get("composite", -1.0) for f in foms], dtype=np.float64)
    return np.argsort(-comps)


def compare_fom(
    fom_a: Dict[str, float],
    fom_b: Dict[str, float],
) -> Dict[str, float]:
    """Signed deltas (b − a) for key metrics; positive composite_delta favors b."""
    keys = ["composite", "R_pos", "skewness", "excess_kurtosis", "max_over_sigma", "pos_frac"]
    out = {}
    for k in keys:
        if k in fom_a and k in fom_b:
            out[f"delta_{k}"] = float(fom_b[k] - fom_a[k])
    return out
