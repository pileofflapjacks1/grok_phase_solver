"""
Simulated measured-phase noise for the partial-φ information budget.

The experiment measures intensities. Phases φ are not data unless a seed, a
heavy atom, a fragment, SIR/MAD, a predicted model, or a *measured* phase
puts them there. This module is a **simulator**. A phase-sensitive detector
does not exist in this repository; do not quote σ_φ as a beamline typical.

Noise model
-----------
φ_meas = wrap(φ_true + ε),  ε ~ wrapped noise with circular std ≈ σ_φ.

- ``von_mises`` (default): von Mises on the circle, concentration κ chosen so
  the circular standard deviation √(−2 ln R), R = I₁(κ)/I₀(κ), equals σ_φ.
- ``wrapped_normal``: add N(0, σ_φ) in degrees, then wrap.

Optional |E|-dependent width (simulator schedule, not a calibration)::

    σ(|E|) = max(floor_deg, σ0 / max(|E| / E_ref, ε))

Measured fraction is among the **strong-|E|** set (default: top 30% by |E|,
the C4 / AI-PhaSeed seed set). frac≤20° is reported both among measured
strong |E| and among all strong |E| (unmeasured count as not-within-20°).

The 20° window is the non-centrosymmetric TREF/C4 bar. Centrosymmetric
Patterson is a 0/π sign problem on |F|²; do not apply this window to PATT.

Wrap convention reuses :func:`grok_phase_solver.metrics.phase_error.wrap_phase`
(radians → (−π, π], mapped to degrees (−180, 180]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from grok_phase_solver.metrics.phase_error import wrap_phase

PathLike = Union[str, Path]

# C4 / AI-PhaSeed strong-|E| seed-set size (fraction of all reflections)
DEFAULT_STRONG_FRACTION = 0.30
# Non-centrosymmetric TREF window used by the C4 bar
WINDOW_DEG = 20.0
C4_FRAC_LE_20 = 0.30


def wrap_angle_deg(phi: Union[float, np.ndarray]) -> np.ndarray:
    """Wrap angles in degrees to (−180, 180].

    Uses :func:`wrap_phase` on radians, then maps the −180° endpoint to +180°
    so the documented interval is closed on the right.
    """
    arr = np.asarray(phi, dtype=np.float64)
    w = np.rad2deg(wrap_phase(np.deg2rad(arr)))
    w = np.where(np.isclose(w, -180.0, atol=1e-12), 180.0, w)
    return w


def kappa_from_circular_std_rad(
    sigma_rad: Union[float, np.ndarray],
) -> np.ndarray:
    """von Mises concentration κ with circular std ≈ σ (radians).

    Circular std σ = √(−2 ln R) with R = I₁(κ)/I₀(κ), so R = exp(−σ²/2).
    Invert R → κ with the Sra / Fisher piecewise approximation:

    - R < 0.53:  κ ≈ 2R + R³ + 5R⁵/6
    - 0.53 ≤ R < 0.85:  κ ≈ −0.4 + 1.39 R + 0.43/(1−R)
    - R ≥ 0.85:  κ ≈ 1/(R³ − 4R² + 3R)

    σ ≤ 0 → κ = +∞ (delta at the mean). Large σ → κ → 0 (uniform).
    """
    s = np.asarray(sigma_rad, dtype=np.float64)
    out = np.full(s.shape, np.inf, dtype=np.float64)
    pos = np.isfinite(s) & (s > 0)
    if not np.any(pos):
        return out
    R = np.exp(-0.5 * np.square(s[pos]))
    R = np.clip(R, 0.0, 1.0 - 1e-15)
    k = np.empty_like(R)
    lo = R < 0.53
    mid = (R >= 0.53) & (R < 0.85)
    hi = ~lo & ~mid
    k[lo] = 2.0 * R[lo] + R[lo] ** 3 + 5.0 * R[lo] ** 5 / 6.0
    k[mid] = -0.4 + 1.39 * R[mid] + 0.43 / (1.0 - R[mid])
    den = R[hi] ** 3 - 4.0 * R[hi] ** 2 + 3.0 * R[hi]
    k[hi] = np.where(np.abs(den) < 1e-18, 1e12, 1.0 / np.maximum(den, 1e-18))
    k = np.where(R < 1e-12, 0.0, k)
    out[pos] = k
    return out


def add_measured_phase_noise(
    phi_true_deg: Union[float, np.ndarray],
    sigma_phi_deg: Union[float, np.ndarray],
    rng: np.random.Generator,
    model: str = "von_mises",
) -> np.ndarray:
    """φ_meas = φ_true + ε, ε ~ wrapped noise with circular std ≈ σ_φ (degrees).

    Parameters
    ----------
    phi_true_deg
        True phases in degrees (any wrap).
    sigma_phi_deg
        Circular standard deviation in degrees (scalar or per-reflection).
    rng
        NumPy Generator.
    model
        ``von_mises`` (default) or ``wrapped_normal``.

    Returns
    -------
    ndarray
        Wrapped measured phases in degrees, shape matching ``phi_true_deg``.
    """
    phi = np.asarray(phi_true_deg, dtype=np.float64)
    sigma = np.asarray(sigma_phi_deg, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(phi.shape, float(sigma), dtype=np.float64)
    elif sigma.shape != phi.shape:
        sigma = np.broadcast_to(sigma, phi.shape).astype(np.float64, copy=False)

    model_l = (model or "von_mises").lower()
    if model_l in ("wrapped_normal", "wrapped-normal", "gaussian", "normal"):
        eps = sigma * rng.standard_normal(size=phi.shape)
        return wrap_angle_deg(phi + eps)

    if model_l not in ("von_mises", "vonmises", "von-mises"):
        raise ValueError(
            f"unknown measured-phase noise model {model!r}; "
            "use 'von_mises' or 'wrapped_normal'"
        )

    mu_rad = np.deg2rad(phi)
    kap = kappa_from_circular_std_rad(np.deg2rad(sigma))
    out_rad = np.array(mu_rad, copy=True)
    # numpy vonmises is unstable / pointless at huge κ; treat as delta
    finite = np.isfinite(kap) & (kap < 1e6)
    zero_k = np.isfinite(kap) & (kap <= 0)
    if np.any(zero_k):
        out_rad[zero_k] = rng.uniform(-np.pi, np.pi, size=int(np.count_nonzero(zero_k)))
    draw = finite & ~zero_k
    if np.any(draw):
        out_rad[draw] = rng.vonmises(mu_rad[draw], kap[draw])
    return wrap_angle_deg(np.rad2deg(out_rad))


def sigma_phi_from_E(
    E: np.ndarray,
    sigma0_deg: float,
    E_ref: float = 1.0,
    floor_deg: float = 1.0,
) -> np.ndarray:
    """Optional |E|-dependent phase-noise width (simulator schedule).

    Formula::

        σ(|E|) = max(floor_deg, σ0 / max(|E| / E_ref, ε))

    Stronger |E| → narrower σ. This is **not** a beamline calibration and
    must not be quoted as experimental σ_φ.
    """
    e = np.abs(np.asarray(E, dtype=np.float64))
    e_ref = max(float(E_ref), 1e-12)
    ratio = np.maximum(e / e_ref, 1e-12)
    return np.maximum(float(floor_deg), float(sigma0_deg) / ratio)


def _strong_mask(E: np.ndarray, *, strong_fraction: float) -> np.ndarray:
    e = np.abs(np.asarray(E, dtype=np.float64))
    n = int(e.size)
    n_strong = int(np.clip(round(float(strong_fraction) * n), 1, max(n, 1)))
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    order = np.argsort(-e, kind="stable")
    mask[order[:n_strong]] = True
    return mask


def make_measured_phase_mask(
    E: np.ndarray,
    frac_strong: float,
    rng: np.random.Generator,
    e_min: Optional[float] = None,
    *,
    strong_fraction: float = DEFAULT_STRONG_FRACTION,
) -> np.ndarray:
    """Boolean mask of reflections that carry a simulated measured phase.

    Measured fraction is among the **strong-|E|** set:

    - If ``e_min`` is set, candidates are |E| ≥ e_min.
    - Else candidates are the top ``strong_fraction`` of reflections by |E|
      (default 0.30, the C4 / AI-PhaSeed seed set).

    Then the highest-|E| ``frac_strong`` fraction of those candidates is
    selected (ties broken with a tiny RNG jitter). ``frac_strong=1`` measures
    the whole strong set; ``frac_strong=0.3`` measures 30% of the strong set.
    """
    e = np.abs(np.asarray(E, dtype=np.float64).ravel())
    n = int(e.size)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or float(frac_strong) <= 0:
        return mask

    if e_min is not None:
        cand = e >= float(e_min)
        if not np.any(cand):
            cand = np.ones(n, dtype=bool)
    else:
        cand = _strong_mask(e, strong_fraction=strong_fraction)

    idx = np.where(cand)[0]
    n_pick = int(round(float(frac_strong) * len(idx)))
    n_pick = int(np.clip(n_pick, 0, len(idx)))
    if n_pick <= 0:
        return mask
    jitter = rng.uniform(0.0, 1e-12, size=len(idx))
    order = np.argsort(-(e[idx] + jitter), kind="stable")
    pick = idx[order[:n_pick]]
    mask[pick] = True
    return mask


def seed_quality_from_noisy_phases(
    phi_true_deg: np.ndarray,
    phi_meas_deg: np.ndarray,
    E: np.ndarray,
    e_cut: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    *,
    strong_fraction: float = DEFAULT_STRONG_FRACTION,
    window_deg: float = WINDOW_DEG,
) -> Dict[str, Any]:
    """Seed-quality dict for a noisy measured-φ mask.

    ``frac_le_20`` is among **strong |E| that were measured**.
    ``frac_le_20_all_strong`` is among **all** strong |E|, counting unmeasured
    as not-within-20° (the C4 metric).

    Parameters
    ----------
    e_cut
        If given, strong = |E| ≥ e_cut. Else strong = top ``strong_fraction``.
    mask
        Boolean measured mask aligned to the reflection table. If omitted,
        every reflection is treated as measured.
    """
    phi_t = np.asarray(phi_true_deg, dtype=np.float64).ravel()
    phi_m = np.asarray(phi_meas_deg, dtype=np.float64).ravel()
    e = np.abs(np.asarray(E, dtype=np.float64).ravel())
    n = len(e)
    if mask is None:
        meas = np.ones(n, dtype=bool)
    else:
        meas = np.asarray(mask, dtype=bool).ravel()
        if meas.size != n:
            raise ValueError("mask length must match E / phases")

    if e_cut is not None:
        strong = e >= float(e_cut)
        if not np.any(strong):
            strong = _strong_mask(e, strong_fraction=strong_fraction)
    else:
        strong = _strong_mask(e, strong_fraction=strong_fraction)

    dphi = wrap_angle_deg(phi_m - phi_t)
    ad = np.abs(dphi)
    within = ad <= float(window_deg)

    measured_strong = meas & strong
    n_strong = int(strong.sum())
    n_measured = int(meas.sum())
    n_measured_strong = int(measured_strong.sum())

    if n_measured_strong > 0:
        frac_le_20 = float(np.mean(within[measured_strong]))
        circ_abs = float(np.mean(ad[measured_strong]))
        z = np.mean(np.exp(1j * np.deg2rad(dphi[measured_strong])))
        circ_signed = float(np.rad2deg(np.angle(z)))
    else:
        frac_le_20 = float("nan")
        circ_abs = float("nan")
        circ_signed = float("nan")

    if n_strong > 0:
        # unmeasured strong count as not-within-20
        frac_all = float(np.sum(within & measured_strong) / n_strong)
    else:
        frac_all = float("nan")

    return {
        "frac_le_20": frac_le_20,
        "frac_le_20_all_strong": frac_all,
        "circular_mean_error_deg": circ_abs,
        "circular_mean_signed_deg": circ_signed,
        "n_strong": n_strong,
        "n_measured": n_measured,
        "n_measured_strong": n_measured_strong,
        "window_deg": float(window_deg),
        "c4_bar": C4_FRAC_LE_20,
        "meets_c4_bar": bool(np.isfinite(frac_all) and frac_all >= C4_FRAC_LE_20),
    }


def apply_measured_phase_to_seed(
    seed_phases_rad: np.ndarray,
    mask: np.ndarray,
    E: np.ndarray,
    *,
    sigma_phi_deg: float,
    frac_strong: float,
    rng: np.random.Generator,
    model: str = "von_mises",
    e_min: Optional[float] = None,
    strong_fraction: float = DEFAULT_STRONG_FRACTION,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Corrupt an existing seed as a measured-φ simulator.

    1. Among currently seeded reflections (or all, if the mask is empty),
       take the strong-|E| pool (``e_min`` or top ``strong_fraction``).
    2. Keep a ``frac_strong`` subset preferring high |E|.
    3. Add wrapped noise with circular std ≈ ``sigma_phi_deg``.
    4. Dropped reflections are unmasked and random-filled.

    Returns ``(seed_phases_rad, new_mask, meta)`` suitable for
    :func:`grok_phase_solver.solvers.partial_seed.partial_phaseed_solve`
    and :func:`grok_phase_solver.solvers.partial_seed.write_phase_seed_csv`.
    """
    seed_ph = np.asarray(seed_phases_rad, dtype=np.float64).copy()
    mask_in = np.asarray(mask, dtype=bool).ravel()
    e = np.abs(np.asarray(E, dtype=np.float64).ravel())
    n = len(e)
    if mask_in.size != n:
        raise ValueError("mask length must match E / seed phases")

    if e_min is not None:
        strong = e >= float(e_min)
        if not np.any(strong):
            strong = _strong_mask(e, strong_fraction=strong_fraction)
    else:
        strong = _strong_mask(e, strong_fraction=strong_fraction)

    pool = strong & (mask_in if mask_in.any() else np.ones(n, dtype=bool))
    if not pool.any():
        pool = mask_in if mask_in.any() else strong

    n_pick = int(round(float(frac_strong) * int(pool.sum())))
    n_pick = int(np.clip(n_pick, 0, int(pool.sum())))
    idx_pool = np.where(pool)[0]
    jitter = rng.uniform(0.0, 1e-12, size=len(idx_pool)) if len(idx_pool) else np.array([])
    order = np.argsort(-(e[idx_pool] + jitter), kind="stable") if len(idx_pool) else np.array([], dtype=int)
    pick = idx_pool[order[:n_pick]] if n_pick else np.array([], dtype=int)
    new_mask = np.zeros(n, dtype=bool)
    new_mask[pick] = True

    phi_deg = np.rad2deg(seed_ph)
    if new_mask.any() and float(sigma_phi_deg) > 0:
        noisy = add_measured_phase_noise(
            phi_deg[new_mask], float(sigma_phi_deg), rng, model=model
        )
        seed_ph[new_mask] = np.deg2rad(noisy)
        phi_deg[new_mask] = noisy

    dropped = mask_in & ~new_mask
    if dropped.any():
        seed_ph[dropped] = rng.uniform(-np.pi, np.pi, size=int(dropped.sum()))

    meta = {
        "kind": "measured_phase",
        "source": "measured_phi_simulator",
        "information_source": "measured-φ",
        "sigma_phi_deg": float(sigma_phi_deg),
        "frac_strong_requested": float(frac_strong),
        "model": model,
        "n_known": int(new_mask.sum()),
        "fraction": float(new_mask.mean()) if n else 0.0,
        "note": (
            "Simulated measured phases. A phase-sensitive detector does not "
            "exist in this repository."
        ),
    }
    return seed_ph, new_mask, meta


def write_measured_phase_seed_csv(
    path: PathLike,
    hkl: np.ndarray,
    seed_phases_rad: np.ndarray,
    mask: np.ndarray,
) -> Path:
    """Write the measured-φ seed as the same CSV ``partial_phaseed`` already loads."""
    from grok_phase_solver.solvers.partial_seed import write_phase_seed_csv

    return write_phase_seed_csv(path, hkl, seed_phases_rad, mask, phase_unit="deg")
