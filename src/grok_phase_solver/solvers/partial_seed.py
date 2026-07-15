"""
Partial-φ / fragment seed API for hard-region phase extension.

Science motivation
------------------
Oracle AI-PhaSeed tests show that *when a fraction of strong phases is correct*,
classical extension + free-FOM polish can climb out of basin-B failure. Full
ab initio priors (GraphPhaseNet, hard-P1) still land ~mapCC 0.5 on hard cells.
This module makes that path a **first-class API**:

1. **Oracle partial-φ** — known phases on a subset (benchmark / MAD-like masks)
2. **Noisy partial-φ** — controlled MPE on the seed set (robustness curve)
3. **Fragment seed** — Fcalc phases from a partial atomic model (MR-lite / heavy atom)
4. **File seed** — load h,k,l,φ from CSV and extend

All paths feed ``ai_phaseed_solve``. Not a claim of general protein solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.solvers.ai_phaseed import (
    ai_phaseed_solve,
    select_seed_indices,
)
from grok_phase_solver.solvers.direct_methods import normalize_E
from grok_phase_solver.physics.reciprocal import d_spacing

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Seed construction
# ---------------------------------------------------------------------------


def _hkl_key(h: Sequence[int]) -> Tuple[int, int, int]:
    return (int(h[0]), int(h[1]), int(h[2]))


def select_partial_mask(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    fraction: Optional[float] = None,
    n_known: Optional[int] = None,
    mode: str = "strong_E",
    d_min_known: Optional[float] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Boolean mask of reflections treated as *known* phases.

    mode
    ----
    strong_E : top |E| (default; matches AI-PhaSeed seed set)
    strong_F : top |F|
    random   : uniform random subset
    low_res  : lowest-resolution shells first (d ≥ d_min_known or top by d)
    """
    hkl = np.asarray(hkl)
    amp = np.asarray(amplitudes, dtype=np.float64)
    n = len(amp)
    if n_known is None:
        frac = 0.25 if fraction is None else float(fraction)
        n_known = int(np.clip(round(frac * n), 1, n))
    n_known = int(min(max(n_known, 1), n))
    rng = np.random.default_rng(seed)

    if mode == "strong_E":
        idx = select_seed_indices(hkl, amp, cell, n_seed=n_known, by="E")
    elif mode == "strong_F":
        idx = select_seed_indices(hkl, amp, cell, n_seed=n_known, by="F")
    elif mode == "random":
        idx = rng.choice(n, size=n_known, replace=False)
    elif mode == "low_res":
        d = d_spacing(hkl, cell)
        if d_min_known is not None:
            cand = np.where(d >= d_min_known)[0]
            if len(cand) < n_known:
                # fill with next-highest d
                order = np.argsort(-d)
                idx = order[:n_known]
            else:
                # strongest |E| among low-res
                E = normalize_E(hkl, amp, cell)
                sub = cand[np.argsort(-E[cand])[:n_known]]
                idx = sub
        else:
            order = np.argsort(-d)  # large d first
            idx = order[:n_known]
    else:
        raise ValueError(f"unknown mask mode: {mode}")

    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(idx, dtype=int)] = True
    return mask


def build_partial_phase_vector(
    n_refl: int,
    known_idx: np.ndarray,
    known_phases: np.ndarray,
    *,
    fill: str = "random",
    seed: int = 0,
) -> np.ndarray:
    """Full phase array with known phases set; rest random or zero."""
    rng = np.random.default_rng(seed)
    if fill == "random":
        ph = rng.uniform(-np.pi, np.pi, size=n_refl)
    elif fill == "zero":
        ph = np.zeros(n_refl, dtype=np.float64)
    else:
        raise ValueError(fill)
    known_idx = np.asarray(known_idx, dtype=int)
    ph[known_idx] = np.asarray(known_phases, dtype=np.float64)
    return ph


def oracle_partial_seed(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases_true: np.ndarray,
    *,
    fraction: float = 0.25,
    n_known: Optional[int] = None,
    mode: str = "strong_E",
    phase_noise_deg: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Oracle partial seed: true phases on a mask, optional Gaussian phase noise.

    Returns (seed_phases_full, mask, meta).
    """
    mask = select_partial_mask(
        hkl, amplitudes, cell,
        fraction=fraction, n_known=n_known, mode=mode, seed=seed,
    )
    idx = np.where(mask)[0]
    ph_k = np.asarray(phases_true, dtype=np.float64)[idx].copy()
    if phase_noise_deg > 0:
        rng = np.random.default_rng(seed + 1)
        ph_k = ph_k + np.deg2rad(phase_noise_deg) * rng.standard_normal(len(ph_k))
    seed_ph = build_partial_phase_vector(
        len(amplitudes), idx, ph_k, fill="random", seed=seed + 2
    )
    # also put noisy/true values only on mask; keep random elsewhere (already)
    seed_ph[idx] = ph_k
    meta = {
        "kind": "oracle_partial",
        "mode": mode,
        "n_known": int(mask.sum()),
        "fraction": float(mask.mean()),
        "phase_noise_deg": float(phase_noise_deg),
        "mean_seed_mpe_vs_truth": _mpe(ph_k, phases_true[idx]),
    }
    return seed_ph, mask, meta


def fragment_seed_phases(
    hkl: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    cell: np.ndarray,
    *,
    b_iso: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phases (and |Fcalc|) from a partial atomic model.

    Returns (phases, |Fcalc|).
    """
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    if len(fracs) == 0:
        return np.zeros(len(hkl)), np.zeros(len(hkl))
    b = np.full(len(fracs), b_iso, dtype=np.float64)
    F = compute_structure_factors(
        hkl, fracs, list(elements), cell, b_isos=b
    )
    return np.angle(F), np.abs(F)


def fragment_partial_seed(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    *,
    n_atoms: Optional[int] = None,
    atom_fraction: Optional[float] = None,
    seed: int = 0,
    b_iso: float = 8.0,
    use_fcalc_weight: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Keep a random subset of true atoms → Fcalc phases as full seed vector.

    For benchmarks with known truth coordinates. For real use, pass your
    fragment ``fracs`` / ``elements`` directly to ``fragment_seed_phases``.
    """
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    elements = list(elements)
    n = len(fracs)
    rng = np.random.default_rng(seed)
    if n_atoms is None:
        frac = 0.35 if atom_fraction is None else float(atom_fraction)
        n_atoms = int(np.clip(round(frac * n), 1, n))
    n_atoms = int(min(max(n_atoms, 1), n))
    pick = rng.choice(n, size=n_atoms, replace=False)
    fr_sub = fracs[pick]
    el_sub = [elements[i] for i in pick]
    ph, fcalc = fragment_seed_phases(hkl, fr_sub, el_sub, cell, b_iso=b_iso)
    if use_fcalc_weight:
        # weak Fcalc → leave more freedom: blend toward random
        rng2 = np.random.default_rng(seed + 3)
        ph_r = rng2.uniform(-np.pi, np.pi, size=len(ph))
        w = fcalc / (fcalc.max() + 1e-16)
        w = np.clip(w, 0.0, 1.0) ** 0.5
        z = w * np.exp(1j * ph) + (1 - w) * np.exp(1j * ph_r)
        ph = np.angle(z)
    meta = {
        "kind": "fragment",
        "n_atoms_model": n_atoms,
        "n_atoms_true": n,
        "atom_fraction": n_atoms / max(n, 1),
        "elements": el_sub[:20],
    }
    return ph, fr_sub, meta


def load_phase_seed_csv(
    path: PathLike,
    hkl: np.ndarray,
    *,
    phase_unit: str = "deg",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load partial phases from CSV: h,k,l,phase[,weight].

    Unmatched reflections get random phases. Returns (seed_phases, mask, meta).
    """
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    # allow headerless
    if data.dtype.names is None:
        raw = np.loadtxt(path, delimiter=",")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        hkl_f = raw[:, :3].astype(int)
        ph = raw[:, 3].astype(float)
    else:
        names = [n.lower() for n in data.dtype.names]
        def col(*cands):
            for c in cands:
                if c in names:
                    return data[data.dtype.names[names.index(c)]]
            raise KeyError(cands)
        hkl_f = np.column_stack(
            [col("h", "index_h"), col("k", "index_k"), col("l", "index_l")]
        ).astype(int)
        ph = np.asarray(col("phase", "phase_deg", "phi", "ph"), dtype=float)

    if phase_unit in ("deg", "degree", "degrees"):
        ph = np.deg2rad(ph)
    elif phase_unit in ("rad", "radian", "radians"):
        pass
    else:
        raise ValueError(phase_unit)

    key_to_ph = {_hkl_key(h): p for h, p in zip(hkl_f, ph)}
    # Friedel mates
    for h, p in list(key_to_ph.items()):
        fm = (-h[0], -h[1], -h[2])
        if fm not in key_to_ph:
            key_to_ph[fm] = -p

    n = len(hkl)
    rng = np.random.default_rng(0)
    seed_ph = rng.uniform(-np.pi, np.pi, size=n)
    mask = np.zeros(n, dtype=bool)
    mapped = 0
    for i, h in enumerate(hkl):
        k = _hkl_key(h)
        if k in key_to_ph:
            seed_ph[i] = key_to_ph[k]
            mask[i] = True
            mapped += 1
    meta = {
        "kind": "file",
        "path": str(path),
        "n_file": len(hkl_f),
        "n_mapped": mapped,
        "fraction": mapped / max(n, 1),
    }
    return seed_ph, mask, meta


def write_phase_seed_csv(
    path: PathLike,
    hkl: np.ndarray,
    phases: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    phase_unit: str = "deg",
) -> Path:
    """Write h,k,l,phase_deg for masked (or all) reflections."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None:
        mask = np.ones(len(hkl), dtype=bool)
    ph = np.asarray(phases, dtype=np.float64)
    if phase_unit.startswith("deg"):
        ph_out = np.rad2deg(ph)
        hdr = "h,k,l,phase_deg"
    else:
        ph_out = ph
        hdr = "h,k,l,phase_rad"
    lines = [hdr]
    for i in np.where(mask)[0]:
        h, k, l = map(int, hkl[i])
        lines.append(f"{h},{k},{l},{ph_out[i]:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _mpe(a: np.ndarray, b: np.ndarray) -> float:
    d = np.angle(np.exp(1j * (a - b)))
    return float(np.rad2deg(np.mean(np.abs(d))))


# ---------------------------------------------------------------------------
# Solve wrappers
# ---------------------------------------------------------------------------


def partial_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    n_seed: Optional[int] = None,
    seed_fraction: float = 0.30,
    n_extend: int = 20,
    polish: str = "charge_flipping",
    n_polish: int = 60,
    n_starts: int = 2,
    seed: int = 0,
    d_min: Optional[float] = None,
    prior_weight: float = 0.35,
    use_free_fom_gate: bool = True,
    select_by: str = "E",
    verbose: bool = False,
    meta: Optional[Dict] = None,
):
    """
    Extend from a partial phase seed via AI-PhaSeed.

    If ``mask`` is given, only masked reflections are treated as the hard seed
    set (``n_seed`` ignored for selection); ``seed_phases`` must be full-length.
    """
    seed_phases = np.asarray(seed_phases, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        idx = np.where(mask)[0]
        # Build a full prior that is strong on known, weak elsewhere:
        # for selection, pass full seed_phases but force seed set = mask
        # by using n_seed and re-selecting — better: use seed_fraction path
        # with custom seed indices via temporary override.
        # We call ai_phaseed with full prior and set n_seed = |mask| after
        # putting known phases only; select_seed_indices would re-pick.
        # So implement thin wrapper that uses ai_phaseed internals.
        from grok_phase_solver.solvers.ai_phaseed import (
            build_initial_phases,
            phase_extend,
            reimpose_seed,
        )
        from grok_phase_solver.solvers.hybrid import blend_phases
        from grok_phase_solver.solvers.free_fom import free_fom
        from grok_phase_solver.solvers.conditional_hybrid import conditional_polish

        hkl = np.asarray(hkl, dtype=int)
        amp = np.asarray(amplitudes, dtype=np.float64)
        sp = seed_phases[idx]
        trials = []
        best = None
        for s in range(max(1, n_starts)):
            rng = np.random.default_rng(seed + s)
            ph0 = build_initial_phases(len(amp), idx, sp, rng)
            if prior_weight > 0:
                ph0 = blend_phases(
                    seed_phases, ph0, np.full(len(amp), 0.5 * prior_weight)
                )
                ph0 = reimpose_seed(ph0, idx, sp, weight=1.0)
            ph, rho, hist = phase_extend(
                hkl, amp, cell, ph0, idx, sp,
                n_cycles=n_extend,
                seed_weight=1.0,
                seed_weight_final=0.8,
                d_min=d_min,
                full_prior=seed_phases if prior_weight > 0 else None,
                prior_weight=prior_weight,
                verbose=verbose and s == 0,
            )
            if polish and polish != "none" and use_free_fom_gate:
                ph, rho, pinfo = conditional_polish(
                    hkl, amp, cell, ph, polish=polish, n_iter=n_polish,
                    seed=seed + s, d_min=d_min, verbose=verbose and s == 0,
                )
            else:
                pinfo = {"accepted_polish": False}
            fom = free_fom(hkl, amp, ph, cell, density=rho)
            trial = {"start": s, "composite": fom["composite"], "polish": pinfo, "extend": hist}
            trials.append(trial)
            if best is None or fom["composite"] > best[0]:
                best = (fom["composite"], ph, rho, trial)
        assert best is not None
        info = {
            "algorithm": "partial_phaseed",
            "n_seed": int(mask.sum()),
            "seed_fraction_actual": float(mask.mean()),
            "seed_source": "partial_mask",
            "trials": trials,
            "best_trial": best[3],
            "fom_final": free_fom(hkl, amp, best[1], cell, density=best[2]),
            "meta": meta or {},
        }
        return best[1], best[2], info

    # Default: treat seed_phases as full AI prior; PhaSeed picks strong set
    ph, rho, info = ai_phaseed_solve(
        hkl, amplitudes, cell, seed_phases,
        n_seed=n_seed,
        seed_fraction=seed_fraction,
        n_extend=n_extend,
        polish=polish,
        n_polish=n_polish,
        n_starts=n_starts,
        seed=seed,
        d_min=d_min,
        prior_weight=prior_weight,
        use_free_fom_gate=use_free_fom_gate,
        select_by=select_by,
        verbose=verbose,
    )
    info["algorithm"] = "partial_phaseed"
    info["seed_source"] = "full_vector"
    if meta:
        info["meta"] = meta
    return ph, rho, info


def oracle_partial_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases_true: np.ndarray,
    *,
    fraction: float = 0.25,
    n_known: Optional[int] = None,
    mode: str = "strong_E",
    phase_noise_deg: float = 0.0,
    seed: int = 0,
    **kwargs,
):
    """Benchmark helper: oracle mask → partial_phaseed_solve."""
    seed_ph, mask, meta = oracle_partial_seed(
        hkl, amplitudes, cell, phases_true,
        fraction=fraction, n_known=n_known, mode=mode,
        phase_noise_deg=phase_noise_deg, seed=seed,
    )
    return partial_phaseed_solve(
        hkl, amplitudes, cell, seed_ph, mask=mask, seed=seed, meta=meta, **kwargs
    )


def fragment_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    *,
    seed: int = 0,
    b_iso: float = 8.0,
    seed_fraction: float = 0.30,
    **kwargs,
):
    """
    Partial model atoms → Fcalc phases → AI-PhaSeed extension.

    ``fracs`` / ``elements`` are the *known fragment* (not full structure).
    """
    ph_seed, fcalc = fragment_seed_phases(
        hkl, fracs, elements, cell, b_iso=b_iso
    )
    # Prefer strong |Fcalc| reflections as seed set
    meta = {
        "kind": "fragment_model",
        "n_atoms": len(np.asarray(fracs).reshape(-1, 3)),
        "mean_fcalc": float(np.mean(fcalc)),
    }
    return partial_phaseed_solve(
        hkl, amplitudes, cell, ph_seed,
        seed_fraction=seed_fraction,
        seed=seed,
        meta=meta,
        **kwargs,
    )
