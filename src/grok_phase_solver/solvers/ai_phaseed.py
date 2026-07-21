"""
AI-PhaSeed: AI phase seed + classical phase extension.

Implements the hybrid protocol of Carrozzini *et al.* (J. Appl. Cryst. 2025)
as used with PhAI (Larsen *et al.*, Science 2024) under the name AI-PhaSeed:

1. Obtain seed phases from an AI model (PhAI fair, PhaseMLP, or external).
2. Select a strong reflection subset by |E| (or |F|) as the *seed set*.
3. Initialize remaining phases randomly (optionally discretized; centro / bins).
4. Iterate: density → positivity / solvent flatten → FFT → |F_obs| projection,
   re-imposing seed phases with weight (hard fix or soft blend).
5. Optional **DM+AI hybrid** modified tangent (AI phases as a priori info with
   reliability weights; Carrozzini eqs. 3–5 style).
6. Optional free-FOM–gated classical polish (CF / RAAR).
7. Optional **seed-quality** prediction (Class 0/1 style) before extension.

This attacks **basin (B)** failure: a good seed puts search near the correct
density, while free-FOM gating prevents polish from destroying the prior
(see free_fom v2.1, conditional_hybrid).

Honest limits
-------------
Hard ab initio remains unsolved in general; Class 1 seeds and hybrid EDM help
most in the paper's regime (e.g. P2₁/c organics, Vol ~1000–3500 Å³). Partial-φ
requirements for hard cells are unchanged.

References
----------
- Larsen, Rekis, Madsen (2024). PhAI. Science 385, 522–528.
- Carrozzini et al. (2025). J. Appl. Cryst. 58, 1859–1869.
  DOI: 10.1107/S1600576725008271
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.density import (
    density_from_structure_factors,
    grid_shape_from_resolution,
)
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.direct_methods import normalize_E, dm_ai_hybrid_refine
from grok_phase_solver.solvers.density_modification import solvent_flatten
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.hybrid import blend_phases
from grok_phase_solver.solvers.phase_recycle import fourier_modulus_projection
from grok_phase_solver.solvers.conditional_hybrid import conditional_polish
from grok_phase_solver.solvers.projectors import (
    density_to_F,
    project_positivity,
    r_factor_moduli,
    unit_cell_volume,
)


def select_seed_indices(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_seed: Optional[int] = None,
    seed_fraction: float = 0.15,
    min_seed: int = 20,
    max_seed: int = 200,
    by: str = "E",
) -> np.ndarray:
    """
    Indices of strongest reflections for the seed set.

    by:
      - ``"E"``: Wilson-normalized |E| (default, classical DM style)
      - ``"F"``: raw |F|
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    n = len(amp)
    if n_seed is None:
        n_seed = int(np.clip(seed_fraction * n, min_seed, max_seed))
    n_seed = int(min(max(n_seed, 1), n))
    if by == "E":
        score = normalize_E(hkl, amp, cell)
    elif by == "F":
        score = amp
    else:
        raise ValueError(by)
    order = np.argsort(-score)
    return order[:n_seed].astype(int)


def discretize_phases(
    phases: np.ndarray,
    mode: str = "none",
    n_bins: int = 4,
) -> np.ndarray:
    """
    Optional phase discretization (AI-PhaSeed / non-centrosymmetric seeds).

    mode:
      - ``none``: continuous
      - ``centro``: {0, π} (centrosymmetric)
      - ``bins``: n_bins equal bins on (-π, π]
    """
    ph = np.asarray(phases, dtype=np.float64)
    if mode == "none":
        return ph.copy()
    if mode == "centro":
        # nearest of 0 or π (map to [0, π] then fold)
        c = np.cos(ph)
        return np.where(c >= 0.0, 0.0, np.pi)
    if mode == "bins":
        # quantize to midpoints of equal bins
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        # wrap ph to (-π, π]
        phw = (ph + np.pi) % (2 * np.pi) - np.pi
        idx = np.digitize(phw, edges[1:-1])
        return mids[np.clip(idx, 0, n_bins - 1)]
    raise ValueError(mode)


def build_initial_phases(
    n_refl: int,
    seed_idx: np.ndarray,
    seed_phases: np.ndarray,
    rng: np.random.Generator,
    nonseed_mode: str = "random",
    discrete: str = "none",
    n_bins: int = 4,
) -> np.ndarray:
    """
    Full phase vector: seed reflections from AI, rest random/zero.

    nonseed_mode: ``random`` | ``zero``
    """
    phases = np.zeros(n_refl, dtype=np.float64)
    if nonseed_mode == "random":
        phases[:] = rng.uniform(-np.pi, np.pi, size=n_refl)
    elif nonseed_mode == "zero":
        phases[:] = 0.0
    else:
        raise ValueError(nonseed_mode)
    seed_idx = np.asarray(seed_idx, dtype=int)
    sp = discretize_phases(seed_phases, mode=discrete, n_bins=n_bins)
    phases[seed_idx] = sp
    return phases


def reimpose_seed(
    phases: np.ndarray,
    seed_idx: np.ndarray,
    seed_phases: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """
    Re-apply seed phases after an extension step.

    weight=1: hard fix. weight∈(0,1): circular blend toward seed.
    """
    out = np.asarray(phases, dtype=np.float64).copy()
    seed_idx = np.asarray(seed_idx, dtype=int)
    sp = np.asarray(seed_phases, dtype=np.float64)
    w = float(np.clip(weight, 0.0, 1.0))
    if w >= 1.0 - 1e-12:
        out[seed_idx] = sp
        return out
    if w <= 1e-12:
        return out
    # blend only on seed indices
    cur = out[seed_idx]
    out[seed_idx] = blend_phases(sp, cur, np.full(len(seed_idx), w))
    return out


def phase_extend(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: np.ndarray,
    seed_idx: np.ndarray,
    seed_phases: np.ndarray,
    n_cycles: int = 15,
    seed_weight: float = 1.0,
    seed_weight_final: float = 0.75,
    use_positivity: bool = True,
    solvent_fraction: Optional[float] = None,
    d_min: Optional[float] = None,
    discrete: str = "none",
    n_bins: int = 4,
    full_prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.25,
    dm_ai_weight: float = 0.0,
    dm_ai_every: int = 3,
    low_res_path: bool = False,
    n_atoms_approx: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Classical phase extension with seed re-imposition each cycle.

    Schedule: seed_weight linearly anneals to seed_weight_final over cycles
    (early: hard seed; late: slightly softer). Optional ``full_prior`` softly
    pulls *all* reflections toward the AI phase set each cycle.

    Parameters
    ----------
    dm_ai_weight : float
        If >0, periodically run modified tangent with AI prior (Carrozzini
        eqs. 3–5 style) via ``dm_ai_hybrid_refine``. Typical 0.3–0.7.
    dm_ai_every : int
        Apply DM+AI hybrid every N cycles (default 3).
    low_res_path : bool
        Paper-inspired EDM-friendly schedule for lower resolution / larger
        volumes: more solvent flatten, longer seed anneal, slightly more
        DM hybrid steps. Auto-used when ``ai_phaseed_solve(..., low_res_path=True)``.
    discrete : str
        ``none`` | ``centro`` | ``bins`` — non-centro binning hooks for
        seed re-imposition (see ``discretize_phases``).
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64).copy()
    seed_idx = np.asarray(seed_idx, dtype=int)
    seed_phases = discretize_phases(seed_phases, mode=discrete, n_bins=n_bins)
    full_ai = (
        np.asarray(full_prior, dtype=np.float64)
        if full_prior is not None
        else None
    )

    if d_min is None:
        d_min = float(np.min(d_spacing(hkl, cell)))
    shape = grid_shape_from_resolution(cell, d_min, sampling=3.0)

    # Low-res / large-volume EDM schedule (Carrozzini hybrid protocols)
    sol_frac = solvent_fraction
    n_cyc = int(n_cycles)
    dm_every = int(max(dm_ai_every, 1))
    dm_w = float(dm_ai_weight)
    sw0, sw1 = float(seed_weight), float(seed_weight_final)
    if low_res_path:
        if sol_frac is None:
            sol_frac = 0.35
        n_cyc = max(n_cyc, 20)
        # Hold seed harder longer, then release more gently
        sw0 = max(sw0, 1.0)
        sw1 = min(sw1, 0.55)
        if dm_w <= 0:
            dm_w = 0.45
        dm_every = min(dm_every, 2)
        history_alg = "phase_extend_lowres_edm"
    else:
        history_alg = "phase_extend"

    history: Dict = {
        "R": [],
        "seed_weight": [],
        "algorithm": history_alg,
        "dm_ai_weight": dm_w,
        "low_res_path": bool(low_res_path),
        "dm_ai_steps": [],
    }

    def density_op(rho: np.ndarray) -> np.ndarray:
        out = rho
        if use_positivity:
            out = project_positivity(out)
        if sol_frac is not None and sol_frac > 0:
            out = solvent_flatten(out, solvent_fraction=sol_frac)
        return out

    # N_atoms heuristic for Cochran κ
    if n_atoms_approx is None:
        try:
            vol = float(unit_cell_volume(cell))
            n_atoms_approx = max(vol / (4.0 * 18.0), 10.0)
        except Exception:
            n_atoms_approx = 30.0

    for c in range(n_cyc):
        # anneal seed weight (optionally hold-then-ease for low-res)
        t = 0.0 if n_cyc <= 1 else c / (n_cyc - 1)
        if n_cyc <= 1:
            w = sw1
        elif low_res_path:
            # stay near sw0 for first half, then ease to sw1
            t_eff = 0.0 if t < 0.4 else (t - 0.4) / 0.6
            w = (1 - t_eff) * sw0 + t_eff * sw1
        else:
            w = (1 - t) * sw0 + t * sw1

        phases, rho = fourier_modulus_projection(
            hkl, amp, phases, cell, density_op=density_op, d_min=d_min
        )
        # soft full-map prior from AI (weak reflections)
        if full_ai is not None and prior_weight > 0:
            pw = float(prior_weight) * (1.0 - 0.5 * (c / max(n_cyc - 1, 1)))
            phases = blend_phases(
                full_ai,
                phases,
                np.full(len(phases), pw),
            )
        phases = reimpose_seed(phases, seed_idx, seed_phases, weight=w)
        if discrete != "none":
            phases[seed_idx] = discretize_phases(
                phases[seed_idx], mode=discrete, n_bins=n_bins
            )

        # DM + AI modified tangent (optional hybrid)
        if dm_w > 0 and full_ai is not None and (c % dm_every == 0):
            lam = dm_w * (1.0 - 0.4 * t)
            phases, dm_info = dm_ai_hybrid_refine(
                hkl,
                amp,
                cell,
                phases,
                full_ai,
                ai_weight=lam,
                seed_idx=seed_idx,
                n_atoms_approx=float(n_atoms_approx),
                tangent_iter=6 if low_res_path else 5,
            )
            # re-lock seeds after tangent so AI seed is not washed out
            phases = reimpose_seed(phases, seed_idx, seed_phases, weight=w)
            history["dm_ai_steps"].append({"cycle": c, **dm_info})

        F = density_to_F(project_positivity(rho), hkl, cell)
        R = r_factor_moduli(F, amp)
        history["R"].append(R)
        history["seed_weight"].append(w)
        if verbose and (c % 5 == 0 or c == n_cyc - 1):
            print(f"  phase_extend {c+1}/{n_cyc}  R₊≈{R:.4f}  w_seed={w:.2f}")

    F_final = amp * np.exp(1j * phases)
    rho_final = density_from_structure_factors(
        hkl, F_final, cell, shape=shape, d_min=d_min
    )
    history["final_R"] = history["R"][-1] if history["R"] else None
    history["n_cycles"] = n_cyc
    history["n_seed"] = int(len(seed_idx))
    history["solvent_fraction"] = sol_frac
    return phases, rho_final, history


def ai_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    n_seed: Optional[int] = None,
    seed_fraction: float = 0.25,
    n_extend: int = 15,
    seed_weight: float = 1.0,
    seed_weight_final: float = 0.75,
    use_positivity: bool = True,
    solvent_fraction: Optional[float] = None,
    polish: str = "charge_flipping",
    n_polish: int = 60,
    n_starts: int = 1,
    seed: int = 0,
    d_min: Optional[float] = None,
    discrete: str = "none",
    nonseed_mode: str = "random",
    select_by: str = "E",
    prior_weight: float = 0.30,
    use_full_prior: bool = True,
    use_free_fom_gate: bool = True,
    dm_ai_weight: float = 0.0,
    low_res_path: bool = False,
    assess_seed_quality: bool = True,
    seed_quality_filter: bool = False,
    n_atoms_user: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Full AI-PhaSeed pipeline from an external AI phase vector.

    Parameters
    ----------
    seed_phases : (M,) full-length phase array from AI (aligned with hkl)
    n_seed / seed_fraction : size of strong-reflection seed subset (|E|-ranked)
    n_extend : phase-extension cycles
    prior_weight : soft pull of *all* phases toward AI each cycle (in addition
        to hard/soft reimpose on the strong seed set)
    polish : final polish method for conditional_polish (or ``"none"``)
    n_starts : multistart of non-seed random phases; free-FOM pick
    use_free_fom_gate : if True, final polish only accepted if free FOM improves
    dm_ai_weight : if >0, modified-tangent AI hybrid during extension
        (Carrozzini 2025 eqs. 3–5 style). Typical 0.5. Backward compatible default 0.
    low_res_path : EDM-optimized anneal / solvent schedule for lower res / larger Vol
    assess_seed_quality : run ``predict_seed_quality`` and store in info
    seed_quality_filter : if True and Class 0, emit warning; still runs (does not
        hard-abort — user / CLI may choose fallback)
    n_atoms_user : optional ASU non-H count for seed-quality features

    Returns
    -------
    phases, density, info
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    seed_phases = np.asarray(seed_phases, dtype=np.float64)
    if len(seed_phases) != len(amp):
        raise ValueError("seed_phases length must match amplitudes")

    seed_idx = select_seed_indices(
        hkl, amp, cell, n_seed=n_seed, seed_fraction=seed_fraction, by=select_by
    )
    sp = seed_phases[seed_idx]
    full_prior = seed_phases if use_full_prior else None

    # Auto low-res path when d_min is coarse and volume is large (paper regime)
    low_res = bool(low_res_path)
    if not low_res and d_min is not None:
        try:
            vol = float(unit_cell_volume(cell))
            if d_min >= 1.35 and vol >= 1000.0:
                low_res = True
        except Exception:
            pass

    seed_quality: Optional[Dict] = None
    if assess_seed_quality:
        try:
            from grok_phase_solver.metrics.seed_quality import predict_seed_quality

            seed_quality = predict_seed_quality(
                hkl,
                amp,
                cell,
                seed_phases,
                seed_idx=seed_idx,
                seed_fraction=float(len(seed_idx) / max(len(amp), 1)),
                n_atoms_user=n_atoms_user,
                d_min=d_min,
            )
            if verbose and seed_quality is not None:
                print(
                    f"  seed_quality: Class {seed_quality['predicted_class']} "
                    f"P(success)≈{seed_quality['success_probability']:.2f} "
                    f"est.MPE≈{seed_quality['predicted_mpe_deg']:.0f}°"
                )
                if seed_quality.get("warning"):
                    print(f"  WARNING: {seed_quality['warning']}")
            if seed_quality_filter and seed_quality.get("recommend_fallback"):
                if verbose:
                    print(
                        "  seed_quality_filter: low quality — continuing with "
                        "AI-PhaSeed but recommend partial-φ / ensemble fallback"
                    )
        except Exception as e:
            seed_quality = {"error": str(e), "method": "failed"}

    trials: List[Dict] = []
    best = None  # (composite, phases, rho, info)

    for s in range(max(1, n_starts)):
        rng = np.random.default_rng(seed + s)
        ph0 = build_initial_phases(
            len(amp), seed_idx, sp, rng,
            nonseed_mode=nonseed_mode, discrete=discrete,
        )
        # initialize non-seed with soft AI prior when available
        if use_full_prior and prior_weight > 0:
            ph0 = blend_phases(
                seed_phases, ph0, np.full(len(amp), 0.5 * prior_weight)
            )
            ph0 = reimpose_seed(ph0, seed_idx, sp, weight=1.0)

        # Optional initial DM+AI pass before EDM extension
        if dm_ai_weight > 0 and full_prior is not None:
            ph0, _ = dm_ai_hybrid_refine(
                hkl, amp, cell, ph0, seed_phases,
                ai_weight=float(dm_ai_weight),
                seed_idx=seed_idx,
            )
            ph0 = reimpose_seed(ph0, seed_idx, sp, weight=1.0)

        ph, rho, hist = phase_extend(
            hkl, amp, cell, ph0, seed_idx, sp,
            n_cycles=n_extend,
            seed_weight=seed_weight,
            seed_weight_final=seed_weight_final,
            use_positivity=use_positivity,
            solvent_fraction=solvent_fraction,
            d_min=d_min,
            discrete=discrete,
            full_prior=full_prior,
            prior_weight=prior_weight,
            dm_ai_weight=float(dm_ai_weight),
            low_res_path=low_res,
            n_atoms_approx=float(n_atoms_user) if n_atoms_user else None,
            verbose=verbose and s == 0,
        )
        fom_ext = free_fom(hkl, amp, ph, cell, density=rho)

        polish_info: Dict = {"accepted_polish": False, "polish": polish}
        if polish and polish != "none":
            if use_free_fom_gate:
                ph2, rho2, polish_info = conditional_polish(
                    hkl, amp, cell, ph, polish=polish, n_iter=n_polish,
                    seed=seed + s, d_min=d_min, verbose=verbose and s == 0,
                )
            else:
                # unconditional polish via conditional with min_delta=-1 force path
                # use hybrid-style direct CF
                from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
                from grok_phase_solver.solvers.iterative_retrieval import raar_solve

                if polish == "charge_flipping":
                    ph2, rho2, h = charge_flipping_solve(
                        hkl, amp, cell, n_iter=n_polish, seed=seed + s,
                        d_min=d_min, phase_init=ph,
                    )
                elif polish == "raar":
                    ph2, rho2, h = raar_solve(
                        hkl, amp, cell, n_iter=n_polish, seed=seed + s,
                        d_min=d_min, phase_init=ph,
                    )
                else:
                    ph2, rho2, polish_info = conditional_polish(
                        hkl, amp, cell, ph, polish=polish, n_iter=n_polish,
                        seed=seed + s, d_min=d_min,
                    )
                    h = polish_info.get("history", {})
                f1 = free_fom(hkl, amp, ph2, cell, density=rho2)
                polish_info = {
                    "accepted_polish": True,
                    "polish": polish,
                    "fom_seed": fom_ext,
                    "fom_polished": f1,
                    "fom_final": f1,
                    "history": h,
                }
            ph, rho = ph2, rho2

        fom_final = free_fom(hkl, amp, ph, cell, density=rho)
        trial = {
            "start": s,
            "composite": fom_final["composite"],
            "R_pos": fom_final["R_pos"],
            "fom_extended": fom_ext,
            "fom_final": fom_final,
            "extend_history": hist,
            "polish": polish_info,
        }
        trials.append(trial)
        if verbose:
            print(
                f"  AI-PhaSeed start {s}: composite={fom_final['composite']:.3f} "
                f"R₊={fom_final['R_pos']:.3f} polish_accept="
                f"{polish_info.get('accepted_polish')}"
            )
        if best is None or fom_final["composite"] > best[0]:
            best = (fom_final["composite"], ph, rho, trial)

    assert best is not None
    _, ph, rho, trial = best
    info = {
        "algorithm": "ai_phaseed",
        "n_seed": int(len(seed_idx)),
        "seed_fraction_actual": float(len(seed_idx) / len(amp)),
        "seed_indices": seed_idx.tolist(),
        "n_starts": n_starts,
        "n_extend": n_extend,
        "polish": polish,
        "use_free_fom_gate": use_free_fom_gate,
        "best_trial": trial,
        "all_trials": trials,
        "fom_final": trial["fom_final"],
        "discrete": discrete,
        "select_by": select_by,
        "dm_ai_weight": float(dm_ai_weight),
        "low_res_path": bool(low_res),
        "seed_quality": seed_quality,
    }
    return ph, rho, info


def phai_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_phai_cycles: int = 5,
    n_seed: Optional[int] = None,
    seed_fraction: float = 0.25,
    n_extend: int = 15,
    polish: str = "charge_flipping",
    n_polish: int = 60,
    n_starts: int = 2,
    seed: int = 0,
    d_min: Optional[float] = None,
    discrete: str = "none",
    solvent_fraction: Optional[float] = None,
    prior_weight: float = 0.30,
    use_free_fom_gate: bool = True,
    dm_ai_weight: float = 0.0,
    low_res_path: bool = False,
    assess_seed_quality: bool = True,
    seed_quality_filter: bool = False,
    verbose: bool = False,
    fallback: str = "cf",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    PhAI fair packing → AI-PhaSeed extension → optional gated polish.

    If PhAI is unavailable, fall back to classical CF or pure phase_extend
    from random seeds (``fallback``: ``"cf"`` | ``"random_seed"``).

    Extra kwargs (``dm_ai_weight``, ``low_res_path``, seed-quality flags)
    forward to ``ai_phaseed_solve``; defaults preserve pre-0.4.0 behaviour
    when left at 0 / False.
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)

    common = dict(
        n_seed=n_seed,
        seed_fraction=seed_fraction,
        n_extend=n_extend,
        polish=polish,
        n_polish=n_polish,
        n_starts=n_starts,
        seed=seed,
        d_min=d_min,
        discrete=discrete,
        solvent_fraction=solvent_fraction,
        prior_weight=prior_weight,
        use_free_fom_gate=use_free_fom_gate,
        dm_ai_weight=dm_ai_weight,
        low_res_path=low_res_path,
        assess_seed_quality=assess_seed_quality,
        seed_quality_filter=seed_quality_filter,
        verbose=verbose,
    )

    phai_meta = None
    try:
        from grok_phase_solver.models.phai_fair import run_phai_fair
        from grok_phase_solver.models.phai_runner import phai_available

        if phai_available():
            ph_ai, phai_meta = run_phai_fair(
                hkl, amp, n_cycles=n_phai_cycles, seed=seed
            )
            if verbose:
                print(
                    f"  PhAI fair seed: mapped={phai_meta.get('frac_input_mapped')} "
                    f"n_mapped={phai_meta.get('n_mapped')}"
                )
            # Auto centro discretization if structure looks centrosymmetric
            # (user can override via discrete=)
            ph, rho, info = ai_phaseed_solve(
                hkl, amp, cell, ph_ai, **common,
            )
            info["seed_source"] = "phai_fair"
            info["phai_meta"] = phai_meta
            # also store pure PhAI free FOM for comparison
            rho_ai = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_ai), cell, d_min=d_min
            )
            info["fom_phai_only"] = free_fom(hkl, amp, ph_ai, cell, density=rho_ai)
            return ph, rho, info
    except Exception as e:
        if verbose:
            print(f"  PhAI unavailable ({e}); fallback={fallback}")

    if fallback == "cf":
        from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

        ph, rho, hist = charge_flipping_solve(
            hkl, amp, cell, n_iter=n_polish, seed=seed, d_min=d_min, verbose=verbose
        )
        return ph, rho, {
            "algorithm": "ai_phaseed",
            "seed_source": "cf_fallback",
            "history": hist,
        }

    # random seed phases as pseudo-AI
    rng = np.random.default_rng(seed)
    ph_rand = rng.uniform(-np.pi, np.pi, size=len(amp))
    ph, rho, info = ai_phaseed_solve(
        hkl, amp, cell, ph_rand, **common,
    )
    info["seed_source"] = "random_fallback"
    return ph, rho, info
