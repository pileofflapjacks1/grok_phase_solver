"""
Main structure-solution pipeline for experimental data.

Scientist-facing flow:
  load reflections → choose method → phase → density → peaks → export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.io.experiment import load_experiment, summarize_experiment
from grok_phase_solver.io.hkl import ReflectionTable
from grok_phase_solver.io.ins import ShelxIns
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.pipeline.peaks import DensityPeak, pick_density_peaks
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.direct_methods import direct_methods_solve
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval
from grok_phase_solver.solvers.phase_recycle import phase_recycle


# Methods accepted by gps-solve
KNOWN_METHODS = (
    "auto",
    "charge_flipping",
    "ensemble",
    "raar",
    "phai",
    "phai+cf",
    "phai+cf_cond",
    "phai_phaseed",
    "phai+recycle",
    "hard_p1_phaseed",
    "strong_prior_phaseed",
    "recycle",
    "direct_methods",
    "hio",
    "dual_space",
    "shelxd",
    "shelxd_or_dual",
    "partial_phaseed",
)


@dataclass
class SolveConfig:
    """User options for gps-solve."""

    method: str = "auto"
    d_min: Optional[float] = None
    n_iter: int = 120
    n_recycle: int = 8
    n_extend: int = 12
    n_starts: int = 2
    seed: int = 0
    n_peaks: int = 40
    min_peak_sigma: float = 2.5
    solvent_fraction: Optional[float] = None
    verbose: bool = True
    # Partial-φ seed file (CSV: h,k,l,phase_deg) for method=partial_phaseed
    phase_seed_csv: Optional[str] = None
    seed_fraction: float = 0.30


@dataclass
class SolveResult:
    hkl: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    density: np.ndarray
    cell: np.ndarray
    space_group_hm: Optional[str]
    method: str
    d_min: Optional[float]
    peaks: List[DensityPeak] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    table: Optional[ReflectionTable] = None
    ins: Optional[ShelxIns] = None


def _sg_norm(space_group: Optional[str]) -> str:
    return (space_group or "").replace(" ", "").upper()


def _is_p21c_like(sg: str) -> bool:
    s = _sg_norm(sg)
    return any(x in s for x in ("P21/C", "P121/C1", "P21/C1", "P121/C"))


def _is_p1(sg: str) -> bool:
    s = _sg_norm(sg)
    return s in ("P1", "P 1", "") or s == "P1"


def _phai_ok() -> bool:
    try:
        from grok_phase_solver.models.phai_runner import phai_available

        return bool(phai_available())
    except Exception:
        return False


def _hard_p1_prior_available() -> bool:
    try:
        from grok_phase_solver.models.hard_p1_prior import default_hard_p1_path

        return default_hard_p1_path().exists()
    except Exception:
        return False


def resolve_method(
    method: str,
    space_group: Optional[str],
    data_dmin: float,
    n_refl: int,
) -> Tuple[str, str]:
    """
    Resolve ``auto`` to a concrete method.

    Policy (highest expected impact given repo benchmarks):
    - P2₁/c-like + PhAI weights → ``phai_phaseed`` (AI-PhaSeed + gated polish)
    - P1 + hard-P1 prior weights + low res → ``hard_p1_phaseed``
    - high res (d_min ≤ 1.0) small-ish data → ``ensemble`` multistart CF+RAAR
    - else → ``charge_flipping``
    """
    m = method.lower().strip()
    if m != "auto":
        if m not in KNOWN_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {KNOWN_METHODS}")
        return m, "user-selected"

    sg = space_group or ""
    phai = _phai_ok()
    hp1 = _hard_p1_prior_available()

    if phai and _is_p21c_like(sg):
        return "phai_phaseed", "auto: P21/c-like + PhAI → AI-PhaSeed"
    # Prefer graph strong prior when available on hard P1-like data
    try:
        from grok_phase_solver.models.strong_prior import default_strong_prior_path

        strong_ok = default_strong_prior_path().exists()
    except Exception:
        strong_ok = False
    if strong_ok and (_is_p1(sg) or not sg) and data_dmin >= 1.3:
        return "strong_prior_phaseed", "auto: P1-ish hard-res + GraphPhaseNet prior"
    if hp1 and (_is_p1(sg) or not sg) and data_dmin >= 1.3:
        return "hard_p1_phaseed", "auto: P1-ish hard-res + hard_p1 prior"
    if data_dmin <= 1.0 and n_refl >= 80:
        return "ensemble", "auto: high-res multistart CF+RAAR free-FOM"
    if phai and data_dmin <= 1.2:
        return "phai+cf_cond", "auto: PhAI + free-FOM–gated CF"
    return "charge_flipping", "auto: classical charge flipping default"


def _filter_dmin(table: ReflectionTable, d_min: Optional[float]) -> ReflectionTable:
    if d_min is None:
        return table
    return table.filter_resolution(d_min=d_min)


def _run_phasing(
    method: str,
    hkl: np.ndarray,
    amp: np.ndarray,
    cell_arr: np.ndarray,
    d_use: float,
    cfg: SolveConfig,
    centro: bool,
    warnings: List[str],
) -> Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Dispatch phasing; may rewrite method on fallback. Returns (method, phases, density, history)."""
    history: Dict[str, Any] = {}

    if method == "charge_flipping":
        phases, density, history = charge_flipping_solve(
            hkl, amp, cell_arr, n_iter=cfg.n_iter, seed=cfg.seed, d_min=d_use,
            centrosymmetric=centro, verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "ensemble":
        from grok_phase_solver.solvers.ensemble import ensemble_cf_raar

        phases, density, history = ensemble_cf_raar(
            hkl, amp, cell_arr,
            n_starts=max(2, cfg.n_starts),
            n_iter=cfg.n_iter,
            base_seed=cfg.seed,
            d_min=d_use,
            verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "raar":
        from grok_phase_solver.solvers.iterative_retrieval import raar_solve

        phases, density, history = raar_solve(
            hkl, amp, cell_arr, n_iter=cfg.n_iter, seed=cfg.seed, d_min=d_use,
            verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "recycle":
        phases, density, history = phase_recycle(
            hkl, amp, cell_arr, n_cycles=cfg.n_recycle, seed=cfg.seed, d_min=d_use,
            use_positivity=True, solvent_fraction=cfg.solvent_fraction, verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "direct_methods":
        dm = direct_methods_solve(
            hkl, amp, cell_arr, n_atoms_approx=max(20, cfg.n_peaks),
            n_trials=max(20, cfg.n_iter // 4), seed=cfg.seed, verbose=cfg.verbose,
        )
        phases = dm.phases_full
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases), cell_arr, d_min=d_use
        )
        history = dm.history
        return method, phases, density, history

    if method == "dual_space":
        from grok_phase_solver.solvers.dual_space import dual_space_solve

        phases, density, history = dual_space_solve(
            hkl, amp, cell_arr,
            n_atoms=max(8, min(cfg.n_peaks, 30)),
            n_cycles=max(20, cfg.n_iter // 2),
            n_starts=max(4, cfg.n_starts * 2),
            seed=cfg.seed,
            d_min=d_use,
            polish_cf_iters=min(30, cfg.n_iter // 3),
            verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "shelxd":
        from grok_phase_solver.solvers.shelxd_runner import shelxd_solve

        phases, density, history = shelxd_solve(
            hkl, amp, cell_arr,
            n_atoms=max(8, min(cfg.n_peaks, 40)),
            n_try=max(20, cfg.n_starts * 25),
            seed=max(1, cfg.seed),
            d_min=d_use,
            verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "shelxd_or_dual":
        from grok_phase_solver.solvers.shelxd_runner import shelxd_or_dual_space

        phases, density, history = shelxd_or_dual_space(
            hkl, amp, cell_arr,
            n_atoms=max(8, min(cfg.n_peaks, 40)),
            n_try=max(20, cfg.n_starts * 25),
            seed=cfg.seed,
            d_min=d_use,
            dual_space_starts=max(4, cfg.n_starts * 2),
            dual_space_cycles=max(20, cfg.n_iter // 2),
            verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "hio":
        from grok_phase_solver.solvers.hio import hio_solve

        phases, density, history = hio_solve(
            hkl, amp, cell_arr, n_iter=cfg.n_iter, seed=cfg.seed, d_min=d_use,
            verbose=cfg.verbose,
        )
        return method, phases, density, history

    if method == "hard_p1_phaseed":
        try:
            from grok_phase_solver.models.hard_p1_prior import hard_p1_phaseed_solve

            phases, density, history = hard_p1_phaseed_solve(
                hkl, amp, cell_arr,
                n_extend=cfg.n_extend, polish="charge_flipping", n_polish=cfg.n_iter,
                n_starts=cfg.n_starts, seed=cfg.seed, d_min=d_use, verbose=cfg.verbose,
            )
            return method, phases, density, history
        except Exception as e:
            warnings.append(f"hard_p1_phaseed failed ({e}); falling back to charge_flipping")
            return _run_phasing(
                "charge_flipping", hkl, amp, cell_arr, d_use, cfg, centro, warnings
            )

    if method == "strong_prior_phaseed":
        try:
            from grok_phase_solver.models.strong_prior import strong_prior_phaseed_solve

            phases, density, history = strong_prior_phaseed_solve(
                hkl, amp, cell_arr,
                n_extend=cfg.n_extend, polish="charge_flipping", n_polish=cfg.n_iter,
                n_starts=cfg.n_starts, seed=cfg.seed, d_min=d_use, verbose=cfg.verbose,
            )
            return method, phases, density, history
        except Exception as e:
            warnings.append(f"strong_prior_phaseed failed ({e}); falling back to charge_flipping")
            return _run_phasing(
                "charge_flipping", hkl, amp, cell_arr, d_use, cfg, centro, warnings
            )

    if method == "partial_phaseed":
        from grok_phase_solver.solvers.partial_seed import (
            load_phase_seed_csv,
            partial_phaseed_solve,
        )

        if not cfg.phase_seed_csv:
            raise ValueError(
                "method=partial_phaseed requires SolveConfig.phase_seed_csv "
                "(CSV with h,k,l,phase_deg). See solvers/partial_seed.py."
            )
        seed_ph, mask, meta = load_phase_seed_csv(cfg.phase_seed_csv, hkl)
        if mask.sum() < 5:
            warnings.append(
                f"phase seed CSV mapped only {int(mask.sum())} reflections; "
                "results may be poor"
            )
        phases, density, history = partial_phaseed_solve(
            hkl, amp, cell_arr, seed_ph,
            mask=mask if mask.sum() >= 5 else None,
            seed_fraction=cfg.seed_fraction,
            n_extend=cfg.n_extend,
            polish="charge_flipping",
            n_polish=cfg.n_iter,
            n_starts=cfg.n_starts,
            seed=cfg.seed,
            d_min=d_use,
            verbose=cfg.verbose,
            meta=meta,
        )
        return method, phases, density, history

    if method == "phai_phaseed":
        try:
            from grok_phase_solver.solvers.ai_phaseed import phai_phaseed_solve

            phases, density, history = phai_phaseed_solve(
                hkl, amp, cell_arr,
                n_phai_cycles=cfg.n_recycle,
                n_extend=cfg.n_extend,
                polish="charge_flipping",
                n_polish=cfg.n_iter,
                n_starts=cfg.n_starts,
                seed=cfg.seed,
                d_min=d_use,
                discrete="centro" if centro else "none",
                verbose=cfg.verbose,
            )
            if history.get("seed_source") == "cf_fallback":
                warnings.append("PhAI unavailable inside phai_phaseed; used CF fallback")
            return method, phases, density, history
        except Exception as e:
            warnings.append(f"phai_phaseed failed ({e}); falling back to charge_flipping")
            return _run_phasing(
                "charge_flipping", hkl, amp, cell_arr, d_use, cfg, centro, warnings
            )

    if method == "phai+cf_cond":
        try:
            from grok_phase_solver.solvers.conditional_hybrid import phai_conditional_solve

            phases, density, history = phai_conditional_solve(
                hkl, amp, cell_arr, polish="charge_flipping", n_iter=cfg.n_iter,
                n_phai_cycles=cfg.n_recycle, seed=cfg.seed, d_min=d_use, verbose=cfg.verbose,
            )
            return method, phases, density, history
        except Exception as e:
            warnings.append(f"phai+cf_cond failed ({e}); falling back to charge_flipping")
            return _run_phasing(
                "charge_flipping", hkl, amp, cell_arr, d_use, cfg, centro, warnings
            )

    if method in ("phai", "phai+cf", "phai+recycle"):
        try:
            from grok_phase_solver.models.phai_fair import run_phai_fair
            from grok_phase_solver.models.phai_runner import phai_available

            if not phai_available():
                warnings.append("PhAI requested but weights unavailable; falling back to charge_flipping")
                return _run_phasing(
                    "charge_flipping", hkl, amp, cell_arr, d_use, cfg, centro, warnings
                )
            phases, info = run_phai_fair(
                hkl, amp, n_cycles=cfg.n_recycle, random_init=True, seed=cfg.seed
            )
            history = {"phai": info}
            if method == "phai":
                density = density_from_structure_factors(
                    hkl, amp * np.exp(1j * phases), cell_arr, d_min=d_use
                )
            elif method == "phai+cf":
                phases, density, h2 = hybrid_phase_retrieval(
                    hkl, amp, cell_arr, phases, polish="charge_flipping",
                    n_iter=cfg.n_iter, seed=cfg.seed, verbose=cfg.verbose,
                )
                history.update(h2)
            else:
                phases, density, h2 = phase_recycle(
                    hkl, amp, cell_arr, n_cycles=cfg.n_recycle, phase_init=phases,
                    seed=cfg.seed, d_min=d_use, verbose=cfg.verbose,
                )
                history.update(h2)
            return method, phases, density, history
        except Exception as e:
            warnings.append(f"PhAI failed ({e}); falling back to charge_flipping")
            return _run_phasing(
                "charge_flipping", hkl, amp, cell_arr, d_use, cfg, centro, warnings
            )

    raise ValueError(f"Unknown method '{method}'. Choose from {KNOWN_METHODS}")


def solve_structure(
    hkl_path: str,
    ins_path: Optional[str] = None,
    cell: Optional[str] = None,
    space_group: Optional[str] = None,
    wavelength: Optional[float] = None,
    config: Optional[SolveConfig] = None,
) -> SolveResult:
    """
    Phase experimental amplitudes and build density + peak list.
    """
    cfg = config or SolveConfig()
    table, ins = load_experiment(
        hkl_path, ins=ins_path, cell=cell, space_group=space_group, wavelength=wavelength
    )
    warnings: List[str] = []

    if cfg.verbose:
        print("=== Loaded experiment ===")
        print(summarize_experiment(table, ins))

    table = _filter_dmin(table, cfg.d_min)
    if len(table) < 20:
        raise ValueError(f"Too few reflections after filtering: {len(table)}")

    amp = table.amplitudes.copy()
    bad = ~np.isfinite(amp) | (amp < 0)
    if np.any(bad):
        warnings.append(f"Removed {int(bad.sum())} non-finite/negative amplitudes")
        table = table.subset(~bad)
        amp = table.amplitudes.copy()

    hkl = table.hkl
    cell_arr = table.cell
    assert cell_arr is not None
    sg = table.space_group_hm

    d_all = d_spacing(hkl, cell_arr)
    data_dmin = float(d_all.min())
    d_use = cfg.d_min if cfg.d_min is not None else data_dmin

    if data_dmin > 1.5:
        warnings.append(
            f"Data d_min ≈ {data_dmin:.2f} Å is relatively low for classical ab initio; "
            "expect partial maps. Prefer PhAI/AI-PhaSeed (P21/c) or experimental phasing / MR."
        )
    if len(table) < 100:
        warnings.append(f"Only {len(table)} reflections — solution may be unreliable.")

    method, auto_reason = resolve_method(cfg.method, sg, data_dmin, len(table))
    if cfg.verbose:
        print(f"\n=== Phasing with method: {method} ===")
        if cfg.method.lower() == "auto":
            print(f"    ({auto_reason})")

    diagnostics: Dict[str, Any] = {
        "n_reflections": len(table),
        "data_dmin": data_dmin,
        "requested_dmin": cfg.d_min,
        "method_requested": cfg.method,
        "method_used": method,
        "auto_reason": auto_reason,
    }

    centro = False
    if sg:
        try:
            import gemmi

            centro = gemmi.SpaceGroup(sg).is_centrosymmetric()
        except Exception:
            sgu = sg.replace(" ", "").upper()
            centro = "P21/C" in sgu or "P-1" in sgu or "P21/C" in sgu

    method, phases, density, history = _run_phasing(
        method, hkl, amp, cell_arr, d_use, cfg, centro, warnings
    )
    diagnostics["method_used"] = method

    if history.get("R"):
        diagnostics["final_R"] = history["R"][-1]
    if history.get("final_R") is not None:
        diagnostics["final_R"] = history.get("final_R")
    if history.get("best_fom") is not None:
        diagnostics["dm_fom"] = history.get("best_fom")
    if history.get("best_method"):
        diagnostics["ensemble_pick"] = history.get("best_method")
    if history.get("seed_source"):
        diagnostics["seed_source"] = history.get("seed_source")
    if history.get("accepted_polish") is not None:
        diagnostics["accepted_polish"] = history.get("accepted_polish")
    if history.get("best_trial"):
        bt = history["best_trial"]
        if isinstance(bt, dict) and bt.get("polish"):
            diagnostics["accepted_polish"] = bt["polish"].get("accepted_polish")
    # free FOM of final map (truth-free)
    try:
        fom = free_fom(hkl, amp, phases, cell_arr, density=density)
        diagnostics["free_fom_composite"] = fom["composite"]
        diagnostics["free_fom_R_pos"] = fom["R_pos"]
        diagnostics["free_fom_version"] = fom.get("fom_version")
    except Exception as e:
        warnings.append(f"free FOM failed: {e}")

    # Optional solvent DM polish after CF-family methods
    if cfg.solvent_fraction is not None and method in (
        "charge_flipping", "phai+cf", "hio", "ensemble", "phai_phaseed", "phai+cf_cond",
    ):
        if cfg.verbose:
            print(f"Density modification polish (solvent_fraction={cfg.solvent_fraction})")
        from grok_phase_solver.solvers.density_modification import density_modification_cycle

        phases, density, dm_hist = density_modification_cycle(
            hkl, amp, phases, cell_arr, n_iter=max(5, cfg.n_recycle),
            solvent_fraction=cfg.solvent_fraction, d_min=d_use, verbose=cfg.verbose,
        )
        diagnostics["dm_R"] = (dm_hist.get("R") or [None])[-1]

    peaks = pick_density_peaks(
        density, n_peaks=cfg.n_peaks, min_sigma=cfg.min_peak_sigma
    )
    diagnostics["n_peaks"] = len(peaks)
    diagnostics["map_sigma"] = float(density.std())
    diagnostics["map_max"] = float(density.max())
    diagnostics["frac_negative"] = float((density < 0).mean())

    if cfg.verbose:
        print(f"\nPeaks found (σ>{cfg.min_peak_sigma}): {len(peaks)}")
        for p in peaks[:8]:
            print(
                f"  #{p.rank:02d}  ({p.fract[0]:.3f}, {p.fract[1]:.3f}, {p.fract[2]:.3f})  "
                f"{p.height_sigma:.1f} σ"
            )
        if "free_fom_composite" in diagnostics:
            print(
                f"Free FOM composite={diagnostics['free_fom_composite']:.3f} "
                f"R₊={diagnostics.get('free_fom_R_pos', float('nan')):.3f}"
            )
        for w in warnings:
            print(f"WARNING: {w}")

    return SolveResult(
        hkl=hkl,
        amplitudes=amp,
        phases=phases,
        density=density,
        cell=cell_arr,
        space_group_hm=sg,
        method=method,
        d_min=d_use,
        peaks=peaks,
        diagnostics=diagnostics,
        warnings=warnings,
        table=table,
        ins=ins,
    )
