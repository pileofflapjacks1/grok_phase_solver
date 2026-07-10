"""
Main structure-solution pipeline for experimental data.

Scientist-facing flow:
  load reflections → choose method → phase → density → peaks → export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from grok_phase_solver.io.experiment import load_experiment, summarize_experiment
from grok_phase_solver.io.hkl import ReflectionTable
from grok_phase_solver.io.ins import ShelxIns
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.pipeline.peaks import DensityPeak, pick_density_peaks
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.direct_methods import direct_methods_solve
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval
from grok_phase_solver.solvers.phase_recycle import phase_recycle


@dataclass
class SolveConfig:
    """User options for gps-solve."""

    method: str = "auto"  # auto|charge_flipping|phai|phai+cf|recycle|direct_methods|hio
    d_min: Optional[float] = None  # None = use data limit (optional cutoff)
    n_iter: int = 120
    n_recycle: int = 8
    seed: int = 0
    n_peaks: int = 40
    min_peak_sigma: float = 2.5
    solvent_fraction: Optional[float] = None  # for DM polish
    verbose: bool = True


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


def _resolve_method(method: str, space_group: Optional[str]) -> str:
    m = method.lower().strip()
    if m != "auto":
        return m
    # Prefer PhAI for P21/c-like when available
    sg = (space_group or "").replace(" ", "").upper()
    try:
        from grok_phase_solver.models.phai_runner import phai_available

        phai_ok = phai_available()
    except Exception:
        phai_ok = False
    if phai_ok and ("P21/C" in sg or "P121/C1" in sg or "P1 21/C 1" in sg.replace(" ", "")):
        return "phai+cf"
    return "charge_flipping"


def _filter_dmin(table: ReflectionTable, d_min: Optional[float]) -> ReflectionTable:
    if d_min is None:
        return table
    return table.filter_resolution(d_min=d_min)


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
    # Clean negatives / NaNs
    bad = ~np.isfinite(amp) | (amp < 0)
    if np.any(bad):
        warnings.append(f"Removed {int(bad.sum())} non-finite/negative amplitudes")
        table = table.subset(~bad)
        amp = table.amplitudes.copy()

    hkl = table.hkl
    cell_arr = table.cell
    assert cell_arr is not None
    sg = table.space_group_hm

    # Data resolution stats
    d_all = d_spacing(hkl, cell_arr)
    data_dmin = float(d_all.min())
    d_use = cfg.d_min if cfg.d_min is not None else data_dmin

    if data_dmin > 1.5:
        warnings.append(
            f"Data d_min ≈ {data_dmin:.2f} Å is relatively low for classical ab initio; "
            "expect partial maps. Consider PhAI (P21/c) or experimental phasing / MR."
        )
    if len(table) < 100:
        warnings.append(f"Only {len(table)} reflections — solution may be unreliable.")

    method = _resolve_method(cfg.method, sg)
    if cfg.verbose:
        print(f"\n=== Phasing with method: {method} ===")

    diagnostics: Dict[str, Any] = {
        "n_reflections": len(table),
        "data_dmin": data_dmin,
        "requested_dmin": cfg.d_min,
        "method_requested": cfg.method,
        "method_used": method,
    }

    phases: np.ndarray
    density: np.ndarray
    history: Dict = {}

    centro = False
    if sg:
        try:
            import gemmi

            centro = gemmi.SpaceGroup(sg).is_centrosymmetric()
        except Exception:
            centro = "P21/C" in sg.replace(" ", "").upper() or "P-1" in sg.replace(" ", "").upper()

    if method == "charge_flipping":
        phases, density, history = charge_flipping_solve(
            hkl,
            amp,
            cell_arr,
            n_iter=cfg.n_iter,
            seed=cfg.seed,
            d_min=d_use,
            centrosymmetric=centro,
            verbose=cfg.verbose,
        )
        if history.get("R"):
            diagnostics["final_R"] = history["R"][-1]

    elif method == "recycle":
        phases, density, history = phase_recycle(
            hkl,
            amp,
            cell_arr,
            n_cycles=cfg.n_recycle,
            seed=cfg.seed,
            d_min=d_use,
            use_positivity=True,
            solvent_fraction=cfg.solvent_fraction,
            verbose=cfg.verbose,
        )
        diagnostics["final_R"] = history.get("final_R")

    elif method == "direct_methods":
        dm = direct_methods_solve(
            hkl,
            amp,
            cell_arr,
            n_atoms_approx=max(20, cfg.n_peaks),
            n_trials=max(20, cfg.n_iter // 4),
            seed=cfg.seed,
            verbose=cfg.verbose,
        )
        phases = dm.phases_full
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases), cell_arr, d_min=d_use
        )
        diagnostics["dm_fom"] = dm.history.get("best_fom")
        diagnostics["dm_n_strong"] = dm.history.get("n_strong")

    elif method == "hio":
        from grok_phase_solver.solvers.hio import hio_solve

        phases, density, history = hio_solve(
            hkl, amp, cell_arr, n_iter=cfg.n_iter, seed=cfg.seed, d_min=d_use, verbose=cfg.verbose
        )
        if history.get("R"):
            diagnostics["final_R"] = history["R"][-1]

    elif method in ("phai", "phai+cf", "phai+recycle"):
        try:
            from grok_phase_solver.models.phai_runner import PhAIRunner, phai_available

            if not phai_available():
                warnings.append("PhAI requested but weights unavailable; falling back to charge_flipping")
                method = "charge_flipping"
                phases, density, history = charge_flipping_solve(
                    hkl, amp, cell_arr, n_iter=cfg.n_iter, seed=cfg.seed, d_min=d_use,
                    centrosymmetric=centro, verbose=cfg.verbose,
                )
            else:
                runner = PhAIRunner(device="cpu")
                phases, info = runner.predict(
                    hkl, amp, n_cycles=cfg.n_recycle, random_init=True, seed=cfg.seed
                )
                diagnostics["phai"] = info
                if method == "phai":
                    density = density_from_structure_factors(
                        hkl, amp * np.exp(1j * phases), cell_arr, d_min=d_use
                    )
                elif method == "phai+cf":
                    phases, density, history = hybrid_phase_retrieval(
                        hkl, amp, cell_arr, phases, polish="charge_flipping",
                        n_iter=cfg.n_iter, seed=cfg.seed, verbose=cfg.verbose,
                    )
                else:
                    phases, density, history = phase_recycle(
                        hkl, amp, cell_arr, n_cycles=cfg.n_recycle, phase_init=phases,
                        seed=cfg.seed, d_min=d_use, verbose=cfg.verbose,
                    )
        except Exception as e:
            warnings.append(f"PhAI failed ({e}); falling back to charge_flipping")
            method = "charge_flipping"
            phases, density, history = charge_flipping_solve(
                hkl, amp, cell_arr, n_iter=cfg.n_iter, seed=cfg.seed, d_min=d_use,
                centrosymmetric=centro, verbose=cfg.verbose,
            )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose auto|charge_flipping|phai|phai+cf|"
            "phai+recycle|recycle|direct_methods|hio"
        )

    # Optional solvent DM polish after CF
    if cfg.solvent_fraction is not None and method in ("charge_flipping", "phai+cf", "hio"):
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
