#!/usr/bin/env python3
"""
Frontier benchmark: new algorithms vs CF on hard + easy synthetic cells.

Hard region (from solvability diagram): n≥12, d_min≥1.5
Easy region: n≤8, d_min≤1.0

Methods: CF, RAAR, DiffMap, DM(κ), recycle, PhAI-cond (if weights)

Strict success: SuccessThresholds
Writes data/processed/frontier_benchmark.{json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.direct_methods import direct_methods_solve
from grok_phase_solver.solvers.iterative_retrieval import (
    difference_map_solve,
    raar_solve,
)
from grok_phase_solver.solvers.phase_recycle import phase_recycle
from grok_phase_solver.solvers.conditional_hybrid import phai_conditional_solve
from grok_phase_solver.models.phai_runner import phai_available


def run_case(n_atoms, d_min, seed, method, n_iter=100):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]
    t0 = time.time()
    if method == "cf":
        ph, rho, hist = charge_flipping_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min, verbose=False
        )
    elif method == "raar":
        ph, rho, hist = raar_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min, verbose=False
        )
    elif method == "diffmap":
        ph, rho, hist = difference_map_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min, verbose=False
        )
    elif method == "recycle":
        ph, rho, hist = phase_recycle(
            hkl, amp, st.cell, n_cycles=max(5, n_iter // 10), seed=seed, d_min=d_min
        )
    elif method == "dm_kappa":
        dm = direct_methods_solve(
            hkl, amp, st.cell, n_atoms_approx=n_atoms, n_trials=40, seed=seed, verbose=False
        )
        ph = dm.phases_full
        from grok_phase_solver.physics.density import density_from_structure_factors

        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), st.cell, d_min=d_min
        )
        hist = dm.history
    elif method == "phai_cond":
        ph, rho, hist = phai_conditional_solve(
            hkl, amp, st.cell, polish="raar", n_iter=n_iter, seed=seed, d_min=d_min
        )
    else:
        raise ValueError(method)

    rep = evaluate_success(
        hkl, amp, ph, ph_true, st.cell, data["fracs"], density=rho,
        elements=data["elements"], thresholds=SuccessThresholds(),
    )
    return {
        "n_atoms": n_atoms,
        "d_min": d_min,
        "seed": seed,
        "method": method,
        "mapcc_oi": rep.mapcc_oi,
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": rep.solved,
        "seconds": time.time() - t0,
        "n_refl": len(amp),
    }


def main():
    easy = [(4, 0.9), (6, 1.0), (8, 0.9)]
    hard = [(12, 1.5), (12, 2.0), (16, 1.5), (20, 1.5)]
    seeds = [0, 1, 2]
    methods = ["cf", "raar", "diffmap", "recycle", "dm_kappa"]
    if phai_available():
        methods.append("phai_cond")

    rows = []
    cases = [("easy", easy), ("hard", hard)]
    for region, grid in cases:
        print(f"\n===== {region.upper()} REGION =====")
        for n_atoms, d_min in grid:
            for seed in seeds:
                for method in methods:
                    try:
                        row = run_case(n_atoms, d_min, seed, method)
                        row["region"] = region
                        rows.append(row)
                        flag = "SOLVED" if row["solved"] else "fail"
                        print(
                            f"  {flag:6s} {method:10s} n={n_atoms:2d} d={d_min:.1f} "
                            f"s={seed} CC={row['mapcc_oi']:.2f} peak={row['peak_recovery']:.2f} "
                            f"R1={row['r1']:.2f}"
                        )
                    except Exception as e:
                        print(f"  ERROR {method} n={n_atoms} d={d_min}: {e}")
                        rows.append({
                            "region": region, "n_atoms": n_atoms, "d_min": d_min,
                            "seed": seed, "method": method, "error": str(e), "solved": False,
                        })

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "frontier_benchmark.json"
    jp.write_text(json.dumps(rows, indent=2, default=float))

    md = [
        "# Frontier algorithm benchmark",
        "",
        "Compares **RAAR**, **Difference Map**, κ-weighted DM, recycle, CF "
        "(+ PhAI conditional if available) under **strict SuccessThresholds**.",
        "",
        "## Easy region (n≤8, d_min≤1.0) success rates",
        "",
        "| Method | Solved | Total | Rate | mean mapCC |",
        "|--------|--------|-------|------|------------|",
    ]
    for method in methods:
        sub = [r for r in rows if r.get("region") == "easy" and r.get("method") == method and "error" not in r]
        if not sub:
            continue
        n_ok = sum(1 for r in sub if r["solved"])
        mcc = np.mean([r["mapcc_oi"] for r in sub])
        md.append(f"| `{method}` | {n_ok} | {len(sub)} | {n_ok/len(sub):.0%} | {mcc:.3f} |")

    md.extend([
        "",
        "## Hard region (n≥12, d_min≥1.5) success rates",
        "",
        "| Method | Solved | Total | Rate | mean mapCC |",
        "|--------|--------|-------|------|------------|",
    ])
    for method in methods:
        sub = [r for r in rows if r.get("region") == "hard" and r.get("method") == method and "error" not in r]
        if not sub:
            continue
        n_ok = sum(1 for r in sub if r["solved"])
        mcc = np.mean([r["mapcc_oi"] for r in sub])
        md.append(f"| `{method}` | {n_ok} | {len(sub)} | {n_ok/len(sub):.0%} | {mcc:.3f} |")

    md.extend([
        "",
        "## Notes",
        "",
        "- Easy region should remain CF-competitive (no regressions).",
        "- Hard region is where new methods must show gains vs CF.",
        "- PhAI conditional uses fair packing + free-FOM gate for polish.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp = out / "frontier_benchmark.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}")


if __name__ == "__main__":
    main()
