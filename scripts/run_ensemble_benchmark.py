#!/usr/bin/env python3
"""
Multistart CF+RAAR ensemble vs single-start CF/RAAR.

Ranks by free-FOM (truth-free) at selection time; reports strict SuccessThresholds
against truth for analysis only.

Writes data/processed/ensemble_benchmark.{json,md}
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
from grok_phase_solver.solvers.ensemble import ensemble_cf_raar
from grok_phase_solver.solvers.iterative_retrieval import raar_solve


def run_case(n_atoms, d_min, seed, method, n_iter=80, n_starts=4):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]
    t0 = time.time()
    if method == "cf":
        ph, rho, hist = charge_flipping_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min
        )
        info = {"final_R": hist.get("final_R"), "algorithm": "cf"}
    elif method == "raar":
        ph, rho, hist = raar_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min
        )
        info = {"final_R": hist.get("final_R"), "algorithm": "raar"}
    elif method == "ensemble_cf_raar":
        ph, rho, info = ensemble_cf_raar(
            hkl, amp, st.cell,
            n_starts=n_starts, n_iter=n_iter, base_seed=seed, d_min=d_min,
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
        "best_method": info.get("best_method"),
        "best_composite": (info.get("best_fom") or {}).get("composite"),
        "n_trials": info.get("n_trials"),
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fewer cases for CI")
    args = p.parse_args()

    if args.quick:
        easy = [(4, 0.9), (6, 1.0)]
        hard = [(12, 1.5)]
        seeds = [0, 1]
        n_starts = 2
        n_iter = 50
    else:
        easy = [(4, 0.9), (6, 1.0), (8, 0.9)]
        hard = [(12, 1.5), (16, 1.5)]
        seeds = [0, 1, 2]
        n_starts = 4
        n_iter = 80

    methods = ["cf", "raar", "ensemble_cf_raar"]
    rows = []
    for region, grid in [("easy", easy), ("hard", hard)]:
        print(f"\n===== {region.upper()} =====")
        for n_atoms, d_min in grid:
            for seed in seeds:
                for method in methods:
                    try:
                        row = run_case(
                            n_atoms, d_min, seed, method,
                            n_iter=n_iter, n_starts=n_starts,
                        )
                        row["region"] = region
                        rows.append(row)
                        flag = "SOLVED" if row["solved"] else "fail"
                        print(
                            f"  {flag:6s} {method:18s} n={n_atoms:2d} d={d_min:.1f} "
                            f"s={seed} CC={row['mapcc_oi']:.2f} peak={row['peak_recovery']:.2f}"
                        )
                    except Exception as e:
                        print(f"  ERROR {method}: {e}")
                        rows.append({
                            "region": region, "n_atoms": n_atoms, "d_min": d_min,
                            "seed": seed, "method": method, "error": str(e), "solved": False,
                        })

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "ensemble_benchmark.json"
    jp.write_text(json.dumps(rows, indent=2, default=float))

    md = [
        "# Multistart ensemble benchmark (CF + RAAR, free-FOM pick)",
        "",
        "Selection uses **truth-free composite FOM**; success rates use strict "
        "SuccessThresholds against synthetic truth.",
        "",
    ]
    for region in ("easy", "hard"):
        md.append(f"## {region.capitalize()} region")
        md.append("")
        md.append("| Method | Solved | Total | Rate | mean mapCC |")
        md.append("|--------|--------|-------|------|------------|")
        for method in methods:
            sub = [
                r for r in rows
                if r.get("region") == region and r.get("method") == method and "error" not in r
            ]
            if not sub:
                continue
            n_ok = sum(1 for r in sub if r["solved"])
            mcc = float(np.mean([r["mapcc_oi"] for r in sub]))
            md.append(
                f"| `{method}` | {n_ok} | {len(sub)} | {n_ok/len(sub):.0%} | {mcc:.3f} |"
            )
        md.append("")

    md.extend([
        "## Notes",
        "",
        "- Ensemble runs multiple CF and RAAR starts; picks highest free-FOM composite.",
        "- Expect higher wall-clock (~ n_starts × 2 methods) and modest success gains.",
        f"- JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp = out / "ensemble_benchmark.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}")


if __name__ == "__main__":
    main()
