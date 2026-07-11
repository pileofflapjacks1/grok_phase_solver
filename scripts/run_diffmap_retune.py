#!/usr/bin/env python3
"""
Difference Map hyperparameter retune: β, real-space projector, charge-flip δσ.

Ranks candidates by free-FOM (truth-free). Reports strict success vs truth for
analysis. Writes data/processed/diffmap_retune.{json,md}
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
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.iterative_retrieval import (
    difference_map_solve,
    retune_difference_map,
)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        cases = [(6, 1.0, 0), (12, 1.5, 0)]
        beta_grid = (0.7, 1.0)
        real_proj_options = ("positivity", "charge_flip")
        delta_sigma_grid = (0.5, 1.0)
        seeds = (0,)
        n_iter = 40
    else:
        cases = [(6, 1.0, 0), (8, 0.9, 1), (12, 1.5, 0), (12, 1.5, 1), (16, 1.5, 0)]
        beta_grid = (0.5, 0.7, 1.0, 1.2)
        real_proj_options = ("positivity", "charge_flip")
        delta_sigma_grid = (0.0, 0.5, 1.0)
        seeds = (0, 1)
        n_iter = 80

    all_rows = []
    param_scores = {}  # key -> list of composites / solved

    for n_atoms, d_min, seed in cases:
        st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]
        print(f"\n=== n={n_atoms} d={d_min} seed={seed} Nrefl={len(amp)} ===")

        t0 = time.time()
        ret = retune_difference_map(
            hkl, amp, st.cell,
            beta_grid=beta_grid,
            real_proj_options=real_proj_options,
            delta_sigma_grid=delta_sigma_grid,
            n_iter=n_iter,
            seeds=seeds,
            d_min=d_min,
            verbose=True,
        )
        bp = ret["best_params"]
        # Evaluate best free-FOM pick with truth
        ph = ret["phases"]
        rho = ret["density"]
        rep = evaluate_success(
            hkl, amp, ph, ph_true, st.cell, data["fracs"], density=rho,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )
        # CF baseline
        ph_cf, rho_cf, _ = charge_flipping_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min
        )
        rep_cf = evaluate_success(
            hkl, amp, ph_cf, ph_true, st.cell, data["fracs"], density=rho_cf,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )

        # Also evaluate default DiffMap (β=1, positivity)
        ph_d, rho_d, _ = difference_map_solve(
            hkl, amp, st.cell, n_iter=n_iter, beta=1.0, real_proj="positivity",
            seed=seed, d_min=d_min,
        )
        rep_d = evaluate_success(
            hkl, amp, ph_d, ph_true, st.cell, data["fracs"], density=rho_d,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )

        row = {
            "n_atoms": n_atoms,
            "d_min": d_min,
            "seed": seed,
            "best_params": bp,
            "best_composite": ret["best_fom"]["composite"],
            "retuned_mapcc": rep.mapcc_oi,
            "retuned_peak": rep.peak_recovery,
            "retuned_r1": rep.r1,
            "retuned_solved": rep.solved,
            "default_dm_mapcc": rep_d.mapcc_oi,
            "default_dm_solved": rep_d.solved,
            "cf_mapcc": rep_cf.mapcc_oi,
            "cf_solved": rep_cf.solved,
            "seconds": time.time() - t0,
            "grid_size": len(ret["grid_results"]),
        }
        all_rows.append(row)
        print(
            f"  BEST free-FOM: β={bp['beta']} Ps={bp['real_proj']} "
            f"δσ={bp['delta_sigma']} → CC={rep.mapcc_oi:.3f} solved={rep.solved}"
        )
        print(f"  default DM CC={rep_d.mapcc_oi:.3f}  CF CC={rep_cf.mapcc_oi:.3f}")

        # Aggregate grid by params (mean composite over seeds for this structure)
        for g in ret["grid_results"]:
            key = (g["beta"], g["real_proj"], g["delta_sigma"])
            param_scores.setdefault(key, []).append(g["composite"])

    # Global ranking of parameter combinations
    ranking = []
    for key, comps in param_scores.items():
        ranking.append({
            "beta": key[0],
            "real_proj": key[1],
            "delta_sigma": key[2],
            "mean_composite": float(np.mean(comps)),
            "n": len(comps),
        })
    ranking.sort(key=lambda x: -x["mean_composite"])

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"cases": all_rows, "param_ranking": ranking}
    jp = out / "diffmap_retune.json"
    jp.write_text(json.dumps(payload, indent=2, default=float))

    md = [
        "# Difference Map retune (β, charge-flip P_S, δσ)",
        "",
        "Grid search ranked by **truth-free free-FOM composite**. "
        "Truth metrics reported for diagnostics only.",
        "",
        "## Per-structure best (free-FOM pick)",
        "",
        "| n | d_min | best β | P_S | δσ | retuned CC | default DM CC | CF CC | retuned solved |",
        "|---|-------|--------|-----|----|------------|---------------|-------|----------------|",
    ]
    for r in all_rows:
        bp = r["best_params"]
        md.append(
            f"| {r['n_atoms']} | {r['d_min']} | {bp['beta']} | {bp['real_proj']} | "
            f"{bp['delta_sigma']} | {r['retuned_mapcc']:.3f} | {r['default_dm_mapcc']:.3f} | "
            f"{r['cf_mapcc']:.3f} | {r['retuned_solved']} |"
        )

    md.extend([
        "",
        "## Global parameter ranking (mean free-FOM composite)",
        "",
        "| Rank | β | P_S | δσ | mean composite | n |",
        "|------|---|-----|----|----------------|---|",
    ])
    for i, r in enumerate(ranking[:12], 1):
        md.append(
            f"| {i} | {r['beta']} | {r['real_proj']} | {r['delta_sigma']} | "
            f"{r['mean_composite']:.3f} | {r['n']} |"
        )

    md.extend([
        "",
        "## Recommended defaults (from this search)",
        "",
    ])
    if ranking:
        top = ranking[0]
        md.append(
            f"- **β** = {top['beta']}, **real_proj** = `{top['real_proj']}`, "
            f"**delta_sigma** = {top['delta_sigma']} "
            f"(mean composite {top['mean_composite']:.3f})"
        )
    md.extend([
        "",
        "- Still multistart + free-FOM for production; retune is case-dependent.",
        f"- JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp = out / "diffmap_retune.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}")
    if ranking:
        print(
            f"Top params: β={ranking[0]['beta']} Ps={ranking[0]['real_proj']} "
            f"δσ={ranking[0]['delta_sigma']}"
        )


if __name__ == "__main__":
    main()
