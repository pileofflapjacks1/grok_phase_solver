#!/usr/bin/env python3
"""
PhAI-seeded failure taxonomy vs classical random multistart.

On the hard solvability grid (n≥12, d_min≥1.5), compare:

  classical: multistart CF+RAAR from random phases
  phai:      PhAI fair seed + multistart CF+RAAR from that seed (+ pure seed trial)

Reports how many A+B / B+C shift to solved or near.

Writes data/processed/phai_taxonomy.{json,md}
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
from grok_phase_solver.metrics.failure_taxonomy import (
    TaxonomyResult,
    diagnose_structure,
    inversion_rate,
    summarize_taxonomy,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def _try_phai(hkl, amp, seed=0):
    try:
        from grok_phase_solver.models.phai_fair import run_phai_fair
        from grok_phase_solver.models.phai_runner import phai_available

        if not phai_available():
            return None, "phai_unavailable"
        ph, meta = run_phai_fair(hkl, amp, n_cycles=5, seed=seed)
        return ph, meta
    except Exception as e:
        return None, str(e)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        hard = [(12, 1.5), (16, 1.5)]
        seeds = [0, 1]
        n_starts = 2
        n_iter = 40
    else:
        hard = [(12, 1.5), (12, 2.0), (16, 1.5), (20, 1.5)]
        seeds = [0, 1, 2]
        n_starts = 3
        n_iter = 60

    rows = []
    classical_objs = []
    phai_objs = []
    t0 = time.time()

    print(f"Hard grid: {hard}, seeds={seeds}, starts={n_starts}, iter={n_iter}")
    phai_ok = None

    for n_atoms, d_min in hard:
        for seed in seeds:
            st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
            data = structure_to_fcalc(st, d_min=d_min)
            n_cell = int(data.get("n_atoms_cell", n_atoms))
            hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]

            # Classical
            res_c = diagnose_structure(
                hkl, amp, ph_t, st.cell, n_atoms=n_cell, d_min=d_min,
                structure_seed=seed, n_starts=n_starts, n_iter=n_iter,
                methods=("cf", "raar"), phase_init=None, init_label="random",
            )
            classical_objs.append(res_c)

            # PhAI seed
            ph0, meta = _try_phai(hkl, amp, seed=seed)
            if ph0 is None:
                phai_ok = False
                res_p = None
                phai_status = meta
            else:
                phai_ok = True
                res_p = diagnose_structure(
                    hkl, amp, ph_t, st.cell, n_atoms=n_cell, d_min=d_min,
                    structure_seed=seed, n_starts=n_starts, n_iter=n_iter,
                    methods=("cf", "raar"), phase_init=ph0, init_label="phai_fair",
                )
                phai_objs.append(res_p)
                phai_status = "ok"

            row = {
                "n_atoms": n_atoms,
                "d_min": d_min,
                "seed": seed,
                "n_refl": len(amp),
                "classical_primary": res_c.primary,
                "classical_bestCC": res_c.mapcc_best_trial,
                "classical_fomPickCC": res_c.mapcc_fom_pick,
                "classical_Ctrue": res_c.composite_true,
                "classical_fom_inversion": res_c.flags.get("fom_inversion_vs_true"),
                "phai_status": phai_status,
            }
            if res_p is not None:
                row.update({
                    "phai_primary": res_p.primary,
                    "phai_bestCC": res_p.mapcc_best_trial,
                    "phai_fomPickCC": res_p.mapcc_fom_pick,
                    "phai_seed_mapcc": next(
                        (t["mapcc_oi"] for t in res_p.trials if t["method"] == "seed"),
                        None,
                    ),
                    "phai_Ctrue": res_p.composite_true,
                    "phai_fom_inversion": res_p.flags.get("fom_inversion_vs_true"),
                    "improved": (
                        res_p.mapcc_best_trial > res_c.mapcc_best_trial + 0.05
                        or (
                            res_p.primary in ("solved", "near")
                            and res_c.primary not in ("solved", "near")
                        )
                    ),
                    "label_upgrade": res_c.primary != res_p.primary,
                })
            rows.append(row)
            msg = (
                f"  n={n_atoms:2d} d={d_min:.1f} s={seed}  "
                f"classical={res_c.primary:6s} CC={res_c.mapcc_best_trial:.2f}"
            )
            if res_p is not None:
                msg += (
                    f"  → phai={res_p.primary:6s} CC={res_p.mapcc_best_trial:.2f} "
                    f"seedCC={row.get('phai_seed_mapcc')}"
                )
            else:
                msg += f"  → PhAI skip ({phai_status})"
            print(msg)

    sum_c = summarize_taxonomy(classical_objs)
    sum_p = summarize_taxonomy(phai_objs) if phai_objs else None

    # Transition matrix classical → phai
    transitions = {}
    for r in rows:
        if "phai_primary" not in r:
            continue
        key = f"{r['classical_primary']}→{r['phai_primary']}"
        transitions[key] = transitions.get(key, 0) + 1

    n_improve = sum(1 for r in rows if r.get("improved"))
    n_both = sum(1 for r in rows if "phai_primary" in r)

    payload = {
        "rows": rows,
        "summary_classical": sum_c,
        "summary_phai": sum_p,
        "transitions": transitions,
        "n_improved": n_improve,
        "n_compared": n_both,
        "phai_available": phai_ok,
        "inversion_rate_classical": inversion_rate(classical_objs),
        "inversion_rate_phai": inversion_rate(phai_objs) if phai_objs else None,
        "seconds": time.time() - t0,
        "config": {
            "hard": hard, "seeds": seeds, "n_starts": n_starts, "n_iter": n_iter,
        },
        "fom_version": 2.1,
        "note": (
            "Compares random multistart vs PhAI-seeded multistart under free-FOM v2.1. "
            "Primary labels from failure_taxonomy."
        ),
    }

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "phai_taxonomy.json"
    jp.write_text(json.dumps(payload, indent=2, default=float))

    md = _render(payload)
    mp = out / "phai_taxonomy.md"
    mp.write_text(md)
    print(f"\nClassical: {sum_c['counts']}")
    if sum_p:
        print(f"PhAI:      {sum_p['counts']}")
    print(f"Improved: {n_improve}/{n_both}")
    print(f"Wrote {jp}\nWrote {mp}")


def _render(payload) -> str:
    lines = [
        "# PhAI-seeded failure taxonomy (hard region)",
        "",
        "Compare **classical random multistart** vs **PhAI fair seed + multistart** "
        "under free-FOM v2.1 (anti-false-atomicity).",
        "",
        f"PhAI available: **{payload['phai_available']}**",
        f"Improved cases (mapCC +0.05 or label upgrade to solved/near): "
        f"**{payload['n_improved']}/{payload['n_compared']}**",
        "",
        "## Label counts",
        "",
        "### Classical (random init)",
        "",
        "| Label | Count | Rate |",
        "|-------|-------|------|",
    ]
    sc = payload["summary_classical"]
    for lab, c in sc["counts"].items():
        if c:
            lines.append(f"| `{lab}` | {c} | {c/sc['n']:.0%} |")
    lines.append(f"\nmean best mapCC = {sc['mean_best_mapcc']:.3f}")
    lines.append(
        f"FOM inversion rate = {payload['inversion_rate_classical']:.0%}"
    )

    if payload["summary_phai"]:
        sp = payload["summary_phai"]
        lines.extend([
            "",
            "### PhAI-seeded",
            "",
            "| Label | Count | Rate |",
            "|-------|-------|------|",
        ])
        for lab, c in sp["counts"].items():
            if c:
                lines.append(f"| `{lab}` | {c} | {c/sp['n']:.0%} |")
        lines.append(f"\nmean best mapCC = {sp['mean_best_mapcc']:.3f}")
        if payload["inversion_rate_phai"] is not None:
            lines.append(
                f"FOM inversion rate = {payload['inversion_rate_phai']:.0%}"
            )

    lines.extend([
        "",
        "## Transitions (classical → PhAI)",
        "",
        "| Transition | Count |",
        "|------------|-------|",
    ])
    for k, v in sorted(payload["transitions"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |")

    lines.extend([
        "",
        "## Per-case",
        "",
        "| n | d_min | seed | classical | bestCC | phai | bestCC | seedCC | improved |",
        "|---|-------|------|-----------|--------|------|--------|--------|----------|",
    ])
    for r in payload["rows"]:
        lines.append(
            f"| {r['n_atoms']} | {r['d_min']} | {r['seed']} | "
            f"**{r['classical_primary']}** | {r['classical_bestCC']:.2f} | "
            f"**{r.get('phai_primary', '—')}** | "
            f"{r.get('phai_bestCC', float('nan')):.2f} | "
            f"{r.get('phai_seed_mapcc', float('nan')):.2f} | "
            f"{r.get('improved', False)} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If PhAI shifts **A+B / B+C → solved/near**, the hard cliff is mainly a "
        "**basin/prior** problem that neural seeds help.",
        "- If labels stay **A+B** with higher mapCC but free-FOM inversion remains, "
        "need both priors and AFA free FOM.",
        "- If PhAI seed mapCC is high but polish destroys it, conditional gate matters "
        "(see free-FOM rewrite trust-region).",
        "",
        f"JSON: `data/processed/phai_taxonomy.json`",
        f"Runtime: {payload['seconds']:.1f}s",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
