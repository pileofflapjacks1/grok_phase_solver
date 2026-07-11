#!/usr/bin/env python3
"""
Solvability failure taxonomy benchmark.

For each synthetic cell, run multistart CF+RAAR, score free FOM vs truth mapCC,
and label unsolved cases:

  A — selection (FOM missed a better basin / FOM inversion)
  B — basin (search never found truth-like free FOM)
  C — information (low refl/atom, weak κ, true FOM ≲ random)

Writes data/processed/failure_taxonomy.{json,md}
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
    diagnose_structure,
    summarize_taxonomy,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-starts", type=int, default=None)
    p.add_argument("--n-iter", type=int, default=None)
    args = p.parse_args()

    if args.quick:
        easy = [(4, 0.9), (6, 1.0)]
        hard = [(12, 1.5), (16, 1.5)]
        seeds = [0, 1]
        n_starts = args.n_starts or 2
        n_iter = args.n_iter or 40
    else:
        easy = [(4, 0.9), (6, 1.0), (8, 0.9)]
        hard = [(12, 1.5), (12, 2.0), (16, 1.5), (20, 1.5)]
        seeds = [0, 1, 2]
        n_starts = args.n_starts or 3
        n_iter = args.n_iter or 60

    results = []
    t0 = time.time()

    for region, grid in [("easy", easy), ("hard", hard)]:
        print(f"\n===== {region.upper()} =====")
        for n_atoms, d_min in grid:
            for seed in seeds:
                st = generate_random_organic(
                    n_atoms=n_atoms, seed=seed, space_group="P1"
                )
                data = structure_to_fcalc(st, d_min=d_min)
                n_cell = int(data.get("n_atoms_cell", n_atoms))
                try:
                    res = diagnose_structure(
                        data["hkl"],
                        data["amplitudes"],
                        data["phases"],
                        st.cell,
                        n_atoms=n_cell,
                        d_min=d_min,
                        structure_seed=seed,
                        n_starts=n_starts,
                        n_iter=n_iter,
                    )
                except Exception as e:
                    print(f"  ERROR n={n_atoms} d={d_min} s={seed}: {e}")
                    continue
                row = res.to_dict()
                row["region"] = region
                results.append(row)
                flag = res.primary
                print(
                    f"  {flag:6s} n={n_atoms:2d} d={d_min:.1f} s={seed} "
                    f"bestCC={res.mapcc_best_trial:.2f} fomPickCC={res.mapcc_fom_pick:.2f} "
                    f"rpa={res.refl_per_atom:.1f} κ={res.mean_kappa:.2f} "
                    f"Ctrue={res.composite_true:.2f} Crand={res.composite_random:.2f}"
                )
                for reason in res.reasons[:2]:
                    print(f"         · {reason}")

    # Rebuild TaxonomyResult-like summary from dicts
    from grok_phase_solver.metrics.failure_taxonomy import TaxonomyResult

    objects = []
    for r in results:
        objects.append(
            TaxonomyResult(
                n_atoms=r["n_atoms"],
                d_min=r["d_min"],
                structure_seed=r["structure_seed"],
                n_refl=r["n_refl"],
                refl_per_atom=r["refl_per_atom"],
                mean_kappa=r["mean_kappa"],
                n_triplets=r["n_triplets"],
                mapcc_best_trial=r["mapcc_best_trial"],
                mapcc_fom_pick=r["mapcc_fom_pick"],
                composite_best_trial=r["composite_best_trial"],
                composite_fom_pick=r["composite_fom_pick"],
                composite_true=r["composite_true"],
                composite_random=r["composite_random"],
                R_pos_true=r["R_pos_true"],
                R_pos_random=r["R_pos_random"],
                solved_strict=r["solved_strict"],
                primary=r["primary"],
                flags=r["flags"],
                reasons=r["reasons"],
                trials=r.get("trials", []),
            )
        )
    summary = summarize_taxonomy(objects)
    # region from rows
    for reg in ("easy", "hard"):
        sub = [o for o, r in zip(objects, results) if r.get("region") == reg]
        summary[f"summary_{reg}"] = summarize_taxonomy(sub)

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": results,
        "summary": summary,
        "seconds": time.time() - t0,
        "config": {
            "n_starts": n_starts,
            "n_iter": n_iter,
            "methods": ["cf", "raar"],
            "easy": easy,
            "hard": hard,
            "seeds": seeds,
        },
        "note": (
            "Primary labels: solved | A (selection) | B (basin) | C (information) | "
            "A+B | B+C | unknown. See docs/math/failure_taxonomy.md."
        ),
    }
    jp = out_dir / "failure_taxonomy.json"
    jp.write_text(json.dumps(payload, indent=2, default=float))

    md = _render_md(payload, results, summary)
    mp = out_dir / "failure_taxonomy.md"
    mp.write_text(md)
    print(f"\nSummary counts: {summary['counts']}")
    print(f"Wrote {jp}\nWrote {mp}")


def _render_md(payload, results, summary) -> str:
    lines = [
        "# Solvability failure taxonomy",
        "",
        "Classification of multistart CF+RAAR outcomes under free-FOM ranking.",
        "",
        "## Labels",
        "",
        "| Code | Meaning | Actionable implication |",
        "|------|---------|------------------------|",
        "| **solved** | Best trial mapCC ≥ 0.7 | Classical path works |",
        "| **near** | mapCC in [0.55, 0.7) | More polish / iters, not more data alone |",
        "| **A** | Selection failure | Improve free FOM / selection |",
        "| **B** | Basin / optimization failure | Better search, multistart, or priors |",
        "| **C** | Information / underdetermination | Need more data or stronger atomic prior |",
        "| **A+B** / **B+C** | Multi-factor | Address both |",
        "",
        "## Overall counts",
        "",
        "| Label | Count | Rate |",
        "|-------|-------|------|",
    ]
    n = summary["n"]
    for lab, c in summary["counts"].items():
        if c == 0:
            continue
        lines.append(f"| `{lab}` | {c} | {c/n:.0%} |")

    for reg in ("easy", "hard"):
        key = f"summary_{reg}"
        if key not in summary:
            continue
        sub = summary[key]
        lines.extend([
            "",
            f"## {reg.capitalize()} region",
            "",
            f"n = {sub['n']}, mean best mapCC = {sub['mean_best_mapcc']:.3f}, "
            f"mean refl/atom = {sub['mean_refl_per_atom']:.1f}",
            "",
            "| Label | Count | Rate |",
            "|-------|-------|------|",
        ])
        sn = sub["n"] or 1
        for lab, c in sub["counts"].items():
            if c == 0:
                continue
            lines.append(f"| `{lab}` | {c} | {c/sn:.0%} |")

    lines.extend([
        "",
        "## Per-case table",
        "",
        "| region | n | d_min | seed | primary | bestCC | fomPickCC | rpa | κ | C_true | C_rand |",
        "|--------|---|-------|------|---------|--------|-----------|-----|---|--------|--------|",
    ])
    for r in results:
        lines.append(
            f"| {r['region']} | {r['n_atoms']} | {r['d_min']} | {r['structure_seed']} | "
            f"**{r['primary']}** | {r['mapcc_best_trial']:.2f} | {r['mapcc_fom_pick']:.2f} | "
            f"{r['refl_per_atom']:.1f} | {r['mean_kappa']:.2f} | "
            f"{r['composite_true']:.2f} | {r['composite_random']:.2f} |"
        )

    # Implications
    counts = summary["counts"]
    lines.extend([
        "",
        "## Scientific implications",
        "",
    ])
    n_a = counts.get("A", 0) + counts.get("A+B", 0)
    n_b = counts.get("B", 0) + counts.get("A+B", 0) + counts.get("B+C", 0)
    n_c = counts.get("C", 0) + counts.get("B+C", 0)
    lines.append(
        f"- **Selection (A-family):** {n_a} cases — free-FOM ranking is the bottleneck; "
        "continue FOM calibration / ensemble diversity metrics."
    )
    lines.append(
        f"- **Basin (B-family):** {n_b} cases — need better initialization (DM κ, PhAI) "
        "or more aggressive multistart, not only better FOMs."
    )
    lines.append(
        f"- **Information (C-family):** {n_c} cases — classical ab initio is underdetermined; "
        "neural/atomic priors or higher resolution are required."
    )
    lines.extend([
        "",
        "Hard region dominated by **B** and/or **C** means the cliff is not mainly "
        "a free-FOM bug. Dominance of **A** would mean the solution is often found "
        "but discarded.",
        "",
        f"JSON: `data/processed/failure_taxonomy.json`",
        f"Runtime: {payload['seconds']:.1f}s",
        "",
        "Math write-up: [`docs/math/failure_taxonomy.md`](../../docs/math/failure_taxonomy.md)",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
