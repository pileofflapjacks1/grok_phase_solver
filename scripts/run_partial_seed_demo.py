#!/usr/bin/env python3
"""
Run the packaged partial-φ demo (hard-ish synthetic + 30% oracle seed).

Compares:
  - auto (ab initio path)
  - partial_phaseed with known_phases_30pct.csv

Writes under examples/partial_seed_demo/out_* (gitignored).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.pipeline.export import export_solution
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure


def main():
    demo = ROOT / "examples" / "partial_seed_demo"
    hkl = demo / "demo_hard.hkl"
    ins = demo / "demo_hard.ins"
    seed_csv = demo / "known_phases_30pct.csv"
    if not hkl.exists():
        print(f"Missing demo files under {demo}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for name, cfg in [
        (
            "auto",
            SolveConfig(
                method="auto",
                n_iter=40,
                n_starts=2,
                n_extend=10,
                verbose=False,
                seed=0,
                n_peaks=20,
            ),
        ),
        (
            "partial_30",
            SolveConfig(
                method="partial_phaseed",
                phase_seed_csv=str(seed_csv),
                n_iter=50,
                n_starts=2,
                n_extend=15,
                verbose=False,
                seed=0,
                n_peaks=20,
            ),
        ),
    ]:
        out = demo / f"out_{name}"
        t0 = time.time()
        print(f"Running {name}…", flush=True)
        result = solve_structure(str(hkl), ins_path=str(ins), config=cfg)
        export_solution(result, out)
        row = {
            "run": name,
            "method": result.method,
            "seconds": time.time() - t0,
            "free_fom": result.diagnostics.get("free_fom_composite"),
            "n_peaks": len(result.peaks),
            "auto_reason": result.diagnostics.get("auto_reason"),
            "seed_source": result.diagnostics.get("seed_source"),
            "out": str(out.relative_to(ROOT)),
        }
        rows.append(row)
        print(
            f"  → method={result.method} FOM={row['free_fom']} "
            f"peaks={row['n_peaks']} t={row['seconds']:.1f}s",
            flush=True,
        )

    summary = demo / "demo_run_summary.json"
    summary.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSummary: {summary}")
    print(
        "Expect: partial_30 free-FOM much higher / map cleaner than auto on this hard-ish cell."
    )


if __name__ == "__main__":
    main()
