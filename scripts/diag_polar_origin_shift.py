#!/usr/bin/env python3
"""Diagnostic (read-only): is a FAILing Q-peak list correct up to a free origin shift?

``metrics.qpeak_benchmark`` only tries origins {0, 1/2}^3 x inversion. In polar
space groups (P21: free origin along b; Pna21: along c; ...) any shift along the
polar axis leaves all intensities unchanged, so a correct solution at an
arbitrary polar-axis origin scores FAIL. This script scans a continuous shift
along one axis (and inversion) and reports the best hit@0.5 A. It does not
change the scorer or any result.

    python scripts/diag_polar_origin_shift.py --cif 2020456.cif --res trial_winner.res --axis b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grok_phase_solver.pipeline.peaks  # noqa: F401,E402
import numpy as np  # noqa: E402

from grok_phase_solver.metrics.qpeak_benchmark import (  # noqa: E402
    evaluate, orbit, orth_matrix, read_cif_nonh, read_res_q, space_group_ops,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif", required=True, type=Path)
    ap.add_argument("--res", required=True, type=Path)
    ap.add_argument("--axis", choices="abc", default="b")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    cell, hm, atoms = read_cif_nonh(a.cif)
    ops = space_group_ops(hm)
    orth = orth_matrix(cell)
    af = np.array([x[2] for x in atoms])
    a_orb = orbit(af, ops)
    q = read_res_q(a.res)
    ax = "abc".index(a.axis)
    base = evaluate(af, a_orb, q, ops, orth)["hit05"]
    best = (base, 1, 0.0)
    for sgn in (1, -1):
        for t in np.linspace(0.0, 1.0, a.steps + 1):
            shift = np.zeros(3)
            shift[ax] = t
            h = evaluate(af, a_orb, sgn * q + shift, ops, orth)["hit05"]
            if h > best[0]:
                best = (h, sgn, float(t))
    print(f"{a.cif.name} {hm} N={len(af)}: hit05 as scored = {base}; "
          f"best with free shift along {a.axis} (+/- inversion) = {best[0]} "
          f"(inversion={'yes' if best[1] < 0 else 'no'}, shift={best[2]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
