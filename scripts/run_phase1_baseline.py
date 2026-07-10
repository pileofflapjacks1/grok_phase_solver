#!/usr/bin/env python3
"""
Phase 1 baseline demonstration.

Runs:
1. Synthetic organic → charge flipping / HIO / random
2. COD 2100301 (if present) → Fcalc phasing benchmark
3. COD 2017775 HKL load stats (experimental amplitudes)

Writes JSON results to data/processed/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure src on path when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.cod import download_phase1_samples
from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.io.hkl import load_hkl_cif
from grok_phase_solver.models.phai_interface import PhAIInterface, describe_phai_architecture
from grok_phase_solver.solvers.baseline import run_physics_baseline, structure_to_fcalc


def main() -> None:
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("grok_phase_solver — Phase 1 baseline run")
    print("=" * 70)

    # Ensure COD samples
    print("\n[1] COD samples")
    try:
        paths = download_phase1_samples(dest_dir=ROOT / "data" / "raw" / "cod")
        for p in paths:
            print(f"  OK {p} ({p.stat().st_size} bytes)")
    except Exception as e:
        print(f"  COD download warning: {e}")

    results = []

    # Synthetic suite
    print("\n[2] Synthetic structure baselines")
    for n_atoms, d_min, n_iter in [(6, 1.0, 100), (8, 1.2, 120)]:
        st = generate_random_organic(n_atoms=n_atoms, seed=42)
        print(f"\n  {st.summary()}")
        for method in ("random", "charge_flipping", "hio"):
            res = run_physics_baseline(
                st,
                method=method,
                d_min=d_min,
                n_iter=n_iter if method != "random" else 1,
                seed=42,
                verbose=True,
            )
            results.append(
                {
                    "suite": "synthetic",
                    "n_atoms": n_atoms,
                    "method": res.method,
                    "d_min": res.d_min,
                    "n_reflections": res.n_reflections,
                    "mpe": res.mean_phase_error_deg,
                    "mpe_oi": res.mean_phase_error_origin_invariant_deg,
                    "map_cc": res.map_cc,
                    "R": res.r_factor,
                }
            )

    # COD 2100301
    cif_path = ROOT / "data" / "raw" / "cod" / "2100301.cif"
    print("\n[3] COD 2100301 (dinicotinic acid) Fcalc baseline")
    if cif_path.exists():
        st = load_cif(cif_path)
        print(f"  {st.summary()}")
        data = structure_to_fcalc(st, d_min=1.2)
        print(
            f"  Expanded cell atoms={data['n_atoms_cell']}, "
            f"reflections@1.2Å={len(data['amplitudes'])}"
        )
        for method in ("random", "charge_flipping"):
            res = run_physics_baseline(
                st,
                method=method,
                d_min=1.2,
                n_iter=100 if method != "random" else 1,
                seed=0,
                verbose=True,
            )
            results.append(
                {
                    "suite": "COD_2100301",
                    "method": res.method,
                    "d_min": res.d_min,
                    "n_reflections": res.n_reflections,
                    "n_atoms_cell": res.n_atoms_cell,
                    "mpe": res.mean_phase_error_deg,
                    "mpe_oi": res.mean_phase_error_origin_invariant_deg,
                    "map_cc": res.map_cc,
                    "R": res.r_factor,
                    "notes": res.notes,
                }
            )
        # Resolution stress test
        print("\n  Resolution series (charge flipping)")
        for d_min in (0.9, 1.2, 1.5, 2.0):
            res = run_physics_baseline(
                st, method="charge_flipping", d_min=d_min, n_iter=80, seed=0, verbose=False
            )
            print(f"    {res.summary()}")
            results.append(
                {
                    "suite": "COD_2100301_res_series",
                    "method": res.method,
                    "d_min": res.d_min,
                    "n_reflections": res.n_reflections,
                    "mpe_oi": res.mean_phase_error_origin_invariant_deg,
                    "map_cc": res.map_cc,
                    "R": res.r_factor,
                }
            )
    else:
        print("  CIF missing — skip")

    # Experimental HKL
    hkl_path = ROOT / "data" / "raw" / "cod" / "2017775.hkl"
    print("\n[4] COD 2017775 experimental HKL")
    if hkl_path.exists():
        table = load_hkl_cif(hkl_path)
        amp = table.amplitudes
        print(f"  Reflections: {len(table)}")
        print(f"  |F| range: {amp.min():.2f} – {amp.max():.2f}")
        print(f"  Cell: {table.cell}")
        print(f"  SG: {table.space_group_hm}")
        if table.cell is not None:
            d = table.resolution_d()
            print(f"  Resolution: {d.min():.2f} – {d.max():.2f} Å")
        results.append(
            {
                "suite": "COD_2017775_hkl",
                "n_reflections": len(table),
                "F_min": float(amp.min()),
                "F_max": float(amp.max()),
                "cell": table.cell.tolist() if table.cell is not None else None,
                "space_group": table.space_group_hm,
            }
        )
    else:
        print("  HKL missing — skip")

    # PhAI status
    print("\n[5] PhAI integration status")
    phai = PhAIInterface()
    print(f"  {phai.status()}")
    print(describe_phai_architecture()[:400] + "...")

    out_json = out_dir / "phase1_baseline_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
