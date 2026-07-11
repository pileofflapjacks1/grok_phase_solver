#!/usr/bin/env python3
"""
Experimental HKL scoreboard (not only Fcalc).

Runs gps-solve methods on:
  - Demo small molecule (examples/demo_solve)
  - COD 2016452 Fcalc control (PhAI hybrid reference)
  - COD 2017775 experimental Fobs (large macrolide; capped resolution)

Metrics vs deposited structure Fcalc phases where available:
  mapCC_OI, free FOM (truth-free), peak recovery, R1, wall time.

Writes data/processed/experimental_scoreboard.{json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.io.hkl import ReflectionTable, write_hkl_simple
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def _match_truth_phases(hkl_obs, cell, st, d_min):
    """Fcalc truth phases on experimental hkl list (by Miller index)."""
    # Match resolution of the solve to keep Fcalc tractable
    data = structure_to_fcalc(st, d_min=max(d_min or 0.9, 0.85))
    key = {tuple(map(int, h)): i for i, h in enumerate(data["hkl"])}
    ph = np.zeros(len(hkl_obs), dtype=np.float64)
    mapped = 0
    for i, h in enumerate(hkl_obs):
        t = tuple(map(int, h))
        if t in key:
            ph[i] = data["phases"][key[t]]
            mapped += 1
        else:
            tf = (-t[0], -t[1], -t[2])
            if tf in key:
                ph[i] = -data["phases"][key[tf]]
                mapped += 1
    return ph, mapped / max(len(hkl_obs), 1), data


def run_case(
    label: str,
    hkl_path: Path,
    methods: list,
    cell: str = None,
    sg: str = None,
    cif_truth: Path = None,
    ins: Path = None,
    d_min: float = None,
    n_iter: int = 50,
    n_starts: int = 2,
    n_extend: int = 10,
    do_success: bool = True,
):
    rows = []
    print(f"\n===== {label} =====", flush=True)
    st = load_cif(cif_truth) if cif_truth and cif_truth.exists() else None

    for method in methods:
        t0 = time.time()
        print(f"  … {method}", flush=True)
        try:
            cfg = SolveConfig(
                method=method,
                d_min=d_min,
                n_iter=n_iter,
                n_recycle=5,
                n_extend=n_extend,
                n_starts=n_starts,
                n_peaks=25,
                verbose=False,
                seed=0,
            )
            result = solve_structure(
                str(hkl_path),
                ins_path=str(ins) if ins else None,
                cell=cell,
                space_group=sg,
                config=cfg,
            )
            row = {
                "dataset": label,
                "method": result.method,
                "method_requested": method,
                "n_refl": len(result.hkl),
                "d_min": result.d_min,
                "seconds": time.time() - t0,
                "free_fom_composite": result.diagnostics.get("free_fom_composite"),
                "free_fom_R_pos": result.diagnostics.get("free_fom_R_pos"),
                "n_peaks": len(result.peaks),
                "auto_reason": result.diagnostics.get("auto_reason"),
                "seed_source": result.diagnostics.get("seed_source"),
                "warnings": result.warnings[:5] if result.warnings else [],
            }
            if st is not None and do_success:
                try:
                    ph_t, frac, data = _match_truth_phases(
                        result.hkl, result.cell, st, d_min=result.d_min
                    )
                    row["frac_truth_mapped"] = frac
                    rho_t = density_from_structure_factors(
                        result.hkl,
                        result.amplitudes * np.exp(1j * ph_t),
                        result.cell,
                        shape=result.density.shape,
                    )
                    cc, _ = map_correlation_origin_invariant(result.density, rho_t)
                    row["mapcc_oi"] = float(cc)
                    # Peak recovery can be expensive on huge cells — skip if many atoms
                    n_atoms = len(data.get("fracs", []))
                    if n_atoms <= 80:
                        rep = evaluate_success(
                            result.hkl,
                            result.amplitudes,
                            result.phases,
                            ph_t,
                            result.cell,
                            data["fracs"],
                            density=result.density,
                            elements=data["elements"],
                            thresholds=SuccessThresholds(),
                        )
                        row["peak_recovery"] = rep.peak_recovery
                        row["r1"] = rep.r1
                        row["solved"] = rep.solved
                    else:
                        row["solved"] = bool(cc >= 0.70)
                        row["peak_recovery"] = None
                        row["r1"] = None
                        row["note"] = f"large cell ({n_atoms} atoms): mapCC-only solve flag"
                except Exception as e:
                    row["truth_error"] = str(e)
                    row["solved"] = False

            rows.append(row)
            print(
                f"  {result.method:18s} CC={row.get('mapcc_oi', float('nan')):.3f} "
                f"FOM={row.get('free_fom_composite', float('nan')):.3f} "
                f"n={row['n_refl']} solved={row.get('solved')} "
                f"t={row['seconds']:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(f"  ERROR {method}: {e}", flush=True)
            rows.append({"dataset": label, "method": method, "error": str(e)})
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        methods_small = ["charge_flipping", "ensemble", "auto"]
        methods_phai = ["charge_flipping", "auto"]
        n_iter, n_starts = 35, 1
    else:
        methods_small = [
            "charge_flipping",
            "ensemble",
            "phai+cf_cond",
            "phai_phaseed",
            "auto",
        ]
        methods_phai = methods_small
        n_iter, n_starts = 50, 2

    all_rows = []

    # 1) Demo
    demo_hkl = ROOT / "examples/demo_solve/demo.hkl"
    demo_ins = ROOT / "examples/demo_solve/demo.ins"
    if demo_hkl.exists():
        all_rows.extend(
            run_case(
                "demo_solve",
                demo_hkl,
                methods_small,
                ins=demo_ins,
                n_iter=n_iter,
                n_starts=n_starts,
            )
        )

    # 2) COD 2016452 Fcalc control
    cif_2016452 = ROOT / "data/raw/cod/2016452.cif"
    if cif_2016452.exists():
        st = load_cif(cif_2016452)
        dmins = [0.9] if args.quick else [0.9, 1.5]
        for d_min in dmins:
            data = structure_to_fcalc(st, d_min=d_min)
            tmp = ROOT / "data" / "processed" / f"_tmp_2016452_{d_min}.hkl"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            table = ReflectionTable(
                hkl=data["hkl"],
                F_meas=data["amplitudes"],
                cell=st.cell,
                space_group_hm=st.space_group_hm,
            )
            write_hkl_simple(tmp, table)
            cell = ",".join(str(x) for x in st.cell)
            all_rows.extend(
                run_case(
                    f"COD_2016452_Fcalc_{d_min}",
                    tmp,
                    methods_phai,
                    cell=cell,
                    sg=st.space_group_hm,
                    cif_truth=cif_2016452,
                    n_iter=n_iter,
                    n_starts=n_starts,
                )
            )

    # 3) COD 2017775 experimental Fobs (large) — resolution-capped, light methods
    hkl_2017775 = ROOT / "data/raw/cod/2017775.hkl"
    cif_2017775 = ROOT / "data/raw/cod/2017775.cif"
    if hkl_2017775.exists() and cif_2017775.exists():
        st = load_cif(cif_2017775)
        cell = ",".join(str(x) for x in st.cell)
        # Cap at moderate res so grid FFTs stay tractable
        dcut = 1.5 if args.quick else 1.2
        exp_methods = ["charge_flipping", "ensemble", "auto"]
        all_rows.extend(
            run_case(
                f"COD_2017775_exp_{dcut}",
                hkl_2017775,
                exp_methods,
                cell=cell,
                sg=st.space_group_hm or "P212121",
                cif_truth=cif_2017775,
                d_min=dcut,
                n_iter=30 if args.quick else 40,
                n_starts=1,
                do_success=True,
            )
        )

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "experimental_scoreboard.json"
    jp.write_text(json.dumps(all_rows, indent=2, default=float))

    md = [
        "# Experimental HKL scoreboard",
        "",
        "Methods on **experimental-style** data plus COD Fcalc control.",
        "Strict / mapCC success uses deposited structure Fcalc truth when CIF is available.",
        "",
        "## Results",
        "",
        "| Dataset | Method | mapCC | free FOM | peaks | solved | s |",
        "|---------|--------|-------|----------|-------|--------|---|",
    ]
    for r in all_rows:
        if "error" in r:
            md.append(f"| {r['dataset']} | `{r['method']}` | ERROR | — | — | — | — |")
            continue
        md.append(
            f"| {r['dataset']} | `{r.get('method')}` | "
            f"{r.get('mapcc_oi', float('nan')):.3f} | "
            f"{r.get('free_fom_composite', float('nan')):.3f} | "
            f"{r.get('n_peaks', 0)} | {r.get('solved')} | "
            f"{r.get('seconds', 0):.1f} |"
        )
    md.extend([
        "",
        "## Notes",
        "",
        "- **demo_solve**: small synthetic-style demo with INS.",
        "- **COD 2016452 Fcalc**: PhAI/AI-PhaSeed control (should show strong hybrids).",
        "- **COD 2017775**: large experimental Fobs (roxithromycin); ab initio expected "
        "to struggle — free FOM still ranks without truth; resolution capped for runtime.",
        "- `auto` selects phai_phaseed / ensemble / CF by SG, resolution, and weights.",
        "- `trial.res` is written by `gps-solve` for Olex2/SHELXL loading.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp = out / "experimental_scoreboard.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}", flush=True)


if __name__ == "__main__":
    main()
