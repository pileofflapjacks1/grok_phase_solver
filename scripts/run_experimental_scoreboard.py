#!/usr/bin/env python3
"""
Experimental HKL scoreboard (Lane C review pack).

Runs gps-solve methods on:
  - Demo small molecule (examples/demo_solve)
  - COD 2016452 Fcalc control + **experimental Fobs** (new)
  - COD 2100301 Fcalc control + **experimental Fobs** (new)
  - COD 2017775 experimental Fobs (large macrolide; res-capped)

Optional: SHELXS when ShelX/shelxs is present; strong_prior_phaseed; PhAI.

Metrics vs deposited structure Fcalc phases when CIF is available:
  mapCC_OI, free FOM (truth-free), peak recovery, R1, wall time.

Writes data/processed/experimental_scoreboard.{json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

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
from grok_phase_solver.solvers.partial_seed import (
    oracle_partial_seed,
    write_phase_seed_csv,
)


def _match_truth_phases(hkl_obs, st, d_min):
    """Fcalc truth phases on experimental hkl list (by Miller index)."""
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


def _shelxs_available() -> bool:
    try:
        from grok_phase_solver.solvers.shelxs_runner import shelxs_available

        return bool(shelxs_available())
    except Exception:
        return False


def _strong_prior_available() -> bool:
    try:
        from grok_phase_solver.models.strong_prior import default_strong_prior_path

        return default_strong_prior_path().exists()
    except Exception:
        return False


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
    phase_seed_csv: Path = None,
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
                phase_seed_csv=str(phase_seed_csv) if phase_seed_csv else None,
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
                        result.hkl, st, d_min=result.d_min
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
                f"  {result.method:20s} CC={row.get('mapcc_oi', float('nan')):.3f} "
                f"FOM={row.get('free_fom_composite', float('nan')):.3f} "
                f"n={row['n_refl']} solved={row.get('solved')} "
                f"t={row['seconds']:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(f"  ERROR {method}: {e}", flush=True)
            rows.append({"dataset": label, "method": method, "error": str(e)})
    return rows


def _write_fcalc_hkl(st, d_min: float, path: Path) -> Path:
    data = structure_to_fcalc(st, d_min=d_min)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = ReflectionTable(
        hkl=data["hkl"],
        F_meas=data["amplitudes"],
        cell=st.cell,
        space_group_hm=st.space_group_hm,
    )
    write_hkl_simple(path, table)
    return path


def _oracle_seed_for_fcalc(st, d_min: float, path: Path, fraction: float = 0.30) -> Path:
    data = structure_to_fcalc(st, d_min=d_min)
    seed_ph, mask, _ = oracle_partial_seed(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        data["phases"],
        fraction=fraction,
        mode="strong_E",
        seed=0,
    )
    write_phase_seed_csv(path, data["hkl"], seed_ph, mask=mask)
    return path


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    shelxs_ok = _shelxs_available()
    strong_ok = _strong_prior_available()
    print(f"SHELXS available: {shelxs_ok}; strong_prior: {strong_ok}", flush=True)

    if args.quick:
        methods_core = ["charge_flipping", "ensemble", "auto"]
        methods_phai = ["charge_flipping", "ensemble", "auto"]
        n_iter, n_starts, n_extend = 30, 1, 8
    else:
        methods_core = ["charge_flipping", "ensemble", "auto"]
        methods_phai = [
            "charge_flipping",
            "ensemble",
            "phai+cf_cond",
            "phai_phaseed",
            "auto",
        ]
        if strong_ok:
            methods_core = methods_core + ["strong_prior_phaseed"]
            methods_phai = methods_phai + ["strong_prior_phaseed"]
        if shelxs_ok:
            methods_core = methods_core + ["shelxs"]
            methods_phai = methods_phai + ["shelxs"]
        n_iter, n_starts, n_extend = 50, 2, 10

    all_rows: List[dict] = []
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    # 1) Demo
    demo_hkl = ROOT / "examples/demo_solve/demo.hkl"
    demo_ins = ROOT / "examples/demo_solve/demo.ins"
    if demo_hkl.exists():
        all_rows.extend(
            run_case(
                "demo_solve",
                demo_hkl,
                methods_core,
                ins=demo_ins,
                n_iter=n_iter,
                n_starts=n_starts,
                n_extend=n_extend,
            )
        )

    # Helper: Fcalc + optional oracle partial + experimental HKL for a COD id
    def cod_panel(cod_id: str, dmins: List[float], exp_dmin: Optional[float]):
        cif = ROOT / "data/raw/cod" / f"{cod_id}.cif"
        hkl_exp = ROOT / "data/raw/cod" / f"{cod_id}.hkl"
        if not cif.exists():
            print(f"skip COD {cod_id}: no CIF", flush=True)
            return
        st = load_cif(cif)
        cell = ",".join(str(x) for x in st.cell)
        sg = st.space_group_hm
        methods = methods_phai if "P21" in (sg or "").upper().replace(" ", "") or "P121" in (sg or "").upper().replace(" ", "") else methods_core

        for d_min in dmins:
            tmp = proc / f"_tmp_{cod_id}_{d_min}.hkl"
            _write_fcalc_hkl(st, d_min, tmp)
            all_rows.extend(
                run_case(
                    f"COD_{cod_id}_Fcalc_{d_min}",
                    tmp,
                    methods,
                    cell=cell,
                    sg=sg,
                    cif_truth=cif,
                    n_iter=n_iter,
                    n_starts=n_starts,
                    n_extend=n_extend,
                )
            )
            # Oracle partial-φ control (Lane B path on Fcalc)
            seed_path = proc / f"_tmp_{cod_id}_{d_min}_oracle30.csv"
            _oracle_seed_for_fcalc(st, d_min, seed_path, fraction=0.30)
            all_rows.extend(
                run_case(
                    f"COD_{cod_id}_Fcalc_{d_min}_partial30",
                    tmp,
                    ["partial_phaseed"],
                    cell=cell,
                    sg=sg,
                    cif_truth=cif,
                    n_iter=n_iter,
                    n_starts=max(1, n_starts),
                    n_extend=n_extend,
                    phase_seed_csv=seed_path,
                )
            )

        if exp_dmin is not None and hkl_exp.exists():
            all_rows.extend(
                run_case(
                    f"COD_{cod_id}_exp_{exp_dmin}",
                    hkl_exp,
                    methods_core if args.quick else methods,
                    cell=cell,
                    sg=sg,
                    cif_truth=cif,
                    d_min=exp_dmin,
                    n_iter=n_iter if not args.quick else 30,
                    n_starts=1 if args.quick else n_starts,
                    n_extend=n_extend,
                )
            )

    # 2–3) COD small organics with Fcalc + experimental Fobs
    if args.quick:
        cod_panel("2016452", [0.9], 1.0)
        cod_panel("2100301", [0.9], None)
    else:
        cod_panel("2016452", [0.9, 1.2], 1.0)
        cod_panel("2100301", [0.9, 1.2], 1.0)

    # 4) COD 2017775 large experimental Fobs
    hkl_2017775 = ROOT / "data/raw/cod/2017775.hkl"
    cif_2017775 = ROOT / "data/raw/cod/2017775.cif"
    if hkl_2017775.exists() and cif_2017775.exists():
        st = load_cif(cif_2017775)
        cell = ",".join(str(x) for x in st.cell)
        dcut = 1.5 if args.quick else 1.2
        exp_methods = ["charge_flipping", "ensemble", "auto"]
        if not args.quick and strong_ok:
            exp_methods.append("strong_prior_phaseed")
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
                n_extend=8,
                do_success=True,
            )
        )

    out = proc
    jp = out / "experimental_scoreboard.json"
    jp.write_text(json.dumps(all_rows, indent=2, default=float))

    # Build markdown with results + per-dataset best + honest notes
    md = [
        "# Experimental HKL scoreboard (Lane C)",
        "",
        "Methods on **experimental Fobs** (COD) plus **Fcalc controls** from deposited CIFs.",
        "Strict / mapCC success uses Fcalc truth phases matched by Miller index when CIF is available.",
        "",
        f"- SHELXS binary: **{'found' if shelxs_ok else 'not found'}**",
        f"- GraphPhaseNet strong prior: **{'found' if strong_ok else 'not found'}**",
        f"- SHELXD binary: **not in ShelX/** (dual-space in-repo only; see `shelxd_h2h.md`)",
        "",
        "## Results",
        "",
        "| Dataset | Method | mapCC | free FOM | peaks | solved | s |",
        "|---------|--------|-------|----------|-------|--------|---|",
    ]
    by_ds: dict = {}
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
        ds = r["dataset"]
        by_ds.setdefault(ds, []).append(r)

    md.extend(["", "## Best mapCC per dataset (truth-matched)", "", "| Dataset | Best method | mapCC | solved |", "|---------|-------------|-------|--------|"])
    for ds, rows in by_ds.items():
        scored = [r for r in rows if r.get("mapcc_oi") is not None and "error" not in r]
        if not scored:
            md.append(f"| {ds} | — | n/a (no truth mapCC) | — |")
            continue
        best = max(scored, key=lambda r: r.get("mapcc_oi") or -1)
        md.append(
            f"| {ds} | `{best.get('method')}` | {best.get('mapcc_oi'):.3f} | {best.get('solved')} |"
        )

    md.extend(
        [
            "",
            "## Notes",
            "",
            "- **demo_solve**: packaged easy demo (INS); free FOM ranks without truth.",
            "- **COD 2016452**: small P2₁/c organic; Fcalc control + **experimental Fobs** from COD.",
            "- **COD 2100301**: dinicotinic acid P2₁/c (neutron structure); Fcalc + **experimental Fobs**.",
            "- **`*_partial30`**: oracle 30% strong-|E| phases → `partial_phaseed` (Lane B hard path).",
            "- **COD 2017775**: large experimental Fobs (roxithromycin); ab initio expected to struggle.",
            "- Experimental Fobs mapCC uses Fcalc from deposited model as proxy truth (not refined R1).",
            "- `auto` selects ensemble / PhAI / prior / CF by SG, resolution, and available weights.",
            "- Industrial SHELXD not redistributed; local SHELXS used when present.",
            "",
            "Related scoreboards: `cod_hybrid_benchmark.md`, `shelxs_h2h.md`, `strong_prior.md`, "
            "`partial_seed_benchmark.md`.",
            "",
            f"JSON: `{jp.relative_to(ROOT)}`",
            "",
        ]
    )
    mp = out / "experimental_scoreboard.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}", flush=True)


if __name__ == "__main__":
    main()
