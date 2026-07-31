#!/usr/bin/env python3
"""
Stratified COD experimental / Fcalc hybrid bench skeleton (v0.10).

For each local COD sample under data/raw/cod/ with CIF (+ optional HKL):
  - auto / partial_30 / fragment_half (if non-H atoms available)
  - Report mapCC_OI, free FOM, R1; stratify by volume band and max Z

Writes data/processed/cod_stratified_bench.{json,md}

Honest: small local COD set is not Carrozzini's 1505-structure panel;
partial-φ remains the reliable hard path.
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
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.stratified_prior import max_Z_from_elements, is_ha_bearing
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.partial_seed import oracle_partial_seed, write_phase_seed_csv
from grok_phase_solver.solvers.projectors import unit_cell_volume


def _vol_band(vol: float) -> str:
    if vol < 1000:
        return "vol_lt_1000"
    if vol <= 3500:
        return "vol_1000_3500"
    return "vol_gt_3500"


def main():
    cod_dir = ROOT / "data" / "raw" / "cod"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cifs = sorted(cod_dir.glob("*.cif")) if cod_dir.is_dir() else []
    if not cifs:
        print("No COD CIFs found under data/raw/cod/ — writing empty skeleton.")
    for cif in cifs[:6]:  # cap for laptop runs
        label = cif.stem
        print(f"=== {label} ===", flush=True)
        try:
            st = load_cif(str(cif))
        except Exception as e:
            rows.append({"dataset": label, "error": str(e)})
            continue
        vol = float(unit_cell_volume(np.asarray(st.cell, dtype=np.float64)))
        els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
        max_z = max_Z_from_elements(els)
        ha = is_ha_bearing(els)
        data = structure_to_fcalc(st, d_min=0.9)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        # write temp hkl
        from grok_phase_solver.io.hkl import ReflectionTable, write_hkl_simple

        tmp = out_dir / f"_tmp_strat_{label}.hkl"
        write_hkl_simple(
            tmp,
            ReflectionTable(
                hkl=hkl, F_meas=amp, cell=st.cell, space_group_hm=st.space_group_hm
            ),
        )
        seed30 = out_dir / f"_tmp_strat_{label}_o30.csv"
        sph, mask, _ = oracle_partial_seed(hkl, amp, st.cell, ph_t, fraction=0.30)
        write_phase_seed_csv(seed30, hkl, sph, mask)

        cell_csv = ",".join(str(x) for x in st.cell)
        for name, cfg in [
            ("auto", SolveConfig(method="auto", d_min=0.9, n_iter=40, n_starts=1, verbose=False, seed=0, compute_uncertainty=False)),
            ("partial_30", SolveConfig(method="partial_phaseed", phase_seed_csv=str(seed30), d_min=0.9, n_iter=60, n_extend=18, n_starts=1, prior_weight=0.45, verbose=False, seed=0, compute_uncertainty=False)),
        ]:
            t0 = time.time()
            try:
                res = solve_structure(
                    str(tmp),
                    cell=cell_csv,
                    space_group=st.space_group_hm,
                    config=cfg,
                )
                rho_t = density_from_structure_factors(
                    hkl, amp * np.exp(1j * ph_t), st.cell, shape=res.density.shape, d_min=0.9
                )
                cc, _ = map_correlation_origin_invariant(res.density, rho_t)
                fom = res.diagnostics.get("free_fom_composite")
                row = {
                    "dataset": label,
                    "run": name,
                    "method": res.method,
                    "mapcc_oi": float(cc),
                    "free_fom": fom,
                    "vol": vol,
                    "vol_band": _vol_band(vol),
                    "max_Z": max_z,
                    "ha_bearing": ha,
                    "space_group": st.space_group_hm,
                    "seconds": time.time() - t0,
                }
            except Exception as e:
                row = {
                    "dataset": label,
                    "run": name,
                    "error": str(e),
                    "vol": vol,
                    "vol_band": _vol_band(vol),
                    "max_Z": max_z,
                    "ha_bearing": ha,
                }
            rows.append(row)
            print(f"  {name}: {row.get('mapcc_oi', row.get('error'))}", flush=True)

    # Aggregate by vol band
    by_band: dict = {}
    for r in rows:
        if "mapcc_oi" not in r:
            continue
        key = f"{r.get('vol_band')}/{r.get('run')}"
        by_band.setdefault(key, []).append(float(r["mapcc_oi"]))
    summary = {
        k: {"n": len(v), "mean_mapcc": float(np.mean(v))} for k, v in by_band.items()
    }
    payload = {
        "rows": rows,
        "summary": summary,
        "note": (
            "Local COD stratified skeleton. Not a 1505-structure Carrozzini panel. "
            "partial_30 is oracle control; auto is ab initio."
        ),
    }
    jp = out_dir / "cod_stratified_bench.json"
    jp.write_text(json.dumps(payload, indent=2, default=str))
    md = [
        "# COD stratified bench (v0.10 skeleton)",
        "",
        payload["note"],
        "",
        "| Dataset | Run | mapCC | Vol band | max Z | HA |",
        "|---------|-----|-------|----------|-------|----|",
    ]
    for r in rows:
        if "error" in r and "mapcc_oi" not in r:
            md.append(f"| {r['dataset']} | {r.get('run')} | ERROR | {r.get('vol_band')} | {r.get('max_Z')} | {r.get('ha_bearing')} |")
        else:
            md.append(
                f"| {r.get('dataset')} | {r.get('run')} | **{r.get('mapcc_oi', float('nan')):.3f}** | "
                f"{r.get('vol_band')} | {r.get('max_Z')} | {r.get('ha_bearing')} |"
            )
    md.extend(["", "## Summary by band/run", ""])
    for k, v in sorted(summary.items()):
        md.append(f"- `{k}`: n={v['n']} mean mapCC={v['mean_mapcc']:.3f}")
    md.append("")
    (out_dir / "cod_stratified_bench.md").write_text("\n".join(md))
    print(f"Wrote {jp}", flush=True)


if __name__ == "__main__":
    main()
