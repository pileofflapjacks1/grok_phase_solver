#!/usr/bin/env python3
"""
COD Vol-band stratified experimental panel (v0.13+).

For each local COD sample under data/raw/cod/ with CIF (+ HKL when present):

**Amplitude modes**
  - ``fobs``  experimental HKL (when available)
  - ``fcalc`` deposited-model Fcalc control @ d_min

**Runs**
  - auto
  - partial_15 / partial_30 (oracle strong-|E| seeds)
  - fragment_half (heaviest-cluster ~½ non-H ASU + full Fcalc soft prior)

Reports mapCC_OI vs deposited Fcalc truth, free FOM, R1, peak recovery, strict
success; stratified by Vol band (lt1000 / 1000–3500 / gt3500) and max Z.

Writes:
  data/processed/cod_stratified_bench.{json,md}

Honest: small local COD set is **not** Carrozzini's 1505-structure panel.
partial_30 is oracle control; fragment_half is the no-oracle scientist path.
Pure ab initio (auto) remains weak on hard cells.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.io.cif import AtomSite, CrystalStructure, load_cif
from grok_phase_solver.io.experiment import load_experiment
from grok_phase_solver.io.hkl import ReflectionTable, write_hkl_simple
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.stratified_prior import is_ha_bearing, max_Z_from_elements
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.partial_seed import oracle_partial_seed, write_phase_seed_csv
from grok_phase_solver.solvers.projectors import unit_cell_volume
from grok_phase_solver.solvers.seed_import import seed_from_fragment_atoms, select_fragment_atoms


def _vol_band(vol: float) -> str:
    if vol < 1000:
        return "vol_lt_1000"
    if vol <= 3500:
        return "vol_1000_3500"
    return "vol_gt_3500"


def _write_minimal_cif(path: Path, st: CrystalStructure) -> None:
    a, b, c, al, be, ga = st.cell
    lines = [
        "data_fragment",
        f"_cell_length_a {a}",
        f"_cell_length_b {b}",
        f"_cell_length_c {c}",
        f"_cell_angle_alpha {al}",
        f"_cell_angle_beta {be}",
        f"_cell_angle_gamma {ga}",
        f"_symmetry_space_group_name_H-M '{st.space_group_hm}'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_U_iso_or_equiv",
    ]
    for at in st.atoms:
        u = getattr(at, "u_iso", 0.05) or 0.05
        lines.append(
            f"{at.label} {at.element} {at.fract[0]:.6f} {at.fract[1]:.6f} "
            f"{at.fract[2]:.6f} {getattr(at, 'occupancy', 1.0):.3f} {u:.5f}"
        )
    path.write_text("\n".join(lines) + "\n")


def match_truth_phases(hkl_obs, st, d_min: float):
    data = structure_to_fcalc(st, d_min=max(float(d_min or 0.9), 0.85))
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


def _agg(rows: List[Dict], key_fn) -> Dict[str, Any]:
    buckets: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("mapcc_oi") is None or "error" in r:
            continue
        k = key_fn(r)
        buckets.setdefault(k, []).append(float(r["mapcc_oi"]))
    out = {}
    for k, v in sorted(buckets.items()):
        out[k] = {
            "n": len(v),
            "mean_mapcc": float(np.mean(v)),
            "median_mapcc": float(np.median(v)),
        }
    return out


def run_dataset(
    label: str,
    st: CrystalStructure,
    hkl: np.ndarray,
    amp: np.ndarray,
    ph_true: np.ndarray,
    fdata: Dict,
    frac_mapped: float,
    amp_mode: str,
    d_min: float,
    out_dir: Path,
    *,
    n_starts: int = 1,
    include_fragment: bool = True,
) -> List[Dict]:
    vol = float(unit_cell_volume(np.asarray(st.cell, dtype=np.float64)))
    els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
    max_z = max_Z_from_elements(els)
    ha = is_ha_bearing(els)
    sg = st.space_group_hm
    cell_csv = ",".join(str(x) for x in st.cell)
    tag = f"{label}_{amp_mode}"

    tmp = out_dir / f"_tmp_strat_{tag}.hkl"
    write_hkl_simple(
        tmp,
        ReflectionTable(hkl=hkl, F_meas=amp, cell=st.cell, space_group_hm=sg),
    )
    seed15 = out_dir / f"_tmp_strat_{tag}_o15.csv"
    seed30 = out_dir / f"_tmp_strat_{tag}_o30.csv"
    sph15, m15, meta15 = oracle_partial_seed(hkl, amp, st.cell, ph_true, fraction=0.15)
    sph30, m30, meta30 = oracle_partial_seed(hkl, amp, st.cell, ph_true, fraction=0.30)
    write_phase_seed_csv(seed15, hkl, sph15, m15)
    write_phase_seed_csv(seed30, hkl, sph30, m30)

    frag_cif = None
    n_frag = 0
    if include_fragment and len(els) >= 4:
        fracs = np.array(
            [a.fract for a in st.atoms if a.element.upper() not in ("H", "D")],
            dtype=np.float64,
        )
        n_frag = max(3, len(els) // 2)
        # Cap fragment size for large ASUs (laptop time)
        n_frag = min(n_frag, 40)
        fr_sel, el_sel, fmeta = select_fragment_atoms(
            fracs, els, max_atoms=n_frag, mode="heaviest_cluster", seed=0
        )
        atoms = [
            AtomSite(label=f"{el}{i+1}", element=el, fract=fr_sel[i], b_iso=10.0)
            for i, el in enumerate(el_sel)
        ]
        frag_st = CrystalStructure(
            name=f"{label}_frag", cell=st.cell, space_group_hm=sg, atoms=atoms
        )
        frag_cif = out_dir / f"_tmp_strat_{tag}_frag.cif"
        _write_minimal_cif(frag_cif, frag_st)
        _ = fmeta

    # Shorter budget for large cells
    hard = vol > 3500 or len(els) > 40
    n_iter_auto = 40 if hard else 50
    n_iter_seed = 50 if hard else 70
    n_extend = 12 if hard else 18

    configs: List[Tuple[str, SolveConfig]] = [
        (
            "auto",
            SolveConfig(
                method="auto",
                d_min=d_min,
                n_iter=n_iter_auto,
                n_starts=n_starts,
                n_extend=12,
                verbose=False,
                seed=0,
                compute_uncertainty=False,
            ),
        ),
        (
            "partial_15",
            SolveConfig(
                method="partial_phaseed",
                phase_seed_csv=str(seed15),
                d_min=d_min,
                n_iter=n_iter_seed,
                n_extend=n_extend,
                n_starts=n_starts,
                prior_weight=0.40,
                verbose=False,
                seed=0,
                compute_uncertainty=False,
            ),
        ),
        (
            "partial_30",
            SolveConfig(
                method="partial_phaseed",
                phase_seed_csv=str(seed30),
                d_min=d_min,
                n_iter=n_iter_seed,
                n_extend=n_extend + 2,
                n_starts=n_starts,
                prior_weight=0.45,
                verbose=False,
                seed=0,
                compute_uncertainty=False,
            ),
        ),
    ]
    if frag_cif is not None:
        configs.append(
            (
                "fragment_half",
                SolveConfig(
                    method="partial_phaseed",
                    predicted_model_cif=str(frag_cif),
                    expand_model_symmetry=True,
                    d_min=d_min,
                    n_iter=n_iter_seed + 20,
                    n_extend=n_extend + 8,
                    n_starts=n_starts,
                    prior_weight=0.52,
                    dm_ai_weight=0.42,
                    verbose=False,
                    seed=0,
                    compute_uncertainty=False,
                ),
            )
        )

    rows: List[Dict] = []
    base_meta = {
        "dataset": label,
        "amp_mode": amp_mode,
        "vol": vol,
        "vol_band": _vol_band(vol),
        "max_Z": max_z,
        "ha_bearing": ha,
        "n_nonh": len(els),
        "space_group": sg,
        "d_min": d_min,
        "n_refl": int(len(hkl)),
        "frac_truth_mapped": float(frac_mapped),
        "fragment_n_atoms": n_frag,
    }

    for name, cfg in configs:
        print(f"  {tag} / {name} …", flush=True)
        t0 = time.time()
        try:
            res = solve_structure(
                str(tmp),
                cell=cell_csv,
                space_group=sg,
                config=cfg,
            )
            rho_t = density_from_structure_factors(
                hkl,
                amp * np.exp(1j * ph_true),
                st.cell,
                shape=res.density.shape,
                d_min=d_min,
            )
            cc, _ = map_correlation_origin_invariant(res.density, rho_t)
            try:
                rep = evaluate_success(
                    hkl,
                    amp,
                    res.phases,
                    ph_true,
                    st.cell,
                    fdata["fracs"],
                    density=res.density,
                    elements=fdata["elements"],
                    thresholds=SuccessThresholds(),
                )
                solved = bool(rep.solved)
                r1 = float(rep.r1)
                peak = float(rep.peak_recovery)
            except Exception:
                solved, r1, peak = False, float("nan"), float("nan")
            row = {
                **base_meta,
                "run": name,
                "method": res.method,
                "mapcc_oi": float(cc),
                "free_fom": res.diagnostics.get("free_fom_composite"),
                "r1": r1,
                "peak_recovery": peak,
                "solved": solved,
                "seconds": time.time() - t0,
                "seed_source": res.diagnostics.get("seed_source"),
            }
            print(
                f"    mapCC={row['mapcc_oi']:.3f} FOM={row['free_fom']} "
                f"solved={solved} t={row['seconds']:.1f}s",
                flush=True,
            )
        except Exception as e:
            row = {**base_meta, "run": name, "error": str(e), "seconds": time.time() - t0}
            print(f"    ERROR {e}", flush=True)
        rows.append(row)
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description="COD Vol-band stratified bench")
    p.add_argument("--dmin", type=float, default=1.0, help="Resolution cutoff (Å)")
    p.add_argument("--fcalc-only", action="store_true", help="Skip experimental Fobs")
    p.add_argument("--fobs-only", action="store_true", help="Skip Fcalc controls")
    p.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated COD IDs (default: all with CIF under data/raw/cod)",
    )
    p.add_argument("--skip-large", action="store_true", help="Skip vol_gt_3500 (e.g. 2017775)")
    args = p.parse_args()

    cod_dir = ROOT / "data" / "raw" / "cod"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    d_min = float(args.dmin)

    if args.ids.strip():
        cifs = [cod_dir / f"{i.strip()}.cif" for i in args.ids.split(",") if i.strip()]
    else:
        cifs = sorted(cod_dir.glob("*.cif")) if cod_dir.is_dir() else []

    all_rows: List[Dict] = []
    catalog: List[Dict] = []

    if not cifs:
        print("No COD CIFs under data/raw/cod/")

    for cif in cifs:
        if not cif.exists():
            print(f"skip missing {cif}")
            continue
        label = cif.stem
        print(f"=== {label} ===", flush=True)
        try:
            st = load_cif(str(cif))
        except Exception as e:
            all_rows.append({"dataset": label, "error": f"cif: {e}"})
            continue
        vol = float(unit_cell_volume(np.asarray(st.cell, dtype=np.float64)))
        band = _vol_band(vol)
        if args.skip_large and band == "vol_gt_3500":
            print(f"  skip large vol {vol:.0f}", flush=True)
            continue
        els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
        catalog.append(
            {
                "dataset": label,
                "vol": vol,
                "vol_band": band,
                "max_Z": max_Z_from_elements(els),
                "n_nonh": len(els),
                "space_group": st.space_group_hm,
                "has_hkl": (cod_dir / f"{label}.hkl").exists(),
            }
        )
        print(
            f"  Vol={vol:.0f} Å³ ({band}) n_nonH={len(els)} "
            f"maxZ={max_Z_from_elements(els)} SG={st.space_group_hm}",
            flush=True,
        )

        # --- Fcalc control ---
        if not args.fobs_only:
            data = structure_to_fcalc(st, d_min=d_min)
            hkl_c, amp_c, ph_c = data["hkl"], data["amplitudes"], data["phases"]
            all_rows.extend(
                run_dataset(
                    label,
                    st,
                    hkl_c,
                    amp_c,
                    ph_c,
                    data,
                    1.0,
                    "fcalc",
                    d_min,
                    out_dir,
                    include_fragment=True,
                )
            )

        # --- Experimental Fobs ---
        hkl_path = cod_dir / f"{label}.hkl"
        if not args.fcalc_only and hkl_path.exists():
            try:
                table, _ = load_experiment(
                    str(hkl_path),
                    cell=",".join(str(x) for x in st.cell),
                    space_group=st.space_group_hm,
                )
                d = d_spacing(table.hkl, table.cell)
                keep = d >= (d_min - 1e-9)
                hkl_o = table.hkl[keep]
                amp_o = table.amplitudes[keep]
                ph_t, frac_m, fdata = match_truth_phases(hkl_o, st, d_min)
                all_rows.extend(
                    run_dataset(
                        label,
                        st,
                        hkl_o,
                        amp_o,
                        ph_t,
                        fdata,
                        frac_m,
                        "fobs",
                        d_min,
                        out_dir,
                        include_fragment=True,
                    )
                )
            except Exception as e:
                all_rows.append(
                    {"dataset": label, "amp_mode": "fobs", "error": str(e)}
                )
                print(f"  Fobs failed: {e}", flush=True)

    # Aggregations
    ok = [r for r in all_rows if "mapcc_oi" in r]
    summary = {
        "by_vol_band_run_amp": _agg(
            ok, lambda r: f"{r['vol_band']}/{r['run']}/{r['amp_mode']}"
        ),
        "by_vol_band_run": _agg(ok, lambda r: f"{r['vol_band']}/{r['run']}"),
        "by_run_amp": _agg(ok, lambda r: f"{r['run']}/{r['amp_mode']}"),
        "by_run": _agg(ok, lambda r: r["run"]),
        "by_vol_band": _agg(ok, lambda r: r["vol_band"]),
    }

    # Mid-band focus (AI-PhaSeed / Carrozzini hybrid-friendly)
    mid = [r for r in ok if r.get("vol_band") == "vol_1000_3500"]
    mid_summary = _agg(mid, lambda r: f"{r['run']}/{r['amp_mode']}")

    payload = {
        "version": "0.13.0",
        "d_min": d_min,
        "catalog": catalog,
        "rows": all_rows,
        "summary": summary,
        "vol_1000_3500_focus": mid_summary,
        "n_ok": len(ok),
        "n_datasets": len(catalog),
        "note": (
            "Local COD Vol-band panel (Fobs + Fcalc). Not a 1505-structure "
            "Carrozzini panel. partial_30 = oracle control; fragment_half = "
            "no-oracle scientist path; auto = ab initio. Strict success = "
            "mapCC_OI≥0.7 AND peak_recovery≥0.5 AND R1≤0.45."
        ),
    }
    jp = out_dir / "cod_stratified_bench.json"
    jp.write_text(json.dumps(payload, indent=2, default=str))

    # Markdown scoreboard
    md: List[str] = [
        "# COD Vol-band stratified bench (v0.13)",
        "",
        payload["note"],
        "",
        f"**d_min** = {d_min} Å · **datasets** = {len(catalog)} · **rows OK** = {len(ok)}",
        "",
        "## Catalog",
        "",
        "| COD | Vol (Å³) | Band | max Z | n non-H | SG | HKL |",
        "|-----|----------|------|-------|---------|----|-----|",
    ]
    for c in catalog:
        md.append(
            f"| {c['dataset']} | {c['vol']:.0f} | `{c['vol_band']}` | {c['max_Z']} | "
            f"{c['n_nonh']} | {c['space_group']} | {'yes' if c['has_hkl'] else 'no'} |"
        )

    md.extend(
        [
            "",
            "## Results (all runs)",
            "",
            "| Dataset | Amp | Run | mapCC | free FOM | R1 | solved | Vol band | max Z | s |",
            "|---------|-----|-----|-------|----------|----|--------|----------|-------|---|",
        ]
    )
    for r in all_rows:
        if "error" in r and "mapcc_oi" not in r:
            md.append(
                f"| {r.get('dataset')} | {r.get('amp_mode','')} | {r.get('run','')} | "
                f"ERROR | | | | {r.get('vol_band','')} | {r.get('max_Z','')} | |"
            )
            continue
        fom = r.get("free_fom")
        fom_s = f"{fom:.3f}" if isinstance(fom, (int, float)) and fom is not None else "—"
        r1 = r.get("r1")
        r1_s = f"{r1:.2f}" if isinstance(r1, (int, float)) and r1 == r1 else "—"
        md.append(
            f"| {r.get('dataset')} | {r.get('amp_mode')} | {r.get('run')} | "
            f"**{float(r.get('mapcc_oi', float('nan'))):.3f}** | {fom_s} | {r1_s} | "
            f"{r.get('solved')} | `{r.get('vol_band')}` | {r.get('max_Z')} | "
            f"{float(r.get('seconds', 0)):.1f} |"
        )

    md.extend(["", "## Summary by Vol band × run", ""])
    for k, v in summary["by_vol_band_run"].items():
        md.append(f"- `{k}`: n={v['n']} mean mapCC=**{v['mean_mapcc']:.3f}** (median {v['median_mapcc']:.3f})")

    md.extend(
        [
            "",
            "## Vol 1000–3500 Å³ focus (AI-PhaSeed hybrid-friendly band)",
            "",
        ]
    )
    if mid_summary:
        md.append("| Run/amp | n | mean mapCC | median |")
        md.append("|---------|---|------------|--------|")
        for k, v in mid_summary.items():
            md.append(
                f"| `{k}` | {v['n']} | **{v['mean_mapcc']:.3f}** | {v['median_mapcc']:.3f} |"
            )
    else:
        md.append("_No structures in Vol 1000–3500 in this panel._")

    md.extend(
        [
            "",
            "## Takeaways",
            "",
            "- **auto** (ab initio) is typically weak; mapCC often ≪ 0.5 on hard cells.",
            "- **partial_30** (oracle) is the Lane-B control for the ≥~30% strong-φ bar.",
            "- **partial_15** often under-seeds vs that bar.",
            "- **fragment_half** is the no-oracle path; on coherent half-models it should "
            "approach partial_30 mapCC (see also `cod_hard_path_validation.md`).",
            "- Vol **1000–3500 Å³** is the Carrozzini / AI-PhaSeed hybrid-friendly band.",
            "- Strict multi-criterion *solved* can fail on R1 under short budgets.",
            "",
            "Regenerate:",
            "```bash",
            "python scripts/run_cod_stratified_bench.py --dmin 1.0",
            "```",
            "",
        ]
    )
    mp = out_dir / "cod_stratified_bench.md"
    mp.write_text("\n".join(md))
    print(f"Wrote {jp}", flush=True)
    print(f"Wrote {mp}", flush=True)


if __name__ == "__main__":
    main()
