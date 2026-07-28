#!/usr/bin/env python3
"""
Experimental COD hard-path validation (post v0.7.0).

For each COD sample with CIF + HKL:
  1. Match Fcalc truth phases onto experimental Fobs reflections
  2. Build oracle 30% / 15% strong-|E| seed CSVs
  3. Optional: fragment seed from first N deposited atoms
  4. Run auto vs partial_phaseed; report free FOM + mapCC_OI vs Fcalc truth

Writes:
  data/processed/cod_hard_path_validation.{json,md}
  examples/partial_seed_demo/COD_HARD_PATH.md (short pointer)

Does not claim protein ab initio solution — documents partial-φ on real Fobs.
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
from grok_phase_solver.io.experiment import load_experiment
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.pipeline.export import export_solution
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.partial_seed import oracle_partial_seed, write_phase_seed_csv
from grok_phase_solver.solvers.seed_import import seed_from_fragment_atoms, export_seed_csv


def _write_minimal_cif(path: Path, st) -> None:
    """Write a tiny gemmi/CIF-compatible atom list for predicted-model seeding."""
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


def write_oracle_csv(path: Path, hkl, amp, cell, ph_true, fraction: float):
    seed_ph, mask, meta = oracle_partial_seed(
        hkl, amp, cell, ph_true, fraction=fraction, phase_noise_deg=0.0
    )
    write_phase_seed_csv(path, hkl, seed_ph, mask)
    return meta


def run_one(label, hkl_path, cell, sg, cif_path, d_min, methods_cfg):
    rows = []
    st = load_cif(str(cif_path))
    table, _ = load_experiment(str(hkl_path), cell=cell, space_group=sg)
    # apply d_min cut
    from grok_phase_solver.physics.reciprocal import d_spacing

    d = d_spacing(table.hkl, table.cell)
    keep = d >= (d_min - 1e-9)
    hkl = table.hkl[keep]
    amp = table.amplitudes[keep]
    # write temp hkl for pipeline
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    from grok_phase_solver.io.hkl import ReflectionTable, write_hkl_simple

    tmp_hkl = proc / f"_tmp_hardpath_{label.replace(' ', '_')}.hkl"
    write_hkl_simple(
        tmp_hkl,
        ReflectionTable(hkl=hkl, F_meas=amp, cell=table.cell, space_group_hm=sg),
    )

    ph_true, frac_mapped, fdata = match_truth_phases(hkl, st, d_min)
    cell_csv = ",".join(str(x) for x in table.cell)

    # seeds
    seed30 = proc / f"_tmp_hardpath_{label}_oracle30.csv"
    seed15 = proc / f"_tmp_hardpath_{label}_oracle15.csv"
    m30 = write_oracle_csv(seed30, hkl, amp, table.cell, ph_true, 0.30)
    m15 = write_oracle_csv(seed15, hkl, amp, table.cell, ph_true, 0.15)

    # Fragment CIF: heaviest-cluster half of non-H ASU (pipeline expands SG)
    fracs = np.array(
        [a.fract for a in st.atoms if a.element.upper() not in ("H", "D")],
        dtype=np.float64,
    )
    els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
    n_frag = max(3, len(els) // 2)
    seed_frag_cif = proc / f"_tmp_hardpath_{label}_frag.cif"
    fmeta = {}
    if len(fracs) >= 3:
        from grok_phase_solver.solvers.seed_import import select_fragment_atoms
        from grok_phase_solver.io.cif import AtomSite, CrystalStructure

        fr_sel, el_sel, fmeta = select_fragment_atoms(
            fracs, els, max_atoms=n_frag, mode="heaviest_cluster", seed=0
        )
        atoms = [
            AtomSite(label=f"{el}{i+1}", element=el, fract=fr_sel[i], b_iso=10.0)
            for i, el in enumerate(el_sel)
        ]
        frag_st = CrystalStructure(
            name=f"{label}_frag",
            cell=st.cell,
            space_group_hm=st.space_group_hm,
            atoms=atoms,
        )
        # minimal CIF write
        _write_minimal_cif(seed_frag_cif, frag_st)
        # diagnostic seed stats
        sph, mask, smeta = seed_from_fragment_atoms(
            hkl, amp, table.cell, fr_sel, el_sel,
            expand_symmetry=True, space_group=sg or st.space_group_hm,
            prefer_strong_E=True, fcalc_min_rel=0.06, target_strong_frac=0.32,
            full_fcalc_prior=True, auto_b_iso=True,
            seed=0,
        )
        fmeta = {**fmeta, **smeta}
        print(
            f"  fragment seed: n_asu={fmeta.get('n_selected')} "
            f"n_atoms_after_expand={smeta.get('n_atoms')} "
            f"expanded={smeta.get('expand', {}).get('expanded')} "
            f"n_seed={int(mask.sum())} frac_strong={smeta.get('frac_strong_seeded')}",
            flush=True,
        )
    else:
        seed_frag_cif = None

    configs = [
        ("auto", SolveConfig(method="auto", d_min=d_min, n_iter=60, n_starts=2, n_extend=12, verbose=False, seed=0, compute_uncertainty=False)),
        ("partial_15", SolveConfig(method="partial_phaseed", phase_seed_csv=str(seed15), d_min=d_min, n_iter=80, n_starts=2, n_extend=18, verbose=False, seed=0, compute_uncertainty=False, prior_weight=0.40)),
        ("partial_30", SolveConfig(method="partial_phaseed", phase_seed_csv=str(seed30), d_min=d_min, n_iter=80, n_starts=2, n_extend=20, verbose=False, seed=0, compute_uncertainty=False, prior_weight=0.45)),
    ]
    if seed_frag_cif is not None:
        configs.append(
            ("fragment_half", SolveConfig(
                method="partial_phaseed",
                predicted_model_cif=str(seed_frag_cif),
                expand_model_symmetry=True,
                d_min=d_min,
                n_iter=100,
                n_starts=2,
                n_extend=28,
                verbose=False,
                seed=0,
                compute_uncertainty=False,
                prior_weight=0.52,
                dm_ai_weight=0.42,
            ))
        )

    for name, cfg in configs:
        print(f"  {label} / {name} …", flush=True)
        t0 = time.time()
        try:
            result = solve_structure(
                str(tmp_hkl),
                cell=cell_csv,
                space_group=sg or st.space_group_hm,
                config=cfg,
            )
        except Exception as e:
            rows.append({"dataset": label, "run": name, "error": str(e)})
            print(f"    ERROR {e}", flush=True)
            continue
        rho = result.density
        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_true), table.cell, shape=rho.shape, d_min=d_min
        )
        cc, _ = map_correlation_origin_invariant(rho, rho_t)
        try:
            rep = evaluate_success(
                hkl, amp, result.phases, ph_true, table.cell,
                fdata["fracs"], density=rho, elements=fdata["elements"],
                thresholds=SuccessThresholds(),
            )
            solved = bool(rep.solved)
            r1 = float(rep.r1)
            peak = float(rep.peak_recovery)
        except Exception:
            solved, r1, peak = False, float("nan"), float("nan")
        row = {
            "dataset": label,
            "run": name,
            "method": result.method,
            "d_min": d_min,
            "n_refl": int(len(hkl)),
            "frac_truth_mapped": float(frac_mapped),
            "free_fom": result.diagnostics.get("free_fom_composite"),
            "mapcc_oi": float(cc),
            "r1": r1,
            "peak_recovery": peak,
            "solved": solved,
            "seconds": time.time() - t0,
            "seed_source": result.diagnostics.get("seed_source"),
            "oracle30_n": int(m30.get("n_known", 0)) if isinstance(m30, dict) else None,
            "fragment_n_atoms": n_frag if seed_frag_cif else 0,
        }
        rows.append(row)
        flag = "SOLVED" if solved else "fail"
        print(
            f"    {flag} FOM={row['free_fom']:.3f} mapCC={row['mapcc_oi']:.3f} "
            f"R1={row['r1']:.2f} t={row['seconds']:.1f}s",
            flush=True,
        )
    return rows


def main():
    cod = ROOT / "data" / "raw" / "cod"
    panels = [
        ("COD_2016452_exp", cod / "2016452.hkl", cod / "2016452.cif", "9.748,8.89,7.566,90,112.74,90", "P 1 21/c 1", 1.0),
        ("COD_2100301_exp", cod / "2100301.hkl", cod / "2100301.cif", None, None, 1.0),
    ]
    all_rows = []
    print("=== COD experimental hard-path validation ===", flush=True)
    for label, hklp, cifp, cell, sg, dmin in panels:
        if not hklp.exists() or not cifp.exists():
            print(f"skip {label}: missing files", flush=True)
            continue
        # load cell from cif if needed
        st = load_cif(str(cifp))
        if cell is None:
            cell = ",".join(f"{x:.6g}" for x in st.cell)
        if sg is None:
            sg = st.space_group_hm
        all_rows.extend(run_one(label, hklp, cell, sg, cifp, dmin, None))

    out_dir = ROOT / "data" / "processed"
    payload = {
        "version": "0.7.0",
        "rows": all_rows,
        "note": (
            "Experimental Fobs + oracle partial-φ from deposited CIF Fcalc. "
            "Documents hard-path value on real data; not a protein ab initio claim."
        ),
    }
    jp = out_dir / "cod_hard_path_validation.json"
    jp.write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# COD experimental hard-path validation",
        "",
        "Experimental **Fobs** with mapCC vs deposited-model Fcalc phases.",
        "Oracle partial seeds use true phases on strong |E| only (Lane B control).",
        "Fragment seed uses ~half of non-H atoms from the deposited CIF.",
        "",
        "| Dataset | Run | Method | mapCC | free FOM | R1 | solved | s |",
        "|---------|-----|--------|-------|----------|----|--------|---|",
    ]
    for r in all_rows:
        if "error" in r:
            lines.append(f"| {r['dataset']} | {r['run']} | ERROR | | | | | |")
            continue
        lines.append(
            f"| `{r['dataset']}` | `{r['run']}` | `{r['method']}` | "
            f"**{r['mapcc_oi']:.3f}** | {r['free_fom']:.3f} | {r['r1']:.2f} | "
            f"{r['solved']} | {r['seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "- Compare **auto** vs **partial_30** on each COD set.",
            "- **partial_15** often under-seeds vs the ~30% practical bar.",
            "- **fragment_half** (SG-expanded ~½ ASU + full Fcalc soft prior) is the "
            "no-oracle scientist path; with a coherent half-model it should "
            "**approach or match partial_30** mapCC on these COD cells.",
            "- Strict multi-criterion *solved* can still fail on R1 under short "
            "budget — honesty about residual polish.",
            "- Easy COD cases (e.g. 2016452) may already solve with PhAI hybrids; "
            "partial-φ / fragment is critical when ab initio fails.",
            "",
            "See also: `examples/partial_seed_demo/HARD_PATH_VALIDATION.md` (synthetic).",
            "",
        ]
    )
    mp = out_dir / "cod_hard_path_validation.md"
    mp.write_text("\n".join(lines) + "\n")
    # short pointer in examples
    ptr = ROOT / "examples" / "partial_seed_demo" / "COD_HARD_PATH.md"
    ptr.write_text(
        "\n".join(
            [
                "# COD experimental hard path",
                "",
                "See full table: [`data/processed/cod_hard_path_validation.md`](../../data/processed/cod_hard_path_validation.md).",
                "",
                "```bash",
                "python scripts/run_cod_hard_path_validation.py",
                "```",
                "",
            ]
        )
    )
    print(f"\nWrote {jp}")
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
