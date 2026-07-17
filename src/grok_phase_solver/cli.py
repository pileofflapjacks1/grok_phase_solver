"""Command-line entry points for researchers and experimentalists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def download_cod_main(argv: list[str] | None = None) -> None:
    from grok_phase_solver.data.cod import download_phase1_samples, download_cod_cif, download_cod_hkl

    p = argparse.ArgumentParser(description="Download COD CIF/HKL samples")
    p.add_argument("--id", type=str, default=None, help="COD ID (default: Phase-1 set)")
    p.add_argument("--dest", type=str, default="data/raw/cod")
    p.add_argument("--hkl", action="store_true", help="Also try HKL download")
    args = p.parse_args(argv)

    dest = Path(args.dest)
    if args.id:
        path = download_cod_cif(args.id, dest_dir=dest)
        print(f"Downloaded {path}")
        if args.hkl:
            try:
                h = download_cod_hkl(args.id, dest_dir=dest)
                print(f"Downloaded {h}")
            except Exception as e:
                print(f"HKL download failed: {e}", file=sys.stderr)
    else:
        paths = download_phase1_samples(dest_dir=dest)
        for path in paths:
            print(f"Downloaded {path}")


def baseline_main(argv: list[str] | None = None) -> None:
    from grok_phase_solver.io.cif import load_cif
    from grok_phase_solver.solvers.baseline import run_physics_baseline
    from grok_phase_solver.data.synthetic import generate_random_organic

    p = argparse.ArgumentParser(description="Run physics baseline (synthetic/CIF ground-truth tests)")
    p.add_argument("--cif", type=str, default=None, help="Path to CIF")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic structure")
    p.add_argument("--n-atoms", type=int, default=10)
    p.add_argument(
        "--method",
        type=str,
        default="charge_flipping",
        choices=["charge_flipping", "hio", "random", "direct_methods", "patterson"],
    )
    p.add_argument("--dmin", type=float, default=1.2)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--completeness", type=float, default=1.0)
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.synthetic or args.cif is None:
        structure = generate_random_organic(n_atoms=args.n_atoms, seed=args.seed)
        print(structure.summary())
    else:
        structure = load_cif(args.cif)
        print(structure.summary())

    result = run_physics_baseline(
        structure,
        method=args.method,
        d_min=args.dmin,
        n_iter=args.n_iter,
        seed=args.seed,
        noise_level=args.noise,
        completeness=args.completeness,
        verbose=not args.quiet,
    )
    print(result.summary())

    if args.json_out:
        payload = {
            "name": result.name,
            "method": result.method,
            "d_min": result.d_min,
            "n_reflections": result.n_reflections,
            "n_atoms_cell": result.n_atoms_cell,
            "mean_phase_error_deg": result.mean_phase_error_deg,
            "mean_phase_error_origin_invariant_deg": result.mean_phase_error_origin_invariant_deg,
            "map_cc": result.map_cc,
            "r_factor": result.r_factor,
            "notes": result.notes,
            "final_R_history": (result.history.get("R") or [None])[-1],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json_out}")


def solve_main(argv: list[str] | None = None) -> None:
    """
    End-user entry: phase experimental data and write a results folder.

    Examples
    --------
    gps-solve --hkl data.hkl --ins data.ins --out results/
    gps-solve --hkl data.hkl --cell 10,12,15,90,98,90 --sg "P21/c" --out results/
    gps-solve --hkl data.hkl --ins data.ins --method charge_flipping --n-iter 200
    """
    from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
    from grok_phase_solver.pipeline.export import export_solution

    p = argparse.ArgumentParser(
        prog="gps-solve",
        description=(
            "Solve (phase) a crystal structure from experimental amplitudes. "
            "Provide SHELX .hkl + .ins, or .hkl + cell/space group."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gps-solve --hkl mydata.hkl --ins mydata.ins --out ./solve_out
  gps-solve --hkl mydata.hkl --cell 9.7,8.9,7.6,90,112.7,90 --sg "P 1 21/c 1" --out ./out
  gps-solve --hkl reflections.hkl --ins job.ins --method auto --n-iter 150

Outputs (in --out):
  report.md, phases.csv, structure_factors.F, density.npz, peaks.csv, peaks.xyz, ...

Next: inspect density_slice.png / peaks, then refine in SHELXL or Olex2.
        """,
    )
    p.add_argument("--hkl", required=True, help="Reflection file (.hkl SHELX, CIF HKL, or .mtz)")
    p.add_argument("--ins", default=None, help="SHELX .ins/.res with CELL and SYMM (recommended)")
    p.add_argument(
        "--cell",
        default=None,
        help="Unit cell a,b,c,alpha,beta,gamma (if no .ins / cell in file)",
    )
    p.add_argument("--sg", "--space-group", dest="sg", default=None, help='Space group, e.g. "P21/c"')
    p.add_argument("--wavelength", type=float, default=None, help="Wavelength Å (optional)")
    p.add_argument(
        "--method",
        default="auto",
        choices=[
            "auto",
            "charge_flipping",
            "ensemble",
            "raar",
            "phai",
            "phai+cf",
            "phai+cf_cond",
            "phai_phaseed",
            "phai+recycle",
            "hard_p1_phaseed",
            "strong_prior_phaseed",
            "partial_phaseed",
            "fragment_phaseed",
            "ha_phaseed",
            "recycle",
            "direct_methods",
            "hio",
            "dual_space",
            "shelxd",
            "shelxd_or_dual",
            "shelxs",
            "shelxs+shelxe",
        ],
        help=(
            "Phasing method (default: auto = ensemble on easy, priors/CF on hard; "
            "if a seed source is given, auto → partial_phaseed)"
        ),
    )
    p.add_argument("--dmin", type=float, default=None, help="High-resolution cutoff Å (optional)")
    p.add_argument("--n-iter", type=int, default=120, help="Iterations for CF/HIO/polish")
    p.add_argument("--n-recycle", type=int, default=8, help="Cycles for recycle/PhAI")
    p.add_argument("--n-extend", type=int, default=12, help="AI-PhaSeed extension cycles")
    p.add_argument("--n-starts", type=int, default=2, help="Multistart trials (ensemble / PhaSeed)")
    p.add_argument("--n-peaks", type=int, default=40, help="Max density peaks to list")
    p.add_argument("--min-peak-sigma", type=float, default=2.5, help="Min peak height in map σ")
    p.add_argument(
        "--solvent-fraction",
        type=float,
        default=None,
        help="Optional solvent flatten fraction after phasing (e.g. 0.4)",
    )
    p.add_argument(
        "--phase-seed-csv",
        default=None,
        help="Partial-φ CSV (h,k,l,phase_deg) for partial_phaseed",
    )
    p.add_argument(
        "--phase-seed-res",
        default=None,
        help="SHELXS/SHELXL .res atoms → Fcalc phase seed (fragment path)",
    )
    p.add_argument(
        "--seed-atoms-csv",
        default=None,
        help="Fragment atoms CSV: x,y,z,element (fractional)",
    )
    p.add_argument(
        "--seed-peaks-csv",
        default=None,
        help="Prior gps-solve peaks.csv → light-atom Fcalc seed",
    )
    p.add_argument(
        "--seed-element",
        default="C",
        help="Element for peaks.csv seeds (default C)",
    )
    p.add_argument(
        "--seed-n-atoms",
        type=int,
        default=None,
        help="Max atoms from .res / peaks / atoms CSV",
    )
    p.add_argument(
        "--seed-b-iso",
        type=float,
        default=8.0,
        help="B_iso for Fcalc fragment seeds (default 8)",
    )
    p.add_argument(
        "--export-seed-csv",
        default=None,
        help="Write the mapped seed phases to this CSV path",
    )
    p.add_argument(
        "--native-hkl",
        default=None,
        help="Native amplitudes for isomorphous HA seed (with --derivative-hkl)",
    )
    p.add_argument(
        "--derivative-hkl",
        default=None,
        help="Derivative amplitudes for isomorphous HA seed",
    )
    p.add_argument(
        "--n-ha",
        type=int,
        default=1,
        help="Number of heavy-atom sites to place (HA seed)",
    )
    p.add_argument(
        "--ha-element",
        default="Br",
        help="Heavy-atom element for HA seed (default Br)",
    )
    p.add_argument(
        "--patterson-ha",
        action="store_true",
        help="Single-dataset Patterson→HA heuristic seed (weak; for HA-containing data)",
    )
    p.add_argument(
        "--seed-fraction",
        type=float,
        default=0.30,
        help="Strong-seed fraction for PhaSeed-style methods (default 0.30)",
    )
    p.add_argument(
        "--shelxe-polish",
        action="store_true",
        help="After shelxs, run SHELXE density mod (or use method shelxs+shelxe)",
    )
    p.add_argument(
        "--shelxe-cycles",
        type=int,
        default=15,
        help="SHELXE -m density-modification cycles (default 15)",
    )
    p.add_argument(
        "--shelxe-solvent",
        type=float,
        default=0.45,
        help="SHELXE -s solvent fraction (default 0.45)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", "-o", default="./gps_solve_out", help="Output directory")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = SolveConfig(
        method=args.method,
        d_min=args.dmin,
        n_iter=args.n_iter,
        n_recycle=args.n_recycle,
        n_extend=args.n_extend,
        n_starts=args.n_starts,
        seed=args.seed,
        n_peaks=args.n_peaks,
        min_peak_sigma=args.min_peak_sigma,
        solvent_fraction=args.solvent_fraction,
        verbose=not args.quiet,
        phase_seed_csv=args.phase_seed_csv,
        phase_seed_res=args.phase_seed_res,
        seed_atoms_csv=args.seed_atoms_csv,
        seed_peaks_csv=args.seed_peaks_csv,
        seed_element=args.seed_element,
        seed_n_atoms=args.seed_n_atoms,
        seed_b_iso=args.seed_b_iso,
        seed_fraction=args.seed_fraction,
        export_seed_csv=args.export_seed_csv,
        native_hkl=args.native_hkl,
        derivative_hkl=args.derivative_hkl,
        n_ha=args.n_ha,
        ha_element=args.ha_element,
        patterson_ha=args.patterson_ha,
        shelxe_polish=args.shelxe_polish,
        shelxe_cycles=args.shelxe_cycles,
        shelxe_solvent=args.shelxe_solvent,
    )

    try:
        result = solve_structure(
            hkl_path=args.hkl,
            ins_path=args.ins,
            cell=args.cell,
            space_group=args.sg,
            wavelength=args.wavelength,
            config=cfg,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    paths = export_solution(result, out)
    print("\n=== Done ===")
    print(f"Method used: {result.method}")
    print(f"Results written to: {out.resolve()}")
    for path in paths:
        print(f"  - {path.name}")
    print(f"\nRead {out / 'report.md'} for interpretation and next steps.")


def make_seed_main(argv: list[str] | None = None) -> None:
    """
    Convert lab artifacts → phase-seed CSV without a full solve.

    Examples
    --------
    gps-make-seed --hkl data.hkl --ins data.ins --from-res model.res -o seed.csv
    gps-make-seed --hkl data.hkl --cell ... --from-peaks peaks.csv -o seed.csv
    """
    from grok_phase_solver.io.experiment import load_experiment
    from grok_phase_solver.solvers.seed_import import (
        assess_seed_quality,
        export_seed_csv,
        resolve_phase_seed,
    )

    p = argparse.ArgumentParser(
        prog="gps-make-seed",
        description=(
            "Build a partial-φ seed CSV from .res atoms, peaks.csv, atoms CSV, "
            "or isomorphous HA pair — for use with gps-solve --phase-seed-csv."
        ),
    )
    p.add_argument("--hkl", required=True, help="Reflection file (for indexing)")
    p.add_argument("--ins", default=None)
    p.add_argument("--cell", default=None)
    p.add_argument("--sg", default=None)
    p.add_argument("--from-res", default=None, help="SHELX .res / .ins atoms")
    p.add_argument("--from-peaks", default=None, help="peaks.csv")
    p.add_argument("--from-atoms", default=None, help="x,y,z,element CSV")
    p.add_argument("--from-phases", default=None, help="existing phase CSV (re-map)")
    p.add_argument("--native-hkl", default=None)
    p.add_argument("--derivative-hkl", default=None)
    p.add_argument("--patterson-ha", action="store_true")
    p.add_argument("--seed-element", default="C")
    p.add_argument("--seed-n-atoms", type=int, default=None)
    p.add_argument("--n-ha", type=int, default=1)
    p.add_argument("--ha-element", default="Br")
    p.add_argument("--seed-b-iso", type=float, default=8.0)
    p.add_argument("-o", "--out", default="phase_seed.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    table, _ = load_experiment(
        args.hkl, ins=args.ins, cell=args.cell, space_group=args.sg
    )
    hkl, amp, cell = table.hkl, table.amplitudes, table.cell
    assert cell is not None

    native_amp = deriv_amp = None
    if args.native_hkl and args.derivative_hkl:
        t_n, _ = load_experiment(args.native_hkl, ins=args.ins, cell=args.cell, space_group=args.sg)
        t_d, _ = load_experiment(args.derivative_hkl, ins=args.ins, cell=args.cell, space_group=args.sg)
        key_n = {(int(r[0]), int(r[1]), int(r[2])): float(a) for r, a in zip(t_n.hkl, t_n.amplitudes)}
        key_d = {(int(r[0]), int(r[1]), int(r[2])): float(a) for r, a in zip(t_d.hkl, t_d.amplitudes)}
        native_amp = np.array(
            [key_n.get((int(r[0]), int(r[1]), int(r[2])), 0.0) for r in hkl]
        )
        deriv_amp = np.array(
            [key_d.get((int(r[0]), int(r[1]), int(r[2])), 0.0) for r in hkl]
        )

    try:
        seed_ph, mask, meta = resolve_phase_seed(
            hkl,
            amp,
            cell,
            phase_seed_csv=args.from_phases,
            phase_seed_res=args.from_res,
            seed_atoms_csv=args.from_atoms,
            seed_peaks_csv=args.from_peaks,
            seed_element=args.seed_element,
            seed_n_atoms=args.seed_n_atoms,
            seed_b_iso=args.seed_b_iso,
            native_amp=native_amp,
            derivative_amp=deriv_amp,
            n_ha=args.n_ha,
            ha_element=args.ha_element,
            use_patterson_ha=args.patterson_ha,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    export_seed_csv(out, hkl, seed_ph, mask)
    qual = assess_seed_quality(hkl, amp, cell, seed_ph, mask)
    print(f"Wrote {out.resolve()}  ({int(mask.sum())} reflections)")
    print(
        f"  source={meta.get('source', meta.get('kind'))}  "
        f"frac_strong_seeded={qual['frac_strong_seeded']:.0%}  "
        f"size_meets_bar={qual['size_meets_bar']}"
    )
    for h in qual.get("hints") or []:
        print(f"  note: {h}")


def main(argv: list[str] | None = None) -> None:
    """Multi-command dispatcher: gps solve | baseline | download-cod | make-seed"""
    # Allow `python -m grok_phase_solver.cli solve ...`
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(
            "Usage: gps-solve | gps-make-seed | gps-baseline | gps-download-cod  "
            "(see --help on each)"
        )
        print("  or:  python -m grok_phase_solver.cli solve --hkl ...")
        sys.exit(0)
    cmd = argv[0]
    rest = argv[1:]
    if cmd in ("solve", "gps-solve"):
        solve_main(rest)
    elif cmd in ("make-seed", "gps-make-seed"):
        make_seed_main(rest)
    elif cmd in ("baseline", "gps-baseline"):
        baseline_main(rest)
    elif cmd in ("download-cod", "gps-download-cod"):
        download_cod_main(rest)
    else:
        # treat full argv as solve if starts with --hkl
        if cmd.startswith("-"):
            solve_main(argv)
        else:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            sys.exit(2)
