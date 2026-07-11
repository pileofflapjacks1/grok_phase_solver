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
            "recycle",
            "direct_methods",
            "hio",
        ],
        help="Phasing method (default: auto = pick best available pipeline)",
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


def main(argv: list[str] | None = None) -> None:
    """Multi-command dispatcher: gps solve | baseline | download-cod"""
    # Allow `python -m grok_phase_solver.cli solve ...`
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: gps-solve | gps-baseline | gps-download-cod  (see --help on each)")
        print("  or:  python -m grok_phase_solver.cli solve --hkl ...")
        sys.exit(0)
    cmd = argv[0]
    rest = argv[1:]
    if cmd in ("solve", "gps-solve"):
        solve_main(rest)
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
