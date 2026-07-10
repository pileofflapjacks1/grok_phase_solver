"""Command-line entry points."""

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

    p = argparse.ArgumentParser(description="Run physics baseline phase retrieval")
    p.add_argument("--cif", type=str, default=None, help="Path to CIF")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic structure")
    p.add_argument("--n-atoms", type=int, default=10)
    p.add_argument("--method", type=str, default="charge_flipping",
                   choices=["charge_flipping", "hio", "random"])
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
