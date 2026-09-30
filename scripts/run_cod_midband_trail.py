#!/usr/bin/env python3
"""Scientist-facing COD mid-band Fobs + fragment_half → Olex2 trail.

Picks the in-repo 2012000 experimental HKL (Vol ~1027 Å³, P 1 21 1), writes a
heaviest-cluster ~½ non-H CIF from the deposited model (same selection as the
stratified Vol-band bench), runs ``partial_phaseed``, and exports ``trial.res``
plus ``olex2_handbuild.md``.

This is the no-oracle product path. **Do not invent mapCC** here — gps-solve
does not score against deposited phases on the scientist path. Free FOM is
from this run. mapCC / R1 / strict success are copied from the committed
scoreboard ``data/processed/cod_stratified_bench.md``.

``trial.res`` is Q peaks for Olex2 hand-build, not a SHELXL start.

Usage (from repo root)::

    python scripts/run_cod_midband_trail.py --out ./out_2012000_frag
    python scripts/run_cod_midband_trail.py --dry-run --out ./out_2012000_frag
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.io.cif import AtomSite, CrystalStructure, load_cif
from grok_phase_solver.pipeline.export import export_solution
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
from grok_phase_solver.solvers.projectors import unit_cell_volume
from grok_phase_solver.solvers.seed_import import select_fragment_atoms

COD_ID = "2012000"
# Copied from data/processed/cod_stratified_bench.md (2012000 / fobs / fragment_half).
# Do not recompute or invent these on the scientist path.
COMMITTED_SCOREBOARD = {
    "source": "data/processed/cod_stratified_bench.md",
    "dataset": COD_ID,
    "amp_mode": "fobs",
    "run": "fragment_half",
    "mapcc_oi": 0.675,
    "free_fom": 0.776,
    "r1": 0.53,
    "strict_success": False,
    "vol_band": "vol_1000_3500",
    "d_min": 1.0,
    "note": (
        "Strict success = mapCC_OI ≥ 0.7 AND peak recovery ≥ 0.5 AND R1 ≤ 0.45. "
        "This Fobs fragment_half row is not a strict solve (mapCC 0.675, R1 0.53). "
        "Mid-band mean fragment_half mapCC ~0.71 vs auto ~0.27 (C25)."
    ),
}


def write_minimal_cif(path: Path, st: CrystalStructure) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_fragment_half_cif(
    cif_path: Path,
    out_cif: Path,
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """Heaviest-cluster ~½ non-H ASU — same selection as the Vol-band bench."""
    st = load_cif(str(cif_path))
    nonh = [a for a in st.atoms if a.element.upper() not in ("H", "D")]
    els = [a.element for a in nonh]
    fracs = np.array([a.fract for a in nonh], dtype=np.float64)
    n_frag = max(3, min(len(els) // 2, 40))
    fr_sel, el_sel, fmeta = select_fragment_atoms(
        fracs, els, max_atoms=n_frag, mode="heaviest_cluster", seed=seed
    )
    atoms = [
        AtomSite(label=f"{el}{i + 1}", element=el, fract=fr_sel[i], b_iso=10.0)
        for i, el in enumerate(el_sel)
    ]
    frag = CrystalStructure(
        name=f"{st.name or cif_path.stem}_frag",
        cell=st.cell,
        space_group_hm=st.space_group_hm,
        atoms=atoms,
    )
    write_minimal_cif(out_cif, frag)
    vol = float(unit_cell_volume(np.asarray(st.cell, dtype=np.float64)))
    return {
        "n_nonh": len(els),
        "n_frag": int(len(atoms)),
        "space_group": st.space_group_hm,
        "cell": [float(x) for x in np.asarray(st.cell, dtype=np.float64)],
        "vol": vol,
        "fragment_cif": str(out_cif),
        "select_meta": {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in dict(fmeta).items()},
    }


def _repo_rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def gps_solve_command(hkl: Path, frag_cif: Path, out_dir: Path) -> str:
    return (
        f"gps-solve --hkl {_repo_rel(hkl)} \\\n"
        f"  --predicted-model {_repo_rel(frag_cif)} \\\n"
        f"  --method partial_phaseed --dmin 1.0 \\\n"
        f"  --n-iter 90 --n-extend 26 --n-starts 1 \\\n"
        f"  --prior-weight 0.52 --dm-ai-weight 0.42 --no-uncertainty \\\n"
        f"  --out {_repo_rel(out_dir)}"
    )


def write_processed(payload: Dict[str, Any], processed: Path) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    js = processed / "cod_midband_trail.json"
    md = processed / "cod_midband_trail.md"
    js.write_text(json.dumps(payload, indent=2) + "\n")

    this = payload.get("this_run") or {}
    sb = payload.get("committed_scoreboard") or COMMITTED_SCOREBOARD
    fom = this.get("free_fom_composite")
    try:
        fom_s = f"{float(fom):.3f}" if fom is not None else "—"
    except (TypeError, ValueError):
        fom_s = "—"
    n_peaks = this.get("n_q_written", this.get("n_peaks"))
    nq = "—" if n_peaks is None else str(n_peaks)
    na = this.get("next_action_id") or "—"
    sg_run = this.get("space_group") or payload.get("space_group") or "—"
    lines = [
        "# COD mid-band Fobs fragment trail (this run)",
        "",
        "Scientist path on in-repo **2012000** experimental Fobs + heaviest-cluster",
        "~½ non-H fragment. **Do not invent mapCC** — free FOM is from this run;",
        "mapCC / R1 / strict are copied from the committed Vol-band scoreboard.",
        "",
        f"- Fragment CIF: `{payload.get('fragment_cif')}` ({payload.get('n_frag')} of {payload.get('n_nonh')} non-H)",
        f"- Solve out: `{payload.get('out_dir')}`",
        f"- Seconds: {payload.get('seconds')}",
        "",
        "## This run (truth-free)",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| method | `{this.get('method') or '—'}` |",
        f"| free FOM (rank only) | **{fom_s}** |",
        f"| n Q peaks written | {nq} |",
        f"| next-action id | `{na}` |",
        f"| space group | `{sg_run}` |",
        "",
        "`trial.res` is a Q-peak list. Open `olex2_handbuild.md` in the `--out`",
        "folder: **File → Open** then **View → Work → Info** for Z / Z′.",
        "Do not SHELXL the raw Q list.",
        "",
        "## Committed scoreboard (copied, not recomputed)",
        "",
        f"Source: `{sb.get('source')}`.",
        "",
        "| Amp | Run | mapCC_OI | free FOM | R1 | strict |",
        "|-----|-----|----------|----------|----|--------|",
        f"| {sb.get('amp_mode')} | {sb.get('run')} | **{sb.get('mapcc_oi')}** | "
        f"{sb.get('free_fom')} | {sb.get('r1')} | **{sb.get('strict_success')}** |",
        "",
        sb.get("note") or "",
        "",
        "## gps-solve equivalent",
        "",
        "```bash",
        payload.get("gps_solve_command") or "",
        "```",
        "",
        "Recipe: [`docs/examples/cod_midband_fragment_trail.md`]"
        "(../../docs/examples/cod_midband_fragment_trail.md).",
        "",
    ]
    md.write_text("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="COD 2012000 Fobs fragment → Olex2 trail")
    p.add_argument(
        "--out",
        default="./out_2012000_frag",
        help="gps-solve --out folder (trial.res + olex2_handbuild.md)",
    )
    p.add_argument(
        "--cod-id",
        default=COD_ID,
        help="In-repo COD id (default 2012000 mid-band Fobs)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the half-CIF and print gps-solve; do not phase",
    )
    args = p.parse_args(argv)

    cod_dir = ROOT / "data" / "raw" / "cod"
    cif_path = cod_dir / f"{args.cod_id}.cif"
    hkl_path = cod_dir / f"{args.cod_id}.hkl"
    if not cif_path.exists():
        print(f"ERROR: missing {cif_path}", file=sys.stderr)
        return 1
    if not hkl_path.exists():
        print(f"ERROR: missing {hkl_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frag_cif = out_dir / f"{args.cod_id}_half.cif"

    meta = write_fragment_half_cif(cif_path, frag_cif)
    cmd = gps_solve_command(hkl_path, frag_cif, out_dir)
    print(f"Wrote fragment CIF ({meta['n_frag']} of {meta['n_nonh']} non-H): {frag_cif}")
    print("Scientist command:\n")
    print(cmd)
    print()

    payload: Dict[str, Any] = {
        "cod_id": args.cod_id,
        "hkl": _repo_rel(hkl_path),
        "deposited_cif": _repo_rel(cif_path),
        "fragment_cif": _repo_rel(frag_cif),
        "out_dir": _repo_rel(out_dir),
        "gps_solve_command": cmd,
        "committed_scoreboard": COMMITTED_SCOREBOARD,
        "n_nonh": meta["n_nonh"],
        "n_frag": meta["n_frag"],
        "space_group": meta["space_group"],
        "cell": meta["cell"],
        "vol": round(float(meta["vol"]), 1),
        "this_run": None,
        "seconds": None,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print("Dry run — skipped solve. Run the gps-solve command above to phase.")
        return 0

    cfg = SolveConfig(
        method="partial_phaseed",
        predicted_model_cif=str(frag_cif),
        expand_model_symmetry=True,
        d_min=1.0,
        n_iter=90,
        n_extend=26,
        n_starts=1,
        prior_weight=0.52,
        dm_ai_weight=0.42,
        verbose=True,
        seed=0,
        compute_uncertainty=False,
    )
    t0 = time.time()
    result = solve_structure(str(hkl_path), config=cfg)
    written = export_solution(result, out_dir)
    elapsed = time.time() - t0
    na = result.diagnostics.get("next_action") if isinstance(result.diagnostics, dict) else None
    na_id = na.get("primary_id") if isinstance(na, dict) else None
    payload["seconds"] = round(elapsed, 1)
    fom_raw = result.diagnostics.get("free_fom_composite")
    try:
        fom_val = None if fom_raw is None else float(fom_raw)
    except (TypeError, ValueError):
        fom_val = None
    n_q = sum(
        1
        for line in (out_dir / "trial.res").read_text().splitlines()
        if line.startswith("Q")
    )
    payload["this_run"] = {
        "method": result.method,
        "free_fom_composite": fom_val,
        "n_peaks": int(len(result.peaks)),
        "n_q_written": int(n_q),
        "space_group": result.space_group_hm,
        "next_action_id": na_id,
        "information_source": (na or {}).get("information_source")
        if isinstance(na, dict)
        else result.diagnostics.get("information_source"),
        "written": [p.name for p in written],
    }
    write_processed(payload, ROOT / "data" / "processed")
    names = {p.name for p in written}
    print(f"\nWrote {out_dir} in {elapsed:.1f}s")
    print(f"  trial.res: {'yes' if 'trial.res' in names else 'NO'}")
    print(f"  olex2_handbuild.md: {'yes' if 'olex2_handbuild.md' in names else 'NO'}")
    print(f"  free FOM (rank only): {payload['this_run']['free_fom_composite']}")
    print(f"  next-action: {na_id}")
    print("Committed scoreboard mapCC_OI (copied, not this run): "
          f"{COMMITTED_SCOREBOARD['mapcc_oi']}")
    print(f"Read {out_dir / 'olex2_handbuild.md'} — File → Open, then View → Work → Info.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
