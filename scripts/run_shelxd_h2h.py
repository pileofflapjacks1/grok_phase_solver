#!/usr/bin/env python3
"""
SHELXD head-to-head scoreboard.

Compares gps-solve methods against:
  - real external SHELXD (if `shelxd` is on PATH / SHELXD env)
  - educational dual-space baseline (always; SHELXD-inspired, not SHELXD)

Panels:
  1. Synthetic easy (n≈8, d≈1.0 Å) — should be solvable classically
  2. Synthetic hard (n≈14, d≈1.7 Å) — current research cliff
  3. Demo + COD 2016452 Fcalc control (if present)

Metrics (fair, same truth):
  mapCC_OI, peak recovery, R1, strict solved, free FOM, wall time

Writes data/processed/shelxd_h2h.{json,md}

Install SHELXD (optional, academic license):
  https://shelx.uni-goettingen.de/
  export SHELXD=/path/to/shelxd
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

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.direct_methods import direct_methods_solve
from grok_phase_solver.solvers.dual_space import dual_space_solve
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.shelxd_runner import (
    find_shelxd,
    shelxd_available,
    shelxd_solve,
)


def _eval_method(
    name: str,
    hkl,
    amp,
    ph_t,
    cell,
    fracs,
    elements,
    d_min: float,
    n_iter: int,
    n_starts: int,
    seed: int,
    n_atoms: int,
    shelxd_path: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    row: Dict[str, Any] = {"method": name}
    try:
        if name == "charge_flipping":
            ph, rho, info = charge_flipping_solve(
                hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
            )
        elif name == "direct_methods":
            dm = direct_methods_solve(
                hkl, amp, cell,
                n_atoms_approx=n_atoms,
                n_trials=max(20, n_iter // 3),
                seed=seed,
            )
            ph = dm.phases_full
            rho = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph), cell, d_min=d_min
            )
            info = dm.history
        elif name == "dual_space":
            ph, rho, info = dual_space_solve(
                hkl, amp, cell,
                n_atoms=n_atoms,
                n_cycles=max(20, n_iter // 2),
                n_starts=max(4, n_starts * 2),
                seed=seed,
                d_min=d_min,
                polish_cf_iters=min(25, n_iter // 2),
            )
        elif name == "shelxd":
            ph, rho, info = shelxd_solve(
                hkl, amp, cell,
                n_atoms=n_atoms,
                n_try=max(30, n_starts * 20),
                seed=max(1, seed),
                d_min=d_min,
                shelxd_path=shelxd_path,
                keep_files=False,
            )
        elif name == "ensemble":
            from grok_phase_solver.solvers.ensemble import ensemble_cf_raar

            ph, rho, info = ensemble_cf_raar(
                hkl, amp, cell,
                n_starts=max(2, n_starts),
                n_iter=n_iter,
                base_seed=seed,
                d_min=d_min,
            )
        elif name == "strong_prior_phaseed":
            from grok_phase_solver.models.strong_prior import (
                default_strong_prior_path,
                strong_prior_phaseed_solve,
            )

            if not default_strong_prior_path().exists():
                raise FileNotFoundError("strong prior weights missing")
            ph, rho, info = strong_prior_phaseed_solve(
                hkl, amp, cell,
                n_extend=10, polish="charge_flipping", n_polish=n_iter,
                n_starts=n_starts, seed=seed, d_min=d_min,
            )
        elif name == "hard_p1_phaseed":
            from grok_phase_solver.models.hard_p1_prior import (
                default_hard_p1_path,
                hard_p1_phaseed_solve,
            )

            if not default_hard_p1_path().exists():
                raise FileNotFoundError("hard_p1 weights missing")
            ph, rho, info = hard_p1_phaseed_solve(
                hkl, amp, cell,
                n_extend=10, polish="charge_flipping", n_polish=n_iter,
                n_starts=n_starts, seed=seed, d_min=d_min,
            )
        else:
            raise ValueError(f"unknown method {name}")

        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_t), cell, shape=rho.shape
        )
        if rho.shape != rho_t.shape:
            rho = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph), cell, shape=rho_t.shape
            )
        cc, _ = map_correlation_origin_invariant(rho, rho_t)
        rep = evaluate_success(
            hkl, amp, ph, ph_t, cell, fracs, density=rho, elements=elements,
            thresholds=SuccessThresholds(),
        )
        fom = free_fom(hkl, amp, ph, cell, density=rho, include_shells=False)
        row.update(
            {
                "mapcc_oi": float(cc),
                "peak_recovery": rep.peak_recovery,
                "r1": rep.r1,
                "solved": bool(rep.solved),
                "free_fom": float(fom["composite"]),
                "seconds": time.time() - t0,
                "backend": info.get("backend") or info.get("method") or name,
                "R_partial": info.get("R_partial"),
                "ok": True,
            }
        )
    except Exception as e:
        row.update(
            {
                "ok": False,
                "error": str(e),
                "seconds": time.time() - t0,
                "mapcc_oi": None,
                "solved": False,
            }
        )
    return row


def run_synthetic_panel(
    label: str,
    n_cases: int,
    n_atoms_range,
    d_min_choices,
    methods: List[str],
    seed: int,
    n_iter: int,
    n_starts: int,
    shelxd_path: Optional[Path],
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows = []
    print(f"\n===== {label} ({n_cases} cases) =====", flush=True)
    for i in range(n_cases):
        n_atoms = int(rng.integers(n_atoms_range[0], n_atoms_range[1] + 1))
        d_min = float(rng.choice(d_min_choices))
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell
        print(f"  case {i+1}/{n_cases}: n={data['n_atoms_cell']} d={d_min:.2f}", flush=True)
        for m in methods:
            row = _eval_method(
                m, hkl, amp, ph_t, cell, data["fracs"], data["elements"],
                d_min=d_min, n_iter=n_iter, n_starts=n_starts, seed=s,
                n_atoms=n_atoms, shelxd_path=shelxd_path,
            )
            row.update(
                {
                    "panel": label,
                    "case": i,
                    "n_atoms": data["n_atoms_cell"],
                    "d_min": d_min,
                    "structure_seed": s,
                }
            )
            rows.append(row)
            cc = row.get("mapcc_oi")
            cc_s = f"{cc:.3f}" if isinstance(cc, float) else "ERR"
            print(
                f"    {m:22s} CC={cc_s} sol={row.get('solved')} "
                f"t={row.get('seconds', 0):.1f}s"
                + (f"  ({row.get('error', '')[:60]})" if not row.get("ok") else ""),
                flush=True,
            )
    return rows


def run_cod_control(methods: List[str], n_iter: int, n_starts: int, shelxd_path) -> List[Dict]:
    rows = []
    cif = ROOT / "data/raw/cod/2016452.cif"
    if not cif.exists():
        return rows
    from grok_phase_solver.io.cif import load_cif

    st = load_cif(cif)
    d_min = 0.9
    data = structure_to_fcalc(st, d_min=d_min)
    print(f"\n===== COD_2016452_Fcalc d={d_min} =====", flush=True)
    for m in methods:
        row = _eval_method(
            m,
            data["hkl"],
            data["amplitudes"],
            data["phases"],
            st.cell,
            data["fracs"],
            data["elements"],
            d_min=d_min,
            n_iter=n_iter,
            n_starts=n_starts,
            seed=0,
            n_atoms=min(40, data["n_atoms_cell"]),
            shelxd_path=shelxd_path,
        )
        row.update({"panel": "COD_2016452_Fcalc", "n_atoms": data["n_atoms_cell"], "d_min": d_min})
        rows.append(row)
        cc = row.get("mapcc_oi")
        cc_s = f"{cc:.3f}" if isinstance(cc, float) else "ERR"
        print(f"  {m:22s} CC={cc_s} sol={row.get('solved')} t={row.get('seconds', 0):.1f}s", flush=True)
    return rows


def summarize(rows: List[Dict]) -> Dict[str, Dict]:
    by = {}
    for r in rows:
        key = (r.get("panel"), r.get("method"))
        by.setdefault(key, []).append(r)
    summary = {}
    for (panel, method), rs in sorted(by.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        ok = [r for r in rs if r.get("ok") and r.get("mapcc_oi") is not None]
        n = len(rs)
        n_ok = len(ok)
        n_sol = sum(1 for r in ok if r.get("solved"))
        mcc = float(np.mean([r["mapcc_oi"] for r in ok])) if ok else None
        mpeak = float(np.mean([r["peak_recovery"] for r in ok if r.get("peak_recovery") is not None])) if ok else None
        mt = float(np.mean([r.get("seconds", 0) for r in rs]))
        summary[f"{panel}::{method}"] = {
            "panel": panel,
            "method": method,
            "n": n,
            "n_ok": n_ok,
            "n_solved": n_sol,
            "solve_rate": n_sol / max(n_ok, 1) if n_ok else 0.0,
            "mean_mapcc": mcc,
            "mean_peak_recovery": mpeak,
            "mean_seconds": mt,
            "errors": [r.get("error") for r in rs if not r.get("ok")][:3],
        }
    return summary


def write_report(rows, summary, shelxd_path, out_json: Path, out_md: Path):
    payload = {
        "shelxd_available": shelxd_path is not None,
        "shelxd_path": str(shelxd_path) if shelxd_path else None,
        "n_rows": len(rows),
        "summary": summary,
        "rows": rows,
        "note": (
            "SHELXD is not redistributed. dual_space is an educational "
            "SHELXD-inspired baseline, not claimed identical to SHELXD."
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2, default=float))

    lines = [
        "# SHELXD head-to-head",
        "",
        "Fair comparison of **gps-solve** methods vs **external SHELXD** "
        "(when installed) and an in-repo **dual-space** educational baseline.",
        "",
        f"- SHELXD binary: **{'found — `' + str(shelxd_path) + '`' if shelxd_path else 'not found'}**",
        "- Install: [SHELX academic distribution](https://shelx.uni-goettingen.de/) "
        "then `export SHELXD=/path/to/shelxd`",
        "- dual_space: multi-start peak↔phase recycling (Sheldrick-style idea); "
        "**not** SHELXD",
        "",
        "## Summary",
        "",
        "| Panel | Method | n | solved | rate | mean mapCC | mean peak | t (s) |",
        "|-------|--------|---|--------|------|------------|-----------|-------|",
    ]
    for k, s in summary.items():
        mcc = s["mean_mapcc"]
        mcc_s = f"{mcc:.3f}" if mcc is not None else "—"
        pk = s["mean_peak_recovery"]
        pk_s = f"{pk:.2f}" if pk is not None else "—"
        err = ""
        if s["errors"]:
            err = " ⚠️"
        lines.append(
            f"| {s['panel']} | `{s['method']}` | {s['n_ok']}/{s['n']} | "
            f"{s['n_solved']} | {s['solve_rate']:.0%} | {mcc_s} | {pk_s} | "
            f"{s['mean_seconds']:.1f}{err} |"
        )

    # Per-panel ranking
    panels = sorted({s["panel"] for s in summary.values()})
    lines.extend(["", "## Rankings by panel (mean mapCC)", ""])
    for panel in panels:
        lines.append(f"### {panel}")
        lines.append("")
        items = [s for s in summary.values() if s["panel"] == panel and s["mean_mapcc"] is not None]
        items.sort(key=lambda x: -x["mean_mapcc"])
        for rank, s in enumerate(items, 1):
            lines.append(
                f"{rank}. **`{s['method']}`** — mapCC {s['mean_mapcc']:.3f}, "
                f"solved {s['n_solved']}/{s['n_ok']}, {s['mean_seconds']:.1f}s"
            )
        lines.append("")

    lines.extend(
        [
            "## Fairness notes",
            "",
            "- Same synthetic structures and COD Fcalc control for all methods.",
            "- Success uses strict `SuccessThresholds` (mapCC_OI + peak recovery + R1).",
            "- SHELXD wall time depends on NTRY; dual_space uses multi-start × cycles.",
            "- If SHELXD is missing, the `shelxd` method rows show errors; dual_space still runs.",
            "- This is **not** a claim that gps-solve replaces SHELXD/SHELXT in production.",
            "",
            f"JSON: `{out_json.relative_to(ROOT)}`",
            "",
        ]
    )
    out_md.write_text("\n".join(lines))


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Fewer cases / iterations")
    p.add_argument("--shelxd", type=str, default=None, help="Path to shelxd binary")
    p.add_argument("--skip-hard", action="store_true")
    p.add_argument("--skip-cod", action="store_true")
    p.add_argument("--n-easy", type=int, default=None)
    p.add_argument("--n-hard", type=int, default=None)
    args = p.parse_args()

    shelxd_path = find_shelxd(args.shelxd)
    print(
        f"SHELXD: {shelxd_path if shelxd_path else 'NOT FOUND — dual_space only for that column'}",
        flush=True,
    )

    if args.quick:
        n_easy, n_hard = 2, 2
        n_iter, n_starts = 35, 2
    else:
        n_easy, n_hard = 4, 4
        n_iter, n_starts = 50, 2
    if args.n_easy is not None:
        n_easy = args.n_easy
    if args.n_hard is not None:
        n_hard = args.n_hard

    methods = [
        "charge_flipping",
        "direct_methods",
        "dual_space",
        "ensemble",
    ]
    # optional AI priors if weights exist
    try:
        from grok_phase_solver.models.strong_prior import default_strong_prior_path
        from grok_phase_solver.models.hard_p1_prior import default_hard_p1_path

        if default_strong_prior_path().exists():
            methods.append("strong_prior_phaseed")
        if default_hard_p1_path().exists():
            methods.append("hard_p1_phaseed")
    except Exception:
        pass
    if shelxd_path is not None:
        methods.append("shelxd")

    all_rows: List[Dict] = []
    all_rows.extend(
        run_synthetic_panel(
            "synthetic_easy",
            n_cases=n_easy,
            n_atoms_range=(6, 9),
            d_min_choices=[0.9, 1.0, 1.1],
            methods=methods,
            seed=11,
            n_iter=n_iter,
            n_starts=n_starts,
            shelxd_path=shelxd_path,
        )
    )
    if not args.skip_hard:
        all_rows.extend(
            run_synthetic_panel(
                "synthetic_hard",
                n_cases=n_hard,
                n_atoms_range=(12, 16),
                d_min_choices=[1.5, 1.7, 2.0],
                methods=methods,
                seed=22,
                n_iter=n_iter,
                n_starts=n_starts,
                shelxd_path=shelxd_path,
            )
        )
    if not args.skip_cod:
        all_rows.extend(
            run_cod_control(methods, n_iter=n_iter, n_starts=n_starts, shelxd_path=shelxd_path)
        )

    summary = summarize(all_rows)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "shelxd_h2h.json"
    mp = out / "shelxd_h2h.md"
    write_report(all_rows, summary, shelxd_path, jp, mp)
    print(f"\nWrote {jp}\nWrote {mp}", flush=True)

    # Console mini-summary
    print("\n=== Head-to-head summary ===", flush=True)
    for k, s in summary.items():
        mcc = s["mean_mapcc"]
        mcc_s = f"{mcc:.3f}" if mcc is not None else "n/a"
        print(
            f"  {s['panel']:18s} {s['method']:22s} "
            f"CC={mcc_s} solved={s['n_solved']}/{s['n_ok']} t={s['mean_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
