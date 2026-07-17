#!/usr/bin/env python3
"""
SHELXS head-to-head scoreboard.

Fair comparison of gps-solve methods vs external **SHELXS** (Sheldrick
direct-methods solver) when ``ShelX/shelxs`` or PATH/SHELXS is available.

Panels: synthetic easy, synthetic hard, optional COD Fcalc control.

Writes data/processed/shelxs_h2h.{json,md}

Setup (academic license — do not commit binaries):
  1. Download macOS SHELXS from https://shelx.uni-goettingen.de/
  2. Place at ShelX/shelxs  (or export SHELXS=/path/to/shelxs)
  3. xattr -dr com.apple.quarantine ShelX   # if macOS blocks execution
  4. chmod +x ShelX/shelxs
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
from grok_phase_solver.solvers.dual_space import dual_space_solve
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.shelxs_runner import find_shelxs, shelxs_solve


def _score(hkl, amp, ph, ph_t, cell, fracs, elements, rho, d_min) -> Dict[str, Any]:
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
    return {
        "mapcc_oi": float(cc),
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": bool(rep.solved),
        "free_fom": float(fom["composite"]),
    }


def _eval_method(
    name: str,
    hkl, amp, ph_t, cell, fracs, elements, d_min, n_atoms,
    n_iter, n_starts, seed, shelxs_path, n_try,
) -> Dict[str, Any]:
    t0 = time.time()
    row: Dict[str, Any] = {"method": name}
    try:
        if name == "shelxs":
            ph, rho, info = shelxs_solve(
                hkl, amp, cell,
                n_atoms=n_atoms,
                n_try=n_try,
                d_min=d_min,
                shelxs_path=shelxs_path,
                timeout_s=180.0,
                verbose=False,
            )
            row["backend_info"] = {
                k: info.get(k)
                for k in ("returncode", "n_peaks_parsed", "R_partial", "cfom_line", "status")
            }
        elif name == "charge_flipping":
            ph, rho, info = charge_flipping_solve(
                hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
            )
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
                raise FileNotFoundError("strong prior missing")
            ph, rho, info = strong_prior_phaseed_solve(
                hkl, amp, cell,
                n_extend=12, n_polish=n_iter, n_starts=n_starts,
                seed=seed, d_min=d_min,
            )
        else:
            raise ValueError(name)

        sc = _score(hkl, amp, ph, ph_t, cell, fracs, elements, rho, d_min)
        row.update(sc)
        row["ok"] = True
        row["seconds"] = time.time() - t0
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


def run_panel(
    label: str,
    n_cases: int,
    n_atoms_range,
    d_min_choices,
    methods: List[str],
    seed: int,
    n_iter: int,
    n_starts: int,
    shelxs_path,
    n_try: int,
) -> List[Dict]:
    rows = []
    rng = np.random.default_rng(seed)
    print(f"\n===== {label} ({n_cases} cases) =====", flush=True)
    for i in range(n_cases):
        n_atoms = int(rng.integers(n_atoms_range[0], n_atoms_range[1] + 1))
        d_min = float(rng.choice(d_min_choices))
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        print(
            f"  case {i+1}/{n_cases}: n={data['n_atoms_cell']} d={d_min:.2f} "
            f"nrefl={len(hkl)}",
            flush=True,
        )
        for m in methods:
            row = _eval_method(
                m, hkl, amp, ph_t, st.cell, data["fracs"], data["elements"],
                d_min, data["n_atoms_cell"], n_iter, n_starts, s, shelxs_path, n_try,
            )
            row.update(
                {
                    "panel": label,
                    "case": i,
                    "n_atoms": data["n_atoms_cell"],
                    "d_min": d_min,
                    "structure_seed": s,
                    "n_refl": len(hkl),
                }
            )
            rows.append(row)
            cc = row.get("mapcc_oi")
            cc_s = f"{cc:.3f}" if isinstance(cc, float) else "ERR"
            err = f" ({row.get('error', '')[:70]})" if not row.get("ok") else ""
            print(
                f"    {m:22s} CC={cc_s} sol={row.get('solved')} "
                f"t={row.get('seconds', 0):.1f}s{err}",
                flush=True,
            )
    return rows


def summarize(rows: List[Dict]) -> Dict:
    from collections import defaultdict

    g = defaultdict(list)
    for r in rows:
        g[(r.get("panel"), r.get("method"))].append(r)
    out = {}
    for (panel, method), rs in sorted(g.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        ok = [r for r in rs if r.get("ok") and r.get("mapcc_oi") is not None]
        n_sol = sum(1 for r in ok if r.get("solved"))
        out[f"{panel}::{method}"] = {
            "panel": panel,
            "method": method,
            "n": len(rs),
            "n_ok": len(ok),
            "n_solved": n_sol,
            "solve_rate": n_sol / max(len(ok), 1) if ok else 0.0,
            "mean_mapcc": float(np.mean([r["mapcc_oi"] for r in ok])) if ok else None,
            "mean_peak": float(
                np.mean([r["peak_recovery"] for r in ok if r.get("peak_recovery") is not None])
            )
            if ok
            else None,
            "mean_seconds": float(np.mean([r.get("seconds", 0) for r in rs])),
            "errors": [r.get("error") for r in rs if not r.get("ok")][:3],
        }
    return out


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--shelxs", type=str, default=None)
    p.add_argument("--n-easy", type=int, default=None)
    p.add_argument("--n-hard", type=int, default=None)
    p.add_argument("--n-try", type=int, default=None, help="SHELXS TREF trials")
    p.add_argument("--skip-hard", action="store_true")
    args = p.parse_args()

    shelxs_path = find_shelxs(args.shelxs)
    print(
        f"SHELXS: {shelxs_path if shelxs_path else 'NOT FOUND — install to ShelX/shelxs'}",
        flush=True,
    )

    if args.quick:
        n_easy, n_hard = 2, 2
        n_iter, n_starts, n_try = 40, 2, 50
    else:
        n_easy, n_hard = 4, 4
        n_iter, n_starts, n_try = 50, 2, 100
    if args.n_easy is not None:
        n_easy = args.n_easy
    if args.n_hard is not None:
        n_hard = args.n_hard
    if args.n_try is not None:
        n_try = args.n_try

    methods = ["charge_flipping", "dual_space", "ensemble"]
    try:
        from grok_phase_solver.models.strong_prior import default_strong_prior_path

        if default_strong_prior_path().exists():
            methods.append("strong_prior_phaseed")
    except Exception:
        pass
    if shelxs_path is not None:
        methods.append("shelxs")

    all_rows: List[Dict] = []
    all_rows.extend(
        run_panel(
            "synthetic_easy",
            n_easy,
            (6, 9),
            [0.9, 1.0, 1.1],
            methods,
            seed=11,
            n_iter=n_iter,
            n_starts=n_starts,
            shelxs_path=shelxs_path,
            n_try=n_try,
        )
    )
    if not args.skip_hard:
        all_rows.extend(
            run_panel(
                "synthetic_hard",
                n_hard,
                (12, 16),
                [1.5, 1.7, 2.0],
                methods,
                seed=22,
                n_iter=n_iter,
                n_starts=n_starts,
                shelxs_path=shelxs_path,
                n_try=n_try,
            )
        )

    summary = summarize(all_rows)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "shelxs_h2h.json"
    mp = out / "shelxs_h2h.md"
    payload = {
        "shelxs_available": shelxs_path is not None,
        "shelxs_path": str(shelxs_path) if shelxs_path else None,
        "n_try": n_try,
        "summary": summary,
        "rows": all_rows,
        "note": (
            "SHELXS is not redistributed. Binaries live in local ShelX/ (gitignored). "
            "Fixed-format HKLF-4 + TREF multi-trial solution; peaks → Fcalc phases."
        ),
    }
    jp.write_text(json.dumps(payload, indent=2, default=float))

    lines = [
        "# SHELXS head-to-head",
        "",
        "Fair comparison of **gps-solve** methods vs external **SHELXS 2024/1** "
        "(Sheldrick direct methods) on the same synthetic cases.",
        "",
        f"- SHELXS binary: **{'`' + str(shelxs_path) + '`' if shelxs_path else 'not found'}**",
        f"- TREF trials: **{n_try}**",
        "",
        "## Summary",
        "",
        "| Panel | Method | n | solved | rate | mean mapCC | mean peak | t (s) |",
        "|-------|--------|---|--------|------|------------|-----------|-------|",
    ]
    for s in summary.values():
        mcc = s["mean_mapcc"]
        mcc_s = f"{mcc:.3f}" if mcc is not None else "—"
        pk = s["mean_peak"]
        pk_s = f"{pk:.2f}" if pk is not None else "—"
        warn = " ⚠️" if s["errors"] else ""
        lines.append(
            f"| {s['panel']} | `{s['method']}` | {s['n_ok']}/{s['n']} | "
            f"{s['n_solved']} | {s['solve_rate']:.0%} | {mcc_s} | {pk_s} | "
            f"{s['mean_seconds']:.1f}{warn} |"
        )

    panels = sorted({s["panel"] for s in summary.values()})
    lines.extend(["", "## Rankings by panel (mean mapCC)", ""])
    for panel in panels:
        lines.append(f"### {panel}")
        lines.append("")
        items = [
            s for s in summary.values()
            if s["panel"] == panel and s["mean_mapcc"] is not None
        ]
        items.sort(key=lambda x: -x["mean_mapcc"])
        for rank, s in enumerate(items, 1):
            lines.append(
                f"{rank}. **`{s['method']}`** — mapCC {s['mean_mapcc']:.3f}, "
                f"solved {s['n_solved']}/{s['n_ok']}, {s['mean_seconds']:.1f}s"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- **SHELXS** = classical direct methods (not SHELXD dual-space).",
            "- HKL written as fixed-format (3i4,2f8.2) with scaled intensities.",
            "- Peaks from `.res` → equal-atom Fcalc phases for mapCC vs truth.",
            "- Do not commit `ShelX/` binaries (academic license, gitignored).",
            "- If SHELXS is missing, only in-repo methods are scored.",
            "",
            f"JSON: `{jp.relative_to(ROOT)}`",
            "",
        ]
    )
    mp.write_text("\n".join(lines))
    print(f"\nWrote {jp}\nWrote {mp}", flush=True)
    print("\n=== Summary ===", flush=True)
    for s in summary.values():
        mcc = s["mean_mapcc"]
        mcc_s = f"{mcc:.3f}" if mcc is not None else "n/a"
        print(
            f"  {s['panel']:16s} {s['method']:22s} "
            f"CC={mcc_s} solved={s['n_solved']}/{s['n_ok']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
