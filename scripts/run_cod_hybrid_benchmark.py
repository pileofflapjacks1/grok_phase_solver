#!/usr/bin/env python3
"""
COD 2016452 conditional hybrid benchmark: PhAI + RAAR (and CF) polish.

Compares:
  - CF alone
  - RAAR alone
  - ensemble CF+RAAR (free-FOM)
  - PhAI fair alone
  - PhAI + unconditional CF
  - PhAI + unconditional RAAR
  - PhAI + conditional CF / RAAR (free-FOM gate)

Writes data/processed/cod_hybrid_benchmark.{json,md}
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
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.models.phai_fair import run_phai_fair
from grok_phase_solver.models.phai_runner import phai_available, find_model_path
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.conditional_hybrid import conditional_polish
from grok_phase_solver.solvers.ensemble import ensemble_cf_raar
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval
from grok_phase_solver.solvers.iterative_retrieval import raar_solve


def load_cod_2016452():
    path = ROOT / "data" / "raw" / "cod" / "2016452.cif"
    if not path.exists():
        # try download
        path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(
            "https://www.crystallography.net/cod/2016452.cif", path
        )
    return load_cif(path)


def eval_row(label, method, hkl, amp, ph_true, cell, true_fracs, elements, phases, density,
             d_min, seconds, **extra):
    rep = evaluate_success(
        hkl, amp, phases, ph_true, cell, true_fracs, density=density,
        elements=elements, thresholds=SuccessThresholds(),
    )
    row = {
        "dataset": label,
        "method": method,
        "d_min": d_min,
        "mapcc_oi": rep.mapcc_oi,
        "mpe_oi_deg": rep.mpe_oi_deg,
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": rep.solved,
        "seconds": seconds,
    }
    row.update(extra)
    return row


def run_at_resolution(st, d_min, seed=0, n_iter=80, n_starts=3):
    data = structure_to_fcalc(st, d_min=d_min)
    hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]
    cell = st.cell
    true_fracs = data["fracs"]
    elements = data["elements"]
    label = "COD_2016452"
    rows = []
    print(f"\n=== COD 2016452 Fcalc @ {d_min} Å  Nrefl={len(amp)} ===")

    def _e(method, phases, density, t0, **extra):
        row = eval_row(
            label, method, hkl, amp, ph_true, cell, true_fracs, elements,
            phases, density, d_min, time.time() - t0, **extra,
        )
        print(
            f"  {method:22s} CC={row['mapcc_oi']:.3f} peak={row['peak_recovery']:.2f} "
            f"R1={row['r1']:.2f} solved={row['solved']}"
        )
        rows.append(row)
        return row

    # CF
    t0 = time.time()
    ph, rho, hist = charge_flipping_solve(
        hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min, centrosymmetric=True
    )
    _e("charge_flipping", ph, rho, t0, final_R=hist.get("final_R"))

    # RAAR
    t0 = time.time()
    ph, rho, hist = raar_solve(
        hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
    )
    _e("raar", ph, rho, t0, final_R=hist.get("final_R"))

    # Ensemble
    t0 = time.time()
    ph, rho, info = ensemble_cf_raar(
        hkl, amp, cell, n_starts=n_starts, n_iter=n_iter, base_seed=seed, d_min=d_min
    )
    _e(
        "ensemble_cf_raar", ph, rho, t0,
        best_method=info.get("best_method"),
        best_composite=(info.get("best_fom") or {}).get("composite"),
    )

    if not phai_available():
        rows.append({
            "dataset": label, "d_min": d_min, "method": "phai_fair",
            "status": "unavailable", "weights": str(find_model_path()),
        })
        print("  PhAI unavailable — skipping hybrid PhAI arms")
        return rows

    # PhAI fair
    t0 = time.time()
    ph0, meta = run_phai_fair(hkl, amp, n_cycles=5, random_init=True, seed=seed)
    rho0 = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph0), cell, d_min=d_min
    )
    _e(
        "phai_fair", ph0, rho0, t0,
        frac_mapped=meta.get("frac_input_mapped"),
        n_mapped=meta.get("n_mapped"),
    )

    # Unconditional PhAI+CF
    t0 = time.time()
    ph, rho, _ = hybrid_phase_retrieval(
        hkl, amp, cell, ph0, polish="charge_flipping", n_iter=n_iter, seed=seed
    )
    _e("phai+CF_uncond", ph, rho, t0)

    # Unconditional PhAI+RAAR
    t0 = time.time()
    ph, rho, hist = raar_solve(
        hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min, phase_init=ph0
    )
    _e("phai+RAAR_uncond", ph, rho, t0, final_R=hist.get("final_R"))

    # Conditional PhAI+CF
    t0 = time.time()
    ph, rho, info = conditional_polish(
        hkl, amp, cell, ph0, polish="charge_flipping", n_iter=n_iter,
        seed=seed, d_min=d_min,
    )
    _e(
        "phai+CF_cond", ph, rho, t0,
        accepted_polish=info.get("accepted_polish"),
        fom_seed=(info.get("fom_seed") or {}).get("composite"),
        fom_final=(info.get("fom_final") or {}).get("composite"),
    )

    # Conditional PhAI+RAAR (primary target of this benchmark)
    t0 = time.time()
    ph, rho, info = conditional_polish(
        hkl, amp, cell, ph0, polish="raar", n_iter=n_iter,
        seed=seed, d_min=d_min,
    )
    _e(
        "phai+RAAR_cond", ph, rho, t0,
        accepted_polish=info.get("accepted_polish"),
        fom_seed=(info.get("fom_seed") or {}).get("composite"),
        fom_final=(info.get("fom_final") or {}).get("composite"),
    )

    return rows


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dmin", type=float, nargs="*", default=None)
    args = p.parse_args()

    st = load_cod_2016452()
    print(f"Loaded COD 2016452: SG={getattr(st, 'space_group_hm', '?')} cell={st.cell}")

    if args.dmin:
        dmins = args.dmin
    elif args.quick:
        dmins = [0.9, 1.5]
    else:
        dmins = [0.9, 1.2, 1.5, 2.0]

    n_iter = 50 if args.quick else 80
    n_starts = 2 if args.quick else 3

    all_rows = []
    for d_min in dmins:
        try:
            all_rows.extend(
                run_at_resolution(st, d_min, seed=0, n_iter=n_iter, n_starts=n_starts)
            )
        except Exception as e:
            print(f"  ERROR at d_min={d_min}: {e}")
            all_rows.append({
                "dataset": "COD_2016452", "d_min": d_min, "error": str(e),
            })

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "cod_hybrid_benchmark.json"
    jp.write_text(json.dumps(all_rows, indent=2, default=float))

    methods_order = [
        "charge_flipping", "raar", "ensemble_cf_raar", "phai_fair",
        "phai+CF_uncond", "phai+RAAR_uncond", "phai+CF_cond", "phai+RAAR_cond",
    ]
    md = [
        "# COD 2016452 conditional hybrid benchmark",
        "",
        "Fcalc from deposited structure; **strict SuccessThresholds**. "
        "Conditional polish keeps PhAI seed unless free-FOM composite improves.",
        "",
        f"PhAI weights available: **{phai_available()}**",
        "",
    ]
    for d_min in dmins:
        md.append(f"## d_min = {d_min} Å")
        md.append("")
        md.append("| Method | mapCC | peak | R1 | solved | notes |")
        md.append("|--------|-------|------|----|--------|-------|")
        sub = [r for r in all_rows if r.get("d_min") == d_min and "error" not in r]
        by_m = {r["method"]: r for r in sub if "method" in r}
        for m in methods_order:
            if m not in by_m:
                continue
            r = by_m[m]
            if r.get("status") == "unavailable":
                md.append(f"| `{m}` | — | — | — | — | unavailable |")
                continue
            notes = []
            if "accepted_polish" in r:
                notes.append(f"accept={r['accepted_polish']}")
            if r.get("best_method"):
                notes.append(f"pick={r['best_method']}")
            md.append(
                f"| `{m}` | {r['mapcc_oi']:.3f} | {r['peak_recovery']:.2f} | "
                f"{r.get('r1', float('nan')):.2f} | {r['solved']} | {'; '.join(notes)} |"
            )
        md.append("")

    md.extend([
        "## Interpretation",
        "",
        "- Prefer **phai+RAAR_cond** when PhAI seed is already good at low res "
        "(conditional gate rejects harmful polish).",
        "- At high res (~0.9 Å), unconditional or conditional polish may both solve.",
        "- Ensemble CF+RAAR is a classical multistart baseline without PhAI.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp = out / "cod_hybrid_benchmark.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}")


if __name__ == "__main__":
    main()
