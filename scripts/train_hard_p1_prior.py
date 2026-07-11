#!/usr/bin/env python3
"""
Train domain-matched PhaseMLP on hard P1 synthetic cells.

Saves data/processed/hard_p1_prior.{npz,json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.phase_error import (
    mean_phase_error,
    mean_phase_error_origin_invariant,
)
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.models.hard_p1_prior import (
    hard_p1_phaseed_solve,
    predict_phases_hard_p1,
    save_hard_p1_prior,
    train_hard_p1_prior,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.ai_phaseed import ai_phaseed_solve
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve


def eval_holdout(model, n_eval=6, seed=12345, n_iter=50, n_extend=12):
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n_eval):
        n_atoms = int(rng.integers(12, 18))
        d_min = float(rng.choice([1.5, 1.7, 2.0]))
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell

        # prior only (with free-FOM origin search)
        ph_p = predict_phases_hard_p1(model, hkl, amp, cell, origin_fom_search=True)
        rho_p = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_p), cell, d_min=d_min
        )
        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_t), cell, shape=rho_p.shape
        )
        cc_p, _ = map_correlation_origin_invariant(rho_p, rho_t)
        mpe, _ = mean_phase_error_origin_invariant(ph_p, ph_t, hkl, weights=amp)
        mpe = float(mpe)

        # prior + AI-PhaSeed
        ph_h, rho_h, info = hard_p1_phaseed_solve(
            hkl, amp, cell, model=model,
            n_extend=n_extend, polish="charge_flipping", n_polish=n_iter,
            n_starts=2, seed=s, d_min=d_min,
        )
        if rho_h.shape != rho_t.shape:
            rho_h = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_h), cell, shape=rho_t.shape
            )
        rep_h = evaluate_success(
            hkl, amp, ph_h, ph_t, cell, data["fracs"], density=rho_h,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )

        # CF baseline
        ph_c, rho_c, _ = charge_flipping_solve(
            hkl, amp, cell, n_iter=n_iter, seed=s, d_min=d_min
        )
        if rho_c.shape != rho_t.shape:
            rho_c = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_c), cell, shape=rho_t.shape
            )
        rep_c = evaluate_success(
            hkl, amp, ph_c, ph_t, cell, data["fracs"], density=rho_c,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )

        # random + phaseed (control)
        ph_r = np.random.default_rng(s).uniform(-np.pi, np.pi, len(amp))
        ph_rp, rho_rp, _ = ai_phaseed_solve(
            hkl, amp, cell, ph_r, n_extend=n_extend, polish="charge_flipping",
            n_polish=n_iter, n_starts=1, seed=s, d_min=d_min,
        )
        if rho_rp.shape != rho_t.shape:
            rho_rp = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_rp), cell, shape=rho_t.shape
            )
        rep_rp = evaluate_success(
            hkl, amp, ph_rp, ph_t, cell, data["fracs"], density=rho_rp,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )

        row = {
            "n_atoms": n_atoms,
            "d_min": d_min,
            "seed": s,
            "prior_mpe_deg": mpe,
            "prior_mapcc": cc_p,
            "hard_p1_phaseed_mapcc": rep_h.mapcc_oi,
            "hard_p1_phaseed_solved": rep_h.solved,
            "hard_p1_phaseed_peak": rep_h.peak_recovery,
            "cf_mapcc": rep_c.mapcc_oi,
            "cf_solved": rep_c.solved,
            "random_phaseed_mapcc": rep_rp.mapcc_oi,
            "random_phaseed_solved": rep_rp.solved,
        }
        rows.append(row)
        print(
            f"  holdout n={n_atoms} d={d_min:.1f}: "
            f"prior MPE={mpe:.0f}° CC={cc_p:.2f} | "
            f"hP1+PhaSeed CC={rep_h.mapcc_oi:.2f} sol={rep_h.solved} | "
            f"CF={rep_c.mapcc_oi:.2f} | rand+PS={rep_rp.mapcc_oi:.2f}"
        )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-structures", type=int, default=80)
    p.add_argument("--epochs-per", type=int, default=45)
    p.add_argument("--epochs-refine", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default="data/processed/hard_p1_prior.npz")
    p.add_argument("--n-eval", type=int, default=6)
    args = p.parse_args()

    if args.quick:
        args.n_structures = 16
        args.epochs_per = 25
        args.epochs_refine = 8
        args.hidden = 64
        args.n_eval = 3

    t0 = time.time()
    model, meta = train_hard_p1_prior(
        n_structures=args.n_structures,
        epochs_per=args.epochs_per,
        epochs_refine=args.epochs_refine,
        hidden=args.hidden,
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    save_hard_p1_prior(model, out, meta=meta)
    print(f"Saved {out}  train MPE={meta['mean_train_mpe']:.1f}°  "
          f"hold MPE={meta.get('mean_holdout_mpe')}")

    print("\nHold-out hard-P1 solve comparison:")
    t1 = time.time()
    hold = eval_holdout(
        model, n_eval=args.n_eval, seed=args.seed + 777,
        n_iter=40 if args.quick else 50,
        n_extend=8 if args.quick else 12,
    )
    meta["holdout_solve"] = hold
    meta["eval_seconds"] = time.time() - t1
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))

    # summary rates
    n = len(hold)
    n_h = sum(1 for r in hold if r["hard_p1_phaseed_solved"])
    n_c = sum(1 for r in hold if r["cf_solved"])
    n_r = sum(1 for r in hold if r["random_phaseed_solved"])
    mcc_h = float(np.mean([r["hard_p1_phaseed_mapcc"] for r in hold]))
    mcc_c = float(np.mean([r["cf_mapcc"] for r in hold]))
    mcc_p = float(np.mean([r["prior_mapcc"] for r in hold]))
    mpe = float(np.mean([r["prior_mpe_deg"] for r in hold]))

    md = [
        "# Hard-P1 domain-matched phase prior",
        "",
        "PhaseMLP trained on synthetic **P1** cells in the hard region "
        "(\(n \\in [12,20]\), \(d_{\\min} \\in [1.5, 2.0]\)), plus bridge samples.",
        "",
        f"- Structures: **{args.n_structures}** (bridge every 4th)",
        f"- Hidden: {args.hidden}, epochs/structure: {args.epochs_per}+{args.epochs_refine}",
        f"- Mean train MPE: **{meta['mean_train_mpe']:.1f}°**",
        f"- Mean hold-out MPE: **{meta.get('mean_holdout_mpe', float('nan')):.1f}°**",
        f"- Weights: `{out.relative_to(ROOT)}`",
        "",
        "## Hold-out hard-region solve rates",
        "",
        f"| Method | Solved | Rate | mean mapCC |",
        f"|--------|--------|------|------------|",
        f"| hard_p1 prior only | — | — | {mcc_p:.3f} (MPE {mpe:.0f}°) |",
        f"| **hard_p1 + AI-PhaSeed** | {n_h} | {n_h/n:.0%} | **{mcc_h:.3f}** |",
        f"| CF | {n_c} | {n_c/n:.0%} | {mcc_c:.3f} |",
        f"| random + AI-PhaSeed | {n_r} | {n_r/n:.0%} | "
        f"{float(np.mean([r['random_phaseed_mapcc'] for r in hold])):.3f} |",
        "",
        "## Per-case",
        "",
        "| n | d_min | prior MPE | prior CC | hP1+PS CC | solved | CF CC |",
        "|---|-------|-----------|----------|-----------|--------|-------|",
    ]
    for r in hold:
        md.append(
            f"| {r['n_atoms']} | {r['d_min']} | {r['prior_mpe_deg']:.0f}° | "
            f"{r['prior_mapcc']:.2f} | {r['hard_p1_phaseed_mapcc']:.2f} | "
            f"{r['hard_p1_phaseed_solved']} | {r['cf_mapcc']:.2f} |"
        )
    md.extend([
        "",
        "## Scope",
        "",
        "In-domain prior for **synthetic hard P1** only. Transfer to experimental "
        "or non-P1 space groups is unproven. Use via "
        "`hard_p1_phaseed_solve` / `predict_phases_hard_p1`.",
        "",
        f"Train time: {meta['train_seconds']:.1f}s + eval {meta['eval_seconds']:.1f}s",
        "",
    ])
    mp = out.parent / "hard_p1_prior.md"
    mp.write_text("\n".join(md))
    print(f"Wrote {mp}")
    print(
        f"\nSummary: hard_p1+PhaSeed solved {n_h}/{n} (mean CC {mcc_h:.2f}) "
        f"vs CF {n_c}/{n} (mean CC {mcc_c:.2f})"
    )


if __name__ == "__main__":
    main()
