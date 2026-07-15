#!/usr/bin/env python3
"""
Train stronger GraphPhaseNet prior on hard multi-SG synthetic cells.

Compares hold-out: CF | hard_p1 PhaseMLP+PhaSeed | strong GraphNet+PhaSeed.

Saves data/processed/strong_prior.{npz,json,md}
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
from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.models.strong_prior import (
    predict_full_phases,
    save_strong_prior,
    strong_prior_phaseed_solve,
    train_strong_prior,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve


def holdout_solve_compare(model, n_eval=6, seed=4242, n_iter=50, n_extend=12, max_refl=100):
    rows = []
    rng = np.random.default_rng(seed)
    # optional hard_p1 for comparison
    hp1 = None
    try:
        from grok_phase_solver.models.hard_p1_prior import (
            default_hard_p1_path,
            hard_p1_phaseed_solve,
            load_hard_p1_prior,
        )
        p = default_hard_p1_path()
        if p.exists():
            hp1 = load_hard_p1_prior(p)
    except Exception:
        pass

    for i in range(n_eval):
        n_atoms = int(rng.integers(12, 18))
        d_min = float(rng.choice([1.5, 1.7, 2.0]))
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell

        # strong prior only
        ph_p = predict_full_phases(model, hkl, amp, cell, max_reflections=max_refl)
        rho_p = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_p), cell, d_min=d_min
        )
        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_t), cell, shape=rho_p.shape
        )
        cc_p, _ = map_correlation_origin_invariant(rho_p, rho_t)
        mpe, _ = mean_phase_error_origin_invariant(ph_p, ph_t, hkl, weights=amp)

        # strong + PhaSeed
        ph_s, rho_s, _ = strong_prior_phaseed_solve(
            hkl, amp, cell, model=model,
            n_extend=n_extend, polish="charge_flipping", n_polish=n_iter,
            n_starts=2, seed=s, d_min=d_min, max_reflections=max_refl,
        )
        if rho_s.shape != rho_t.shape:
            rho_s = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_s), cell, shape=rho_t.shape
            )
        rep_s = evaluate_success(
            hkl, amp, ph_s, ph_t, cell, data["fracs"], density=rho_s,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )

        # CF
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

        row = {
            "n_atoms": n_atoms,
            "d_min": d_min,
            "seed": s,
            "graph_prior_mpe_oi": float(mpe),
            "graph_prior_mapcc": float(cc_p),
            "strong_phaseed_mapcc": rep_s.mapcc_oi,
            "strong_phaseed_solved": rep_s.solved,
            "strong_phaseed_peak": rep_s.peak_recovery,
            "cf_mapcc": rep_c.mapcc_oi,
            "cf_solved": rep_c.solved,
        }

        if hp1 is not None:
            ph_h, rho_h, _ = hard_p1_phaseed_solve(
                hkl, amp, cell, model=hp1,
                n_extend=n_extend, polish="charge_flipping", n_polish=n_iter,
                n_starts=1, seed=s, d_min=d_min,
            )
            if rho_h.shape != rho_t.shape:
                rho_h = density_from_structure_factors(
                    hkl, amp * np.exp(1j * ph_h), cell, shape=rho_t.shape
                )
            rep_h = evaluate_success(
                hkl, amp, ph_h, ph_t, cell, data["fracs"], density=rho_h,
                elements=data["elements"], thresholds=SuccessThresholds(),
            )
            row["hard_p1_phaseed_mapcc"] = rep_h.mapcc_oi
            row["hard_p1_phaseed_solved"] = rep_h.solved

        rows.append(row)
        extra = ""
        if "hard_p1_phaseed_mapcc" in row:
            extra = f" | hP1={row['hard_p1_phaseed_mapcc']:.2f}"
        print(
            f"  n={n_atoms} d={d_min:.1f}: graph prior MPE={mpe:.0f}° CC={cc_p:.2f} | "
            f"strong+PS CC={rep_s.mapcc_oi:.2f} sol={rep_s.solved} | "
            f"CF={rep_c.mapcc_oi:.2f}{extra}"
        )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-structures", type=int, default=50)
    p.add_argument("--epochs-per", type=int, default=30)
    p.add_argument("--epochs-refine", type=int, default=10)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--max-refl", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default="data/processed/strong_prior.npz")
    p.add_argument("--n-eval", type=int, default=6)
    args = p.parse_args()

    if args.quick:
        args.n_structures = 12
        args.epochs_per = 15
        args.epochs_refine = 5
        args.hidden = 48
        args.max_refl = 60
        args.n_eval = 3

    t0 = time.time()
    model, meta = train_strong_prior(
        n_structures=args.n_structures,
        epochs_per=args.epochs_per,
        epochs_refine=args.epochs_refine,
        hidden=args.hidden,
        max_reflections=args.max_refl,
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    save_strong_prior(model, out, meta=meta)
    print(f"Saved {out}")

    print("\nHold-out hard solve comparison:")
    t1 = time.time()
    hold = holdout_solve_compare(
        model, n_eval=args.n_eval, seed=args.seed + 333,
        n_iter=40 if args.quick else 50,
        n_extend=8 if args.quick else 12,
        max_refl=args.max_refl,
    )
    meta["holdout_solve"] = hold
    meta["eval_seconds"] = time.time() - t1
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))

    n = len(hold)
    n_s = sum(1 for r in hold if r["strong_phaseed_solved"])
    n_c = sum(1 for r in hold if r["cf_solved"])
    mcc_s = float(np.mean([r["strong_phaseed_mapcc"] for r in hold]))
    mcc_c = float(np.mean([r["cf_mapcc"] for r in hold]))
    mcc_p = float(np.mean([r["graph_prior_mapcc"] for r in hold]))
    mpe = float(np.mean([r["graph_prior_mpe_oi"] for r in hold]))
    mcc_h = None
    if any("hard_p1_phaseed_mapcc" in r for r in hold):
        mcc_h = float(np.mean([r["hard_p1_phaseed_mapcc"] for r in hold if "hard_p1_phaseed_mapcc" in r]))

    md = [
        "# Strong phase prior (GraphPhaseNet)",
        "",
        "Triplet-graph message-passing network trained on **hard multi-SG** "
        "synthetic cells (P1 + P−1) with origin-invariant loss.",
        "",
        f"- Structures: **{args.n_structures}**",
        f"- Hidden={args.hidden}, layers=2, max strong reflections={args.max_refl}",
        f"- Mean train MPE_OI: **{meta.get('mean_train_mpe_oi', float('nan')):.1f}°**",
        f"- Mean hold-out MPE_OI: **{meta.get('mean_holdout_mpe_oi', float('nan')):.1f}°**",
        f"- Weights: `{out.relative_to(ROOT)}`",
        "",
        "## Hold-out hard-region comparison",
        "",
        "| Method | Solved | Rate | mean mapCC |",
        "|--------|--------|------|------------|",
        f"| Graph prior only | — | — | {mcc_p:.3f} (MPE_OI {mpe:.0f}°) |",
        f"| **Graph + AI-PhaSeed** | {n_s} | {n_s/max(n,1):.0%} | **{mcc_s:.3f}** |",
        f"| CF | {n_c} | {n_c/max(n,1):.0%} | {mcc_c:.3f} |",
    ]
    if mcc_h is not None:
        md.append(f"| hard_p1 PhaseMLP + PhaSeed | — | — | {mcc_h:.3f} |")
    md.extend([
        "",
        "## Per-case",
        "",
        "| n | d_min | prior MPE | prior CC | strong+PS CC | solved | CF CC |",
        "|---|-------|-----------|----------|--------------|--------|-------|",
    ])
    for r in hold:
        md.append(
            f"| {r['n_atoms']} | {r['d_min']} | {r['graph_prior_mpe_oi']:.0f}° | "
            f"{r['graph_prior_mapcc']:.2f} | {r['strong_phaseed_mapcc']:.2f} | "
            f"{r['strong_phaseed_solved']} | {r['cf_mapcc']:.2f} |"
        )
    md.extend([
        "",
        "## Scope",
        "",
        "Stronger than per-reflection PhaseMLP by using **Cochran triplet graph** "
        "message passing. Still a synthetic hard-region prior — not a claimed "
        "general experimental solver. Use via `strong_prior_phaseed_solve`.",
        "",
        f"Train {meta['train_seconds']:.1f}s + eval {meta['eval_seconds']:.1f}s",
        "",
    ])
    mp = out.parent / "strong_prior.md"
    mp.write_text("\n".join(md))
    print(f"Wrote {mp}")
    print(
        f"\nSummary: strong+PhaSeed solved {n_s}/{n} (CC {mcc_s:.2f}) "
        f"vs CF {n_c}/{n} (CC {mcc_c:.2f})"
        + (f" vs hP1 CC {mcc_h:.2f}" if mcc_h is not None else "")
    )


if __name__ == "__main__":
    main()
