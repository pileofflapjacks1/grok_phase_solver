#!/usr/bin/env python3
"""
Train physics-recycle net (PhaseMLP in phase_recycle loop) on hard synthetic cells.

Hard region: n_atoms ∈ [12, 20], d_min ∈ [1.5, 2.0].

Saves weights + meta + optional small eval table.
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
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.phase_recycle import phase_recycle
from grok_phase_solver.solvers.recycle_net import (
    recycle_net_solve,
    save_recycle_net,
    train_recycle_net_hard,
)


def eval_holdout(model, n_eval=4, seed=999, n_iter_cf=60):
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n_eval):
        n_atoms = int(rng.integers(12, 17))
        d_min = float(rng.choice([1.5, 1.8, 2.0]))
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]

        ph_n, rho_n, _ = recycle_net_solve(
            hkl, amp, st.cell, model, n_cycles=8, seed=s, d_min=d_min
        )
        rep_n = evaluate_success(
            hkl, amp, ph_n, ph_true, st.cell, data["fracs"], density=rho_n,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )
        ph_r, rho_r, _ = phase_recycle(
            hkl, amp, st.cell, n_cycles=8, seed=s, d_min=d_min
        )
        rep_r = evaluate_success(
            hkl, amp, ph_r, ph_true, st.cell, data["fracs"], density=rho_r,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )
        ph_c, rho_c, _ = charge_flipping_solve(
            hkl, amp, st.cell, n_iter=n_iter_cf, seed=s, d_min=d_min
        )
        rep_c = evaluate_success(
            hkl, amp, ph_c, ph_true, st.cell, data["fracs"], density=rho_c,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )
        rows.append({
            "n_atoms": n_atoms,
            "d_min": d_min,
            "seed": s,
            "recycle_net_mapcc": rep_n.mapcc_oi,
            "recycle_net_solved": rep_n.solved,
            "recycle_mapcc": rep_r.mapcc_oi,
            "recycle_solved": rep_r.solved,
            "cf_mapcc": rep_c.mapcc_oi,
            "cf_solved": rep_c.solved,
        })
        print(
            f"  holdout n={n_atoms} d={d_min:.1f}: "
            f"net CC={rep_n.mapcc_oi:.2f} recycle={rep_r.mapcc_oi:.2f} CF={rep_c.mapcc_oi:.2f}"
        )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-structures", type=int, default=20)
    p.add_argument("--epochs-per", type=int, default=50)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default="data/processed/recycle_net.npz")
    args = p.parse_args()

    if args.quick:
        args.n_structures = 6
        args.epochs_per = 25
        n_eval = 2
    else:
        n_eval = 4

    print(
        f"Training recycle net on hard cells: "
        f"n_structures={args.n_structures}, epochs_per={args.epochs_per}"
    )
    t0 = time.time()
    model, meta = train_recycle_net_hard(
        n_structures=args.n_structures,
        epochs_per=args.epochs_per,
        hidden=args.hidden,
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    save_recycle_net(model, out, meta=meta)
    print(f"Saved {out}  mean MPE={meta['mean_mpe']:.1f}°")

    print("\nHold-out hard-region eval:")
    hold = eval_holdout(model, n_eval=n_eval, seed=args.seed + 1000)
    meta["holdout"] = hold
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))

    md = [
        "# Physics-recycle net (hard region)",
        "",
        f"- Trained on **{args.n_structures}** synthetic structures with "
        f"**n ∈ [12,20]**, **d_min ∈ [1.5, 2.0]**",
        f"- Hidden={args.hidden}, epochs/structure={args.epochs_per}",
        f"- Mean train MPE: **{meta['mean_mpe']:.1f}°**",
        f"- Weights: `{out.relative_to(ROOT)}`",
        "",
        "## Hold-out comparison (strict success)",
        "",
        "| n | d_min | recycle_net CC | pure recycle CC | CF CC | net solved |",
        "|---|-------|----------------|-----------------|-------|------------|",
    ]
    for r in hold:
        md.append(
            f"| {r['n_atoms']} | {r['d_min']} | {r['recycle_net_mapcc']:.3f} | "
            f"{r['recycle_mapcc']:.3f} | {r['cf_mapcc']:.3f} | {r['recycle_net_solved']} |"
        )
    md.extend([
        "",
        "## Scope",
        "",
        "Supervised on synthetic φ_true in the hard region; physics recycle enforces "
        "|F| consistency. Not a claimed general experimental solver.",
        "",
    ])
    mp = out.with_suffix(".md")
    if mp.name.endswith(".npz.md"):
        mp = out.parent / "recycle_net.md"
    else:
        mp = out.parent / "recycle_net.md"
    mp.write_text("\n".join(md))
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
