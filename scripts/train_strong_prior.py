#!/usr/bin/env python3
"""
Train scaled GraphPhaseNet prior on hard multi-SG synthetic cells.

Scale features: more structures, deeper/wider GNN, curriculum multi-pass,
triplet auxiliary loss, vectorized message passing.

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


def holdout_solve_compare(
    model, n_eval=8, seed=4242, n_iter=50, n_extend=12, max_refl=120
):
    rows = []
    rng = np.random.default_rng(seed)
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

        ph_p = predict_full_phases(model, hkl, amp, cell, max_reflections=max_refl)
        rho_p = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_p), cell, d_min=d_min
        )
        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_t), cell, shape=rho_p.shape
        )
        cc_p, _ = map_correlation_origin_invariant(rho_p, rho_t)
        mpe, _ = mean_phase_error_origin_invariant(ph_p, ph_t, hkl, weights=amp)

        ph_s, rho_s, _ = strong_prior_phaseed_solve(
            hkl,
            amp,
            cell,
            model=model,
            n_extend=n_extend,
            polish="charge_flipping",
            n_polish=n_iter,
            n_starts=2,
            seed=s,
            d_min=d_min,
            max_reflections=max_refl,
        )
        if rho_s.shape != rho_t.shape:
            rho_s = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_s), cell, shape=rho_t.shape
            )
        rep_s = evaluate_success(
            hkl,
            amp,
            ph_s,
            ph_t,
            cell,
            data["fracs"],
            density=rho_s,
            elements=data["elements"],
            thresholds=SuccessThresholds(),
        )

        ph_c, rho_c, _ = charge_flipping_solve(
            hkl, amp, cell, n_iter=n_iter, seed=s, d_min=d_min
        )
        if rho_c.shape != rho_t.shape:
            rho_c = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_c), cell, shape=rho_t.shape
            )
        rep_c = evaluate_success(
            hkl,
            amp,
            ph_c,
            ph_t,
            cell,
            data["fracs"],
            density=rho_c,
            elements=data["elements"],
            thresholds=SuccessThresholds(),
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
                hkl,
                amp,
                cell,
                model=hp1,
                n_extend=n_extend,
                polish="charge_flipping",
                n_polish=n_iter,
                n_starts=1,
                seed=s,
                d_min=d_min,
            )
            if rho_h.shape != rho_t.shape:
                rho_h = density_from_structure_factors(
                    hkl, amp * np.exp(1j * ph_h), cell, shape=rho_t.shape
                )
            rep_h = evaluate_success(
                hkl,
                amp,
                ph_h,
                ph_t,
                cell,
                data["fracs"],
                density=rho_h,
                elements=data["elements"],
                thresholds=SuccessThresholds(),
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
    p.add_argument("--n-structures", type=int, default=None)
    p.add_argument("--epochs-per", type=int, default=None)
    p.add_argument("--epochs-refine", type=int, default=None)
    p.add_argument("--n-passes", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--max-refl", type=int, default=None)
    p.add_argument("--triplet-weight", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke run (tests / CI)",
    )
    p.add_argument(
        "--scale",
        action="store_true",
        help="Full scale defaults (250 structs, H=128, L=3, 3 passes)",
    )
    p.add_argument(
        "--wilson-match",
        action="store_true",
        help="Match synthetic |F| to experimental Wilson template before training",
    )
    p.add_argument("--out", type=str, default="data/processed/strong_prior.npz")
    p.add_argument("--n-eval", type=int, default=None)
    args = p.parse_args()

    # Defaults: medium (legacy-ish) unless --scale or --quick
    cfg = dict(
        n_structures=80,
        epochs_per=20,
        epochs_refine=8,
        n_global_passes=2,
        hidden=96,
        n_layers=2,
        max_refl=100,
        triplet_weight=0.15,
        n_eval=6,
        curriculum=True,
        wilson_match=False,
    )
    if args.scale:
        cfg.update(
            n_structures=250,
            epochs_per=18,
            epochs_refine=8,
            n_global_passes=3,
            hidden=128,
            n_layers=3,
            max_refl=120,
            triplet_weight=0.18,
            n_eval=8,
        )
    if args.quick:
        cfg.update(
            n_structures=10,
            epochs_per=8,
            epochs_refine=3,
            n_global_passes=1,
            hidden=48,
            n_layers=2,
            max_refl=50,
            triplet_weight=0.1,
            n_eval=3,
        )

    # CLI overrides
    if args.n_structures is not None:
        cfg["n_structures"] = args.n_structures
    if args.epochs_per is not None:
        cfg["epochs_per"] = args.epochs_per
    if args.epochs_refine is not None:
        cfg["epochs_refine"] = args.epochs_refine
    if args.n_passes is not None:
        cfg["n_global_passes"] = args.n_passes
    if args.hidden is not None:
        cfg["hidden"] = args.hidden
    if args.n_layers is not None:
        cfg["n_layers"] = args.n_layers
    if args.max_refl is not None:
        cfg["max_refl"] = args.max_refl
    if args.triplet_weight is not None:
        cfg["triplet_weight"] = args.triplet_weight
    if args.n_eval is not None:
        cfg["n_eval"] = args.n_eval
    if args.wilson_match:
        cfg["wilson_match"] = True

    print(
        f"Config: structs={cfg['n_structures']} hidden={cfg['hidden']} "
        f"layers={cfg['n_layers']} passes={cfg['n_global_passes']} "
        f"epochs/struct={cfg['epochs_per']} max_refl={cfg['max_refl']} "
        f"triplet_w={cfg['triplet_weight']} wilson_match={cfg['wilson_match']}"
    )

    t0 = time.time()
    model, meta = train_strong_prior(
        n_structures=cfg["n_structures"],
        epochs_per=cfg["epochs_per"],
        epochs_refine=cfg["epochs_refine"],
        n_global_passes=cfg["n_global_passes"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        max_reflections=cfg["max_refl"],
        triplet_weight=cfg["triplet_weight"],
        curriculum=cfg["curriculum"],
        wilson_match=cfg["wilson_match"],
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    meta["cli_config"] = cfg
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    save_strong_prior(model, out, meta=meta)
    print(f"Saved {out}")

    print("\nHold-out hard solve comparison:")
    t1 = time.time()
    hold = holdout_solve_compare(
        model,
        n_eval=cfg["n_eval"],
        seed=args.seed + 333,
        n_iter=40 if args.quick else 50,
        n_extend=8 if args.quick else 12,
        max_refl=cfg["max_refl"],
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
    n_h = None
    if any("hard_p1_phaseed_mapcc" in r for r in hold):
        mcc_h = float(
            np.mean(
                [
                    r["hard_p1_phaseed_mapcc"]
                    for r in hold
                    if "hard_p1_phaseed_mapcc" in r
                ]
            )
        )
        n_h = sum(
            1
            for r in hold
            if r.get("hard_p1_phaseed_solved")
        )

    md = [
        "# Strong phase prior (GraphPhaseNet) — scaled",
        "",
        "Triplet-graph message-passing network trained on **hard multi-SG** "
        "synthetic cells (P1 + P−1) with origin-invariant + triplet auxiliary loss, "
        "curriculum multi-pass training, and vectorized adjacency aggregation.",
        "",
        f"- Structures: **{cfg['n_structures']}** (packs used: {meta.get('n_packs', '?')})",
        f"- Hidden={cfg['hidden']}, layers={cfg['n_layers']}, "
        f"max strong reflections={cfg['max_refl']}",
        f"- Global passes: **{cfg['n_global_passes']}**, "
        f"epochs/struct≈{cfg['epochs_per']}, triplet_w={cfg['triplet_weight']}",
        f"- Curriculum: **{cfg['curriculum']}**",
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
        md.append(
            f"| hard_p1 PhaseMLP + PhaSeed | {n_h if n_h is not None else '—'} | "
            f"{(n_h/max(n,1) if n_h is not None else 0):.0%} | {mcc_h:.3f} |"
        )
    md.extend(
        [
            "",
            "## Per-case",
            "",
            "| n | d_min | prior MPE | prior CC | strong+PS CC | solved | CF CC |",
            "|---|-------|-----------|----------|--------------|--------|-------|",
        ]
    )
    for r in hold:
        md.append(
            f"| {r['n_atoms']} | {r['d_min']} | {r['graph_prior_mpe_oi']:.0f}° | "
            f"{r['graph_prior_mapcc']:.2f} | {r['strong_phaseed_mapcc']:.2f} | "
            f"{r['strong_phaseed_solved']} | {r['cf_mapcc']:.2f} |"
        )
    md.extend(
        [
            "",
            "## Scale notes",
            "",
            "Compared to the first GraphPhaseNet pass (50 structs / H=80 / L=2), this "
            "run increases data, capacity, curriculum multi-pass SGD, and adds a "
            "Cochran triplet-consistency auxiliary. Strict hard-region success remains "
            "difficult; report mapCC honestly vs CF and hard-P1.",
            "",
            "Still a synthetic hard-region prior — not a claimed general experimental "
            "solver. Use via `strong_prior_phaseed_solve` or "
            "`gps-solve --method strong_prior_phaseed`.",
            "",
            f"Train {meta['train_seconds']:.1f}s + eval {meta['eval_seconds']:.1f}s",
            "",
        ]
    )
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
