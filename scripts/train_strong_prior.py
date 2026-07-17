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
from grok_phase_solver.metrics.strong_seed import full_and_strong_metrics
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
        sm = full_and_strong_metrics(
            ph_p, ph_t, hkl, amp, cell, fraction=0.30, within_deg=20.0
        )

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
        sm_ps = full_and_strong_metrics(
            ph_s, ph_t, hkl, amp, cell, fraction=0.30, within_deg=20.0
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
            "graph_prior_mpe_oi": sm["full_mpe_oi"],
            "graph_prior_mapcc": float(cc_p),
            "graph_prior_strong_mpe_oi": sm["strong_mpe_oi"],
            "graph_prior_frac_within_20": sm["frac_within_deg"],
            "graph_prior_would_seed_solve": sm["would_seed_solve"],
            "strong_phaseed_mapcc": rep_s.mapcc_oi,
            "strong_phaseed_solved": rep_s.solved,
            "strong_phaseed_peak": rep_s.peak_recovery,
            "strong_phaseed_frac_within_20": sm_ps["frac_within_deg"],
            "strong_phaseed_would_seed_solve": sm_ps["would_seed_solve"],
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
            sm_h = full_and_strong_metrics(
                ph_h, ph_t, hkl, amp, cell, fraction=0.30, within_deg=20.0
            )
            row["hard_p1_phaseed_mapcc"] = rep_h.mapcc_oi
            row["hard_p1_phaseed_solved"] = rep_h.solved
            row["hard_p1_frac_within_20"] = sm_h["frac_within_deg"]

        rows.append(row)
        extra = ""
        if "hard_p1_phaseed_mapcc" in row:
            extra = f" | hP1={row['hard_p1_phaseed_mapcc']:.2f}"
        print(
            f"  n={n_atoms} d={d_min:.1f}: prior strongMPE={sm['strong_mpe_oi']:.0f}° "
            f"frac20={sm['frac_within_deg']:.0%} seedOK={sm['would_seed_solve']} "
            f"CC={cc_p:.2f} | +PS CC={rep_s.mapcc_oi:.2f} sol={rep_s.solved} "
            f"frac20={sm_ps['frac_within_deg']:.0%} | CF={rep_c.mapcc_oi:.2f}{extra}"
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
        help="v3-scale defaults (250 structs, H=128, L=3, 3 passes)",
    )
    p.add_argument(
        "--scale-xl",
        action="store_true",
        help=(
            "Lane A XL scale: ~1200 structs, H=192, L=4 residual+Adam, "
            "hard oversample, stronger seed loss (Wilson recommended)"
        ),
    )
    p.add_argument(
        "--wilson-match",
        action="store_true",
        help="Match synthetic |F| to experimental Wilson template before training",
    )
    p.add_argument("--out", type=str, default="data/processed/strong_prior.npz")
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--hard-oversample", type=float, default=None)
    p.add_argument("--top-boost", type=float, default=None)
    p.add_argument("--within-weight", type=float, default=None)
    p.add_argument("--e-power", type=float, default=None)
    p.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "sgd"],
        help="Optimizer (default adam for v4)",
    )
    p.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Fine-tune from existing strong_prior.npz",
    )
    p.add_argument(
        "--hard-only",
        action="store_true",
        help="No bridge curriculum cells (hard region only)",
    )
    p.add_argument(
        "--seed-focus",
        action="store_true",
        help="Concentrate loss on top ~30% |E| nodes (seed set)",
    )
    args = p.parse_args()

    # Defaults: medium (legacy-ish) unless --scale / --scale-xl / --quick
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
        residual=True,
        optimizer="adam",
        e_power=2.0,
        top_frac=0.50,
        top_boost=3.0,
        within_weight=0.35,
        hard_oversample=1.0,
        scale_tag="v4_scale_seed",
        lr=1.5e-3,
        lr_refine=4e-4,
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
            scale_tag="v4_scale_seed",
        )
    if args.scale_xl:
        cfg.update(
            n_structures=1200,
            epochs_per=10,
            epochs_refine=5,
            n_global_passes=4,
            hidden=192,
            n_layers=4,
            max_refl=140,
            triplet_weight=0.22,
            n_eval=12,
            e_power=2.5,
            top_frac=0.40,
            top_boost=4.0,
            within_weight=0.55,
            hard_oversample=1.4,
            residual=True,
            optimizer="adam",
            scale_tag="v4_scale_xl",
            lr=1.2e-3,
            lr_refine=3.5e-4,
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
            scale_tag="v4_quick",
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
    if args.hard_oversample is not None:
        cfg["hard_oversample"] = args.hard_oversample
    if args.top_boost is not None:
        cfg["top_boost"] = args.top_boost
    if args.within_weight is not None:
        cfg["within_weight"] = args.within_weight
    if args.e_power is not None:
        cfg["e_power"] = args.e_power
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    if args.hard_only:
        cfg["bridge_frac"] = 0.0
        cfg["hard_oversample"] = max(cfg.get("hard_oversample", 1.0), 1.0)
        cfg["scale_tag"] = cfg.get("scale_tag", "v4") + "_hard"
    else:
        cfg["bridge_frac"] = 0.30
    if args.seed_focus:
        cfg["top_frac"] = 0.30
        cfg["top_boost"] = max(cfg.get("top_boost", 3.0), 6.0)
        cfg["within_weight"] = max(cfg.get("within_weight", 0.35), 0.75)
        cfg["e_power"] = max(cfg.get("e_power", 2.0), 3.0)
        cfg["scale_tag"] = cfg.get("scale_tag", "v4") + "_seedfocus"

    init_model = None
    if args.continue_from:
        from grok_phase_solver.models.strong_prior import load_strong_prior

        cp = Path(args.continue_from)
        if not cp.is_absolute():
            cp = ROOT / cp
        init_model = load_strong_prior(cp)
        cfg["hidden"] = init_model.hidden
        cfg["n_layers"] = init_model.n_layers
        cfg["scale_tag"] = cfg.get("scale_tag", "v4") + "_ft"
        print(f"Continue from {cp} (H={init_model.hidden}, L={init_model.n_layers})")

    print(
        f"Config: structs={cfg['n_structures']} hidden={cfg['hidden']} "
        f"layers={cfg['n_layers']} residual={cfg['residual']} "
        f"passes={cfg['n_global_passes']} epochs/struct={cfg['epochs_per']} "
        f"max_refl={cfg['max_refl']} triplet_w={cfg['triplet_weight']} "
        f"wilson_match={cfg['wilson_match']} opt={cfg['optimizer']} "
        f"hard_os={cfg['hard_oversample']} tag={cfg['scale_tag']}"
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
        e_power=cfg["e_power"],
        top_frac=cfg["top_frac"],
        top_boost=cfg["top_boost"],
        within_weight=cfg["within_weight"],
        residual=cfg["residual"],
        optimizer=cfg["optimizer"],
        hard_oversample=cfg["hard_oversample"],
        scale_tag=cfg["scale_tag"],
        lr=cfg["lr"] * (0.4 if init_model is not None else 1.0),
        lr_refine=cfg["lr_refine"] * (0.5 if init_model is not None else 1.0),
        init_model=init_model,
        bridge_frac=cfg.get("bridge_frac", 0.30),
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
    smpe = float(np.mean([r["graph_prior_strong_mpe_oi"] for r in hold]))
    frac20 = float(np.mean([r["graph_prior_frac_within_20"] for r in hold]))
    n_seed_ok = sum(1 for r in hold if r.get("graph_prior_would_seed_solve"))
    frac20_ps = float(np.mean([r["strong_phaseed_frac_within_20"] for r in hold]))
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
        n_h = sum(1 for r in hold if r.get("hard_p1_phaseed_solved"))

    md = [
        f"# Strong phase prior (GraphPhaseNet) — {meta.get('scale', 'v4')}",
        "",
        "Triplet-graph prior with residual MP + Adam (v4), **Wilson-matched |F|** "
        "(optional), **strong-|E| loss reweight**, and **within-20°** focus. "
        "Success bar for the hard cliff: ≥30% of strong |E| phases within 20° of "
        "truth (oracle AI-PhaSeed threshold).",
        "",
        f"- Structures: **{cfg['n_structures']}** (packs: {meta.get('n_packs', '?')}, "
        f"Wilson-matched train: {meta.get('n_wilson_matched', 0)})",
        f"- Hidden={cfg['hidden']}, layers={cfg['n_layers']}, "
        f"residual={meta.get('residual', cfg.get('residual'))}, "
        f"d_in={meta.get('d_in', '?')}, max_refl={cfg['max_refl']}",
        f"- optimizer=**{meta.get('optimizer', cfg.get('optimizer'))}**, "
        f"hard_oversample=**{meta.get('hard_oversample', cfg.get('hard_oversample'))}**",
        f"- wilson_match=**{cfg.get('wilson_match')}**, "
        f"scale=**{meta.get('scale')}**",
        f"- Train strong MPE_OI: **{meta.get('mean_train_strong_mpe_oi', float('nan')):.1f}°**, "
        f"frac≤20°: **{meta.get('mean_train_frac_within_20', float('nan')):.0%}**",
        f"- Hold-out full MPE_OI: **{meta.get('mean_holdout_mpe_oi', float('nan')):.1f}°**, "
        f"strong MPE: **{meta.get('mean_holdout_strong_mpe_oi', float('nan')):.1f}°**, "
        f"frac≤20°: **{meta.get('mean_holdout_frac_within_20', float('nan')):.0%}**, "
        f"would_seed_solve: **{meta.get('holdout_would_seed_solve_rate', float('nan')):.0%}**",
        f"- Weights: `{out.relative_to(ROOT)}`",
        "",
        "## Strong-seed bar (hold-out hard)",
        "",
        "| Metric | Graph prior | Graph+PhaSeed |",
        "|--------|-------------|---------------|",
        f"| strong MPE_OI | {smpe:.1f}° | — |",
        f"| frac ≤20° (top 30% \\|E\\|) | {frac20:.0%} | {frac20_ps:.0%} |",
        f"| would_seed_solve (≥30% within 20°) | {n_seed_ok}/{n} | "
        f"{sum(1 for r in hold if r.get('strong_phaseed_would_seed_solve'))}/{n} |",
        "",
        "## Hold-out hard-region comparison",
        "",
        "| Method | Solved | Rate | mean mapCC |",
        "|--------|--------|------|------------|",
        f"| Graph prior only | — | — | {mcc_p:.3f} (full MPE {mpe:.0f}°) |",
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
            "| n | d_min | strongMPE | frac20 | seedOK | prior CC | +PS CC | solved | CF |",
            "|---|-------|-----------|--------|--------|----------|--------|--------|-----|",
        ]
    )
    for r in hold:
        md.append(
            f"| {r['n_atoms']} | {r['d_min']} | {r['graph_prior_strong_mpe_oi']:.0f}° | "
            f"{r['graph_prior_frac_within_20']:.0%} | {r['graph_prior_would_seed_solve']} | "
            f"{r['graph_prior_mapcc']:.2f} | {r['strong_phaseed_mapcc']:.2f} | "
            f"{r['strong_phaseed_solved']} | {r['cf_mapcc']:.2f} |"
        )
    md.extend(
        [
            "",
            "## Notes (Lane A / v4)",
            "",
            "- **Scale:** `--scale-xl` ≈10³ cells, H=192, L=4 residual MP, Adam.",
            "- **Wilson:** `--wilson-match` aligns synthetic |F| to experimental template.",
            "- **Seed loss:** E^p weights, top-|E| boost, within-20° reweight, hard oversample.",
            "- Oracle bar: ≥30% strong phases within 20° → AI-PhaSeed can strict-solve hard cells.",
            "- v3 baseline was ~21% frac≤20° on hard hold-out; v4 targets ≥30%.",
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
