#!/usr/bin/env python3
"""
Train / evaluate GraphPhaseNet **v7** strong prior (GraPhAI HA-aware features).

v7 upgrades vs v6:
  - d_in=22 node features: v5 + ha_E_tail, low_res_w, E·low_res, κ_centrality
  - Stronger κ-gated edges + self-loops (GraPhAI physics edges)
  - Melgalvis curriculum defaults (cod / hard presets, low-res, Wilson match)
  - Cluster-friendly --scale / --scale-xl with --continue-from resume

Scale targets (wall-clock dependent):
  --quick     N≈80    (smoke / CI)
  --pilot     N≈300   (default laptop pilot — honest numbers)
  --scale     N≈2000
  --scale-xl  N≈5000–10000 (cluster; long)

Writes:
  data/processed/strong_prior_v7.{npz,json,md}

Honest: does not claim ≥30% seed bar unless hold-out metrics show it.
Does not claim general macromolecular ab initio solution.
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


def holdout_panel(model, n_eval=12, seed=4242, max_refl=140, n_extend=10, n_iter=40):
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

        ph_p = predict_full_phases(model, hkl, amp, cell, max_reflections=max_refl)
        sm = full_and_strong_metrics(
            ph_p, ph_t, hkl, amp, cell, fraction=0.30, within_deg=20.0
        )
        ph_cf, rho_cf, _ = charge_flipping_solve(
            hkl, amp, cell, n_iter=n_iter, seed=s, d_min=d_min
        )
        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_t), cell, d_min=d_min
        )
        if rho_cf.shape != rho_t.shape:
            rho_cf = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_cf), cell, shape=rho_t.shape
            )
        cc_cf, _ = map_correlation_origin_invariant(rho_cf, rho_t)
        rho_p = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_p), cell, shape=rho_t.shape
        )
        cc_p, _ = map_correlation_origin_invariant(rho_p, rho_t)

        ph_s, rho_s, _ = strong_prior_phaseed_solve(
            hkl, amp, cell, model=model,
            n_extend=n_extend, polish="none", n_polish=n_iter,
            n_starts=1, seed=s, d_min=d_min, max_reflections=max_refl,
        )
        if rho_s.shape != rho_t.shape:
            rho_s = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_s), cell, shape=rho_t.shape
            )
        rep = evaluate_success(
            hkl, amp, ph_s, ph_t, cell, data["fracs"], density=rho_s,
            elements=data["elements"], thresholds=SuccessThresholds(),
        )
        rows.append(
            {
                "n_atoms": n_atoms,
                "d_min": d_min,
                "frac_within_20": sm["frac_within_deg"],
                "strong_mpe_oi": sm["strong_mpe_oi"],
                "seedOK": bool(sm["would_seed_solve"]),
                "mapcc_prior": float(cc_p),
                "mapcc_cf": float(cc_cf),
                "mapcc_phaseed": rep.mapcc_oi,
                "solved": rep.solved,
            }
        )
        print(
            f"  hold {i+1}/{n_eval} frac≤20°={sm['frac_within_deg']:.0%} "
            f"strongMPE={sm['strong_mpe_oi']:.0f}° seedOK={sm['would_seed_solve']} "
            f"CC_prior={cc_p:.2f} CC_ps={rep.mapcc_oi:.2f}",
            flush=True,
        )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description="GraphPhaseNet v7 train + hold-out")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--scale", action="store_true")
    p.add_argument("--scale-xl", action="store_true")
    p.add_argument("--n-structures", type=int, default=None)
    p.add_argument("--out", type=str, default="data/processed/strong_prior_v7.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--no-melgalvis", action="store_true")
    p.add_argument("--no-wilson-match", action="store_true")
    p.add_argument(
        "--melgalvis-preset",
        type=str,
        default="acta2026",
        choices=["none", "cod", "hard", "acta2026"],
        help="Melgalvis curriculum preset (v0.8 default: cod-like volumes)",
    )
    p.add_argument(
        "--low-res-frac",
        type=float,
        default=0.18,
        help="Fraction of samples forced to d_min 1.8–2.5 Å (GraPhAI-like)",
    )
    p.add_argument("--continue-from", type=str, default=None)
    args = p.parse_args()

    if args.scale_xl:
        cfg = dict(
            n_structures=5000, epochs_per=12, epochs_refine=6, n_global_passes=3,
            hidden=192, n_layers=4, max_refl=160, scale_tag="v7_xl",
            hard_oversample=1.5, n_eval=16,
        )
    elif args.scale:
        cfg = dict(
            n_structures=2000, epochs_per=14, epochs_refine=6, n_global_passes=3,
            hidden=160, n_layers=4, max_refl=140, scale_tag="v7_scale",
            hard_oversample=1.4, n_eval=14,
        )
    elif args.quick:
        cfg = dict(
            n_structures=80, epochs_per=8, epochs_refine=4, n_global_passes=1,
            hidden=96, n_layers=3, max_refl=100, scale_tag="v7_quick",
            hard_oversample=1.25, n_eval=8,
        )
    else:
        # pilot default — laptop-friendly (~tens of minutes)
        cfg = dict(
            n_structures=300, epochs_per=12, epochs_refine=6, n_global_passes=2,
            hidden=160, n_layers=4, max_refl=140, scale_tag="v7_pilot",
            hard_oversample=1.35, n_eval=12,
        )

    if args.n_structures is not None:
        cfg["n_structures"] = int(args.n_structures)
    if args.n_eval is not None:
        cfg["n_eval"] = int(args.n_eval)

    init_model = None
    if args.continue_from:
        from grok_phase_solver.models.strong_prior import load_strong_prior

        cp = Path(args.continue_from)
        if not cp.is_absolute():
            cp = ROOT / cp
        init_model = load_strong_prior(cp)
        print(f"Continue from {cp}", flush=True)

    print("=== GraphPhaseNet v7 (GraPhAI multipath + bin CE) ===", flush=True)
    print(json.dumps({k: cfg[k] for k in cfg}, indent=2), flush=True)

    t0 = time.time()
    model, meta = train_strong_prior(
        n_structures=cfg["n_structures"],
        epochs_per=cfg["epochs_per"],
        epochs_refine=cfg["epochs_refine"],
        n_global_passes=cfg["n_global_passes"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        max_reflections=cfg["max_refl"],
        triplet_weight=0.24,
        curriculum=True,
        wilson_match=not args.no_wilson_match,
        e_power=2.6,
        top_frac=0.30,
        top_boost=5.5,
        within_weight=0.58,
        residual=True,
        optimizer="adam",
        d_in=22,
        hard_oversample=cfg["hard_oversample"],
        scale_tag=cfg["scale_tag"],
        init_model=init_model,
        bridge_frac=0.28,
        use_melgalvis_gen=not args.no_melgalvis,
        melgalvis_mode="hybrid",
        melgalvis_large_vol=True,
        melgalvis_preset=None if args.melgalvis_preset == "none" else args.melgalvis_preset,
        include_low_res_frac=float(args.low_res_frac),
        feature_version=7,
        bin_weight=0.18,
        n_phase_bins=4,
        bin_mode="auto",
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    meta["cli"] = cfg
    meta["feature_version"] = 6
    meta["d_in"] = 18

    print("\n=== Hold-out panel ===", flush=True)
    rows = holdout_panel(
        model, n_eval=cfg["n_eval"], seed=args.seed + 99, max_refl=cfg["max_refl"]
    )
    frac = float(np.mean([r["frac_within_20"] for r in rows]))
    seed_ok = float(np.mean([1.0 if r["seedOK"] else 0.0 for r in rows]))
    mpe = float(np.mean([r["strong_mpe_oi"] for r in rows]))
    solved = float(np.mean([1.0 if r["solved"] else 0.0 for r in rows]))
    summary = {
        "mean_frac_within_20": frac,
        "seedOK_rate": seed_ok,
        "mean_strong_mpe_oi": mpe,
        "strict_solve_rate": solved,
        "n_holdout": len(rows),
        "bar_30pct": 0.30,
        "above_legacy_22pct": frac > 0.22,
        "clears_30pct_bar": frac >= 0.30,
        "feature_version": 7,
        "d_in": 22,
        "note": (
            "Honest hold-out on synthetic hard P1 cells. "
            "Legacy plateau ~21–22% frac≤20°. v7 GraPhAI multipath + bin CE; "
            "not a claim of general macromolecular solution."
        ),
    }
    meta["holdout_v7"] = summary
    meta["holdout_rows"] = rows

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    save_strong_prior(model, out, meta=meta)

    js = out.with_suffix(".json")
    js.write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2, default=str)
    )
    md = out.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# GraphPhaseNet v7 strong prior",
                "",
                f"**Scale tag:** `{cfg['scale_tag']}` · **N train:** {cfg['n_structures']}",
                f"**Features:** v7 d_in=22 · GraPhAI multipath · Melgalvis gen · κ-gated edges",
                f"**Preset:** `{args.melgalvis_preset}` · low-res frac={args.low_res_frac}",
                "",
                "## Hold-out seed quality",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| mean frac ≤20° | **{100*frac:.1f}%** |",
                f"| seedOK rate (≥30% of strong ≤20°) | **{100*seed_ok:.1f}%** |",
                f"| mean strong MPE OI | {mpe:.1f}° |",
                f"| strict solve rate (PhaSeed polish none) | {100*solved:.1f}% |",
                f"| clears 30% oracle bar? | {'YES' if frac >= 0.30 else '**NO**'} |",
                f"| above legacy ~22% plateau? | {'yes' if frac > 0.22 else 'no / comparable'} |",
                "",
                f"Train wall time: {meta['train_seconds']:.0f}s",
                "",
                "## Honest limits",
                "",
                "- Hard ab initio strict solves remain rare without partial-φ.",
                "- Numbers are synthetic hold-out; experimental COD may differ.",
                "- Cluster scale: "
                "`python scripts/run_strong_prior_v7.py --scale-xl --melgalvis-preset cod`",
                "- Resume: `--continue-from data/processed/strong_prior_v7.npz`",
                "",
                f"Weights: `{out.name}`",
                "",
            ]
        )
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {md}", flush=True)


if __name__ == "__main__":
    main()
