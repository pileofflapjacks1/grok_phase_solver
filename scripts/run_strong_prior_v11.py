#!/usr/bin/env python3
"""
Train / evaluate GraphPhaseNet **v11** strong prior (v0.12).

v11 upgrades vs v9:
  - d_in=34 node features: v9 multipath depth (hop3, multipath span, Wilson B, E-outlier)
  - Stronger multipath κ-gated edges (feature_version=10, κ power ≥1.65)
  - Higher Carrozzini-style bin CE weight (default 0.24)
  - Melgalvis large-cell / HA curricula; mixed hold-out with P−1 centro path
  - Cluster-friendly --scale / --scale-xl with --continue-from resume

Scale targets (wall-clock dependent):
  --quick     N≈80    (smoke / CI)
  --pilot     N≈300   (default laptop pilot — honest numbers)
  --scale     N≈2000
  --scale-xl  N≈5000–10000 (cluster; long)

Writes:
  data/processed/strong_prior_v11.{npz,json,md}

Honest: does not claim ≥30% seed bar unless hold-out metrics show it.
Does not claim general macromolecular ab initio solution.
Does not redistribute PhAI / SHELX / GraPhAI weights.
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
from grok_phase_solver.solvers.projectors import unit_cell_volume


def holdout_panel(model, n_eval=12, seed=4242, max_refl=140, n_extend=10, n_iter=40):
    from grok_phase_solver.data.synthetic_melgalvis import (
        actas2026_config,
        generate_melgalvis_structure,
        ha_heavy_config,
        large_cell_config,
    )
    from grok_phase_solver.metrics.stratified_prior import (
        is_ha_bearing,
        max_Z_from_elements,
        stratify_by_volume,
        stratify_holdout_rows,
    )

    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n_eval):
        n_atoms = int(rng.integers(12, 20))
        d_min = float(rng.choice([1.4, 1.6, 1.8, 2.0]))
        s = int(rng.integers(0, 2**31 - 1))
        # v0.12: large-cell + HA + optional P−1 (GraPhAI centro path)
        u = rng.random()
        centro = False
        if u < 0.35:
            cfg = large_cell_config()
            st = generate_melgalvis_structure(seed=s, cfg=cfg, space_group="P1")
        elif u < 0.70:
            cfg = ha_heavy_config() if rng.random() < 0.65 else actas2026_config()
            st = generate_melgalvis_structure(seed=s, cfg=cfg, space_group="P1")
        else:
            st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        if rng.random() < 0.30:
            try:
                from grok_phase_solver.data.synthetic_v2 import make_centrosymmetric_copy

                st = make_centrosymmetric_copy(st)
                centro = True
            except Exception:
                pass
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell
        els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
        vol = float(unit_cell_volume(np.asarray(cell, dtype=np.float64)))

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
            hkl,
            amp,
            cell,
            model=model,
            n_extend=n_extend,
            polish="none",
            n_polish=n_iter,
            n_starts=1,
            seed=s,
            d_min=d_min,
            max_reflections=max_refl,
        )
        if rho_s.shape != rho_t.shape:
            rho_s = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_s), cell, shape=rho_t.shape
            )
        rep = evaluate_success(
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
                "max_Z": max_Z_from_elements(els),
                "ha_bearing": is_ha_bearing(els),
                "space_group": getattr(st, "space_group_hm", "P1"),
                "centrosymmetric": bool(centro or "P-1" in str(getattr(st, "space_group_hm", "")).upper()),
                "cell_volume": vol,
            }
        )
        print(
            f"  hold {i+1}/{n_eval} frac≤20°={sm['frac_within_deg']:.0%} "
            f"strongMPE={sm['strong_mpe_oi']:.0f}° seedOK={sm['would_seed_solve']} "
            f"CC_prior={cc_p:.2f} Vol={vol:.0f} maxZ={max_Z_from_elements(els)}",
            flush=True,
        )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description="GraphPhaseNet v11 train + hold-out")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--scale", action="store_true")
    p.add_argument("--scale-xl", action="store_true")
    p.add_argument("--n-structures", type=int, default=None)
    p.add_argument("--out", type=str, default="data/processed/strong_prior_v11.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--no-melgalvis", action="store_true")
    p.add_argument("--no-wilson-match", action="store_true")
    p.add_argument(
        "--melgalvis-preset",
        type=str,
        default="large",
        choices=["none", "cod", "hard", "acta2026", "ha", "large", "xdxd"],
        help="Melgalvis curriculum (default large = Vol~1000–3500 + HA bias)",
    )
    p.add_argument(
        "--low-res-frac",
        type=float,
        default=0.20,
        help="Fraction of samples forced to d_min 1.8–2.5 Å",
    )
    p.add_argument("--continue-from", type=str, default=None)
    args = p.parse_args()

    if args.scale_xl:
        cfg = dict(
            n_structures=5000,
            epochs_per=12,
            epochs_refine=6,
            n_global_passes=3,
            hidden=192,
            n_layers=4,
            max_refl=160,
            scale_tag="v11_xl",
            hard_oversample=1.5,
            n_eval=16,
        )
    elif args.scale:
        cfg = dict(
            n_structures=2000,
            epochs_per=14,
            epochs_refine=6,
            n_global_passes=3,
            hidden=160,
            n_layers=4,
            max_refl=140,
            scale_tag="v11_scale",
            hard_oversample=1.4,
            n_eval=14,
        )
    elif args.quick:
        cfg = dict(
            n_structures=80,
            epochs_per=8,
            epochs_refine=4,
            n_global_passes=1,
            hidden=96,
            n_layers=3,
            max_refl=100,
            scale_tag="v11_quick",
            hard_oversample=1.25,
            n_eval=8,
        )
    else:
        cfg = dict(
            n_structures=300,
            epochs_per=12,
            epochs_refine=6,
            n_global_passes=2,
            hidden=160,
            n_layers=4,
            max_refl=140,
            scale_tag="v11_pilot",
            hard_oversample=1.35,
            n_eval=12,
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

    print("=== GraphPhaseNet v11 (d_in=34 GraPhAI moments + large/HA) ===", flush=True)
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
        d_in=34,
        hard_oversample=cfg["hard_oversample"],
        scale_tag=cfg["scale_tag"],
        init_model=init_model,
        bridge_frac=0.25,
        use_melgalvis_gen=not args.no_melgalvis,
        melgalvis_mode="hybrid",
        melgalvis_large_vol=True,
        melgalvis_preset=None if args.melgalvis_preset == "none" else args.melgalvis_preset,
        include_low_res_frac=float(args.low_res_frac),
        feature_version=10,
        bin_weight=0.26,
        n_phase_bins=4,
        bin_mode="auto",
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    meta["cli"] = cfg
    meta["feature_version"] = 10
    meta["d_in"] = 34
    meta["release"] = "0.13.0"

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
        "feature_version": 10,
        "d_in": 34,
        "note": (
            "Honest hold-out on synthetic hard + large-cell / HA panels. "
            "Legacy plateau ~21–22% frac≤20°. v11 d_in=34 + Melgalvis large/HA; "
            "not a claim of general macromolecular solution."
        ),
    }
    meta["holdout_v11"] = summary
    meta["holdout_rows"] = rows
    from grok_phase_solver.metrics.stratified_prior import (
        format_stratified_md,
        stratify_by_volume,
        stratify_holdout_rows,
    )

    strat = stratify_holdout_rows(rows)
    vol_strat = stratify_by_volume(rows)
    strat["by_volume"] = vol_strat
    meta["holdout_stratified"] = strat
    summary["stratified"] = strat

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
    vol_lines = [
        "## Volume-band stratification",
        "",
        "| Band | n | frac≤20° | seedOK | strong MPE |",
        "|------|---|----------|--------|------------|",
    ]
    for k, lab in [
        ("vol_lt_1000", "Vol < 1000"),
        ("vol_1000_3500", "Vol 1000–3500"),
        ("vol_gt_3500", "Vol > 3500"),
    ]:
        s = vol_strat.get(k) or {"n": 0}
        if s.get("n", 0) == 0:
            vol_lines.append(f"| `{lab}` | 0 | — | — | — |")
        else:
            vol_lines.append(
                f"| `{lab}` | {s['n']} | **{100*s['mean_frac_within_20']:.1f}%** | "
                f"{100*s['seedOK_rate']:.1f}% | {s['mean_strong_mpe_oi']:.1f}° |"
            )
    md.write_text(
        "\n".join(
            [
                "# GraphPhaseNet v11 strong prior",
                "",
                f"**Scale tag:** `{cfg['scale_tag']}` · **N train:** {cfg['n_structures']}",
                f"**Features:** v11 d_in=34 · GraPhAI moments + large/HA · Melgalvis gen",
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
                "- Practical hard path: `partial_phaseed` / fragment / HA (≥~30% strong ≤20°).",
                "- Cluster scale: "
                "`python scripts/run_strong_prior_v11.py --scale-xl --melgalvis-preset large`",
                "- Resume: `--continue-from data/processed/strong_prior_v11.npz`",
                "",
                f"Weights: `{out.name}`",
                "",
                format_stratified_md(strat),
                "",
                *vol_lines,
            ]
        )
    )
    print(json.dumps({k: summary[k] for k in summary if k != "stratified"}, indent=2), flush=True)
    print(json.dumps(strat.get("all", {}), indent=2), flush=True)
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {md}", flush=True)


if __name__ == "__main__":
    main()
