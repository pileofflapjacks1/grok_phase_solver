#!/usr/bin/env python3
"""
Calibrate / validate free-FOM against truth mapCC.

Builds phase sets: true, random, CF, RAAR, (optional PhAI) on synthetic +
COD Fcalc. Reports:

  - Spearman ρ(composite, mapCC_OI)
  - Ranking accuracy: P(FOM ranks better map higher)
  - Conditional polish gate: TP/FP vs truth mapCC improvement
  - v1 pathology check: old R_after_ER was always ~0 (documented)

Writes data/processed/free_fom_calibration.{json,md}
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
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.conditional_hybrid import conditional_polish
from grok_phase_solver.solvers.free_fom import free_fom, should_accept_polish
from grok_phase_solver.solvers.iterative_retrieval import raar_solve
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    if den < 1e-16:
        return 0.0
    return float(np.dot(rx, ry) / den)


def pairwise_rank_accuracy(composites, mapccs):
    """Fraction of pairs where FOM order agrees with mapCC order."""
    n = len(composites)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(mapccs[i] - mapccs[j]) < 0.02:
                continue  # skip near-ties on truth
            total += 1
            fom_i_better = composites[i] > composites[j]
            cc_i_better = mapccs[i] > mapccs[j]
            if fom_i_better == cc_i_better:
                agree += 1
    return agree / total if total else float("nan"), total


def eval_phase_set(label, hkl, amp, cell, phases, rho_true, density=None):
    if density is None:
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases), cell
        )
    # match shapes for mapCC
    if density.shape != rho_true.shape:
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases), cell, shape=rho_true.shape
        )
    cc, _ = map_correlation_origin_invariant(density, rho_true)
    fom = free_fom(hkl, amp, phases, cell, density=density)
    return {
        "label": label,
        "mapcc_oi": cc,
        "composite": fom["composite"],
        "R_pos": fom["R_pos"],
        "skewness": fom["skewness"],
        "excess_kurtosis": fom["excess_kurtosis"],
        "max_over_sigma": fom["max_over_sigma"],
        "pos_frac": fom["pos_frac"],
        "score_R_pos": fom["score_R_pos"],
        "score_peakiness": fom["score_peakiness"],
        "R_sayre": fom.get("R_sayre"),
    }


def collect_structure(name, st, d_min, seed=0, n_iter=60):
    data = structure_to_fcalc(st, d_min=d_min)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    cell = st.cell
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), cell, d_min=d_min
    )
    rows = []
    rows.append(eval_phase_set("true", hkl, amp, cell, ph_t, rho_t, rho_t))

    rng = np.random.default_rng(seed)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    rows.append(eval_phase_set("random", hkl, amp, cell, ph_r, rho_t))

    ph_cf, rho_cf, _ = charge_flipping_solve(
        hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
    )
    rows.append(eval_phase_set("cf", hkl, amp, cell, ph_cf, rho_t, rho_cf))

    ph_ra, rho_ra, _ = raar_solve(
        hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
    )
    rows.append(eval_phase_set("raar", hkl, amp, cell, ph_ra, rho_t, rho_ra))

    # Gate study: seed = true (good) → CF polish; and seed = random → CF
    gate_cases = []
    for seed_name, ph0 in [("true_seed", ph_t), ("random_seed", ph_r)]:
        ph_p, rho_p, info = conditional_polish(
            hkl, amp, cell, ph0, polish="charge_flipping",
            n_iter=n_iter, seed=seed, d_min=d_min,
        )
        rho0 = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph0), cell, shape=rho_t.shape
        )
        if rho_p.shape != rho_t.shape:
            rho_p = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph_p), cell, shape=rho_t.shape
            )
        cc0, _ = map_correlation_origin_invariant(rho0, rho_t)
        cc1, _ = map_correlation_origin_invariant(rho_p, rho_t)
        truth_improve = cc1 > cc0 + 0.02
        truth_worsen = cc1 < cc0 - 0.02
        gate_cases.append({
            "structure": name,
            "d_min": d_min,
            "seed_type": seed_name,
            "accepted": info["accepted_polish"],
            "mapcc_seed": cc0,
            "mapcc_final": cc1,
            "truth_improve": truth_improve,
            "truth_worsen": truth_worsen,
            "delta_composite": info["fom_delta"].get("delta_composite"),
            "delta_R_pos": info["fom_delta"].get("delta_R_pos"),
            # gate correctness
            "true_positive": bool(info["accepted_polish"] and truth_improve),
            "false_positive": bool(info["accepted_polish"] and truth_worsen),
            "true_negative": bool((not info["accepted_polish"]) and (not truth_improve)),
            "false_negative": bool((not info["accepted_polish"]) and truth_improve),
        })

    # Also: PhAI-like good intermediate seed if available — skip heavy PhAI here
    # Synthetic "partial" seed: blend true with random
    blend = 0.55
    ph_partial = np.angle(
        blend * np.exp(1j * ph_t) + (1 - blend) * np.exp(1j * ph_r)
    )
    rows.append(eval_phase_set("partial", hkl, amp, cell, ph_partial, rho_t))

    for r in rows:
        r["structure"] = name
        r["d_min"] = d_min
        r["n_atoms"] = getattr(st, "n_atoms", len(getattr(st, "atoms", [])) or data.get("n_atoms_cell"))
        r["n_refl"] = len(amp)

    return rows, gate_cases


def cod_gate_study():
    """Reproduce COD 2016452 free-FOM gate on PhAI seed if possible."""
    path = ROOT / "data/raw/cod/2016452.cif"
    if not path.exists():
        return []
    st = load_cif(path)
    results = []
    try:
        from grok_phase_solver.models.phai_fair import run_phai_fair
        from grok_phase_solver.models.phai_runner import phai_available
        has_phai = phai_available()
    except Exception:
        has_phai = False

    for d_min in (0.9, 1.2, 1.5, 2.0):
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell
        rho_t = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_t), cell, d_min=d_min
        )
        if has_phai:
            ph0, _ = run_phai_fair(hkl, amp, n_cycles=5, seed=0)
        else:
            # partial seed proxy
            rng = np.random.default_rng(0)
            ph_r = rng.uniform(-np.pi, np.pi, len(amp))
            ph0 = np.angle(0.6 * np.exp(1j * ph_t) + 0.4 * np.exp(1j * ph_r))

        for polish in ("charge_flipping", "raar"):
            ph_p, rho_p, info = conditional_polish(
                hkl, amp, cell, ph0, polish=polish, n_iter=60, seed=0, d_min=d_min
            )
            rho0 = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph0), cell, shape=rho_t.shape
            )
            if rho_p.shape != rho_t.shape:
                rho_p = density_from_structure_factors(
                    hkl, amp * np.exp(1j * ph_p), cell, shape=rho_t.shape
                )
            cc0, _ = map_correlation_origin_invariant(rho0, rho_t)
            cc1, _ = map_correlation_origin_invariant(rho_p, rho_t)
            # good_gate: accept only when mapCC not hurt; reject when mapCC not helped
            if info["accepted_polish"]:
                good = cc1 >= cc0 - 0.02
            else:
                good = cc1 <= cc0 + 0.05  # reject ok unless polish would have helped a lot
            results.append({
                "structure": "COD_2016452",
                "d_min": d_min,
                "seed": "phai_fair" if has_phai else "partial",
                "polish": polish,
                "accepted": info["accepted_polish"],
                "mapcc_seed": cc0,
                "mapcc_final": cc1,
                "delta_mapcc": cc1 - cc0,
                "composite_seed": info["fom_seed"]["composite"],
                "composite_polished": info["fom_polished"]["composite"],
                "R_pos_seed": info["fom_seed"]["R_pos"],
                "R_pos_polished": info["fom_polished"]["R_pos"],
                "phase_displacement": info.get("phase_displacement"),
                "good_gate": good,
            })
            print(
                f"  COD d={d_min} {polish}: accept={info['accepted_polish']} "
                f"CC {cc0:.3f}→{cc1:.3f} R₊ {info['fom_seed']['R_pos']:.3f}→"
                f"{info['fom_polished']['R_pos']:.3f} "
                f"disp={info.get('phase_displacement', float('nan')):.3f} good={good}"
            )
    return results


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    all_rows = []
    all_gates = []

    # Synthetic grid
    if args.quick:
        cases = [(6, 1.0, 0), (12, 1.5, 0)]
        n_iter = 40
    else:
        cases = [
            (4, 0.9, 0), (6, 1.0, 0), (6, 1.0, 1),
            (8, 1.0, 0), (8, 1.2, 1),
            (12, 1.5, 0), (12, 1.5, 1), (16, 1.5, 0),
        ]
        n_iter = 60

    print("=== Synthetic ranking calibration ===")
    for n_atoms, d_min, seed in cases:
        st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
        name = f"synth_n{n_atoms}_d{d_min}_s{seed}"
        print(f"  {name}")
        rows, gates = collect_structure(name, st, d_min, seed=seed, n_iter=n_iter)
        all_rows.extend(rows)
        all_gates.extend(gates)

    comps = [r["composite"] for r in all_rows]
    ccs = [r["mapcc_oi"] for r in all_rows]
    rho_s = spearman(comps, ccs)
    pair_acc, n_pairs = pairwise_rank_accuracy(comps, ccs)

    # true vs random ranking
    true_beats_random = 0
    tr_n = 0
    by_struct = {}
    for r in all_rows:
        by_struct.setdefault((r["structure"], r["d_min"]), {})[r["label"]] = r
    for key, d in by_struct.items():
        if "true" in d and "random" in d:
            tr_n += 1
            if d["true"]["composite"] > d["random"]["composite"]:
                true_beats_random += 1

    # Gate stats
    n_fp = sum(1 for g in all_gates if g["false_positive"])
    n_tp = sum(1 for g in all_gates if g["true_positive"])
    n_fn = sum(1 for g in all_gates if g["false_negative"])
    n_tn = sum(1 for g in all_gates if g["true_negative"])
    n_g = len(all_gates)

    print("\n=== COD 2016452 hybrid gate ===")
    cod = cod_gate_study()

    # FOM inversion: among phase-set types, does any wrong set beat true composite?
    inversion_events = 0
    inversion_total = 0
    hard_inv = 0
    hard_n = 0
    for key, d in by_struct.items():
        if "true" not in d:
            continue
        name = key[0] if isinstance(key, tuple) else str(key)
        is_hard = ("n12" in name or "n16" in name or "n20" in name or
                   (isinstance(key, tuple) and len(key) > 1 and key[1] >= 1.5))
        # check from all_rows for this structure
        sub = [r for r in all_rows if r["structure"] == d["true"]["structure"]]
        if not sub:
            continue
        c_true = d["true"]["composite"]
        wrong = [r for r in sub if r["label"] in ("cf", "raar", "random", "partial") and r["mapcc_oi"] < 0.45]
        if not wrong:
            continue
        inversion_total += 1
        if is_hard:
            hard_n += 1
        if any(r["composite"] > c_true + 0.02 for r in wrong):
            inversion_events += 1
            if is_hard:
                hard_inv += 1

    # Also: mean composite gap (true - cf) on hard-ish
    gaps = []
    for key, d in by_struct.items():
        if "true" in d and "cf" in d:
            gaps.append(d["true"]["composite"] - d["cf"]["composite"])

    payload = {
        "spearman_composite_vs_mapcc": rho_s,
        "pairwise_rank_accuracy": pair_acc,
        "n_pairs": n_pairs,
        "true_beats_random_rate": true_beats_random / tr_n if tr_n else None,
        "n_structures_true_rand": tr_n,
        "fom_inversion_rate": inversion_events / inversion_total if inversion_total else None,
        "fom_inversion_events": inversion_events,
        "fom_inversion_total": inversion_total,
        "hard_fom_inversion_rate": hard_inv / hard_n if hard_n else None,
        "mean_true_minus_cf_composite": float(np.mean(gaps)) if gaps else None,
        "gate": {
            "n": n_g,
            "true_positive": n_tp,
            "false_positive": n_fp,
            "true_negative": n_tn,
            "false_negative": n_fn,
            "precision": n_tp / (n_tp + n_fp) if (n_tp + n_fp) else None,
            "false_positive_rate": n_fp / n_g if n_g else None,
        },
        "rows": all_rows,
        "gates": all_gates,
        "cod_gates": cod,
        "seconds": time.time() - t0,
        "fom_version": 2.1,
        "notes": (
            "v2.1: anti-false-atomicity (inverted-U kurtosis/peakiness, peak balance, AFA). "
            "R_pos is positivity residual BEFORE modulus re-imposition. "
            "Gate requires composite gain, no serious R_pos regression, rewrite trust-region."
        ),
    }

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "free_fom_calibration.json"
    jp.write_text(json.dumps(payload, indent=2, default=float))

    md = [
        "# Free-FOM calibration (v2)",
        "",
        "## Math fix",
        "",
        "Old `R_after_ER` was computed **after** modulus projection → always ≈ 0 "
        "(vacuous). New **R₊** = R-factor of `|FFT(max(ρ,0))|` vs `|F_obs|` "
        "(positivity residual) — informative and truth-free.",
        "",
        "Composite combines scored R₊, excess kurtosis, peakiness (max/σ + top mass), "
        "skew, weak positivity fraction, plus light shell-R₊ and Sayre terms.",
        "",
        "## Ranking vs truth mapCC",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Spearman ρ(composite, mapCC_OI) | **{rho_s:.3f}** |",
        f"| Pairwise rank accuracy | **{pair_acc:.1%}** (n={n_pairs}) |",
        f"| P(true FOM > random FOM) | **{true_beats_random/tr_n if tr_n else float('nan'):.1%}** |",
        f"| FOM inversion rate (wrong beats true) | "
        f"**{payload['fom_inversion_rate'] if payload['fom_inversion_rate'] is not None else float('nan'):.1%}** "
        f"({payload['fom_inversion_events']}/{payload['fom_inversion_total']}) |",
        f"| mean (C_true − C_cf) | "
        f"**{payload['mean_true_minus_cf_composite'] if payload['mean_true_minus_cf_composite'] is not None else float('nan'):.3f}** |",
        f"| free-FOM version | {payload['fom_version']} |",
        "",
        "## Conditional polish gate (synthetic)",
        "",
        "Accept only if composite↑ and R₊ does not regress badly.",
        "",
        f"| TP | FP | TN | FN | precision | FP rate |",
        f"|----|----|----|----|-----------|---------|",
        f"| {n_tp} | {n_fp} | {n_tn} | {n_fn} | "
        f"{(n_tp/(n_tp+n_fp) if n_tp+n_fp else float('nan')):.2f} | "
        f"{(n_fp/n_g if n_g else float('nan')):.2f} |",
        "",
        "## COD 2016452 seed→polish gate",
        "",
        "Rewrite trust-region: large \(D_\\varphi\) requires \(\\Delta R_+ \\ge 0.08\).",
        "",
        "| d_min | polish | seed | accept | mapCC seed→final | R₊ seed→final | disp | good gate |",
        "|-------|--------|------|--------|------------------|---------------|------|-----------|",
    ]
    for r in cod:
        md.append(
            f"| {r['d_min']} | {r['polish']} | {r['seed']} | {r['accepted']} | "
            f"{r['mapcc_seed']:.3f}→{r['mapcc_final']:.3f} | "
            f"{r['R_pos_seed']:.3f}→{r['R_pos_polished']:.3f} | "
            f"{r.get('phase_displacement', float('nan')):.3f} | {r['good_gate']} |"
        )
    if cod:
        n_good = sum(1 for r in cod if r["good_gate"])
        md.append("")
        md.append(f"COD gate correctness: **{n_good}/{len(cod)}** decisions match mapCC interest.")

    # mean composite by label
    md.extend(["", "## Mean composite / mapCC by phase-set type", "",
               "| Label | mean composite | mean mapCC | n |",
               "|-------|----------------|------------|---|"])
    for lab in ("true", "partial", "cf", "raar", "random"):
        sub = [r for r in all_rows if r["label"] == lab]
        if not sub:
            continue
        md.append(
            f"| `{lab}` | {np.mean([r['composite'] for r in sub]):.3f} | "
            f"{np.mean([r['mapcc_oi'] for r in sub]):.3f} | {len(sub)} |"
        )

    md.extend([
        "",
        "## Interpretation",
        "",
        "- Higher Spearman / pairwise accuracy ⇒ free FOM tracks solution quality.",
        "- Low false-positive gate rate ⇒ fewer harmful CF polishes accepted.",
        "- Free FOM remains a **proxy**, not an oracle; always refine experimentally.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        f"Runtime: {payload['seconds']:.1f}s",
        "",
    ])
    mp = out / "free_fom_calibration.md"
    mp.write_text("\n".join(md))
    print(f"\nSpearman ρ={rho_s:.3f}  pair_acc={pair_acc:.1%}  "
          f"true>rand={true_beats_random}/{tr_n}  FP={n_fp}/{n_g}")
    print(f"Wrote {jp}\nWrote {mp}")


if __name__ == "__main__":
    main()
