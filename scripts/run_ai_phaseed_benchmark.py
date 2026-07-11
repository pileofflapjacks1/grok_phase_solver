#!/usr/bin/env python3
"""
AI-PhaSeed benchmark vs PhAI-only, CF, and conditional PhAI+CF.

Datasets:
  - Synthetic easy / hard cells
  - COD 2016452 Fcalc at several d_min (if CIF present)

Writes data/processed/ai_phaseed_benchmark.{json,md}
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
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.ai_phaseed import ai_phaseed_solve, phai_phaseed_solve
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.conditional_hybrid import conditional_polish
from grok_phase_solver.solvers.free_fom import free_fom


def _eval(hkl, amp, ph, ph_true, cell, fracs, elements, rho, d_min):
    if rho is None:
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, d_min=d_min
        )
    rep = evaluate_success(
        hkl, amp, ph, ph_true, cell, fracs, density=rho,
        elements=elements, thresholds=SuccessThresholds(),
    )
    return {
        "mapcc_oi": rep.mapcc_oi,
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": rep.solved,
    }


def run_synthetic(n_atoms, d_min, seed, n_iter=60, n_extend=12, n_starts=2):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    cell = st.cell
    rows = []

    def add(method, ph, rho, t0, **extra):
        r = _eval(hkl, amp, ph, ph_t, cell, data["fracs"], data["elements"], rho, d_min)
        r.update({
            "dataset": f"synth_n{n_atoms}",
            "n_atoms": n_atoms,
            "d_min": d_min,
            "seed": seed,
            "method": method,
            "seconds": time.time() - t0,
        })
        r.update(extra)
        rows.append(r)
        flag = "SOLVED" if r["solved"] else "fail"
        print(
            f"  {flag:6s} {method:18s} CC={r['mapcc_oi']:.3f} "
            f"peak={r['peak_recovery']:.2f} R1={r['r1']:.2f}"
        )
        return r

    # CF
    t0 = time.time()
    ph, rho, _ = charge_flipping_solve(
        hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
    )
    add("cf", ph, rho, t0)

    # Oracle AI-PhaSeed (upper bound with true phases as "AI")
    t0 = time.time()
    ph, rho, info = ai_phaseed_solve(
        hkl, amp, cell, ph_t,
        seed_fraction=0.25, n_extend=n_extend, polish="charge_flipping",
        n_polish=n_iter, n_starts=1, seed=seed, d_min=d_min,
        use_free_fom_gate=True,
    )
    add("oracle_phaseed", ph, rho, t0, n_seed=info.get("n_seed"))

    # Partial seed (60% true) — realistic AI quality
    rng = np.random.default_rng(seed)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    ph_part = np.angle(0.55 * np.exp(1j * ph_t) + 0.45 * np.exp(1j * ph_r))
    t0 = time.time()
    ph, rho, info = ai_phaseed_solve(
        hkl, amp, cell, ph_part,
        seed_fraction=0.25, n_extend=n_extend, polish="charge_flipping",
        n_polish=n_iter, n_starts=n_starts, seed=seed, d_min=d_min,
    )
    add("partial_phaseed", ph, rho, t0, n_seed=info.get("n_seed"))

    # PhAI-PhaSeed if available
    t0 = time.time()
    ph, rho, info = phai_phaseed_solve(
        hkl, amp, cell,
        n_extend=n_extend, polish="charge_flipping", n_polish=n_iter,
        n_starts=n_starts, seed=seed, d_min=d_min, verbose=False,
    )
    add(
        "phai_phaseed", ph, rho, t0,
        seed_source=info.get("seed_source"),
        n_seed=info.get("n_seed"),
        accepted_polish=(info.get("best_trial") or {}).get("polish", {}).get(
            "accepted_polish"
        ),
    )

    # PhAI fair alone (if was used)
    if info.get("seed_source") == "phai_fair" and info.get("fom_phai_only"):
        try:
            from grok_phase_solver.models.phai_fair import run_phai_fair
            t0 = time.time()
            ph0, _ = run_phai_fair(hkl, amp, n_cycles=5, seed=seed)
            rho0 = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph0), cell, d_min=d_min
            )
            add("phai_only", ph0, rho0, t0)
            t0 = time.time()
            phc, rhoc, cinfo = conditional_polish(
                hkl, amp, cell, ph0, polish="charge_flipping",
                n_iter=n_iter, seed=seed, d_min=d_min,
            )
            add(
                "phai_cf_cond", phc, rhoc, t0,
                accepted_polish=cinfo.get("accepted_polish"),
            )
        except Exception as e:
            rows.append({
                "dataset": f"synth_n{n_atoms}", "method": "phai_only",
                "error": str(e), "d_min": d_min, "seed": seed,
            })

    return rows


def run_cod(d_mins, n_iter=60, n_extend=12):
    path = ROOT / "data/raw/cod/2016452.cif"
    if not path.exists():
        print("COD 2016452 missing; skip")
        return []
    st = load_cif(path)
    rows = []
    for d_min in d_mins:
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell
        print(f"\n=== COD 2016452 @ {d_min} Å  N={len(amp)} ===")

        methods = []
        # CF
        t0 = time.time()
        ph, rho, _ = charge_flipping_solve(
            hkl, amp, cell, n_iter=n_iter, seed=0, d_min=d_min, centrosymmetric=True
        )
        methods.append(("cf", ph, rho, t0, {}))

        t0 = time.time()
        ph, rho, info = phai_phaseed_solve(
            hkl, amp, cell, n_extend=n_extend, polish="charge_flipping",
            n_polish=n_iter, n_starts=2, seed=0, d_min=d_min,
            discrete="centro",  # P21/c
        )
        methods.append((
            "phai_phaseed", ph, rho, t0,
            {
                "seed_source": info.get("seed_source"),
                "n_seed": info.get("n_seed"),
                "accepted_polish": (info.get("best_trial") or {})
                .get("polish", {})
                .get("accepted_polish"),
            },
        ))

        if info.get("seed_source") == "phai_fair":
            from grok_phase_solver.models.phai_fair import run_phai_fair
            t0 = time.time()
            ph0, _ = run_phai_fair(hkl, amp, n_cycles=5, seed=0)
            rho0 = density_from_structure_factors(
                hkl, amp * np.exp(1j * ph0), cell, d_min=d_min
            )
            methods.append(("phai_only", ph0, rho0, t0, {}))
            t0 = time.time()
            phc, rhoc, cinfo = conditional_polish(
                hkl, amp, cell, ph0, polish="charge_flipping",
                n_iter=n_iter, seed=0, d_min=d_min,
            )
            methods.append((
                "phai_cf_cond", phc, rhoc, t0,
                {"accepted_polish": cinfo.get("accepted_polish")},
            ))

        for name, ph, rho, t0, extra in methods:
            r = _eval(
                hkl, amp, ph, ph_t, cell, data["fracs"], data["elements"], rho, d_min
            )
            r.update({
                "dataset": "COD_2016452",
                "d_min": d_min,
                "method": name,
                "seconds": time.time() - t0,
            })
            r.update(extra)
            rows.append(r)
            print(
                f"  {name:18s} CC={r['mapcc_oi']:.3f} peak={r['peak_recovery']:.2f} "
                f"solved={r['solved']}"
            )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        synth = [(6, 1.0, 0), (12, 1.5, 0)]
        dmins = [0.9, 1.5]
        n_iter, n_ext, n_st = 40, 8, 1
    else:
        synth = [
            (6, 1.0, 0), (6, 1.0, 1), (8, 0.9, 0),
            (12, 1.5, 0), (12, 1.5, 1), (16, 1.5, 0),
        ]
        dmins = [0.9, 1.2, 1.5, 2.0]
        n_iter, n_ext, n_st = 60, 12, 2

    all_rows = []
    print("===== SYNTHETIC =====")
    for n_atoms, d_min, seed in synth:
        print(f"\n--- n={n_atoms} d={d_min} seed={seed} ---")
        try:
            all_rows.extend(
                run_synthetic(
                    n_atoms, d_min, seed,
                    n_iter=n_iter, n_extend=n_ext, n_starts=n_st,
                )
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            all_rows.append({
                "dataset": f"synth_n{n_atoms}", "d_min": d_min, "seed": seed,
                "error": str(e),
            })

    print("\n===== COD =====")
    try:
        all_rows.extend(run_cod(dmins, n_iter=n_iter, n_extend=n_ext))
    except Exception as e:
        print(f"  COD ERROR: {e}")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "ai_phaseed_benchmark.json"
    jp.write_text(json.dumps(all_rows, indent=2, default=float))

    md = [
        "# AI-PhaSeed benchmark",
        "",
        "Protocol: AI phases → strong-|E| seed subset → positivity phase extension "
        "with seed re-imposition + soft full prior → free-FOM–gated CF polish.",
        "",
        "Literature: PhAI (Larsen *et al.* 2024) + phase-seeding / AI-PhaSeed "
        "(Carrozzini *et al.* 2025).",
        "",
        "## Synthetic",
        "",
        "| n | d_min | seed | method | mapCC | peak | solved |",
        "|---|-------|------|--------|-------|------|--------|",
    ]
    for r in all_rows:
        if not str(r.get("dataset", "")).startswith("synth"):
            continue
        if "error" in r:
            continue
        md.append(
            f"| {r.get('n_atoms')} | {r.get('d_min')} | {r.get('seed')} | "
            f"`{r['method']}` | {r['mapcc_oi']:.3f} | {r['peak_recovery']:.2f} | "
            f"{r['solved']} |"
        )

    md.extend([
        "",
        "## COD 2016452",
        "",
        "| d_min | method | mapCC | peak | R1 | solved | notes |",
        "|-------|--------|-------|------|----|--------|-------|",
    ])
    for r in all_rows:
        if r.get("dataset") != "COD_2016452" or "error" in r:
            continue
        notes = []
        if r.get("seed_source"):
            notes.append(r["seed_source"])
        if "accepted_polish" in r:
            notes.append(f"polish={r['accepted_polish']}")
        md.append(
            f"| {r['d_min']} | `{r['method']}` | {r['mapcc_oi']:.3f} | "
            f"{r['peak_recovery']:.2f} | {r.get('r1', float('nan')):.2f} | "
            f"{r['solved']} | {'; '.join(notes)} |"
        )

    md.extend([
        "",
        "## Notes",
        "",
        "- `oracle_phaseed`: true phases as AI seed (upper bound).",
        "- `partial_phaseed`: 55% true + noise (simulated mediocre AI).",
        "- `phai_phaseed`: PhAI fair + AI-PhaSeed + gated polish.",
        "- Compare to `phai_only` / `phai_cf_cond` to isolate extension gain.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp = out / "ai_phaseed_benchmark.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}\nWrote {mp}")


if __name__ == "__main__":
    main()
