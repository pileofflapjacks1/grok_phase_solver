#!/usr/bin/env python3
"""
Hard-cliff partial-φ / fragment seed benchmark (science track B).

Shows how much known phase / fragment information is needed to solve hard cells
via AI-PhaSeed extension — the path oracle tests already suggested works.

Sweeps:
  1. Oracle fraction of strong |E| phases known (0 → 0.5)
  2. Phase noise (°) at fixed 30% strong seed
  3. Fragment atom fraction → Fcalc seed → PhaSeed
  4. Baselines: CF, strong_prior_phaseed (if weights exist)

Writes data/processed/partial_seed_benchmark.{json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.partial_seed import (
    fragment_partial_seed,
    fragment_phaseed_solve,
    oracle_partial_phaseed_solve,
    partial_phaseed_solve,
)


def _score(hkl, amp, ph, ph_t, cell, fracs, elements, rho, d_min) -> Dict[str, Any]:
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), cell, shape=rho.shape
    )
    if rho.shape != rho_t.shape:
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, shape=rho_t.shape
        )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    rep = evaluate_success(
        hkl, amp, ph, ph_t, cell, fracs, density=rho, elements=elements,
        thresholds=SuccessThresholds(),
    )
    return {
        "mapcc_oi": float(cc),
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": bool(rep.solved),
    }


def make_hard_case(seed: int, n_atoms: int = 14, d_min: float = 1.7):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def run_oracle_fraction_sweep(
    cases: List, fractions: List[float], n_extend: int, n_polish: int, n_starts: int
) -> List[Dict]:
    rows = []
    print("\n=== Oracle strong-|E| fraction sweep ===", flush=True)
    for frac in fractions:
        for i, (st, data) in enumerate(cases):
            t0 = time.time()
            hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
            ph, rho, info = oracle_partial_phaseed_solve(
                hkl, amp, st.cell, ph_t,
                fraction=frac, mode="strong_E", seed=i,
                n_extend=n_extend, n_polish=n_polish, n_starts=n_starts,
                polish="charge_flipping", d_min=data.get("d_min"),
            )
            sc = _score(
                hkl, amp, ph, ph_t, st.cell, data["fracs"], data["elements"], rho,
                data.get("d_min"),
            )
            row = {
                "sweep": "oracle_fraction",
                "fraction": frac,
                "case": i,
                "n_atoms": data["n_atoms_cell"],
                "d_min": float(data.get("d_min") or 1.7),
                "n_seed": info.get("n_seed"),
                "seconds": time.time() - t0,
                **sc,
            }
            rows.append(row)
            print(
                f"  frac={frac:.2f} case={i}: CC={sc['mapcc_oi']:.3f} "
                f"sol={sc['solved']} n_seed={info.get('n_seed')}",
                flush=True,
            )
    return rows


def run_noise_sweep(
    cases: List, noise_degs: List[float], fraction: float,
    n_extend: int, n_polish: int, n_starts: int,
) -> List[Dict]:
    rows = []
    print(f"\n=== Phase noise sweep (frac={fraction}) ===", flush=True)
    for noise in noise_degs:
        for i, (st, data) in enumerate(cases):
            t0 = time.time()
            hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
            ph, rho, info = oracle_partial_phaseed_solve(
                hkl, amp, st.cell, ph_t,
                fraction=fraction, mode="strong_E",
                phase_noise_deg=noise, seed=i,
                n_extend=n_extend, n_polish=n_polish, n_starts=n_starts,
                polish="charge_flipping", d_min=data.get("d_min"),
            )
            sc = _score(
                hkl, amp, ph, ph_t, st.cell, data["fracs"], data["elements"], rho,
                data.get("d_min"),
            )
            row = {
                "sweep": "phase_noise",
                "fraction": fraction,
                "noise_deg": noise,
                "case": i,
                "n_atoms": data["n_atoms_cell"],
                "seconds": time.time() - t0,
                **sc,
            }
            rows.append(row)
            print(
                f"  noise={noise:.0f}° case={i}: CC={sc['mapcc_oi']:.3f} sol={sc['solved']}",
                flush=True,
            )
    return rows


def run_fragment_sweep(
    cases: List, atom_fracs: List[float], n_extend: int, n_polish: int, n_starts: int
) -> List[Dict]:
    rows = []
    print("\n=== Fragment atom-fraction sweep ===", flush=True)
    for af in atom_fracs:
        for i, (st, data) in enumerate(cases):
            t0 = time.time()
            hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
            # random subset of true atoms as "known fragment"
            seed_ph, fr_sub, meta = fragment_partial_seed(
                hkl, amp, st.cell, data["fracs"], data["elements"],
                atom_fraction=af, seed=i, use_fcalc_weight=True,
            )
            ph, rho, info = partial_phaseed_solve(
                hkl, amp, st.cell, seed_ph,
                seed_fraction=0.30, n_extend=n_extend, n_polish=n_polish,
                n_starts=n_starts, seed=i, polish="charge_flipping",
                d_min=data.get("d_min"), meta=meta,
            )
            sc = _score(
                hkl, amp, ph, ph_t, st.cell, data["fracs"], data["elements"], rho,
                data.get("d_min"),
            )
            row = {
                "sweep": "fragment_fraction",
                "atom_fraction": af,
                "n_atoms_model": meta["n_atoms_model"],
                "case": i,
                "n_atoms": data["n_atoms_cell"],
                "seconds": time.time() - t0,
                **sc,
            }
            rows.append(row)
            print(
                f"  atoms={af:.0%} ({meta['n_atoms_model']}) case={i}: "
                f"CC={sc['mapcc_oi']:.3f} sol={sc['solved']}",
                flush=True,
            )
    return rows


def run_baselines(cases: List, n_iter: int, n_starts: int) -> List[Dict]:
    rows = []
    print("\n=== Baselines (CF / strong prior) ===", flush=True)
    strong = None
    try:
        from grok_phase_solver.models.strong_prior import (
            default_strong_prior_path,
            strong_prior_phaseed_solve,
        )
        if default_strong_prior_path().exists():
            strong = strong_prior_phaseed_solve
    except Exception:
        pass

    for i, (st, data) in enumerate(cases):
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        d_min = data.get("d_min")
        t0 = time.time()
        ph, rho, _ = charge_flipping_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=i, d_min=d_min
        )
        sc = _score(hkl, amp, ph, ph_t, st.cell, data["fracs"], data["elements"], rho, d_min)
        rows.append({
            "sweep": "baseline", "method": "charge_flipping", "case": i,
            "n_atoms": data["n_atoms_cell"], "seconds": time.time() - t0, **sc,
        })
        print(f"  CF case={i}: CC={sc['mapcc_oi']:.3f} sol={sc['solved']}", flush=True)

        if strong is not None:
            t0 = time.time()
            ph, rho, _ = strong(
                hkl, amp, st.cell, n_extend=12, n_polish=n_iter,
                n_starts=n_starts, seed=i, d_min=d_min,
            )
            sc = _score(hkl, amp, ph, ph_t, st.cell, data["fracs"], data["elements"], rho, d_min)
            rows.append({
                "sweep": "baseline", "method": "strong_prior_phaseed", "case": i,
                "n_atoms": data["n_atoms_cell"], "seconds": time.time() - t0, **sc,
            })
            print(f"  strong case={i}: CC={sc['mapcc_oi']:.3f} sol={sc['solved']}", flush=True)
    return rows


def summarize(rows: List[Dict]) -> Dict:
    out: Dict[str, Any] = {}
    # group by sweep + key param
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        if r["sweep"] == "oracle_fraction":
            key = f"oracle_f={r['fraction']:.2f}"
        elif r["sweep"] == "phase_noise":
            key = f"noise={r['noise_deg']:.0f}"
        elif r["sweep"] == "fragment_fraction":
            key = f"frag={r['atom_fraction']:.2f}"
        elif r["sweep"] == "baseline":
            key = f"base={r['method']}"
        else:
            key = r["sweep"]
        g[key].append(r)

    for k, rs in sorted(g.items()):
        ccs = [r["mapcc_oi"] for r in rs if r.get("mapcc_oi") is not None]
        sols = sum(1 for r in rs if r.get("solved"))
        out[k] = {
            "n": len(rs),
            "n_solved": sols,
            "solve_rate": sols / max(len(rs), 1),
            "mean_mapcc": float(np.mean(ccs)) if ccs else None,
            "mean_peak": float(np.mean([r["peak_recovery"] for r in rs if r.get("peak_recovery") is not None])) if rs else None,
        }
    return out


def write_md(summary: Dict, rows: List[Dict], path: Path, n_cases: int):
    lines = [
        "# Partial-φ / fragment seed hard-cliff benchmark",
        "",
        "Science track B: how much **known phase or fragment** information is "
        "needed to solve **hard** synthetic cells (n≈12–16, d≈1.5–2.0 Å) via "
        "AI-PhaSeed extension.",
        "",
        f"Cases per condition: **{n_cases}** hard P1 structures.",
        "",
        "## Summary",
        "",
        "| Condition | n | solved | rate | mean mapCC | mean peak |",
        "|-----------|---|--------|------|------------|-----------|",
    ]
    for k, s in summary.items():
        mcc = s["mean_mapcc"]
        mcc_s = f"{mcc:.3f}" if mcc is not None else "—"
        pk = s["mean_peak"]
        pk_s = f"{pk:.2f}" if pk is not None else "—"
        lines.append(
            f"| `{k}` | {s['n']} | {s['n_solved']} | {s['solve_rate']:.0%} | "
            f"{mcc_s} | {pk_s} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Oracle fraction curve**: mapCC / solve rate vs fraction of strong |E| "
        "phases known exactly. This is the theoretical ceiling for a perfect prior "
        "on that seed set.",
        "- **Noise curve**: robustness of a fixed 30% seed to phase error (°).",
        "- **Fragment curve**: random true-atom subsets as MR-lite seeds.",
        "- **Baselines**: CF and GraphPhaseNet prior without partial φ.",
        "",
        "If oracle ≥30–40% strong phases **solves** hard cells while full priors "
        "do not, the bottleneck is **seed quality**, not the extension engine.",
        "",
        "API: `solvers/partial_seed.py` — `oracle_partial_phaseed_solve`, "
        "`fragment_phaseed_solve`, `load_phase_seed_csv`.",
        "",
        f"JSON companion: `{path.with_suffix('.json').name}`",
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-cases", type=int, default=None)
    args = p.parse_args()

    if args.quick:
        n_cases = 2
        n_extend, n_polish, n_starts = 10, 30, 1
        fractions = [0.0, 0.15, 0.30, 0.50]
        noises = [0.0, 30.0, 60.0]
        frags = [0.15, 0.35, 0.55]
    else:
        n_cases = 4
        n_extend, n_polish, n_starts = 15, 50, 2
        fractions = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
        noises = [0.0, 20.0, 40.0, 60.0, 90.0]
        frags = [0.10, 0.25, 0.40, 0.60]
    if args.n_cases is not None:
        n_cases = args.n_cases

    # Fixed hard cases
    rng = np.random.default_rng(2026)
    cases = []
    for i in range(n_cases):
        n_atoms = int(rng.integers(12, 17))
        d_min = float(rng.choice([1.5, 1.7, 2.0]))
        s = int(rng.integers(0, 2**31 - 1))
        st, data = make_hard_case(s, n_atoms=n_atoms, d_min=d_min)
        data["d_min"] = d_min
        cases.append((st, data))
        print(f"case {i}: n={data['n_atoms_cell']} d={d_min}", flush=True)

    all_rows: List[Dict] = []
    all_rows.extend(run_baselines(cases, n_iter=n_polish, n_starts=n_starts))
    all_rows.extend(
        run_oracle_fraction_sweep(cases, fractions, n_extend, n_polish, n_starts)
    )
    all_rows.extend(
        run_noise_sweep(cases, noises, fraction=0.30, n_extend=n_extend,
                        n_polish=n_polish, n_starts=n_starts)
    )
    all_rows.extend(
        run_fragment_sweep(cases, frags, n_extend, n_polish, n_starts)
    )

    summary = summarize(all_rows)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "partial_seed_benchmark.json"
    mp = out / "partial_seed_benchmark.md"
    jp.write_text(json.dumps({"summary": summary, "rows": all_rows}, indent=2, default=float))
    write_md(summary, all_rows, mp, n_cases)
    print(f"\nWrote {jp}\nWrote {mp}", flush=True)
    print("\n=== Summary ===", flush=True)
    for k, s in summary.items():
        mcc = s["mean_mapcc"]
        mcc_s = f"{mcc:.3f}" if mcc is not None else "n/a"
        print(f"  {k:28s} CC={mcc_s} solved={s['n_solved']}/{s['n']}", flush=True)


if __name__ == "__main__":
    main()
