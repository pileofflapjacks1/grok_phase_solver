#!/usr/bin/env python3
"""
Solvability phase diagram for classical ab initio methods.

Sweeps synthetic equal-atom-ish organics over:
  - n_atoms
  - d_min (Å)
  - completeness

Methods: charge_flipping, phase_recycle, direct_methods

Strict success (metrics.success):
  mapCC_OI ≥ 0.70 AND peak_recovery ≥ 0.50 AND R1 ≤ 0.45

Writes:
  data/processed/solvability_diagram.json
  data/processed/solvability_diagram.md
  docs/figures/solvability_heatmap.png (if matplotlib works)
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
from grok_phase_solver.io.cif import expand_asymmetric_unit
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.reciprocal import generate_hkl
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.direct_methods import direct_methods_solve
from grok_phase_solver.solvers.phase_recycle import phase_recycle


def _subsample(hkl, amp, phases, completeness, seed):
    if completeness >= 1.0:
        return hkl, amp, phases
    rng = np.random.default_rng(seed)
    n = max(20, int(completeness * len(amp)))
    idx = rng.choice(len(amp), size=n, replace=False)
    return hkl[idx], amp[idx], phases[idx]


def run_one(n_atoms, d_min, completeness, method, seed=0, n_iter=80):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min, expand_symmetry=True)
    hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]
    hkl, amp, ph_true = _subsample(hkl, amp, ph_true, completeness, seed + 17)
    true_fracs = data["fracs"]

    t0 = time.time()
    if method == "charge_flipping":
        ph, rho, hist = charge_flipping_solve(
            hkl, amp, st.cell, n_iter=n_iter, seed=seed, d_min=d_min, verbose=False
        )
    elif method == "phase_recycle":
        ph, rho, hist = phase_recycle(
            hkl, amp, st.cell, n_cycles=max(5, n_iter // 10), seed=seed, d_min=d_min
        )
    elif method == "direct_methods":
        dm = direct_methods_solve(
            hkl, amp, st.cell, n_atoms_approx=n_atoms, n_trials=20, seed=seed, verbose=False
        )
        ph = dm.phases_full
        from grok_phase_solver.physics.density import density_from_structure_factors

        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), st.cell, d_min=d_min
        )
        hist = dm.history
    else:
        raise ValueError(method)

    thr = SuccessThresholds()
    rep = evaluate_success(
        hkl,
        amp,
        ph,
        ph_true,
        st.cell,
        true_fracs,
        density=rho,
        thresholds=thr,
        elements=data["elements"],
    )
    return {
        "n_atoms": n_atoms,
        "d_min": d_min,
        "completeness": completeness,
        "method": method,
        "n_refl": len(amp),
        "mapcc_oi": rep.mapcc_oi,
        "mpe_oi_deg": rep.mpe_oi_deg,
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": rep.solved,
        "seconds": time.time() - t0,
        "seed": seed,
    }


def main():
    # Compact but informative grid (runtime-conscious)
    n_atoms_list = [4, 8, 12, 20]
    d_min_list = [0.9, 1.2, 1.5, 2.0]
    completeness_list = [1.0, 0.7]
    methods = ["charge_flipping", "phase_recycle", "direct_methods"]
    seeds = [0, 1]  # two seeds for stability

    rows = []
    total = (
        len(n_atoms_list)
        * len(d_min_list)
        * len(completeness_list)
        * len(methods)
        * len(seeds)
    )
    i = 0
    print(f"Solvability sweep: {total} trials")
    print(
        "Success = mapCC_OI≥0.70 AND peak_recovery≥0.50 AND R1≤0.45"
    )
    for n_atoms in n_atoms_list:
        for d_min in d_min_list:
            for comp in completeness_list:
                for method in methods:
                    for seed in seeds:
                        i += 1
                        try:
                            row = run_one(n_atoms, d_min, comp, method, seed=seed)
                        except Exception as e:
                            row = {
                                "n_atoms": n_atoms,
                                "d_min": d_min,
                                "completeness": comp,
                                "method": method,
                                "seed": seed,
                                "error": str(e),
                                "solved": False,
                            }
                        rows.append(row)
                        flag = "SOLVED" if row.get("solved") else "fail"
                        if "error" in row:
                            print(f"[{i}/{total}] ERROR {method} n={n_atoms} d={d_min} c={comp}: {row['error']}")
                        else:
                            print(
                                f"[{i}/{total}] {flag:6s} {method:16s} n={n_atoms:2d} "
                                f"d={d_min:.1f} c={comp:.1f}  "
                                f"CC={row['mapcc_oi']:.2f} peak={row['peak_recovery']:.2f} "
                                f"R1={row['r1']:.2f}"
                            )

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "solvability_diagram.json"
    json_path.write_text(json.dumps(rows, indent=2, default=float))

    # Aggregate success rates
    md = [
        "# Solvability phase diagram",
        "",
        "Synthetic P1 organics. **Strict success:**",
        "",
        "- mapCC_OI ≥ **0.70**",
        "- peak recovery ≥ **0.50** (atoms matched to density peaks, origin-shifted)",
        "- R1 ≤ **0.45** (carbons at top peaks vs |F_obs|)",
        "",
        f"Trials: {len(rows)} (including seeds). JSON: `{json_path.relative_to(ROOT)}`",
        "",
        "## Success rate by method (all conditions)",
        "",
        "| Method | Solved | Total | Rate |",
        "|--------|--------|-------|------|",
    ]
    for method in methods:
        sub = [r for r in rows if r.get("method") == method and "error" not in r]
        n_ok = sum(1 for r in sub if r.get("solved"))
        md.append(f"| `{method}` | {n_ok} | {len(sub)} | {n_ok / max(len(sub), 1):.0%} |")

    md.extend(
        [
            "",
            "## Success rate: charge_flipping vs (n_atoms, d_min) at completeness=1.0",
            "",
            "| n_atoms \\ d_min | "
            + " | ".join(f"{d:.1f}" for d in d_min_list)
            + " |",
            "|"
            + "---|" * (len(d_min_list) + 1),
        ]
    )
    for n_atoms in n_atoms_list:
        cells = []
        for d_min in d_min_list:
            sub = [
                r
                for r in rows
                if r.get("method") == "charge_flipping"
                and r.get("n_atoms") == n_atoms
                and r.get("d_min") == d_min
                and r.get("completeness") == 1.0
                and "error" not in r
            ]
            if not sub:
                cells.append("—")
            else:
                rate = np.mean([r["solved"] for r in sub])
                mean_cc = np.mean([r["mapcc_oi"] for r in sub])
                cells.append(f"{rate:.0%} (CC={mean_cc:.2f})")
        md.append(f"| **{n_atoms}** | " + " | ".join(cells) + " |")

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **High resolution + few atoms:** classical CF should dominate (atomic peak separation).",
            "- **Low resolution or many atoms:** success rate collapses — the open phase problem.",
            "- **Completeness 0.7:** typically harder than full data at same d_min.",
            "- Direct methods here are a **thin educational** multi-start tangent code, not SHELXD.",
            "- Phase recycle (ER positivity) is fast but weaker than CF on atomic-resolution synthetics.",
            "",
            "This diagram defines the **baseline frontier** any new method (PhAI, hybrids, new math)",
            "must beat under the **same** success criterion.",
            "",
        ]
    )
    md_path = out_dir / "solvability_diagram.md"
    md_path.write_text("\n".join(md))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")

    # Heatmap for CF @ completeness 1.0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mat = np.full((len(n_atoms_list), len(d_min_list)), np.nan)
        for ii, n_atoms in enumerate(n_atoms_list):
            for jj, d_min in enumerate(d_min_list):
                sub = [
                    r
                    for r in rows
                    if r.get("method") == "charge_flipping"
                    and r.get("n_atoms") == n_atoms
                    and r.get("d_min") == d_min
                    and r.get("completeness") == 1.0
                    and "error" not in r
                ]
                if sub:
                    mat[ii, jj] = np.mean([r["mapcc_oi"] for r in sub])
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(d_min_list)))
        ax.set_xticklabels([str(d) for d in d_min_list])
        ax.set_yticks(range(len(n_atoms_list)))
        ax.set_yticklabels([str(n) for n in n_atoms_list])
        ax.set_xlabel("d_min (Å)")
        ax.set_ylabel("n_atoms (P1 synthetic)")
        ax.set_title("Charge flipping mean mapCC_OI (completeness=1)")
        fig.colorbar(im, ax=ax, label="mapCC_OI")
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                if np.isfinite(mat[ii, jj]):
                    ax.text(jj, ii, f"{mat[ii, jj]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig_dir = ROOT / "docs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "solvability_heatmap.png"
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        print(f"Wrote {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
