#!/usr/bin/env python3
"""
Wilson domain-gap: synthetic training-like |F| vs experimental COD Fobs.

Quantifies whether hard-region synthetic generators match experimental
amplitude statistics (Wilson B, intensity quantiles, moments).

Writes data/processed/wilson_domain_gap.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.wilson import (
    domain_gap_report,
    mean_domain_gap_vs_experiment,
    wilson_plot,
)
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.io.hkl import load_hkl_shelx, load_hkl_cif
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def synth_pack(n_atoms: int, d_min: float, seed: int, region: str) -> dict:
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    return {
        "name": f"{region}_n{n_atoms}_d{d_min}_{seed}",
        "hkl": data["hkl"],
        "amplitudes": data["amplitudes"],
        "cell": st.cell,
        "region": region,
        "n_atoms": data["n_atoms_cell"],
        "d_min": d_min,
    }


def load_exp_cod_2017775(d_min_cap: float = 1.2) -> dict:
    hkl_path = ROOT / "data/raw/cod/2017775.hkl"
    cif_path = ROOT / "data/raw/cod/2017775.cif"
    st = load_cif(cif_path)
    try:
        table = load_hkl_shelx(hkl_path, cell=st.cell)
    except Exception:
        table = load_hkl_cif(hkl_path)
    # resolution filter
    from grok_phase_solver.physics.reciprocal import d_spacing

    d = d_spacing(table.hkl, st.cell)
    m = d >= d_min_cap
    return {
        "name": f"COD_2017775_exp_dmin{d_min_cap}",
        "hkl": table.hkl[m],
        "amplitudes": table.amplitudes[m],
        "cell": st.cell,
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-synth", type=int, default=None)
    args = p.parse_args()

    n = 8 if args.quick else 20
    if args.n_synth is not None:
        n = args.n_synth

    rng = np.random.default_rng(0)
    easy = [
        synth_pack(int(rng.integers(6, 10)), float(rng.uniform(0.9, 1.2)), int(rng.integers(1e9)), "easy")
        for _ in range(n // 2)
    ]
    hard = [
        synth_pack(int(rng.integers(12, 18)), float(rng.uniform(1.5, 2.0)), int(rng.integers(1e9)), "hard")
        for _ in range(n - n // 2)
    ]
    all_synth = easy + hard

    exp = None
    if (ROOT / "data/raw/cod/2017775.hkl").exists():
        exp = load_exp_cod_2017775(1.2)

    # Also Fcalc control from CIF if present
    fcalc_ctrl = None
    cif_path = ROOT / "data/raw/cod/2016452.cif"
    if cif_path.exists():
        st = load_cif(cif_path)
        data = structure_to_fcalc(st, d_min=0.9)
        fcalc_ctrl = {
            "name": "COD_2016452_Fcalc_0.9",
            "hkl": data["hkl"],
            "amplitudes": data["amplitudes"],
            "cell": st.cell,
        }

    report: dict = {"n_synthetic": len(all_synth), "pairs": []}

    if exp is not None:
        easy_vs = mean_domain_gap_vs_experiment(easy, exp)
        hard_vs = mean_domain_gap_vs_experiment(hard, exp)
        all_vs = mean_domain_gap_vs_experiment(all_synth, exp)
        report["experimental"] = exp["name"]
        report["easy_vs_exp"] = {
            k: v for k, v in easy_vs.items() if k != "per_structure"
        }
        report["hard_vs_exp"] = {
            k: v for k, v in hard_vs.items() if k != "per_structure"
        }
        report["all_vs_exp"] = {
            k: v for k, v in all_vs.items() if k != "per_structure"
        }
        # keep a few examples
        report["example_gaps"] = [
            domain_gap_report(easy[0], exp, label_a=easy[0]["name"], label_b=exp["name"]),
            domain_gap_report(hard[0], exp, label_a=hard[0]["name"], label_b=exp["name"]),
        ]
        # strip huge arrays from examples
        for ex in report["example_gaps"]:
            for key in ("moments_a", "moments_b", "wilson_a", "wilson_b", "resolution_a", "resolution_b", "wilson"):
                if key in ex and isinstance(ex[key], dict):
                    pass  # already scalar-friendly

    if fcalc_ctrl is not None:
        report["fcalc_control_vs_hard"] = {
            k: v
            for k, v in mean_domain_gap_vs_experiment(hard, fcalc_ctrl).items()
            if k != "per_structure"
        }
        report["fcalc_control_vs_easy"] = {
            k: v
            for k, v in mean_domain_gap_vs_experiment(easy, fcalc_ctrl).items()
            if k != "per_structure"
        }

    # Self-consistency: hard vs hard
    if len(hard) >= 2:
        report["hard_vs_hard_self"] = domain_gap_report(
            hard[0], hard[1], label_a=hard[0]["name"], label_b=hard[1]["name"]
        )["domain_gap_score"]

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "wilson_domain_gap.json"
    mp = out / "wilson_domain_gap.md"

    # JSON-safe
    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [scrub(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    jp.write_text(json.dumps(scrub(report), indent=2))

    lines = [
        "# Wilson domain-gap report",
        "",
        "Compare **synthetic** |F| statistics to **experimental** COD reflections.",
        "Lower `domain_gap_score` ⇒ closer match (Wilson slope + intensity quantiles + moments).",
        "",
        f"- Synthetic structures: **{len(all_synth)}** "
        f"(easy≈{len(easy)}, hard≈{len(hard)})",
    ]
    if exp is not None:
        lines.append(f"- Experimental reference: **{exp['name']}**")
        lines.extend([
            "",
            "## Synthetic vs experimental",
            "",
            "| Cohort | mean gap | std | min | max |",
            "|--------|----------|-----|-----|-----|",
        ])
        for label, key in (
            ("easy", "easy_vs_exp"),
            ("hard", "hard_vs_exp"),
            ("all", "all_vs_exp"),
        ):
            s = report[key]
            lines.append(
                f"| {label} | {s['mean_domain_gap_score']:.3f} | "
                f"{s['std_domain_gap_score']:.3f} | "
                f"{s['min_domain_gap_score']:.3f} | "
                f"{s['max_domain_gap_score']:.3f} |"
            )
    if fcalc_ctrl is not None:
        lines.extend([
            "",
            "## Synthetic vs COD Fcalc control (2016452 @ 0.9 Å)",
            "",
            f"- hard mean gap: **{report['fcalc_control_vs_hard']['mean_domain_gap_score']:.3f}**",
            f"- easy mean gap: **{report['fcalc_control_vs_easy']['mean_domain_gap_score']:.3f}**",
        ])
    if "hard_vs_hard_self" in report:
        lines.extend([
            "",
            f"Hard-vs-hard self gap (sanity): **{report['hard_vs_hard_self']:.3f}**",
        ])
    lines.extend([
        "",
        "## Notes",
        "",
        "- Large gap to experimental Fobs is expected (completeness, B-factor, "
        "disorder, measurement noise).",
        "- Use this score when reweighting or filtering synthetic training shards.",
        "- Code: `data/wilson.py` — `domain_gap_report`, `mean_domain_gap_vs_experiment`.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp.write_text("\n".join(lines))
    print(f"Wrote {jp}\nWrote {mp}")
    if exp is not None:
        print(
            f"hard vs exp mean gap = {report['hard_vs_exp']['mean_domain_gap_score']:.3f}"
        )


if __name__ == "__main__":
    main()
