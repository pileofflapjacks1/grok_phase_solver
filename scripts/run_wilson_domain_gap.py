#!/usr/bin/env python3
"""
Wilson domain-gap report + close-the-gap evaluation.

1. Measure raw synthetic vs experimental |F| gap
2. Save experimental template for training-time matching
3. Apply Wilson slope + shell + quantile matching
4. Report before/after scores

Writes:
  data/processed/wilson_domain_gap.{json,md}
  data/processed/wilson_ref_template.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.wilson import domain_gap_report, mean_domain_gap_vs_experiment
from grok_phase_solver.data.wilson_match import (
    WilsonMatchConfig,
    close_wilson_gap,
    match_batch_to_reference,
    save_reference_template,
)
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.io.hkl import load_hkl_shelx
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def synth_pack(n_atoms: int, d_min: float, seed: int, region: str) -> dict:
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    return {
        "name": f"{region}_n{n_atoms}_d{d_min:.2f}_{seed}",
        "hkl": data["hkl"],
        "amplitudes": data["amplitudes"],
        "phases": data["phases"],
        "cell": st.cell,
        "region": region,
        "n_atoms": data["n_atoms_cell"],
        "d_min": d_min,
    }


def load_exp_cod_2017775(d_min_cap: float = 1.2) -> dict:
    hkl_path = ROOT / "data/raw/cod/2017775.hkl"
    cif_path = ROOT / "data/raw/cod/2017775.cif"
    st = load_cif(cif_path)
    table = load_hkl_shelx(hkl_path, cell=st.cell)
    from grok_phase_solver.physics.reciprocal import d_spacing

    d = d_spacing(table.hkl, st.cell)
    m = d >= d_min_cap
    return {
        "name": f"COD_2017775_exp_dmin{d_min_cap}",
        "hkl": table.hkl[m],
        "amplitudes": table.amplitudes[m],
        "cell": st.cell,
    }


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if k != "scales"}
    if isinstance(o, (list, tuple)):
        return [scrub(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o


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
        synth_pack(
            int(rng.integers(6, 10)),
            float(rng.uniform(0.9, 1.2)),
            int(rng.integers(1e9)),
            "easy",
        )
        for _ in range(n // 2)
    ]
    hard = [
        synth_pack(
            int(rng.integers(12, 18)),
            float(rng.uniform(1.5, 2.0)),
            int(rng.integers(1e9)),
            "hard",
        )
        for _ in range(n - n // 2)
    ]
    all_synth = easy + hard

    exp = None
    if (ROOT / "data/raw/cod/2017775.hkl").exists():
        exp = load_exp_cod_2017775(1.2)

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

    report: dict = {"n_synthetic": len(all_synth)}

    # --- raw gaps ---
    if exp is not None:
        tpl_path = save_reference_template(exp)
        report["template_path"] = str(tpl_path.relative_to(ROOT))
        report["experimental"] = exp["name"]
        for label, cohort in (("easy", easy), ("hard", hard), ("all", all_synth)):
            raw = mean_domain_gap_vs_experiment(cohort, exp)
            report[f"{label}_vs_exp_raw"] = {
                k: v for k, v in raw.items() if k != "per_structure"
            }

        # --- close the gap ---
        cfg = WilsonMatchConfig(
            match_slope=True,
            match_shells=True,
            match_quantiles=True,
            noise_level=0.03,
            seed=0,
        )
        # Ablations
        ablations = {
            "slope_only": WilsonMatchConfig(
                match_slope=True, match_shells=False, match_quantiles=False,
                noise_level=0.0, seed=0,
            ),
            "slope_shells": WilsonMatchConfig(
                match_slope=True, match_shells=True, match_quantiles=False,
                noise_level=0.0, seed=0,
            ),
            "full": cfg,
        }
        report["ablations"] = {}
        for name, acfg in ablations.items():
            hard_m, hard_stats = match_batch_to_reference(hard, exp, acfg)
            easy_m, easy_stats = match_batch_to_reference(easy, exp, acfg)
            all_m, all_stats = match_batch_to_reference(all_synth, exp, acfg)
            report["ablations"][name] = {
                "hard": hard_stats,
                "easy": easy_stats,
                "all": all_stats,
            }
            if name == "full":
                report["hard_vs_exp_matched"] = hard_stats
                report["easy_vs_exp_matched"] = easy_stats
                report["all_vs_exp_matched"] = all_stats
                # example single structure detail
                _, ex_rep = close_wilson_gap(hard[0], exp, cfg)
                report["example_hard_close"] = {
                    k: ex_rep[k]
                    for k in (
                        "gap_before",
                        "gap_after",
                        "gap_reduction",
                        "gap_reduction_frac",
                        "wilson_before",
                        "wilson_after",
                        "steps",
                        "config",
                    )
                }

        if len(hard) >= 2:
            report["hard_vs_hard_self"] = domain_gap_report(
                hard[0], hard[1], label_a=hard[0]["name"], label_b=hard[1]["name"]
            )["domain_gap_score"]

    if fcalc_ctrl is not None:
        report["fcalc_control_vs_hard_raw"] = {
            k: v
            for k, v in mean_domain_gap_vs_experiment(hard, fcalc_ctrl).items()
            if k != "per_structure"
        }
        if exp is not None:
            hard_m, _ = match_batch_to_reference(
                hard, exp, WilsonMatchConfig(noise_level=0.0, seed=1)
            )
            # matched-to-exp then gap vs Fcalc (sanity)
            report["fcalc_control_vs_hard_matched_to_exp"] = {
                k: v
                for k, v in mean_domain_gap_vs_experiment(hard_m, fcalc_ctrl).items()
                if k != "per_structure"
            }

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "wilson_domain_gap.json"
    mp = out / "wilson_domain_gap.md"
    jp.write_text(json.dumps(scrub(report), indent=2))

    lines = [
        "# Wilson domain-gap report (close-the-gap)",
        "",
        "Compare **synthetic** |F| to **experimental** COD, then apply "
        "Wilson-slope + shell-mean + quantile matching (`data/wilson_match.py`).",
        "",
        f"- Synthetic structures: **{len(all_synth)}** (easy≈{len(easy)}, hard≈{len(hard)})",
    ]
    if exp is not None:
        lines.append(f"- Experimental reference: **{exp['name']}**")
        lines.append(f"- Training template saved: `{report.get('template_path')}`")
        lines.extend([
            "",
            "## Raw gap (before matching)",
            "",
            "| Cohort | mean gap | std | min | max |",
            "|--------|----------|-----|-----|-----|",
        ])
        for label in ("easy", "hard", "all"):
            s = report[f"{label}_vs_exp_raw"]
            lines.append(
                f"| {label} | {s['mean_domain_gap_score']:.3f} | "
                f"{s['std_domain_gap_score']:.3f} | "
                f"{s['min_domain_gap_score']:.3f} | "
                f"{s['max_domain_gap_score']:.3f} |"
            )

        lines.extend([
            "",
            "## After full matching (slope + shells + quantiles + noise)",
            "",
            "| Cohort | mean gap before | mean gap after | reduction | frac ↓ |",
            "|--------|-----------------|----------------|-----------|--------|",
        ])
        for label, key in (
            ("easy", "easy_vs_exp_matched"),
            ("hard", "hard_vs_exp_matched"),
            ("all", "all_vs_exp_matched"),
        ):
            s = report[key]
            lines.append(
                f"| {label} | {s['mean_gap_before']:.3f} | {s['mean_gap_after']:.3f} | "
                f"{s['mean_reduction']:.3f} | {s['mean_reduction_frac']:.0%} |"
            )

        lines.extend([
            "",
            "## Ablations (hard cohort)",
            "",
            "| Recipe | mean gap after | reduction frac |",
            "|--------|----------------|----------------|",
        ])
        for name, block in report.get("ablations", {}).items():
            h = block["hard"]
            lines.append(
                f"| `{name}` | {h['mean_gap_after']:.3f} | {h['mean_reduction_frac']:.0%} |"
            )

        if "example_hard_close" in report:
            ex = report["example_hard_close"]
            lines.extend([
                "",
                "## Example hard structure",
                "",
                f"- Gap {ex['gap_before']:.3f} → **{ex['gap_after']:.3f}** "
                f"({ex['gap_reduction_frac']:.0%} reduction)",
                f"- Wilson B: {ex['wilson_before'].get('B_a', float('nan')):.1f} → "
                f"target ~{ex['wilson_before'].get('B_b', float('nan')):.1f} "
                f"(after match B_a={ex['wilson_after'].get('B_a', float('nan')):.1f})",
            ])

    if "hard_vs_hard_self" in report:
        lines.append(f"\nHard-vs-hard self gap (sanity): **{report['hard_vs_hard_self']:.3f}**")

    lines.extend([
        "",
        "## How to use in training",
        "",
        "```bash",
        "# rebuild template + report",
        "python scripts/run_wilson_domain_gap.py",
        "",
        "# train GraphPhaseNet with matched |F|",
        "python scripts/train_strong_prior.py --scale  # set wilson_match=True in API",
        "```",
        "",
        "```python",
        "from grok_phase_solver.data.wilson_match import close_wilson_gap, load_reference_template",
        "from grok_phase_solver.models.strong_prior import train_strong_prior",
        "",
        "model, meta = train_strong_prior(n_structures=100, wilson_match=True)",
        "```",
        "",
        "Phases always come from Fcalc truth; only **amplitudes** are matched. "
        "This keeps labels correct while aligning input statistics to experiment.",
        "",
        f"JSON: `{jp.relative_to(ROOT)}`",
        "",
    ])
    mp.write_text("\n".join(lines))
    print(f"Wrote {jp}\nWrote {mp}")
    if exp is not None:
        print(
            f"hard raw gap={report['hard_vs_exp_raw']['mean_domain_gap_score']:.3f} → "
            f"matched={report['hard_vs_exp_matched']['mean_gap_after']:.3f} "
            f"({report['hard_vs_exp_matched']['mean_reduction_frac']:.0%} ↓)"
        )


if __name__ == "__main__":
    main()
