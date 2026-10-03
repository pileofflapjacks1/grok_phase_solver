"""``gps-generate`` command. Does not change ``gps-solve``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from grok_phase_solver.generate.cif_out import write_samples_cif
from grok_phase_solver.generate.cohort import record_from_cif
from grok_phase_solver.generate.propose import propose
from grok_phase_solver.generate.report import (
    METHOD_DENOISER,
    METHOD_RANDOM,
    baseline_comparison_note,
    format_rate,
    render_run_report,
)
from grok_phase_solver.generate.score import (
    pooled_rates,
    proposal_amplitudes,
    reflection_list,
)


def _row(label: str, samples: list) -> dict:
    rates = pooled_rates(samples)
    n_match = sum(
        1
        for s in samples
        if s["score"]["match_fraction"] is not None and s["score"]["match_fraction"] >= 0.5
    )
    return {
        "method": label,
        "n": rates["n"],
        "validity": format_rate(rates["n_valid"], rates["n"]),
        "uniqueness": format_rate(rates["n_unique"], rates["n_valid"]),
        "match": format_rate(n_match, rates["n"]),
        "validity_rate": (rates["n_valid"] / rates["n"]) if rates["n"] else float("nan"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propose fractional coordinates for a known composition and cell, "
            "then score them. Research track. Does not phase and does not replace gps-solve."
        )
    )
    parser.add_argument("--cif", required=True, help="One CIF with the known structure")
    parser.add_argument("--n", type=int, default=8, help="Number of denoiser samples")
    parser.add_argument("--out", required=True, help="Directory for samples.cif and report.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Denoiser steps on this CIF only (not a held-out run)",
    )
    parser.add_argument("--sample-steps", type=int, default=8)
    args = parser.parse_args(argv)

    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    cif_path = Path(args.cif)
    if not cif_path.is_file():
        raise SystemExit(f"CIF not found: {cif_path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    record = record_from_cif(cif_path)
    if len(record["elements"]) == 0:
        raise SystemExit(f"No non-hydrogen atoms in {cif_path}")
    hkl = reflection_list(record["cell"], record["space_group"])
    amplitudes = proposal_amplitudes(
        record["fracs"], record["elements"], record["cell"], record["space_group"], hkl
    )
    common = dict(
        elements=record["elements"],
        cell=record["cell"],
        space_group=record["space_group"],
        z=int(record["z"]),
        n=int(args.n),
        seed=int(args.seed),
        reference_fracs=record["fracs"],
        reference_elements=record["elements"],
        hkl=hkl,
        amplitudes=amplitudes,
        count_reconstruction=False,
    )
    denoiser_samples = propose(
        **common,
        method="denoiser",
        fit_examples=[record],
        fit_steps=int(args.steps),
        sample_steps=int(args.sample_steps),
    )
    random_samples = propose(**common, method="random")
    denoiser_row = _row(METHOD_DENOISER, denoiser_samples)
    random_row = _row(METHOD_RANDOM, random_samples)
    comparison = baseline_comparison_note(
        float(random_row["validity_rate"]), float(denoiser_row["validity_rate"])
    )
    wall = time.perf_counter() - t0
    report = render_run_report(
        cif_name=cif_path.name,
        seed=int(args.seed),
        fit_steps=int(args.steps),
        rows=[random_row, denoiser_row],
        comparison=comparison,
        n_atoms=len(record["elements"]),
        space_group=record["space_group"],
        z=int(record["z"]),
        wall_s=wall,
    )
    write_samples_cif(out / "samples.cif", denoiser_samples)
    (out / "report.md").write_text(report)
    print(
        f"Wrote {out / 'samples.cif'} ({len(denoiser_samples)} blocks) and {out / 'report.md'}"
    )
    print(comparison)
