"""Build the three-row coordinate scoreboard from already loaded structures."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from grok_phase_solver.generate.propose import propose
from grok_phase_solver.generate.report import (
    METHOD_DENOISER,
    METHOD_DIFFUSION,
    METHOD_RANDOM,
    NOVELTY,
    STABILITY,
    baseline_comparison_note,
    format_rate,
    render_scoreboard_md,
)
from grok_phase_solver.generate.score import (
    AMPLITUDE_SOURCE_DEPOSITED_FCALC,
    pooled_rates,
    proposal_amplitudes,
    reflection_list,
)

DIFFUSION_NOTE = (
    "Experimental phase-path flag `diffusion_hybrid`, not the gps-generate default "
    "and not the coordinate denoiser. Amplitudes are Fcalc of the deposited "
    "structure at B=5 Å² and d_min=1.5 Å, so this row is oracle-|F| conditioned. "
    "Peaks are unlabeled; the requested element list is copied in CIF order only "
    "when at least as many peaks as atoms are found. Fewer peaks are not padded, "
    "and composition_match is then false. Settings: n_steps=2, n_starts=1, "
    "n_polish=2, use_learned_score=false. One sample per cell."
)


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.3f}"


def _summarize(method_label: str, groups: Sequence[tuple[dict, list]], note: str) -> dict:
    n = 0
    n_valid = 0
    n_unique = 0
    n_holdout = 0
    n_recovered = 0
    distances: list[float] = []
    per_cell = []
    for record, samples in groups:
        rates = pooled_rates(samples)
        n += rates["n"]
        n_valid += rates["n_valid"]
        n_unique += rates["n_unique"]
        if np.isfinite(rates["mean_min_distance_A"]):
            distances.append(rates["mean_min_distance_A"])
        if record["holdout"]:
            n_holdout += 1
            if any(s["score"].get("reconstructed") is True for s in samples):
                n_recovered += 1
        per_cell.append(
            {
                "structure": record["name"],
                "holdout": bool(record["holdout"]),
                "method": method_label,
                "n_valid": format_rate(rates["n_valid"], rates["n"]),
                "n_unique": format_rate(rates["n_unique"], rates["n_valid"]),
                "dmin": _fmt(rates["mean_min_distance_A"]),
                "match": _fmt(rates["best_match_fraction"]),
                "rfac": _fmt(rates["mean_r_factor"]),
            }
        )
    if n_holdout == 0:
        reconstruction = "not measured (no holdout)"
    else:
        reconstruction = format_rate(n_recovered, n_holdout)
    mean_d = float(np.mean(distances)) if distances else float("nan")
    return {
        "row": {
            "method": method_label,
            "n": n,
            "validity": format_rate(n_valid, n),
            "uniqueness": format_rate(n_unique, n_valid),
            "reconstruction": reconstruction,
            "novelty": NOVELTY,
            "stability": STABILITY,
            "note": note,
            "validity_rate": (n_valid / n) if n else float("nan"),
            "mean_min_distance_A": mean_d,
        },
        "per_cell": per_cell,
    }


def _reference(record: dict) -> tuple[np.ndarray, np.ndarray]:
    hkl = reflection_list(record["cell"], record["space_group"])
    amp = proposal_amplitudes(
        record["fracs"],
        record["elements"],
        record["cell"],
        record["space_group"],
        hkl,
    )
    return hkl, amp


def run_scoreboard(
    records: Sequence[dict],
    *,
    n: int = 8,
    steps: int = 40,
    seed: int = 0,
    run_diffusion: bool = True,
    excluded: Optional[Sequence[str]] = None,
) -> dict:
    """Train on the non-holdout records, score all three methods, return markdown pieces.

    The denoiser does not see |F|. Diffusion, when run, sees deposited Fcalc.
    """
    t0 = time.perf_counter()
    records = list(records)
    train = [r for r in records if not r["holdout"]]
    model = None
    final_loss = float("nan")
    trained_names: list[str] = []
    if train and steps > 0:
        from grok_phase_solver.generate.model import train_denoiser

        model = train_denoiser(train, steps=int(steps), seed=int(seed))
        final_loss = float(getattr(model, "final_loss", float("nan")))
        trained_names = [r["name"] for r in train]

    random_groups = []
    denoiser_groups = []
    diffusion_groups = []
    diffusion_error = None
    for index, record in enumerate(records):
        hkl, amp = _reference(record)
        common = dict(
            elements=record["elements"],
            cell=record["cell"],
            space_group=record["space_group"],
            z=int(record["z"]),
            reference_fracs=record["fracs"],
            reference_elements=record["elements"],
            hkl=hkl,
            amplitudes=amp,
            amplitude_source=AMPLITUDE_SOURCE_DEPOSITED_FCALC,
            count_reconstruction=bool(record["holdout"]),
        )
        random_groups.append(
            (
                record,
                propose(**common, n=n, method="random", seed=int(seed) + 17 * index),
            )
        )
        denoiser_groups.append(
            (
                record,
                propose(
                    **common,
                    n=n,
                    method="denoiser",
                    seed=int(seed) + 100 * index,
                    model=model,
                    fit_steps=0,
                ),
            )
        )
        if run_diffusion:
            try:
                diffusion_groups.append(
                    (
                        record,
                        propose(
                            **common,
                            n=1,
                            method="diffusion_hybrid",
                            seed=int(seed) + 1000 * index,
                            diffusion_steps=2,
                            diffusion_starts=1,
                            diffusion_polish=2,
                        ),
                    )
                )
            except Exception as exc:
                diffusion_error = f"{type(exc).__name__}: {exc}"
                diffusion_groups.append((record, []))

    random_sum = _summarize(METHOD_RANDOM, random_groups, "Uniform fractional coordinates. A draw is kept when the expanded cell has no pair closer than 1.0 Å, up to 80 tries. Composition is the requested multiset.")
    denoiser_note = (
        "Coordinate denoiser. Loss is mean squared minimum-image error of the "
        "predicted clean fractional coordinates. No |F| term. Last linear layer "
        "starts at zero. "
        + (
            f"Trained {int(steps)} steps on {', '.join(trained_names)}. "
            f"Final loss {final_loss:.6f}."
            if trained_names
            else "No training structures; samples come from the zero-initialized network."
        )
    )
    denoiser_sum = _summarize(METHOD_DENOISER, denoiser_groups, denoiser_note)
    if run_diffusion:
        note = DIFFUSION_NOTE
        if diffusion_error:
            note += f" A cell failed: {diffusion_error}"
        diffusion_sum = _summarize(METHOD_DIFFUSION, diffusion_groups, note)
    else:
        diffusion_sum = {
            "row": {
                "method": METHOD_DIFFUSION,
                "n": 0,
                "validity": "not run",
                "uniqueness": "not run",
                "reconstruction": "not run",
                "novelty": NOVELTY,
                "stability": STABILITY,
                "note": DIFFUSION_NOTE + " This invocation did not run it.",
                "validity_rate": float("nan"),
                "mean_min_distance_A": float("nan"),
            },
            "per_cell": [],
        }

    comparison = baseline_comparison_note(
        float(random_sum["row"]["validity_rate"]),
        float(denoiser_sum["row"]["validity_rate"]),
    )
    comparison += (
        f" Mean minimum distance was {_fmt(denoiser_sum['row']['mean_min_distance_A'])} Å "
        f"(denoiser) and {_fmt(random_sum['row']['mean_min_distance_A'])} Å (random)."
    )
    wall = time.perf_counter() - t0
    holdouts = [r["name"] for r in records if r["holdout"]]
    protocol = [
        f"Seed {int(seed)}. Denoiser training steps {int(steps)}. "
        f"Random and denoiser samples per cell: {int(n)}. Diffusion samples per cell: 1.",
        "Clash cutoff 1.0 Å on the symmetry-expanded cell. "
        "Uniqueness cutoff 0.15 fractional RMSD after centroid alignment. "
        "Reconstruction uses a 0.25 origin grid and 0.75 Å, and only the holdout.",
        "Structure-factor residual: both sides use B = 5 Å² and d_min = 1.5 Å. "
        "Reference amplitudes are Fcalc of the deposited coordinates, not measured Fobs.",
        "Holdout: " + (", ".join(holdouts) if holdouts else "none") + ". "
        "Training structures are not counted as reconstruction.",
        f"Wall time {wall:.1f} s.",
    ]
    included = [
        f"{r['name']} (non-H {len(r['elements'])}, {'holdout' if r['holdout'] else 'train'})"
        for r in records
    ]
    markdown = render_scoreboard_md(
        rows=[random_sum["row"], denoiser_sum["row"], diffusion_sum["row"]],
        per_cell=random_sum["per_cell"] + denoiser_sum["per_cell"] + diffusion_sum["per_cell"],
        protocol_lines=protocol,
        comparison=comparison,
        included=included,
        excluded=list(excluded or []),
    )
    return {
        "markdown": markdown,
        "wall_s": wall,
        "comparison": comparison,
        "random_validity": float(random_sum["row"]["validity_rate"]),
        "denoiser_validity": float(denoiser_sum["row"]["validity_rate"]),
        "final_loss": final_loss,
    }
