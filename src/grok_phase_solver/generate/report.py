"""Markdown for a single ``gps-generate`` run and for the COD scoreboard."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

NOVELTY = "not measured"
STABILITY = "not measured — no energy model"

METHOD_RANDOM = "random + clash rejection"
METHOD_DENOISER = "coordinate denoiser"
METHOD_DIFFUSION = "diffusion_hybrid (experimental, not default)"


def metric_definitions_md() -> str:
    """One sentence per metric. The sentences match ``score.py``."""
    return "\n".join(
        [
            "- **Validity.** A sample is valid when the symmetry-expanded cell has no "
            "minimum-image pair closer than 1.0 Å and the element multiset equals the "
            "requested composition.",
            "- **Uniqueness.** A valid sample is unique when, after fractional-centroid "
            "alignment, its greedy same-element minimum-image RMSD to every earlier "
            "unique sample of the same cell is at least 0.15.",
            "- **Reconstruction.** A held-out structure is recovered when at least one "
            "sample places at least half of its non-hydrogen sites within 0.75 Å after "
            "the best shift on a 0.25 fractional origin grid. This is not novelty.",
            f"- **Novelty.** {NOVELTY}.",
            f"- **Stability / synthesizability.** {STABILITY}.",
        ]
    )


def nonclaims_md() -> str:
    return "\n".join(
        [
            "This track does not run DFT, VASP, or LAMMPS. It does not implement a "
            "flow-matching paper, train a foundation model, or claim synthesizability. "
            "It does not design materials for AR or VR. It does not replace `gps-solve` "
            "or solve the phase problem. `gps-solve --method diffusion_hybrid` is a "
            "different experimental flag (Langevin phase completion), not this "
            "coordinate model, and it is not the production default.",
        ]
    )


def baseline_comparison_note(random_validity: float, denoiser_validity: float) -> str:
    """State the measured validity comparison. Cutoffs are not adjusted to the winner."""
    if denoiser_validity < random_validity:
        relation = "below"
        reason = (
            " The denoiser is trained with a minimum-image coordinate loss and never "
            "sees |F|. A few dozen steps on the training asymmetric units do not "
            "enforce the 1.0 Å clash cutoff, while the random baseline rejects draws "
            "that contain such a pair. The cutoff was not changed to hide that."
        )
    elif denoiser_validity > random_validity:
        relation = "above"
        reason = (
            " That comparison is only a clash-and-composition rate on this run. "
            "It is not a novelty, stability, or phasing result."
        )
    else:
        relation = "the same as"
        reason = ""
    return (
        f"The coordinate denoiser validity is {denoiser_validity:.3f}, {relation} "
        f"the random-plus-clash baseline at {random_validity:.3f}.{reason}"
    )


def _fraction(numer: int, denom: int) -> str:
    if denom <= 0:
        return "0/0"
    return f"{numer}/{denom} ({numer / denom:.3f})"


def render_scoreboard_md(
    *,
    rows: Sequence[Mapping],
    per_cell: Sequence[Mapping],
    protocol_lines: Sequence[str],
    comparison: str,
    included: Sequence[str],
    excluded: Sequence[str],
) -> str:
    """Full scoreboard. Every rate in ``rows`` is supplied by the caller."""
    header = [
        "# Coordinate generation scoreboard",
        "",
        "Written by `scripts/run_generate_scoreboard.py`. Do not edit the rates by hand.",
        "",
        "This track proposes fractional coordinates for a known composition and cell. "
        "It does not discover materials and it does not phase a structure.",
        "",
        nonclaims_md(),
        "",
        "## Definitions",
        "",
        metric_definitions_md(),
        "",
        "## Protocol",
        "",
    ]
    header.extend(f"- {line}" for line in protocol_lines)
    header.extend(["", "## Cohort", ""])
    header.append("Included: " + (", ".join(included) if included else "(none)"))
    header.append("")
    header.append("Excluded: " + (", ".join(excluded) if excluded else "(none)"))
    header.extend(
        [
            "",
            "## Results",
            "",
            "| method | n | validity | uniqueness | reconstruction | novelty | stability / synthesizability |",
            "|--------|---|----------|------------|----------------|---------|------------------------------|",
        ]
    )
    for row in rows:
        header.append(
            "| {method} | {n} | {validity} | {uniqueness} | {reconstruction} | {novelty} | {stability} |".format(
                method=row["method"],
                n=row["n"],
                validity=row["validity"],
                uniqueness=row["uniqueness"],
                reconstruction=row["reconstruction"],
                novelty=row.get("novelty", NOVELTY),
                stability=row.get("stability", STABILITY),
            )
        )
    header.extend(
        [
            "",
            "## Per structure",
            "",
            "| structure | holdout | method | n_valid | n_unique | mean min distance (Å) | best match fraction | mean R (Fcalc) |",
            "|-----------|---------|--------|---------|----------|-----------------------|---------------------|----------------|",
        ]
    )
    for row in per_cell:
        header.append(
            "| {structure} | {holdout} | {method} | {n_valid} | {n_unique} | {dmin} | {match} | {rfac} |".format(
                structure=row["structure"],
                holdout="yes" if row["holdout"] else "no",
                method=row["method"],
                n_valid=row["n_valid"],
                n_unique=row["n_unique"],
                dmin=row["dmin"],
                match=row["match"],
                rfac=row["rfac"],
            )
        )
    notes = [str(row["note"]) for row in rows if row.get("note")]
    header.extend(["", "## Notes", ""])
    if notes:
        header.extend(f"- {note}" for note in notes)
    else:
        header.append("- (none)")
    header.extend(["", "## Baseline comparison", "", comparison, ""])
    return "\n".join(header) + "\n"


def render_run_report(
    *,
    cif_name: str,
    seed: int,
    fit_steps: int,
    rows: Sequence[Mapping],
    comparison: str,
    n_atoms: int,
    space_group: str,
    z: int,
    wall_s: Optional[float] = None,
) -> str:
    """Report for one CIF. Match fractions here are in-sample, not reconstruction."""
    lines = [
        "# gps-generate report",
        "",
        f"Input `{cif_name}`: {n_atoms} non-hydrogen asymmetric-unit atoms, "
        f"space group `{space_group}`, Z={z}. Seed {seed}.",
        "",
        f"The coordinate denoiser was trained for {fit_steps} steps on this CIF only. "
        "Coordinate match fractions below are in-sample. They are not held-out "
        "reconstruction. The held-out protocol is `scripts/run_generate_scoreboard.py`.",
        "",
        nonclaims_md(),
        "",
        "## Definitions",
        "",
        metric_definitions_md(),
        "",
        "## Samples",
        "",
        "| method | n | validity | uniqueness | in-sample match ≥ 0.5 | novelty | stability / synthesizability |",
        "|--------|---|----------|------------|------------------------|---------|------------------------------|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {n} | {validity} | {uniqueness} | {match} | {novelty} | {stability} |".format(
                method=row["method"],
                n=row["n"],
                validity=row["validity"],
                uniqueness=row["uniqueness"],
                match=row["match"],
                novelty=NOVELTY,
                stability=STABILITY,
            )
        )
    lines.extend(["", "## Baseline comparison", "", comparison, ""])
    lines.append(
        "`diffusion_hybrid` was not run by this command. It is the experimental "
        "phase-path flag (`gps-solve --method diffusion_hybrid`), scored only on "
        "the COD scoreboard, and it is not the default."
    )
    if wall_s is not None:
        lines.extend(["", f"Wall time: {wall_s:.1f} s on CPU.", ""])
    else:
        lines.append("")
    return "\n".join(lines)


def format_rate(numer: int, denom: int) -> str:
    return _fraction(numer, denom)
