"""Validity, uniqueness, and coordinate residuals. No torch."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, Sequence

import numpy as np

from grok_phase_solver.generate.geometry import (
    CLASH_DISTANCE_A,
    FCALC_B_ISO,
    FCALC_D_MIN,
    RECONSTRUCTION_FRACTION,
    UNIQUENESS_RMSD,
    SpaceGroupError,
    composition_matches,
    expanded_min_distance,
    greedy_fractional_rmsd,
    greedy_match_fraction,
    pairwise_min_distance,
    space_group_consistent,
)
from grok_phase_solver.metrics.rfactor import r_factor
from grok_phase_solver.physics.reciprocal import generate_hkl
from grok_phase_solver.physics.structure_factors import compute_structure_factors

AMPLITUDE_SOURCE_DEPOSITED_FCALC = "deposited_fcalc_B5_dmin_1.5"


def reflection_list(
    cell: np.ndarray,
    space_group: str,
    d_min: float = FCALC_D_MIN,
) -> np.ndarray:
    """Miller indices with d ≥ ``d_min``, Friedel pairs kept, systematic absences dropped."""
    hkl = generate_hkl(np.asarray(cell, dtype=np.float64), d_min=float(d_min), expand_friedel=True)
    try:
        from grok_phase_solver.generate.geometry import symmetry_operations

        ops = symmetry_operations(space_group)
        keep = [
            not ops.is_systematically_absent([int(h), int(k), int(l)])
            for h, k, l in hkl
        ]
        hkl = hkl[np.asarray(keep, dtype=bool)]
    except SpaceGroupError:
        pass
    return hkl


def proposal_amplitudes(
    fracs: np.ndarray,
    elements: Sequence[str],
    cell: np.ndarray,
    space_group: str,
    hkl: np.ndarray,
    b_iso: float = FCALC_B_ISO,
) -> np.ndarray:
    """|Fcalc| of the symmetry-expanded sites at a single isotropic B."""
    from grok_phase_solver.generate.geometry import expand_sites

    expanded, els = expand_sites(fracs, elements, space_group)
    if len(expanded) == 0:
        return np.zeros(len(hkl), dtype=np.float64)
    b_isos = np.full(len(expanded), float(b_iso), dtype=np.float64)
    fcalc = compute_structure_factors(
        hkl, expanded, els, np.asarray(cell, dtype=np.float64), b_isos=b_isos
    )
    return np.abs(fcalc)


def score_sites(
    fracs: np.ndarray,
    elements: Sequence[str],
    requested: Sequence[str],
    cell: np.ndarray,
    space_group: str,
    *,
    reference_fracs: Optional[np.ndarray] = None,
    reference_elements: Optional[Sequence[str]] = None,
    hkl: Optional[np.ndarray] = None,
    amplitudes: Optional[np.ndarray] = None,
    count_reconstruction: bool = False,
    clash_a: float = CLASH_DISTANCE_A,
) -> dict:
    """Score one coordinate set.

    Validity is clash-free expanded-cell packing and an exact element multiset.
    Space-group consistency and the structure-factor residual are reported beside
    that definition and do not change it. ``reconstructed`` is set only when
    ``count_reconstruction`` is true (held-out structures).
    """
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    cell = np.asarray(cell, dtype=np.float64).reshape(6)
    sg_ok = space_group_consistent(fracs, elements, cell, space_group)
    try:
        dmin = expanded_min_distance(fracs, elements, cell, space_group)
    except SpaceGroupError:
        dmin = pairwise_min_distance(fracs, cell)
        sg_ok = False
    clash = bool(dmin < clash_a)
    comp_ok = composition_matches(elements, requested)
    valid = (not clash) and comp_ok

    residual: Optional[float] = None
    if hkl is not None and amplitudes is not None and len(hkl) > 0:
        try:
            f_prop = proposal_amplitudes(fracs, elements, cell, space_group, hkl)
            residual = float(r_factor(amplitudes, f_prop, scale=True))
        except (SpaceGroupError, ValueError):
            residual = None

    match: Optional[float] = None
    reconstructed: Optional[bool] = None
    if reference_fracs is not None and reference_elements is not None:
        match = float(
            greedy_match_fraction(
                fracs,
                elements,
                np.asarray(reference_fracs, dtype=np.float64),
                reference_elements,
                cell,
            )
        )
        if count_reconstruction:
            reconstructed = bool(match >= RECONSTRUCTION_FRACTION)

    return {
        "valid": valid,
        "clash": clash,
        "composition_match": comp_ok,
        "min_distance_A": float(dmin),
        "sg_consistent": bool(sg_ok),
        "r_factor": residual,
        "match_fraction": match,
        "reconstructed": reconstructed,
        "unique": False,
    }


def assign_uniqueness(
    samples: Sequence[MutableMapping],
    cutoff: float = UNIQUENESS_RMSD,
) -> None:
    """Mark valid samples that are not near-duplicates of an earlier valid sample.

    Comparison is inside this list only (one composition and cell). Invalid
    samples are not unique. RMSD is greedy, same-element, after centroid alignment.
    """
    accepted: list[MutableMapping] = []
    for sample in samples:
        score = sample["score"]
        if not score.get("valid"):
            score["unique"] = False
            continue
        duplicate = False
        for earlier in accepted:
            rmsd = greedy_fractional_rmsd(
                earlier["fracs"],
                earlier["elements"],
                sample["fracs"],
                sample["elements"],
            )
            if rmsd < cutoff:
                duplicate = True
                break
        score["unique"] = not duplicate
        if not duplicate:
            accepted.append(sample)


def pooled_rates(samples: Sequence[Mapping]) -> dict:
    """Counts for one method on one cell."""
    n = len(samples)
    n_valid = sum(1 for s in samples if s["score"]["valid"])
    n_unique = sum(1 for s in samples if s["score"].get("unique"))
    distances = [float(s["score"]["min_distance_A"]) for s in samples]
    finite = [d for d in distances if np.isfinite(d)]
    residuals = [
        float(s["score"]["r_factor"])
        for s in samples
        if s["score"]["r_factor"] is not None and np.isfinite(s["score"]["r_factor"])
    ]
    matches = [
        float(s["score"]["match_fraction"])
        for s in samples
        if s["score"]["match_fraction"] is not None
    ]
    n_recon = sum(1 for s in samples if s["score"].get("reconstructed") is True)
    return {
        "n": n,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "mean_min_distance_A": float(np.mean(finite)) if finite else float("nan"),
        "mean_r_factor": float(np.mean(residuals)) if residuals else float("nan"),
        "best_match_fraction": float(max(matches)) if matches else float("nan"),
        "n_reconstructed_samples": n_recon,
    }
