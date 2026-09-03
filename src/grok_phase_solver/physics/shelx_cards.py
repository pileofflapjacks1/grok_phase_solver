"""SHELX LATT / SYMM cards for trial.res (identity omitted)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from grok_phase_solver.physics.symmetry import (
    gemmi_available,
    normalize_space_group_name,
    parse_space_group,
)

# Bragg spec (IT 19). Do not paraphrase these three strings.
_P212121_SYMM = (
    "0.5-X, -Y, 0.5+Z",
    "-X, 0.5+Y, 0.5-Z",
    "0.5+X, 0.5-Y, -Z",
)

_LATT_CENTRING = {"P": 1, "I": 2, "R": 3, "F": 4, "A": 5, "B": 6, "C": 7}


def _hm_compact(name: Optional[str]) -> str:
    return normalize_space_group_name(name).replace(" ", "").upper().replace("−", "-")


def _triplet_is_identity(triplet: str) -> bool:
    parts = [p.strip().replace(" ", "").lower().lstrip("+") for p in triplet.split(",")]
    return parts == ["x", "y", "z"]


def _triplet_to_shelx(triplet: str) -> str:
    pretty = []
    for p in triplet.split(","):
        p = p.strip().replace("1/2", "0.5").replace("1/4", "0.25").replace("3/4", "0.75")
        p = p.replace("x", "X").replace("y", "Y").replace("z", "Z")
        pretty.append(p)
    return ", ".join(pretty)


def _latt_code_from_hm(hm: str, is_centro: bool) -> int:
    letter = (hm.strip() or "P")[0].upper()
    code = _LATT_CENTRING.get(letter, 1)
    return code if is_centro else -code


def shelx_latt_symm(
    space_group: Optional[str] = None,
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
) -> Tuple[int, List[str]]:
    """
    SHELX LATT code + non-identity SYMM cards for a trial.res header.

    If ``symm`` is provided (parsed .ins), those cards and ``lattice`` are used
    unchanged. P2₁2₁2₁ always emits Bragg's three 2₁ strings. P1 → LATT −1
    and no SYMM; P−1 → LATT 1 and no SYMM. Other groups use gemmi ops when
    available; identity is omitted (SHELX convention).
    """
    if symm:
        lat = -1 if lattice is None else int(lattice)
        cards = [str(s).strip() for s in symm if str(s).strip()]
        return lat, cards

    compact = _hm_compact(space_group)
    if compact in ("P1",):
        return -1, []
    if compact in ("P-1", "P1-"):
        return 1, []
    if compact == "P212121":
        return -1, list(_P212121_SYMM)

    info = parse_space_group(space_group)
    hm = info.hm or normalize_space_group_name(space_group)
    if not gemmi_available() or not info.available:
        lat = _latt_code_from_hm(hm, info.is_centrosymmetric)
        return lat, []
    try:
        import gemmi

        sg = gemmi.SpaceGroup(normalize_space_group_name(space_group))
        lat = _latt_code_from_hm(hm, bool(sg.is_centrosymmetric()))
        cards: List[str] = []
        for op in sg.operations():
            trip = op.triplet() if hasattr(op, "triplet") else str(op)
            if _triplet_is_identity(trip):
                continue
            cards.append(_triplet_to_shelx(trip))
        return lat, cards
    except Exception:
        lat = _latt_code_from_hm(hm, info.is_centrosymmetric)
        return lat, []


def format_shelx_latt_symm_lines(
    space_group: Optional[str] = None,
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
) -> List[str]:
    """``LATT n`` plus ``SYMM …`` lines (no trailing newline)."""
    latt, cards = shelx_latt_symm(space_group, lattice=lattice, symm=symm)
    lines = [f"LATT {latt}"]
    for c in cards:
        lines.append(f"SYMM {c}")
    return lines
