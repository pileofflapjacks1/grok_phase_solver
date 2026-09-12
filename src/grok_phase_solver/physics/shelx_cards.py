"""SHELX LATT / SYMM cards for trial.res (identity omitted)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

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


def _triplet_is_inversion(triplet: str) -> bool:
    parts = [p.strip().replace(" ", "").lower().lstrip("+") for p in triplet.split(",")]
    return parts == ["-x", "-y", "-z"]


def _triplet_to_shelx(triplet: str) -> str:
    """Keep gemmi fractions (1/2, 1/4, 3/4). Olex2 rejected -X+0.5 on P21/n."""
    pretty = []
    for p in triplet.split(","):
        p = p.strip()
        p = p.replace("x", "X").replace("y", "Y").replace("z", "Z")
        pretty.append(p)
    return ", ".join(pretty)


def _latt_code_from_hm(hm: str, is_centro: bool) -> int:
    letter = (hm.strip() or "P")[0].upper()
    code = _LATT_CENTRING.get(letter, 1)
    return code if is_centro else -code


def _is_inversion_op(op: Any) -> bool:
    """True for inversion (origin or shifted). LATT > 0 already implies it."""
    if hasattr(op, "rot_type"):
        try:
            if int(op.rot_type()) == -1:
                return True
        except Exception:
            pass
    trip = op.triplet() if hasattr(op, "triplet") else str(op)
    return _triplet_is_inversion(trip)


def _op_key(op: Any) -> Tuple[Any, ...]:
    wrapped = op.wrap() if hasattr(op, "wrap") else op
    den = int(getattr(wrapped, "DEN", 24))
    rot = tuple(tuple(int(x) for x in row) for row in wrapped.rot)
    tran = tuple(int(t) % den for t in wrapped.tran)
    return (rot, tran)


def _primitive_ops(sg: Any) -> List[Any]:
    """Centering lives in LATT — do not emit those translations as SYMM."""
    ops = sg.operations()
    if hasattr(ops, "sym_ops"):
        return list(ops.sym_ops)
    return list(ops)


def _unique_shelx_generators(sg: Any, lat: int) -> List[str]:
    """
    SHELX SYMM cards: omit identity always.

    When LATT > 0, inversion (and its mates) are already implied, so emit
    one representative per {op, inversion∘op} pair. gemmi order matches
    the usual SHELXL generator list (P21/n → one 2₁, not the three-op dump).
    """
    cards: List[str] = []
    if lat > 0:
        import gemmi

        inv = gemmi.Op("-x,-y,-z")
        seen = set()
        for op in _primitive_ops(sg):
            trip = op.triplet() if hasattr(op, "triplet") else str(op)
            if _triplet_is_identity(trip) or _is_inversion_op(op):
                continue
            key = _op_key(op)
            mate = inv * op
            mate_key = _op_key(mate)
            if key in seen or mate_key in seen:
                continue
            seen.add(key)
            seen.add(mate_key)
            cards.append(_triplet_to_shelx(trip))
        return cards

    for op in _primitive_ops(sg):
        trip = op.triplet() if hasattr(op, "triplet") else str(op)
        if _triplet_is_identity(trip):
            continue
        cards.append(_triplet_to_shelx(trip))
    return cards


def shelx_latt_symm(
    space_group: Optional[str] = None,
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
) -> Tuple[int, List[str]]:
    """
    SHELX LATT code + unique-generator SYMM cards for a trial.res header.

    If ``symm`` is provided (parsed .ins), those cards and ``lattice`` are used
    unchanged. P2₁2₁2₁ always emits Bragg's three 2₁ strings. P1 → LATT −1
    and no SYMM; P−1 → LATT 1 and no SYMM. Other groups use gemmi ops when
    available. Identity is omitted. When LATT > 0, inversion and inversion
    mates are omitted (SHELX convention: LATT already implies them).
    Centering translations are omitted (they live in the LATT code).
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
        return lat, _unique_shelx_generators(sg, lat)
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
