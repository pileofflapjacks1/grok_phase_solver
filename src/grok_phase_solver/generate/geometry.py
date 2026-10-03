"""Fractional-coordinate geometry used by the generative track. No torch."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# A pair closer than this (minimum image, Cartesian Å) is a clash.
# Organic bonds are typically 1.2–1.6 Å, so 1.0 Å flags overlaps.
CLASH_DISTANCE_A = 1.0
# Same-element fractional RMSD below this, after centroid alignment, is a duplicate.
UNIQUENESS_RMSD = 0.15
# Held-out site match: Cartesian distance after the best origin-grid shift.
RECONSTRUCTION_DISTANCE_A = 0.75
RECONSTRUCTION_FRACTION = 0.5
ORIGIN_GRID_STEP = 0.25
# Symmetry image of one ASU atom landing on a different ASU atom.
SG_OVERLAP_A = 0.2
RANDOM_MAX_TRIES = 80
FCALC_B_ISO = 5.0
FCALC_D_MIN = 1.5

ELEMENT_VOCAB: tuple[str, ...] = (
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "B",
    "Si",
    "H",
    "X",
)
_HYDROGEN = {"H", "D"}


class SpaceGroupError(ValueError):
    """Hermann–Mauguin symbol was not accepted."""


def element_symbol(raw: str) -> str:
    """Leading chemical symbol. ``O(14)`` → ``O``, ``CL`` → ``Cl``."""
    letters: list[str] = []
    for ch in str(raw).strip():
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    if not letters:
        return "X"
    sym = "".join(letters)
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


def element_index(raw: str) -> int:
    """Index into ``ELEMENT_VOCAB``. Unknown symbols use the ``X`` bin."""
    sym = element_symbol(raw)
    try:
        return ELEMENT_VOCAB.index(sym)
    except ValueError:
        return ELEMENT_VOCAB.index("X")


def is_hydrogen(raw: str) -> bool:
    return element_symbol(raw) in _HYDROGEN


def wrap_frac(fracs: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(fracs, dtype=np.float64), 1.0)


def orth_matrix(cell: np.ndarray) -> np.ndarray:
    """Fractional column → Cartesian Å. Same orthogonalization as ``CrystalStructure``."""
    from grok_phase_solver.io.cif import CrystalStructure

    cell = np.asarray(cell, dtype=np.float64).reshape(6)
    return CrystalStructure("t", cell, "P1").orth_matrix


def pairwise_min_distance(fracs: np.ndarray, cell: np.ndarray) -> float:
    """Smallest minimum-image Cartesian distance among distinct sites. ``inf`` if N < 2."""
    fracs = wrap_frac(np.asarray(fracs, dtype=np.float64).reshape(-1, 3))
    n = int(fracs.shape[0])
    if n < 2:
        return float("inf")
    delta = fracs[:, None, :] - fracs[None, :, :]
    delta -= np.round(delta)
    cart = delta @ orth_matrix(cell).T
    dist = np.linalg.norm(cart, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(dist[iu].min())


def symmetry_operations(space_group: str):
    """gemmi symmetry operators. Raises ``SpaceGroupError`` on a bad symbol."""
    try:
        import gemmi

        return gemmi.SpaceGroup(str(space_group)).operations()
    except Exception as exc:
        raise SpaceGroupError(str(exc)) from exc


def expand_sites(
    fracs: np.ndarray,
    elements: Sequence[str],
    space_group: str,
) -> tuple[np.ndarray, list[str]]:
    """Apply space-group operators and drop sites that round to the same 1e-5 key."""
    fracs = wrap_frac(np.asarray(fracs, dtype=np.float64).reshape(-1, 3))
    els = [element_symbol(e) for e in elements]
    if len(els) != len(fracs):
        raise ValueError("elements and fracs must have the same length")
    ops = symmetry_operations(space_group)
    out_f: list[np.ndarray] = []
    out_e: list[str] = []
    seen: set[tuple] = set()
    for el, frac in zip(els, fracs):
        for op in ops:
            x, y, z = op.apply_to_xyz(frac.tolist())
            wrapped = np.array([x % 1.0, y % 1.0, z % 1.0], dtype=np.float64)
            key = (el, round(float(wrapped[0]), 5), round(float(wrapped[1]), 5), round(float(wrapped[2]), 5))
            if key in seen:
                continue
            seen.add(key)
            out_f.append(wrapped)
            out_e.append(el)
    if not out_f:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.vstack(out_f), out_e


def expanded_min_distance(
    fracs: np.ndarray,
    elements: Sequence[str],
    cell: np.ndarray,
    space_group: str,
) -> float:
    """Minimum-image distance on the symmetry-expanded cell."""
    expanded, _ = expand_sites(fracs, elements, space_group)
    return pairwise_min_distance(expanded, cell)


def space_group_consistent(
    fracs: np.ndarray,
    elements: Sequence[str],
    cell: np.ndarray,
    space_group: str,
    overlap_a: float = SG_OVERLAP_A,
) -> bool:
    """True when gemmi accepts the symbol and no symmetry image of an ASU atom
    lies within ``overlap_a`` of a different ASU atom.

    An image that lands on the same atom within 0.05 Å is a special position
    (or the identity) and is allowed. Element labels are not used.
    """
    del elements  # composition is a separate check
    fracs = wrap_frac(np.asarray(fracs, dtype=np.float64).reshape(-1, 3))
    try:
        ops = symmetry_operations(space_group)
    except SpaceGroupError:
        return False
    metric = orth_matrix(cell)
    for i, frac in enumerate(fracs):
        for op in ops:
            x, y, z = op.apply_to_xyz(frac.tolist())
            image = np.array([x % 1.0, y % 1.0, z % 1.0], dtype=np.float64)
            for j, other in enumerate(fracs):
                delta = image - other
                delta -= np.round(delta)
                cart = float(np.linalg.norm(metric @ delta))
                if i == j and cart < 0.05:
                    continue
                if cart < overlap_a:
                    return False
    return True


def composition_multiset(elements: Sequence[str]) -> list[str]:
    return sorted(element_symbol(e) for e in elements)


def composition_matches(elements: Sequence[str], requested: Sequence[str]) -> bool:
    return composition_multiset(elements) == composition_multiset(requested)


def align_centroid(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Shift ``moving`` so its fractional centroid matches ``reference`` (minimum image)."""
    reference = wrap_frac(np.asarray(reference, dtype=np.float64).reshape(-1, 3))
    moving = wrap_frac(np.asarray(moving, dtype=np.float64).reshape(-1, 3))
    if len(reference) == 0 or len(moving) == 0:
        return moving
    shift = reference.mean(axis=0) - moving.mean(axis=0)
    shift -= np.round(shift)
    return wrap_frac(moving + shift)


def greedy_fractional_rmsd(
    reference: np.ndarray,
    reference_elements: Sequence[str],
    moving: np.ndarray,
    moving_elements: Sequence[str],
) -> float:
    """Greedy same-element minimum-image RMSD after centroid alignment.

    Returns ``inf`` when the element multisets differ.
    """
    if composition_multiset(reference_elements) != composition_multiset(moving_elements):
        return float("inf")
    ref = wrap_frac(np.asarray(reference, dtype=np.float64).reshape(-1, 3))
    mov = align_centroid(ref, np.asarray(moving, dtype=np.float64))
    ref_e = [element_symbol(e) for e in reference_elements]
    mov_e = [element_symbol(e) for e in moving_elements]
    pairs: list[tuple[float, int, int]] = []
    for i, ei in enumerate(ref_e):
        for j, ej in enumerate(mov_e):
            if ei != ej:
                continue
            delta = ref[i] - mov[j]
            delta -= np.round(delta)
            pairs.append((float(np.dot(delta, delta)), i, j))
    pairs.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    errs: list[float] = []
    for dist2, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        errs.append(dist2)
    if len(errs) != len(ref):
        return float("inf")
    return float(np.sqrt(np.mean(errs))) if errs else 0.0


def origin_shifts(step: float = ORIGIN_GRID_STEP) -> list[np.ndarray]:
    """Fractional origin grid. ``step`` 0.25 → 64 shifts, including the origin."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round(1.0 / step))
    shifts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shifts.append(np.array([i * step, j * step, k * step], dtype=np.float64))
    return shifts


def greedy_match_fraction(
    proposal: np.ndarray,
    proposal_elements: Sequence[str],
    target: np.ndarray,
    target_elements: Sequence[str],
    cell: np.ndarray,
    tol_a: float = RECONSTRUCTION_DISTANCE_A,
    step: float = ORIGIN_GRID_STEP,
) -> float:
    """Fraction of target sites matched within ``tol_a`` after the best origin shift.

    The denominator is the number of target sites. Same-element greedy matching.
    """
    target = wrap_frac(np.asarray(target, dtype=np.float64).reshape(-1, 3))
    proposal = wrap_frac(np.asarray(proposal, dtype=np.float64).reshape(-1, 3))
    if len(target) == 0:
        return 1.0
    if len(proposal) == 0:
        return 0.0
    metric = orth_matrix(cell)
    tgt_e = [element_symbol(e) for e in target_elements]
    prop_e = [element_symbol(e) for e in proposal_elements]
    best = 0.0
    for shift in origin_shifts(step):
        moved = wrap_frac(proposal + shift)
        pairs: list[tuple[float, int, int]] = []
        for i, ei in enumerate(tgt_e):
            for j, ej in enumerate(prop_e):
                if ei != ej:
                    continue
                delta = target[i] - moved[j]
                delta -= np.round(delta)
                dist = float(np.linalg.norm(metric @ delta))
                pairs.append((dist, i, j))
        pairs.sort()
        used_i: set[int] = set()
        used_j: set[int] = set()
        hits = 0
        for dist, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            if dist <= tol_a:
                hits += 1
        best = max(best, hits / len(target))
        if best >= 1.0:
            return 1.0
    return float(best)


def sample_random_fracs(
    n_atoms: int,
    elements: Sequence[str],
    cell: np.ndarray,
    space_group: str,
    rng: np.random.Generator,
    max_tries: int = RANDOM_MAX_TRIES,
    clash_a: float = CLASH_DISTANCE_A,
) -> tuple[np.ndarray, bool]:
    """Uniform fractional coordinates. Retry only when the expanded cell clashes.

    Returns the coordinates and whether a clash-free draw was found.
    If every try clashes, the widest draw is returned and the flag is False.
    """
    if len(elements) != n_atoms:
        raise ValueError("elements length must equal n_atoms")
    best: np.ndarray | None = None
    best_d = -1.0
    accepted = False
    for _ in range(max(1, int(max_tries))):
        fracs = rng.random((n_atoms, 3))
        try:
            dmin = expanded_min_distance(fracs, elements, cell, space_group)
        except SpaceGroupError:
            dmin = pairwise_min_distance(fracs, cell)
        if dmin >= clash_a:
            return fracs, True
        if dmin > best_d:
            best = fracs
            best_d = dmin
            accepted = False
    assert best is not None
    return best, accepted
