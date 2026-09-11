"""Fold a P1 peak dump to one unique set under SHELX LATT/SYMM ops."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.shelx_cards import shelx_latt_symm

# SHELX LATT |n|: 1=P, 2=I, 3=R obverse, 4=F, 5=A, 6=B, 7=C
_LATT_TRANSLATIONS = {
    1: [(0.0, 0.0, 0.0)],
    2: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    3: [(0.0, 0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)],
    4: [
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ],
    5: [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],
    6: [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],
    7: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],
}

_TOKEN = re.compile(r"[+-](?:\d+\.?\d*|[XYZ])")


def _orth_matrix(cell: np.ndarray) -> np.ndarray:
    a, b, c, al, be, ga = [float(x) for x in cell]
    al, be, ga = np.deg2rad([al, be, ga])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / (np.sin(ga) + 1e-16)
    cz2 = max(c * c - cx * cx - cy * cy, 0.0)
    vc = np.array([cx, cy, np.sqrt(cz2)])
    return np.column_stack([va, vb, vc])


def _apply_triplet(triplet: str, xyz: np.ndarray) -> np.ndarray:
    """Evaluate a SHELX SYMM card at fractional xyz."""
    parts = [p.strip() for p in triplet.split(",")]
    if len(parts) != 3:
        raise ValueError(f"bad SYMM triplet: {triplet!r}")
    x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    out = np.zeros(3, dtype=np.float64)
    for i, part in enumerate(parts):
        expr = (
            part.upper()
            .replace(" ", "")
            .replace("1/2", "0.5")
            .replace("1/4", "0.25")
            .replace("3/4", "0.75")
        )
        expr = re.sub(r"(?<![+\-])([XYZ])", r"+\1", expr)
        if not expr or expr[0] not in "+-":
            expr = "+" + expr
        val = 0.0
        for t in _TOKEN.findall(expr):
            sign = 1.0 if t[0] == "+" else -1.0
            body = t[1:]
            if body == "X":
                val += sign * x
            elif body == "Y":
                val += sign * y
            elif body == "Z":
                val += sign * z
            else:
                val += sign * float(body)
        out[i] = val
    return out


def space_group_ops(
    space_group: Optional[str] = None,
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
) -> List[Callable[[np.ndarray], np.ndarray]]:
    """Identity + SYMM cards + inversion if LATT>0 + lattice translations."""
    latt, cards = shelx_latt_symm(space_group, lattice=lattice, symm=symm)
    gens: List[Callable[[np.ndarray], np.ndarray]] = [lambda xyz: np.asarray(xyz, dtype=np.float64)]
    for card in cards:
        gens.append(lambda xyz, c=card: _apply_triplet(c, np.asarray(xyz, dtype=np.float64)))
    if int(latt) > 0:
        gens.append(lambda xyz: -np.asarray(xyz, dtype=np.float64))
    trans = _LATT_TRANSLATIONS.get(abs(int(latt)), [(0.0, 0.0, 0.0)])
    ops: List[Callable[[np.ndarray], np.ndarray]] = []
    for g in gens:
        for t in trans:
            tt = np.asarray(t, dtype=np.float64)
            ops.append(lambda xyz, g=g, tt=tt: g(np.asarray(xyz, dtype=np.float64)) + tt)
    return ops


def _min_image_cart(f1: np.ndarray, f2: np.ndarray, M: np.ndarray) -> float:
    d = (f1 - f2 + 0.5) % 1.0 - 0.5
    return float(np.linalg.norm(M @ d))


def unique_asu_fracs(
    fracs: np.ndarray,
    space_group: Optional[str],
    cell: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    tol_angstrom: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Keep one representative per orbit under the space-group ops.

    The kept coordinate is the image that packs toward already-kept atoms
    (origin if empty) so the blob is compact, not four copies.

    Returns (fracs_unique, original_indices, meta).
    """
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    n = len(fracs)
    meta = {"n_in": n, "n_out": n, "n_ops": 1, "folded": False}
    if n == 0:
        return fracs.copy(), np.zeros(0, dtype=int), meta

    ops = space_group_ops(space_group, lattice=lattice, symm=symm)
    meta["n_ops"] = len(ops)
    if len(ops) <= 1:
        return fracs.copy(), np.arange(n, dtype=int), meta

    w = np.zeros(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != n:
        w = np.zeros(n, dtype=np.float64)
    M = _orth_matrix(np.asarray(cell, dtype=np.float64))
    order = np.argsort(-w, kind="stable")

    def images(xyz: np.ndarray) -> List[np.ndarray]:
        return [np.mod(op(xyz), 1.0) for op in ops]

    kept_frac: List[np.ndarray] = []
    kept_idx: List[int] = []

    def orbit_hit(xyz: np.ndarray) -> bool:
        for im in images(xyz):
            for k in kept_frac:
                if _min_image_cart(im, k, M) < tol_angstrom:
                    return True
        return False

    def pack(xyz: np.ndarray) -> np.ndarray:
        ims = images(xyz)
        if not kept_frac:
            return min(ims, key=lambda f: float(np.linalg.norm(M @ ((f + 0.5) % 1.0 - 0.5))))
        centroid = np.mean(np.vstack(kept_frac), axis=0)
        return min(ims, key=lambda f: _min_image_cart(f, centroid, M))

    for i in order:
        xyz = fracs[int(i)]
        if orbit_hit(xyz):
            continue
        kept_idx.append(int(i))
        kept_frac.append(pack(xyz))

    out = np.vstack(kept_frac) if kept_frac else fracs[:0].copy()
    idx = np.asarray(kept_idx, dtype=int)
    meta["n_out"] = int(len(idx))
    meta["folded"] = True
    return out, idx, meta



# Default non-H peak budget for COD 2200001 (C12H18N2O3 × Z′=2).
DEFAULT_N_NON_H_BUDGET = 34


def budget_peaks(
    peaks: Sequence,
    n_non_h_budget: int = DEFAULT_N_NON_H_BUDGET,
) -> Tuple[list, dict]:
    """
    Keep the strongest ``n_non_h_budget`` peaks by ``height_sigma``.

    DensityPeak lists have no hydrogens; CrystalX H placement is skipped for
    the trial.res path that uses this budget. Call after ``unique_peaks``.
    """
    peaks = list(peaks)
    n_in = len(peaks)
    n_keep = max(0, int(n_non_h_budget))
    meta = {
        "n_in": n_in,
        "n_out": min(n_in, n_keep),
        "n_budget": n_keep,
        "budgeted": n_in > n_keep,
    }
    if n_in == 0 or n_in <= n_keep:
        return peaks, meta
    order = sorted(
        range(n_in),
        key=lambda i: -float(getattr(peaks[i], "height_sigma", 0.0) or 0.0),
    )
    kept = [peaks[i] for i in order[:n_keep]]
    meta["n_out"] = len(kept)
    return kept, meta


def unique_typed_atoms(
    typed: Sequence,
    cell: np.ndarray,
    space_group: Optional[str],
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    tol_angstrom: float = 0.6,
) -> Tuple[list, dict]:
    """Fold TypedAtom list; labels/elements of survivors unchanged."""
    typed = list(typed)
    if not typed:
        return [], {"n_in": 0, "n_out": 0, "n_ops": 1, "folded": False}
    fracs = np.vstack([np.asarray(t.fract, dtype=np.float64) for t in typed])
    weights = np.array([float(getattr(t, "height_sigma", 0.0) or 0.0) for t in typed])
    newf, idx, meta = unique_asu_fracs(
        fracs,
        space_group,
        cell,
        weights=weights,
        lattice=lattice,
        symm=symm,
        tol_angstrom=tol_angstrom,
    )
    out = []
    for j, i in enumerate(idx):
        t = typed[int(i)]
        out.append(replace(t, fract=np.asarray(newf[j], dtype=np.float64).copy()))
    return out, meta


def unique_peaks(
    peaks: Sequence,
    cell: np.ndarray,
    space_group: Optional[str],
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    tol_angstrom: float = 0.6,
) -> Tuple[list, dict]:
    """Fold DensityPeak list."""
    peaks = list(peaks)
    if not peaks:
        return [], {"n_in": 0, "n_out": 0, "n_ops": 1, "folded": False}
    fracs = np.vstack([np.asarray(p.fract, dtype=np.float64) for p in peaks])
    weights = np.array([float(getattr(p, "height_sigma", 0.0) or 0.0) for p in peaks])
    newf, idx, meta = unique_asu_fracs(
        fracs,
        space_group,
        cell,
        weights=weights,
        lattice=lattice,
        symm=symm,
        tol_angstrom=tol_angstrom,
    )
    out = []
    for j, i in enumerate(idx):
        p = peaks[int(i)]
        out.append(replace(p, fract=np.asarray(newf[j], dtype=np.float64).copy()))
    return out, meta
