"""Discrete-molecule packing after unique-ASU fold + peak budget (Mark/Bragg).

Treat budgeted peaks as C, build a covalent graph with a fixed C–C cutoff,
allowing neighbours under written SHELX LATT/SYMM ops and short lattice
translations (±1 cell). Keep largest finite connected component(s) until
~n_non_h non-H sites; map survivors back to the ASU. Fail closed on an
infinite polymer under symmetry — never invent a fake discrete trial.res.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.unique_asu import (
    DEFAULT_N_NON_H_BUDGET,
    space_group_ops,
)

# C–C covalent neighbour cutoff (Å). Bragg: ~1.8 Å; one constant, documented.
COVALENT_C_C_CUTOFF_A = 1.8

# Lattice images searched for neighbour contacts (and polymer winding).
_LATTICE_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


class ConnectivityAsuError(RuntimeError):
    """No finite ~N-atom molecule — structure is polymeric under symmetry."""


def format_trial_res_gate(*, sg: str, non_h: int, finite: bool, pass_: bool) -> str:
    """One-line GATE for gps-solve / trial.res (Mark-approved)."""
    return (
        f"GATE sg={sg} non_h={non_h} "
        f"finite={'yes' if finite else 'no'} "
        f"pass={'yes' if pass_ else 'no'}"
    )


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


def _fracs_from_peaks(peaks: Sequence) -> np.ndarray:
    if not peaks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack([np.asarray(p.fract, dtype=np.float64).reshape(3) for p in peaks])


def _build_crystal_bonds(
    fracs: np.ndarray,
    ops: Sequence,
    M: np.ndarray,
    cutoff: float,
) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    """
    Covalent edges (i, j, L) where i bonds to an image of j.

    Image = op(frac_j) + t with t ∈ {-1,0,1}³. L is the integer cell of that
    image (floor of the image fractional coordinate). Identity self-contacts
    (i==j, L==(0,0,0), near-zero distance) are skipped.
    """
    n = len(fracs)
    bonds: List[Tuple[int, int, Tuple[int, int, int]]] = []
    seen = set()
    cutoff2 = float(cutoff) ** 2
    for i in range(n):
        fi = fracs[i]
        for j in range(n):
            fj = fracs[j]
            for op in ops:
                fj0 = np.asarray(op(fj), dtype=np.float64).reshape(3)
                for t in _LATTICE_OFFSETS:
                    img = fj0 + np.asarray(t, dtype=np.float64)
                    d = img - fi
                    cart = M @ d
                    d2 = float(np.dot(cart, cart))
                    if d2 > cutoff2 or d2 < 1e-12:
                        continue
                    L = (
                        int(np.floor(img[0] + 1e-9)),
                        int(np.floor(img[1] + 1e-9)),
                        int(np.floor(img[2] + 1e-9)),
                    )
                    if i == j and L == (0, 0, 0):
                        continue
                    key = (i, j, L)
                    if key in seen:
                        continue
                    seen.add(key)
                    bonds.append((i, j, L))
    return bonds


def _adjacency(
    bonds: Sequence[Tuple[int, int, Tuple[int, int, int]]],
) -> Dict[int, List[Tuple[int, Tuple[int, int, int]]]]:
    adj: Dict[int, List[Tuple[int, Tuple[int, int, int]]]] = defaultdict(list)
    for i, j, L in bonds:
        adj[i].append((j, L))
    return adj


def _walk_component(
    start: int,
    adj: Dict[int, List[Tuple[int, Tuple[int, int, int]]]],
) -> Tuple[bool, List[int]]:
    """
    BFS from start at lattice (0,0,0).

    Returns (is_polymer, member_indices). Polymer iff the same ASU site is
    reached at two distinct lattice vectors (infinite winding under symmetry).
    """
    visited_L: Dict[int, Tuple[int, int, int]] = {start: (0, 0, 0)}
    members = [start]
    q: deque = deque([(start, (0, 0, 0))])
    polymer = False
    while q:
        u, Lu = q.popleft()
        for v, Ledge in adj.get(u, ()):
            Lv = (Lu[0] + Ledge[0], Lu[1] + Ledge[1], Lu[2] + Ledge[2])
            if v in visited_L:
                if visited_L[v] != Lv:
                    polymer = True
                continue
            visited_L[v] = Lv
            members.append(v)
            q.append((v, Lv))
    return polymer, members


def _partition_components(
    n: int,
    adj: Dict[int, List[Tuple[int, Tuple[int, int, int]]]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Return (finite_components, infinite_components)."""
    seen = set()
    finite: List[List[int]] = []
    infinite: List[List[int]] = []
    for i in range(n):
        if i in seen:
            continue
        polymer, members = _walk_component(i, adj)
        for m in members:
            seen.add(m)
        if polymer:
            infinite.append(members)
        else:
            finite.append(members)
    return finite, infinite


def pack_discrete_asu(
    peaks: Sequence,
    cell: np.ndarray,
    space_group: Optional[str] = None,
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    covalent_cutoff: float = COVALENT_C_C_CUTOFF_A,
    n_non_h_budget: int = DEFAULT_N_NON_H_BUDGET,
) -> Tuple[list, dict]:
    """
    Keep largest finite covalent component(s) of budgeted peaks (~n_non_h).

    Peaks are treated as carbon for distances. Neighbours may be generated by
    SHELX LATT/SYMM ops and ±1 cell translations. Survivors keep their ASU
    fractional coordinates (expanded images map back by the inverse of the
    placing op — equivalent to the unique-ASU representative already stored).

    Raises ConnectivityAsuError if bonding under symmetry yields an infinite
    polymer and no finite set of about ``n_non_h_budget`` sites remains.
    """
    peaks = list(peaks)
    n_in = len(peaks)
    target = max(0, int(n_non_h_budget))
    meta = {
        "n_in": n_in,
        "n_out": n_in,
        "n_components": 0,
        "n_infinite": 0,
        "covalent_cutoff_A": float(covalent_cutoff),
        "packed": False,
    }
    if n_in == 0 or target == 0:
        meta["n_out"] = 0
        return [], meta

    fracs = np.mod(_fracs_from_peaks(peaks), 1.0)
    ops = space_group_ops(space_group, lattice=lattice, symm=symm)
    M = _orth_matrix(np.asarray(cell, dtype=np.float64))
    bonds = _build_crystal_bonds(fracs, ops, M, float(covalent_cutoff))
    adj = _adjacency(bonds)
    finite, infinite = _partition_components(n_in, adj)
    meta["n_infinite"] = len(infinite)

    def _cc_key(cc: List[int]):
        max_sig = max(
            float(getattr(peaks[i], "height_sigma", 0.0) or 0.0) for i in cc
        )
        return (-len(cc), -max_sig)

    finite_sorted = sorted(finite, key=_cc_key)

    kept_idx: List[int] = []
    used_components = 0
    for cc in finite_sorted:
        if len(kept_idx) >= target:
            break
        kept_idx.extend(cc)
        used_components += 1
    meta["n_components"] = used_components

    # Fail closed: polymer under symmetry and no finite ~target-atom molecule.
    if infinite and len(kept_idx) < target:
        raise ConnectivityAsuError(
            "connectivity_asu: infinite polymer under symmetry; "
            f"no finite ~{target}-atom molecule "
            f"(finite_sites={len(kept_idx)}, infinite_components={len(infinite)}, "
            f"cutoff={covalent_cutoff}A)"
        )

    kept_idx = sorted(
        set(kept_idx),
        key=lambda i: -float(getattr(peaks[i], "height_sigma", 0.0) or 0.0),
    )
    if len(kept_idx) > target:
        kept_idx = kept_idx[:target]

    # ASU write-back: original unique-ASU fracs (inverse of placing op).
    out = []
    for i in kept_idx:
        p = peaks[i]
        frac = np.mod(np.asarray(p.fract, dtype=np.float64).reshape(3), 1.0)
        out.append(replace(p, fract=frac.copy()))

    meta["n_out"] = len(out)
    meta["packed"] = True
    return out, meta
