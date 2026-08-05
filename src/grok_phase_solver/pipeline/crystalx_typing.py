"""
CrystalX-inspired peak → atom typing + hydrogen placement (v0.13).

After density peak picking, assign element labels and optional hydrogens using
**geometry + peak-height heuristics** (pure NumPy). Optional Torch equivariant
stub is documented but not required.

Inspired by CrystalX (arXiv 2410.13713 / JACS 2026 conceptual): equivariant
Transformer peak typing — we ship a transparent classical fallback that
improves trial.res quality without external weights.

Physics fallback: untyped Q/C peaks via legacy ``write_shelxl_res``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.pipeline.peaks import DensityPeak


# Covalent radii (Å) for geometry checks
_RAD = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


@dataclass
class TypedAtom:
    """Atom assignment for a density peak (or placed H)."""

    label: str
    element: str
    fract: np.ndarray
    height_sigma: float = 0.0
    confidence: float = 0.5
    source: str = "peak"  # peak | hydrogen
    rank: int = 0
    notes: List[str] = field(default_factory=list)


def _orth_matrix(cell: np.ndarray) -> np.ndarray:
    from grok_phase_solver.io.cif import CrystalStructure

    return CrystalStructure("t", cell, "P1").orth_matrix


def _min_image_frac(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    return (f1 - f2 + 0.5) % 1.0 - 0.5


def _dist_cart(f1: np.ndarray, f2: np.ndarray, M: np.ndarray) -> float:
    return float(np.linalg.norm(M @ _min_image_frac(f1, f2)))


def type_peaks_crystalx(
    peaks: Sequence[DensityPeak],
    cell: np.ndarray,
    *,
    composition: Optional[Sequence[str]] = None,
    place_hydrogens: bool = True,
    max_h_per_heavy: int = 2,
    ha_sigma_threshold: float = 6.0,
    method: str = "heuristic",
) -> Tuple[List[TypedAtom], Dict]:
    """
    Assign element types to density peaks (CrystalX-inspired heuristic).

    Rules (heuristic):
    - Highest peaks with height_sigma ≥ ha_sigma_threshold → Cl/Br/S preference
    - Remaining ranked by height: C (majority), O (high neighbor count / mid height),
      N (between C and O height bands)
    - Optional terminal H placement along free valence directions for C/N/O

    ``composition`` if given biases counts (e.g. ``["C"]*10 + ["O"]*3 + ["N"]*2``).

    Returns (typed_atoms, meta).
    """
    peaks = list(peaks)
    meta: Dict = {
        "algorithm": "crystalx_heuristic_v1",
        "method": method,
        "n_peaks_in": len(peaks),
        "research_note": (
            "Geometry/height typing — not trained CrystalX weights. "
            "Improves trial.res labels; refine in SHELXL/Olex2."
        ),
    }
    if not peaks:
        return [], meta

    M = _orth_matrix(np.asarray(cell, dtype=np.float64))
    # Sort by height descending
    order = sorted(range(len(peaks)), key=lambda i: -peaks[i].height_sigma)
    fracs = [np.asarray(peaks[i].fract, dtype=np.float64) for i in order]
    sigs = [float(peaks[i].height_sigma) for i in order]

    # Composition budget
    budget: Dict[str, int] = {}
    if composition:
        for e in composition:
            eu = e if e in _RAD else e.capitalize() if len(e) > 1 else e.upper()
            if eu == "CL":
                eu = "Cl"
            if eu == "BR":
                eu = "Br"
            budget[eu] = budget.get(eu, 0) + 1
    else:
        n = len(peaks)
        budget = {
            "C": max(1, int(0.55 * n)),
            "O": max(0, int(0.18 * n)),
            "N": max(0, int(0.12 * n)),
            "Cl": max(0, int(0.05 * n)),
            "S": max(0, int(0.05 * n)),
            "Br": max(0, int(0.03 * n)),
        }

    typed: List[TypedAtom] = []
    used = {k: 0 for k in budget}

    def _take(el: str) -> bool:
        if budget.get(el, 0) <= used.get(el, 0):
            return False
        used[el] = used.get(el, 0) + 1
        return True

    for i, (f, sig) in enumerate(zip(fracs, sigs)):
        el = "C"
        conf = 0.45
        notes: List[str] = []
        # HA: very strong peaks
        if sig >= ha_sigma_threshold:
            for cand in ("Br", "I", "Cl", "S", "P"):
                if _take(cand):
                    el = cand
                    conf = 0.75
                    notes.append(f"HA by height_sigma={sig:.1f}")
                    break
            else:
                if _take("Cl"):
                    el = "Cl"
                    conf = 0.6
                    notes.append("HA fallback Cl")
        else:
            # Neighbor count at bonding distance (suggests O/N vs C)
            n_bond = 0
            for j, f2 in enumerate(fracs):
                if j == i:
                    continue
                d = _dist_cart(f, f2, M)
                if 1.1 < d < 1.7:
                    n_bond += 1
            # Mid-height + high connectivity → O; mid → N; else C
            rank_frac = i / max(len(fracs) - 1, 1)
            if rank_frac < 0.25 and n_bond >= 1 and _take("O"):
                el = "O"
                conf = 0.55
                notes.append("mid-high peak + bonds → O")
            elif 0.15 < rank_frac < 0.55 and n_bond >= 2 and _take("N"):
                el = "N"
                conf = 0.50
                notes.append("mid peak + bonds → N")
            elif _take("C"):
                el = "C"
                conf = 0.50
            elif _take("O"):
                el = "O"
                conf = 0.40
            elif _take("N"):
                el = "N"
                conf = 0.40
            else:
                el = "C"
                conf = 0.35
                notes.append("over-budget → C")

        label = f"{el}{i+1}"
        typed.append(
            TypedAtom(
                label=label,
                element=el,
                fract=f.copy(),
                height_sigma=sig,
                confidence=conf,
                source="peak",
                rank=i,
                notes=notes,
            )
        )

    # Optional hydrogens (terminal on C/N/O with free valence)
    n_h = 0
    if place_hydrogens:
        heavies = [a for a in typed if a.element != "H"]
        for a in heavies:
            if a.element not in ("C", "N", "O"):
                continue
            # count existing neighbors
            n_nb = 0
            dirs = []
            for b in heavies:
                if b is a:
                    continue
                d = _dist_cart(a.fract, b.fract, M)
                if 1.1 < d < 1.7:
                    n_nb += 1
                    vec = M @ _min_image_frac(b.fract, a.fract)
                    dirs.append(vec / (np.linalg.norm(vec) + 1e-16))
            max_val = 4 if a.element == "C" else (3 if a.element == "N" else 2)
            n_place = min(max_h_per_heavy, max(0, max_val - n_nb))
            if n_place <= 0:
                continue
            # free directions: random orthogonal to mean neighbor dir
            mean_d = np.mean(dirs, axis=0) if dirs else np.array([1.0, 0.0, 0.0])
            mean_d = mean_d / (np.linalg.norm(mean_d) + 1e-16)
            rng = np.random.default_rng(abs(hash((a.label, a.rank))) % (2**31))
            for k in range(n_place):
                v = rng.normal(size=3)
                v = v - np.dot(v, mean_d) * mean_d
                if np.linalg.norm(v) < 1e-8:
                    v = rng.normal(size=3)
                v = v / (np.linalg.norm(v) + 1e-16)
                bl = _RAD[a.element] + _RAD["H"]
                # fractional offset
                try:
                    dfrac = np.linalg.solve(M, bl * v)
                except np.linalg.LinAlgError:
                    continue
                fh = (a.fract + dfrac) % 1.0
                n_h += 1
                typed.append(
                    TypedAtom(
                        label=f"H{n_h}",
                        element="H",
                        fract=fh,
                        height_sigma=0.0,
                        confidence=0.25,
                        source="hydrogen",
                        rank=1000 + n_h,
                        notes=["geometry H"],
                    )
                )

    meta["n_typed"] = len([a for a in typed if a.source == "peak"])
    meta["n_hydrogens"] = n_h
    meta["element_counts"] = {}
    for a in typed:
        meta["element_counts"][a.element] = meta["element_counts"].get(a.element, 0) + 1
    return typed, meta


def typed_atoms_to_shelxl_res(
    typed: Sequence[TypedAtom],
    cell: np.ndarray,
    *,
    method: str = "crystalx_typed",
    space_group: str = "P1",
    wavelength: float = 0.71073,
    free_fom: Optional[float] = None,
) -> str:
    """Write SHELXL-style .res with real SFAC elements from typing."""
    a, b, c, al, be, ga = [float(x) for x in cell]
    # Unique elements for SFAC (H last convention-ish: C H N O then others)
    order_pref = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    els_present = []
    for e in order_pref:
        if any(t.element == e for t in typed):
            els_present.append(e)
    for t in typed:
        if t.element not in els_present:
            els_present.append(t.element)
    if not els_present:
        els_present = ["C"]
    sfac_idx = {e: i + 1 for i, e in enumerate(els_present)}
    unit = " ".join("1" for _ in els_present)
    lines = [
        f"TITL gps-solve trial ({method})",
        f"CELL {wavelength:.5f} {a:.4f} {b:.4f} {c:.4f} {al:.2f} {be:.2f} {ga:.2f}",
        "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        "LATT -1",
        f"SFAC {' '.join(els_present)}",
        f"UNIT {unit}",
        "FVAR 1.0",
        f"REM free_fom_composite={free_fom if free_fom is not None else 'n/a'}",
        f"REM method={method} n_atoms={len(typed)} crystalx_typing=1",
        f"REM space_group_hint={space_group}",
    ]
    for t in typed:
        if t.element == "H":
            continue  # write heavies first
        idx = sfac_idx.get(t.element, 1)
        u = max(0.015, 0.06 / max(t.height_sigma / 3.0, 0.5)) if t.height_sigma > 0 else 0.05
        lines.append(
            f"{t.label:6s} {idx} {t.fract[0]:10.6f} {t.fract[1]:10.6f} {t.fract[2]:10.6f} "
            f"11.00000 {u:.5f}"
        )
    for t in typed:
        if t.element != "H":
            continue
        idx = sfac_idx.get("H", 1)
        lines.append(
            f"{t.label:6s} {idx} {t.fract[0]:10.6f} {t.fract[1]:10.6f} {t.fract[2]:10.6f} "
            f"11.00000 0.08000"
        )
    lines.append("HKLF 4")
    lines.append("END")
    return "\n".join(lines) + "\n"


def crystalx_torch_available() -> bool:
    """True only if a future equivariant checkpoint ships (currently False)."""
    return False
