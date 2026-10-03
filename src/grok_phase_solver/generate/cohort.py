"""Frozen COD subset already in the repo. No downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from grok_phase_solver.generate.geometry import element_symbol, is_hydrogen
from grok_phase_solver.io.cif import CrystalStructure, load_cif

PathLike = Union[str, Path]

MIN_NON_H = 4
MAX_NON_H = 20


def sites_from_structure(structure: CrystalStructure) -> tuple[np.ndarray, list[str]]:
    """Non-hydrogen asymmetric-unit coordinates. Hydrogens and deuteriums are dropped."""
    fracs = []
    elements = []
    for atom in structure.atoms:
        if is_hydrogen(atom.element):
            continue
        fracs.append(np.asarray(atom.fract, dtype=np.float64))
        elements.append(element_symbol(atom.element))
    if not fracs:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.vstack(fracs), elements


def record_from_structure(structure: CrystalStructure, name: Optional[str] = None) -> dict:
    fracs, elements = sites_from_structure(structure)
    return {
        "name": name or structure.name,
        "fracs": fracs,
        "elements": elements,
        "cell": np.asarray(structure.cell, dtype=np.float64).reshape(6),
        "space_group": structure.space_group_hm,
        "z": int(structure.z),
        "holdout": False,
    }


def record_from_cif(path: PathLike) -> dict:
    structure = load_cif(path)
    return record_from_structure(structure, name=Path(path).stem)


def mark_holdout(records: Sequence[dict]) -> None:
    """Largest non-H asymmetric unit is the holdout when at least two records exist.

    Tie-break is the lexicographic name. A single record is not a holdout, so it
    cannot be reported as reconstruction.
    """
    if not records:
        return
    ordered = sorted(records, key=lambda r: (len(r["elements"]), str(r["name"])))
    hold_name = ordered[-1]["name"] if len(records) >= 2 else None
    for record in records:
        record["holdout"] = record["name"] == hold_name


def cohort_from_dir(
    cod_dir: PathLike,
    min_non_h: int = MIN_NON_H,
    max_non_h: int = MAX_NON_H,
) -> tuple[list[dict], list[str]]:
    """Load ``*.cif`` under ``cod_dir``. Return (included records, excluded notes)."""
    root = Path(cod_dir)
    included: list[dict] = []
    excluded: list[str] = []
    for path in sorted(root.glob("*.cif")):
        try:
            record = record_from_cif(path)
        except Exception as exc:
            excluded.append(f"{path.stem} (unreadable: {exc})")
            continue
        n = len(record["elements"])
        if n < min_non_h or n > max_non_h:
            excluded.append(f"{path.stem} (non-H count {n} outside {min_non_h}–{max_non_h})")
            continue
        included.append(record)
    mark_holdout(included)
    return included, excluded
