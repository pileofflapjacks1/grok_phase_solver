"""
Minimal SHELX .ins / .res header parser for experimental setup.

Extracts CELL, LATT/SYMM → space-group hint, wavelength, Z, SFAC list.
Enough to drive phasing when the user has a standard SHELX .ins + .hkl pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class ShelxIns:
    """Parsed subset of a SHELX instruction file."""

    title: str = ""
    cell: Optional[np.ndarray] = None  # a b c α β γ
    wavelength: float = 0.71073
    z: float = 1.0
    lattice: int = 1  # LATT code
    symm: List[str] = field(default_factory=list)
    sfac: List[str] = field(default_factory=list)
    unit: List[float] = field(default_factory=list)
    space_group_hm: Optional[str] = None
    source_path: Optional[str] = None
    raw_lines: List[str] = field(default_factory=list)


def _parse_floats(tokens: List[str]) -> List[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            break
    return out


def load_ins(path: PathLike) -> ShelxIns:
    """Parse CELL / LATT / SYMM / SFAC / UNIT / TITL from a SHELX .ins or .res."""
    path = Path(path)
    text = path.read_text(errors="replace")
    ins = ShelxIns(source_path=str(path.resolve()), raw_lines=text.splitlines())
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("!"):
            continue
        # strip trailing comments after !
        if "!" in raw:
            raw = raw.split("!")[0].strip()
        parts = raw.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "TITL":
            ins.title = " ".join(parts[1:])
        elif key == "CELL" and len(parts) >= 8:
            # CELL wavelength a b c alpha beta gamma
            vals = _parse_floats(parts[1:])
            if len(vals) >= 7:
                ins.wavelength = vals[0]
                ins.cell = np.array(vals[1:7], dtype=np.float64)
        elif key == "ZERR" and len(parts) >= 2:
            try:
                ins.z = float(parts[1])
            except ValueError:
                pass
        elif key == "LATT" and len(parts) >= 2:
            try:
                ins.lattice = int(float(parts[1]))
            except ValueError:
                pass
        elif key == "SYMM":
            ins.symm.append(" ".join(parts[1:]))
        elif key == "SFAC":
            # SFAC C H N O  or  SFAC C  ...
            for t in parts[1:]:
                if re.match(r"^[A-Za-z]{1,2}$", t):
                    ins.sfac.append(t.upper() if len(t) == 1 else t[0].upper() + t[1:].lower())
        elif key == "UNIT":
            ins.unit = _parse_floats(parts[1:])
        elif key in ("HKLF", "END", "TREF", "PATT", "FIND"):
            # stop at data/commands not needed for setup (keep scanning for CELL)
            pass

    ins.space_group_hm = _guess_space_group(ins)
    return ins


def _guess_space_group(ins: ShelxIns) -> Optional[str]:
    """
    Best-effort space-group name from LATT + SYMM count / common patterns.

    SHELX LATT: positive = centro, negative = non-centro; |LATT| encodes lattice type.
    Full SG inference is nontrivial; we map common small-molecule cases and
    fall back to P1 / P-1 based on centrosymmetry flag.
    """
    centro = ins.lattice > 0
    n_symm = len(ins.symm)
    # Common cases from instruction count (asymmetric unit ops excluding identity)
    # P21/c: LATT 1, one SYMM -X, Y+1/2, -Z+1/2  →  n_symm=1, centro
    if n_symm == 0:
        return "P-1" if centro else "P 1"
    # Try gemmi Hall/operators if available
    try:
        import gemmi

        # Build a minimal operations list from SYMM strings is hard;
        # use title or common heuristics
        title = (ins.title or "").upper().replace(" ", "")
        for sg_name in (
            "P21/C",
            "P2(1)/C",
            "P21/N",
            "P212121",
            "P21",
            "C2/C",
            "P-1",
            "P1",
            "PBCA",
            "PNA21",
        ):
            if sg_name.replace("(", "").replace(")", "") in title.replace("(", "").replace(")", ""):
                # normalize
                mapping = {
                    "P21/C": "P 1 21/c 1",
                    "P2(1)/C": "P 1 21/c 1",
                    "P21/N": "P 1 21/n 1",
                    "P212121": "P 21 21 21",
                    "P21": "P 1 21 1",
                    "C2/C": "C 1 2/c 1",
                    "P-1": "P -1",
                    "P1": "P 1",
                    "PBCA": "P b c a",
                    "PNA21": "P n a 21",
                }
                name = mapping.get(sg_name, sg_name)
                gemmi.SpaceGroup(name)  # validate
                return name
    except Exception:
        pass

    if n_symm == 1 and centro:
        return "P 1 21/c 1"  # most common organic monoclinic
    if n_symm == 3 and centro:
        return "P b c a"
    if n_symm == 3 and not centro:
        return "P 21 21 21"
    return "P -1" if centro else "P 1"


def parse_cell_string(s: str) -> np.ndarray:
    """Parse 'a,b,c,alpha,beta,gamma' or space-separated."""
    s = s.replace(",", " ")
    vals = [float(x) for x in s.split()]
    if len(vals) != 6:
        raise ValueError("cell must have 6 numbers: a b c alpha beta gamma")
    return np.array(vals, dtype=np.float64)
