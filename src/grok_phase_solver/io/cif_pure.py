"""
Minimal pure-Python CIF reader for small-molecule COD-style files.

Does **not** replace gemmi for production. Used as a fallback when gemmi is
unavailable and for teaching the data model. Handles:
- key/value pairs
- loops with `_atom_site_*` and simple cell tags
- numeric values with uncertainties: ``1.234(5)``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .cif import AtomSite, CrystalStructure, parse_cif_float

PathLike = Union[str, Path]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def load_cif_pure(path: PathLike) -> CrystalStructure:
    """Parse a restricted subset of CIF into CrystalStructure."""
    path = Path(path)
    text = path.read_text(errors="replace")
    # Remove semicolon text blocks (multi-line strings) → placeholder
    text = re.sub(r";\n.*?\n;", " '?' ", text, flags=re.S)

    tokens: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # crude tokenize preserving quoted strings
        parts = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", line)
        tokens.extend(parts)

    kv: Dict[str, str] = {}
    loops: List[Tuple[List[str], List[List[str]]]] = []
    i = 0
    data_name = path.stem
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("data_"):
            data_name = t[5:] or data_name
            i += 1
            continue
        if t == "loop_":
            i += 1
            tags: List[str] = []
            while i < len(tokens) and tokens[i].startswith("_"):
                tags.append(tokens[i])
                i += 1
            rows: List[List[str]] = []
            while i < len(tokens) and not tokens[i].startswith("_") and tokens[i] != "loop_" and not tokens[i].startswith("data_"):
                row = []
                for _ in tags:
                    if i >= len(tokens):
                        break
                    row.append(_strip_quotes(tokens[i]))
                    i += 1
                if len(row) == len(tags):
                    rows.append(row)
            loops.append((tags, rows))
            continue
        if t.startswith("_"):
            key = t
            i += 1
            if i < len(tokens):
                kv[key] = _strip_quotes(tokens[i])
                i += 1
            continue
        i += 1

    def g(*keys: str, default: Optional[str] = None) -> Optional[str]:
        for k in keys:
            if k in kv:
                return kv[k]
        return default

    a = parse_cif_float(g("_cell_length_a"), 10.0) or 10.0
    b = parse_cif_float(g("_cell_length_b"), 10.0) or 10.0
    c = parse_cif_float(g("_cell_length_c"), 10.0) or 10.0
    al = parse_cif_float(g("_cell_angle_alpha"), 90.0) or 90.0
    be = parse_cif_float(g("_cell_angle_beta"), 90.0) or 90.0
    ga = parse_cif_float(g("_cell_angle_gamma"), 90.0) or 90.0
    cell = np.array([a, b, c, al, be, ga], dtype=np.float64)

    sg = g("_symmetry_space_group_name_H-M", "_space_group_name_H-M_alt") or "P 1"
    z = int(parse_cif_float(g("_cell_formula_units_Z"), 1.0) or 1)
    wl = parse_cif_float(g("_diffrn_radiation_wavelength"), 0.71073) or 0.71073

    atoms: List[AtomSite] = []
    for tags, rows in loops:
        if not any("atom_site_fract_x" in t for t in tags):
            continue
        # map tags
        idx = {t: j for j, t in enumerate(tags)}

        def col(name: str, row: List[str], default: str = "0") -> str:
            for t, j in idx.items():
                if t.endswith(name) or t == name:
                    return row[j]
            return default

        for row in rows:
            label = col("_atom_site_label", row, "X")
            el = col("_atom_site_type_symbol", row, label[:2].strip("0123456789"))
            # strip trailing digits from element if needed
            el = re.match(r"[A-Za-z]+", el or "C")
            el = el.group(0) if el else "C"
            fx = parse_cif_float(col("_atom_site_fract_x", row), 0.0) or 0.0
            fy = parse_cif_float(col("_atom_site_fract_y", row), 0.0) or 0.0
            fz = parse_cif_float(col("_atom_site_fract_z", row), 0.0) or 0.0
            occ = parse_cif_float(col("_atom_site_occupancy", row), 1.0) or 1.0
            u = parse_cif_float(col("_atom_site_U_iso_or_equiv", row), 0.05) or 0.05
            atoms.append(
                AtomSite(
                    label=label,
                    element=el,
                    fract=np.array([fx, fy, fz]),
                    occupancy=occ,
                    u_iso=u,
                )
            )

    return CrystalStructure(
        name=data_name,
        cell=cell,
        space_group_hm=sg,
        atoms=atoms,
        z=z,
        wavelength=wl,
        source_path=str(path.resolve()),
    )
