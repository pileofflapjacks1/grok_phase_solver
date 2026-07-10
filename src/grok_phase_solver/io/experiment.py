"""
Load experimental diffraction setups for end-user phasing.

Supported combinations:
  - SHELX .hkl + .ins (recommended small-molecule workflow)
  - SHELX .hkl + --cell + --sg
  - COD-style structure-factor CIF (.hkl with _refln_*)
  - MTZ (via gemmi)
  - Simple text: h k l F [sigF]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .hkl import ReflectionTable, load_hkl_cif, load_hkl_shelx
from .ins import ShelxIns, load_ins, parse_cell_string
from .mtz import load_mtz

PathLike = Union[str, Path]


def _looks_like_cif_hkl(path: Path) -> bool:
    try:
        head = path.read_text(errors="replace")[:4000]
    except Exception:
        return False
    return "_refln_index_h" in head or "data_" in head[:200] and "loop_" in head


def load_reflections(
    hkl_path: PathLike,
    cell: Optional[np.ndarray] = None,
    space_group: Optional[str] = None,
    wavelength: Optional[float] = None,
) -> ReflectionTable:
    """
    Auto-detect reflection file format and load amplitudes.
    """
    path = Path(hkl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reflection file not found: {path}")

    suffix = path.suffix.lower()
    table: ReflectionTable

    if suffix == ".mtz":
        table = load_mtz(path)
    elif _looks_like_cif_hkl(path):
        table = load_hkl_cif(path)
    else:
        # SHELX free format or simple text
        table = load_hkl_shelx(path, cell=cell)

    if cell is not None:
        table.cell = np.asarray(cell, dtype=np.float64)
    if space_group is not None:
        table.space_group_hm = space_group
    if wavelength is not None:
        table.wavelength = wavelength

    if table.cell is None:
        raise ValueError(
            "Unit cell unknown. Provide --cell a,b,c,al,be,ga or a SHELX .ins with CELL, "
            "or use a CIF/MTZ that includes cell parameters."
        )
    return table


def load_experiment(
    hkl: PathLike,
    ins: Optional[PathLike] = None,
    cell: Optional[str] = None,
    space_group: Optional[str] = None,
    wavelength: Optional[float] = None,
) -> Tuple[ReflectionTable, Optional[ShelxIns]]:
    """
    Full experimental setup for ``gps-solve``.

    Priority for cell/SG:
      1. Explicit CLI cell/sg
      2. SHELX .ins
      3. Cell embedded in CIF HKL / MTZ
    """
    shelx_ins: Optional[ShelxIns] = None
    cell_arr = parse_cell_string(cell) if cell else None
    sg = space_group
    wl = wavelength

    if ins is not None:
        shelx_ins = load_ins(ins)
        if cell_arr is None and shelx_ins.cell is not None:
            cell_arr = shelx_ins.cell
        if sg is None and shelx_ins.space_group_hm:
            sg = shelx_ins.space_group_hm
        if wl is None:
            wl = shelx_ins.wavelength

    table = load_reflections(hkl, cell=cell_arr, space_group=sg, wavelength=wl)
    return table, shelx_ins


def summarize_experiment(table: ReflectionTable, ins: Optional[ShelxIns] = None) -> str:
    """Human-readable summary of loaded data."""
    lines = []
    lines.append(f"Reflections: {len(table)}")
    amp = table.amplitudes
    lines.append(f"|F| range: {amp.min():.3g} – {amp.max():.3g}  (mean {amp.mean():.3g})")
    if table.cell is not None:
        a, b, c, al, be, ga = table.cell
        lines.append(f"Cell: {a:.4f} {b:.4f} {c:.4f}  {al:.2f} {be:.2f} {ga:.2f}")
        try:
            d = table.resolution_d()
            lines.append(f"Resolution: {d.min():.2f} – {d.max():.2f} Å  (d_min = high-angle limit)")
        except Exception:
            pass
    if table.space_group_hm:
        lines.append(f"Space group: {table.space_group_hm}")
    if table.wavelength:
        lines.append(f"Wavelength: {table.wavelength} Å")
    if ins and ins.title:
        lines.append(f"Title: {ins.title}")
    if ins and ins.sfac:
        lines.append(f"Composition (SFAC): {' '.join(ins.sfac)}")
    src = table.meta.get("source") or table.meta.get("format")
    if src:
        lines.append(f"Source: {src}")
    return "\n".join(lines)
