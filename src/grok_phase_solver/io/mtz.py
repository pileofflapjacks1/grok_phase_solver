"""
MTZ reflection file support via gemmi.

MTZ is the CCP4 binary format for amplitudes/phases. This module provides a
thin, correct wrapper: we do not reimplement the binary format.

References: CCP4 MTZ documentation; gemmi Mtz class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .hkl import ReflectionTable

PathLike = Union[str, Path]


def load_mtz(
    path: PathLike,
    f_label: str = "F",
    sigf_label: str = "SIGF",
    free_label: str = "FreeR_flag",
) -> ReflectionTable:
    """
    Load amplitudes from an MTZ file.

    Tries common column names if defaults are missing (FP, FOBS, IMEAN, …).
    """
    path = Path(path)
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError("gemmi is required for MTZ I/O") from exc

    mtz = gemmi.read_mtz_file(str(path))
    # Build array of hkl
    hkl = np.array([[r.h, r.k, r.l] for r in mtz], dtype=np.int32)
    cols = {c.label: c for c in mtz.columns}

    def find_col(*names: str) -> Optional[np.ndarray]:
        for n in names:
            if n in cols:
                return np.array(cols[n].array, dtype=np.float64)
        return None

    F = find_col(f_label, "F", "FP", "FOBS", "F-obs", "F_meas")
    I = find_col("IMEAN", "I", "I-obs", "I_meas")
    sigF = find_col(sigf_label, "SIGF", "SIGFP", "SIGFOBS")
    free = find_col(free_label, "FREE", "R-free-flags")

    if F is None and I is not None:
        F = np.sqrt(np.maximum(I, 0.0))
    if F is None:
        raise ValueError(
            f"No amplitude/intensity column found in {path}. "
            f"Columns: {list(cols)}"
        )

    # Cell from MTZ
    cell = mtz.cell
    cell_arr = np.array(
        [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        dtype=np.float64,
    )
    sg = mtz.spacegroup.hm if mtz.spacegroup else None

    mask = np.isfinite(F) & (F >= 0)
    mask &= ~((hkl[:, 0] == 0) & (hkl[:, 1] == 0) & (hkl[:, 2] == 0))

    return ReflectionTable(
        hkl=hkl[mask],
        F_meas=F[mask],
        sigF=sigF[mask] if sigF is not None else None,
        I_meas=I[mask] if I is not None else None,
        free_flag=free[mask].astype(int) if free is not None else None,
        cell=cell_arr,
        space_group_hm=sg,
        meta={"source": str(path.resolve()), "format": "mtz"},
    )


def write_mtz_stub(path: PathLike, table: ReflectionTable) -> None:
    """
    Write a minimal MTZ with H K L F (and SIGF if present) using gemmi.
    """
    path = Path(path)
    import gemmi

    mtz = gemmi.Mtz(with_base=True)
    if table.cell is not None:
        a, b, c, al, be, ga = table.cell
        mtz.set_cell_for_all(gemmi.UnitCell(a, b, c, al, be, ga))
    if table.space_group_hm:
        mtz.spacegroup = gemmi.SpaceGroup(table.space_group_hm)
    mtz.add_dataset("HKL_base")
    mtz.add_column("F", "F")
    if table.sigF is not None:
        mtz.add_column("SIGF", "Q")

    amp = table.amplitudes
    rows = []
    for i in range(len(table)):
        h, k, l = map(int, table.hkl[i])
        row = [h, k, l, float(amp[i])]
        if table.sigF is not None:
            row.append(float(table.sigF[i]))
        rows.append(row)
    mtz.set_data(np.array(rows, dtype=np.float32))
    mtz.write_to_file(str(path))
