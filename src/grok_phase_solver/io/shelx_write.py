"""
Write SHELX-compatible .hkl / .ins for external SHELXD/SHELXL.

Does not redistribute SHELX binaries. Formats follow common SHELX free-format
conventions used by SHELXD/SHELXL (HKLF 4 = h k l I σ(I)).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from grok_phase_solver.io.hkl import ReflectionTable

PathLike = Union[str, Path]


def write_hkl_shelx(
    path: PathLike,
    hkl: np.ndarray,
    amplitudes: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
    sig_i: Optional[np.ndarray] = None,
    batch: int = 1,
    *,
    fixed_format: bool = True,
    scale_max_i: float = 5000.0,
) -> Path:
    """
    Write SHELX HKLF-4 reflection file, terminated by 0 0 0.

    Parameters
    ----------
    fixed_format
        If True (default), use classic ``(3i4,2f8.2)`` layout that SHELXS/SHELXL
        read reliably. Free-format with wide fields can be mis-parsed.
    scale_max_i
        If >0 and intensities come from amplitudes, scale so max(I) = scale_max_i
        (avoids F8.2 overflow and "no observed E" failures on tiny Fcalc).
    """
    path = Path(path)
    hkl = np.asarray(hkl, dtype=np.int32).reshape(-1, 3)
    if intensities is not None:
        I = np.asarray(intensities, dtype=np.float64).reshape(-1).copy()
    elif amplitudes is not None:
        F = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
        I = F ** 2
        if scale_max_i and scale_max_i > 0 and I.max() > 0:
            I = I * (float(scale_max_i) / float(I.max()))
    else:
        raise ValueError("Need amplitudes or intensities")
    if sig_i is None:
        sig = np.maximum(0.05 * I, 1.0)
    else:
        sig = np.asarray(sig_i, dtype=np.float64).reshape(-1)
    lines = []
    for i in range(len(hkl)):
        h, k, l = map(int, hkl[i])
        Ii = float(min(max(I[i], 0.0), 99999.99))
        si = float(min(max(sig[i], 0.01), 9999.99))
        if fixed_format:
            lines.append(f"{h:4d}{k:4d}{l:4d}{Ii:8.2f}{si:8.2f}")
        else:
            lines.append(f"{h:4d} {k:4d} {l:4d} {Ii:12.4f} {si:12.4f} {int(batch):4d}")
    lines.append(f"{0:4d}{0:4d}{0:4d}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_hkl_shelx_from_table(path: PathLike, table: ReflectionTable, batch: int = 1) -> Path:
    I = table.I_meas
    sig = table.sigI
    if I is None:
        return write_hkl_shelx(
            path, table.hkl, amplitudes=table.amplitudes, batch=batch
        )
    return write_hkl_shelx(
        path, table.hkl, intensities=I, sig_i=sig, batch=batch
    )


def write_shelxd_ins(
    path: PathLike,
    cell: np.ndarray,
    *,
    title: str = "gps shelxd job",
    wavelength: float = 0.71073,
    n_find: int = 12,
    n_try: int = 50,
    seed: int = 1,
    sfac: Optional[Sequence[str]] = None,
    unit: Optional[Sequence[float]] = None,
    lattice: int = -1,
    symm: Optional[Sequence[str]] = None,
    space_group_hm: Optional[str] = None,
    patt: bool = True,
    e_min: Optional[float] = None,
    extra_lines: Optional[Sequence[str]] = None,
) -> Path:
    """
    Minimal SHELXD-style instruction file (small-molecule dual-space FIND).

    SHELXD is not bundled; this file is for an external ``shelxd`` binary.
    Keywords follow common academic usage (FIND / PATT / SEED / HKLF 4).
    Exact keyword acceptance depends on the installed SHELXD version.
    """
    path = Path(path)
    a, b, c, al, be, ga = [float(x) for x in cell]
    sfac = list(sfac) if sfac else ["C", "H", "N", "O"]
    if unit is None:
        # rough: put most atoms on carbon for dual-space FIND
        unit = [float(n_find)] + [0.0] * (len(sfac) - 1)
    while len(unit) < len(sfac):
        unit = list(unit) + [0.0]

    lines: List[str] = [
        f"TITL {title}",
        f"CELL {wavelength:.5f} {a:.4f} {b:.4f} {c:.4f} {al:.3f} {be:.3f} {ga:.3f}",
        f"ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        f"LATT {int(lattice)}",
    ]
    if symm:
        for s in symm:
            lines.append(f"SYMM {s}")
    lines.append("SFAC " + " ".join(sfac))
    lines.append("UNIT " + " ".join(f"{u:.0f}" if float(u) == int(u) else f"{u:.1f}" for u in unit))
    if space_group_hm:
        lines.append(f"REM space_group_hint {space_group_hm}")
    # Dual-space ab initio style
    lines.append(f"FIND {int(n_find)}")
    if patt:
        lines.append("PATT")
    lines.append(f"SEED {int(seed)}")
    # SOME versions accept NTRY / TRY for number of dual-space starts
    lines.append(f"NTRY {int(n_try)}")
    if e_min is not None:
        lines.append(f"ESEL {float(e_min):.2f}")
    if extra_lines:
        lines.extend(list(extra_lines))
    lines.append("HKLF 4")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_shelx_res_atoms(path: PathLike) -> List[dict]:
    """
    Parse atom lines from a SHELX .res / .ins atom list.

    Returns list of {label, sfac, fract, occ_code, u_iso}.
    Skips instruction cards (CELL, SFAC, …).
    """
    path = Path(path)
    if not path.exists():
        return []
    atoms: List[dict] = []
    skip = {
        "TITL", "CELL", "ZERR", "LATT", "SYMM", "SFAC", "UNIT", "FVAR",
        "HKLF", "END", "FIND", "PATT", "SEED", "NTRY", "ESEL", "REM",
        "PLAN", "WGHT", "FMAP", "ACTA", "CONF", "MPLA", "HTAB", "MORE",
        "SIZE", "TEMP", "DISP", "TREF", "MOVE", "ANIS", "AFIX", "PART",
        "BIND", "FREE", "DFIX", "DANG", "SADI", "SAME", "FLAT", "CHIV",
        "DELU", "SIMU", "ISOR", "EADP", "EXYZ", "EADP", "MERG", "OMIT",
        "SHEL", "BASF", "TWIN", "EXTI", "SWAT", "HOPE", "LIST", "BLOC",
        "DAMP", "STIR", "L.S.", "CGLS", "BOND", "RTAB", "MOLE",
    }
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if "!" in line:
            line = line.split("!")[0].strip()
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key in skip or key.startswith("Q") and len(parts) < 5:
            # Q-peaks still useful if full atom format
            pass
        # Atom line: name sfac x y z [sof] [U]
        if len(parts) < 5:
            continue
        if key in skip:
            continue
        # sfac index is integer
        try:
            sfac_i = int(float(parts[1]))
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        # Filter bogus instruction-like names
        if key in skip:
            continue
        u = 0.05
        occ = 11.0
        if len(parts) >= 6:
            try:
                occ = float(parts[5])
            except ValueError:
                pass
        if len(parts) >= 7:
            try:
                u = float(parts[6])
            except ValueError:
                pass
        atoms.append(
            {
                "label": parts[0],
                "sfac": sfac_i,
                "fract": np.array([x, y, z], dtype=np.float64),
                "occ_code": occ,
                "u_iso": u,
            }
        )
    return atoms


def write_shelxs_ins(
    path: PathLike,
    cell: np.ndarray,
    *,
    title: str = "gps shelxs job",
    wavelength: float = 0.71073,
    n_atoms: int = 12,
    n_try: int = 100,
    sfac: Optional[Sequence[str]] = None,
    unit: Optional[Sequence[float]] = None,
    lattice: int = -1,
    symm: Optional[Sequence[str]] = None,
    extra_lines: Optional[Sequence[str]] = None,
) -> Path:
    """
    Minimal SHELXS instruction file (direct-methods TREF).

    Typical small-molecule ab initio: LATT −1 (P1), SFAC C, UNIT n, TREF ntry.
    """
    path = Path(path)
    a, b, c, al, be, ga = [float(x) for x in cell]
    sfac = list(sfac) if sfac else ["C"]
    if unit is None:
        unit = [float(n_atoms)] + [0.0] * (len(sfac) - 1)
    while len(unit) < len(sfac):
        unit = list(unit) + [0.0]
    lines: List[str] = [
        f"TITL {title}",
        f"CELL {wavelength:.5f} {a:.4f} {b:.4f} {c:.4f} {al:.3f} {be:.3f} {ga:.3f}",
        "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        f"LATT {int(lattice)}",
    ]
    if symm:
        for s in symm:
            lines.append(f"SYMM {s}")
    lines.append("SFAC " + " ".join(sfac))
    lines.append(
        "UNIT " + " ".join(f"{u:.0f}" if float(u) == int(u) else f"{u:.1f}" for u in unit)
    )
    lines.append(f"TREF {int(max(n_try, 1))}")
    if extra_lines:
        lines.extend(list(extra_lines))
    lines.append("HKLF 4")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def atoms_to_fracs_elements(
    atoms: List[dict],
    sfac: Optional[Sequence[str]] = None,
    default_element: str = "C",
) -> tuple:
    """Map parsed atoms → (fracs (N,3), elements list)."""
    sfac = list(sfac) if sfac else ["C", "H", "N", "O"]
    fracs = []
    elements = []
    for a in atoms:
        # Skip pure Q-peaks if many atoms? keep them as C for phasing
        lab = str(a["label"]).upper()
        el = default_element
        si = int(a["sfac"]) - 1
        if 0 <= si < len(sfac):
            el = sfac[si]
        if lab.startswith("Q"):
            el = "C"
        fracs.append(a["fract"])
        elements.append(el)
    if not fracs:
        return np.zeros((0, 3)), []
    return np.asarray(fracs, dtype=np.float64), elements
