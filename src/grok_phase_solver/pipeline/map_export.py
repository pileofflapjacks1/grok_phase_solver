"""
CCP4/MRC density map + PyMOL / Coot handoff.

gps-solve already writes ``density.npz`` (NumPy). This module writes a
crystallographer map (``density.map``) covering the unit cell and scripts
to open it with PyMOL or Coot. Not a replacement for Olex2/SHELXL.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from grok_phase_solver.pipeline.peaks import DensityPeak
    from grok_phase_solver.pipeline.solve import SolveResult

CellLike = Union[Sequence[float], np.ndarray]


def _cell6(cell: CellLike) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float64).ravel()
    if arr.size < 6:
        raise ValueError("cell must be a,b,c,alpha,beta,gamma")
    return arr[:6]


def write_ccp4_map(
    path: Union[str, Path],
    rho: np.ndarray,
    cell: CellLike,
    space_group: Optional[str] = None,
) -> Path:
    """
    Write a MODE-2 CCP4 map of the unit-cell grid ``rho[x,y,z]``.

    Axis order matches the in-repo FFT density (x along a, y along b, z along c).
    Prefers gemmi; falls back to a minimal header writer.
    """
    path = Path(path)
    rho = np.asarray(rho, dtype=np.float32)
    if rho.ndim != 3:
        raise ValueError(f"rho must be 3D, got shape {rho.shape}")
    cell6 = _cell6(cell)
    if _write_ccp4_gemmi(path, rho, cell6, space_group):
        return path
    _write_ccp4_raw(path, rho, cell6)
    return path


def _write_ccp4_gemmi(
    path: Path,
    rho: np.ndarray,
    cell6: np.ndarray,
    space_group: Optional[str],
) -> bool:
    try:
        import gemmi
    except Exception:
        return False
    nx, ny, nz = (int(n) for n in rho.shape)
    grid = gemmi.FloatGrid(nx, ny, nz)
    a, b, c, al, be, ga = (float(x) for x in cell6)
    grid.set_unit_cell(gemmi.UnitCell(a, b, c, al, be, ga))
    try:
        from grok_phase_solver.physics.symmetry import normalize_space_group_name

        hm = normalize_space_group_name(space_group or "P 1") or "P 1"
        grid.spacegroup = gemmi.SpaceGroup(hm)
    except Exception:
        grid.spacegroup = gemmi.SpaceGroup("P 1")
    np.array(grid, copy=False)[:] = rho
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))
    return path.is_file() and path.stat().st_size > 1024


def _write_ccp4_raw(path: Path, rho: np.ndarray, cell6: np.ndarray) -> None:
    """Minimal little-endian MODE-2 CCP4 map (MAPC,MAPR,MAPS = 1,2,3)."""
    nx, ny, nz = (int(n) for n in rho.shape)
    data = np.asfortranarray(rho, dtype="<f4")
    amin = float(np.min(rho))
    amax = float(np.max(rho))
    amean = float(np.mean(rho))
    arms = float(np.std(rho))
    a, b, c, al, be, ga = (float(x) for x in cell6)
    words = bytearray(1024)
    def i32(offset: int, val: int) -> None:
        struct.pack_into("<i", words, offset * 4, int(val))
    def f32(offset: int, val: float) -> None:
        struct.pack_into("<f", words, offset * 4, float(val))
    i32(0, nx); i32(1, ny); i32(2, nz)
    i32(3, 2)  # MODE float32
    i32(4, 0); i32(5, 0); i32(6, 0)
    i32(7, nx); i32(8, ny); i32(9, nz)
    f32(10, a); f32(11, b); f32(12, c)
    f32(13, al); f32(14, be); f32(15, ga)
    i32(16, 1); i32(17, 2); i32(18, 3)
    f32(19, amin); f32(20, amax); f32(21, amean)
    i32(22, 1)  # P1
    i32(23, 0)
    words[208:212] = b"MAP "
    words[212:216] = b"\x44\x41\x00\x00"  # little-endian MACHST
    f32(54, arms)
    i32(55, 1)
    label = b"gps-solve density.map (unit cell)"
    words[224:304] = label[:80].ljust(80, b" ")
    path.write_bytes(bytes(words) + data.tobytes(order="F"))


def write_peaks_pdb(
    path: Union[str, Path],
    peaks: Sequence["DensityPeak"],
    cell: CellLike,
    space_group: Optional[str] = None,
) -> Path:
    """Minimal PDB of density peaks (Coot/PyMOL). Coordinates in Å."""
    from grok_phase_solver.io.cif import CrystalStructure

    path = Path(path)
    cell6 = _cell6(cell)
    M = CrystalStructure("t", cell6, "P1").orth_matrix
    sg = (space_group or "P 1").replace("\n", " ")[:40]
    a, b, c, al, be, ga = (float(x) for x in cell6)
    lines = [
        "REMARK   gps-solve density peaks (not a refined model)",
        f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {sg:<11}{1:4d}",
    ]
    for i, p in enumerate(peaks, start=1):
        xyz = M @ np.asarray(p.fract, dtype=np.float64)
        x, y, z = (float(v) for v in xyz)
        occ = 1.0
        bfac = max(5.0, 40.0 - float(getattr(p, "height_sigma", 0.0)))
        lines.append(
            f"HETATM{i:5d}  Q   QPK Q{1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          Q  "
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def map_isolevel(rho: np.ndarray, n_sigma: float = 1.5) -> float:
    rho = np.asarray(rho, dtype=np.float64)
    sigma = float(rho.std())
    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0
    return float(rho.mean() + n_sigma * sigma)


def write_pymol_script(
    path: Union[str, Path],
    *,
    isolevel: float,
    has_pdb: bool,
    has_xyz: bool,
) -> Path:
    path = Path(path)
    pdb_line = "load peaks.pdb, gps_peaks\nshow spheres, gps_peaks\n" if has_pdb else ""
    xyz_line = (
        "load peaks.xyz, gps_peaks_xyz\nshow spheres, gps_peaks_xyz\n"
        if has_xyz and not has_pdb
        else ""
    )
    path.write_text(
        f"""# gps-solve PyMOL handoff — run from this folder:
#   pymol open_in_pymol.pml
# Does not replace SHELXL / Olex2 refinement.
load density.map, gps_map
{pdb_line}{xyz_line}isosurface gps_iso, gps_map, {isolevel:.4g}
color density, gps_iso
orient
"""
    )
    return path


def write_coot_script(path: Union[str, Path], *, has_pdb: bool) -> Path:
    path = Path(path)
    pdb = " peaks.pdb" if has_pdb else ""
    path.write_text(
        f"""#!/bin/sh
# gps-solve Coot handoff — run from this folder:
#   sh open_in_coot.sh
# or: coot --map density.map{('--pdb peaks.pdb' if has_pdb else '')}
exec coot --map density.map{('--pdb' + pdb if has_pdb else '')}
"""
    )
    return path


def write_map_handoff(result: "SolveResult", out_dir: Path) -> List[Path]:
    """Write density.map + viewer scripts next to other gps-solve outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    rho = np.asarray(result.density)
    if rho.ndim != 3 or rho.size == 0:
        return written
    map_path = write_ccp4_map(
        out_dir / "density.map",
        rho,
        result.cell,
        space_group=result.space_group_hm,
    )
    written.append(map_path)
    has_pdb = False
    if result.peaks:
        write_peaks_pdb(
            out_dir / "peaks.pdb",
            result.peaks,
            result.cell,
            space_group=result.space_group_hm,
        )
        written.append(out_dir / "peaks.pdb")
        has_pdb = True
    has_xyz = (out_dir / "peaks.xyz").is_file()
    iso = map_isolevel(rho)
    written.append(
        write_pymol_script(
            out_dir / "open_in_pymol.pml",
            isolevel=iso,
            has_pdb=has_pdb,
            has_xyz=has_xyz,
        )
    )
    written.append(write_coot_script(out_dir / "open_in_coot.sh", has_pdb=has_pdb))
    result.diagnostics["map_export"] = {
        "file": "density.map",
        "shape": [int(n) for n in rho.shape],
        "isolevel_1p5_sigma": iso,
        "pymol": "pymol open_in_pymol.pml",
        "coot": "sh open_in_coot.sh",
    }
    return written
