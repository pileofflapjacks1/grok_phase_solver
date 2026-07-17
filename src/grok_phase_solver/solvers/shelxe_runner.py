"""
External SHELXE density-modification runner (not redistributed).

Typical small-molecule use after SHELXS / gps peaks:
  write job.hkl + job.ins (atoms) → ``shelxe job -m15 -s0.45``
  read phases from job.phs

Binary: ShelX/shelxe or SHELXE env. Gitignore binaries; academic license only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grok_phase_solver.io.shelx_write import write_hkl_shelx
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.shelxs_runner import (
    _ensure_executable,
    _repo_shelx_dir,
    clear_macos_quarantine,
)

PathLike = Union[str, Path]


def find_shelxe(explicit: Optional[PathLike] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    for env in ("SHELXE", "SHELXE_PATH"):
        if os.environ.get(env):
            candidates.append(Path(os.environ[env]))
    bindir = os.environ.get("SHELX_BIN") or os.environ.get("SHELXHOME")
    if bindir:
        for name in ("shelxe", "SHELXE", "shelxe.exe"):
            candidates.append(Path(bindir) / name)
            candidates.append(Path(bindir) / "bin" / name)
    for name in ("shelxe", "SHELXE"):
        candidates.append(_repo_shelx_dir() / name)
    which = shutil.which("shelxe")
    if which:
        candidates.append(Path(which))
    for p in candidates:
        if p.is_file():
            _ensure_executable(p)
            return p.resolve()
    return None


def shelxe_available(explicit: Optional[PathLike] = None) -> bool:
    return find_shelxe(explicit) is not None


def write_model_ins(
    path: PathLike,
    cell: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    *,
    title: str = "gps shelxe model",
    wavelength: float = 0.71073,
    lattice: int = -1,
) -> Path:
    """Write .ins with atomic sites for SHELXE starting model."""
    path = Path(path)
    a, b, c, al, be, ga = [float(x) for x in cell]
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    elements = list(elements)
    # collapse to unique SFAC list
    sfac: List[str] = []
    for el in elements:
        e = el.upper() if len(el) == 1 else el[0].upper() + el[1:].lower()
        if e not in sfac:
            sfac.append(e)
    if not sfac:
        sfac = ["C"]
    unit = [float(elements.count(s) if s in elements else elements.count(s.upper())) for s in sfac]
    # recount properly
    unit = []
    for s in sfac:
        unit.append(float(sum(1 for e in elements if e.upper() == s.upper())))

    lines = [
        f"TITL {title}",
        f"CELL {wavelength:.5f} {a:.4f} {b:.4f} {c:.4f} {al:.3f} {be:.3f} {ga:.3f}",
        "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        f"LATT {int(lattice)}",
        "SFAC " + " ".join(sfac),
        "UNIT " + " ".join(f"{int(u)}" for u in unit),
        "FVAR 1.0",
    ]
    for i, (f, el) in enumerate(zip(fracs, elements)):
        e = el.upper() if len(el) == 1 else el[0].upper() + el[1:].lower()
        si = sfac.index(e) + 1 if e in sfac else 1
        label = f"{e}{i+1}"
        lines.append(
            f"{label:6s} {si} {f[0]:10.6f} {f[1]:10.6f} {f[2]:10.6f} 11.00000 0.05000"
        )
    lines.append("HKLF 4")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_shelxe_phs(path: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse SHELXE .phs file.

    Columns (typical): h k l F FOM phase_deg [extra]
    Returns hkl (N,3), phases_rad (N,), fom (N,).
    """
    path = Path(path)
    rows_h, rows_p, rows_f = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            h, k, l = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
            fom = float(parts[4])
            ph_deg = float(parts[5])
        except ValueError:
            continue
        if h == 0 and k == 0 and l == 0:
            continue
        rows_h.append((h, k, l))
        rows_p.append(np.deg2rad(ph_deg))
        rows_f.append(fom)
    if not rows_h:
        raise ValueError(f"No phases parsed from {path}")
    return (
        np.asarray(rows_h, dtype=np.int32),
        np.asarray(rows_p, dtype=np.float64),
        np.asarray(rows_f, dtype=np.float64),
    )


def map_phs_to_reflections(
    hkl: np.ndarray,
    phs_hkl: np.ndarray,
    phs_phases: np.ndarray,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Align SHELXE phases onto the full reflection list (Friedel-aware)."""
    key = {}
    for i, h in enumerate(phs_hkl):
        t = (int(h[0]), int(h[1]), int(h[2]))
        key[t] = phs_phases[i]
        key[(-t[0], -t[1], -t[2])] = -phs_phases[i]
    if fallback is None:
        out = np.zeros(len(hkl), dtype=np.float64)
    else:
        out = np.asarray(fallback, dtype=np.float64).copy()
    mapped = 0
    for i, h in enumerate(hkl):
        t = (int(h[0]), int(h[1]), int(h[2]))
        if t in key:
            out[i] = key[t]
            mapped += 1
    return out


def shelxe_polish(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    *,
    n_mod_cycles: int = 15,
    solvent_fraction: float = 0.45,
    d_min: Optional[float] = None,
    phase_init: Optional[np.ndarray] = None,
    work_dir: Optional[PathLike] = None,
    shelxe_path: Optional[PathLike] = None,
    timeout_s: float = 300.0,
    keep_files: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Density-modify phases starting from a partial atomic model (SHELXE).

    Returns phases, density, info.
    """
    exe = find_shelxe(shelxe_path)
    if exe is None:
        raise FileNotFoundError(
            "SHELXE not found. Place binary at ShelX/shelxe or set SHELXE=..."
        )
    clear_macos_quarantine(exe.parent)

    hkl = np.asarray(hkl)
    amp = np.asarray(amplitudes, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)

    cleanup = work_dir is None and not keep_files
    if work_dir is None:
        wdir = Path(tempfile.mkdtemp(prefix="gps_shelxe_"))
    else:
        wdir = Path(work_dir)
        wdir.mkdir(parents=True, exist_ok=True)

    job = "job"
    write_hkl_shelx(wdir / f"{job}.hkl", hkl, amplitudes=amp, fixed_format=True)
    write_model_ins(
        wdir / f"{job}.ins",
        cell,
        fracs,
        elements,
        title="gps_shelxe",
    )
    # SHELXE also accepts atoms from .res matching name
    shutil.copy(wdir / f"{job}.ins", wdir / f"{job}.res")

    # -m = density modification cycles; -s = solvent fraction
    cmd = [
        str(exe),
        job,
        f"-m{int(max(n_mod_cycles, 1))}",
        f"-s{float(solvent_fraction):.3f}",
    ]
    t0 = time.time()
    if verbose:
        print(f"  running: {' '.join(cmd)} in {wdir}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(wdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError(f"SHELXE timed out after {timeout_s}s") from e

    elapsed = time.time() - t0
    phs_path = wdir / f"{job}.phs"
    info: Dict = {
        "method": "shelxe",
        "shelxe_path": str(exe),
        "returncode": proc.returncode,
        "seconds": elapsed,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "cmd": cmd,
        "work_dir": str(wdir),
    }
    # pseudo-free CC from listing
    for line in (proc.stdout or "").splitlines():
        if "Pseudo-free CC" in line or "Estimated mean FOM" in line:
            info.setdefault("fom_lines", []).append(line.strip())

    if not phs_path.exists():
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError(
            f"SHELXE did not write .phs (rc={proc.returncode}). "
            f"stdout: {(proc.stdout or '')[-400:]!r}"
        )

    phs_hkl, phs_ph, phs_fom = parse_shelxe_phs(phs_path)
    phases = map_phs_to_reflections(hkl, phs_hkl, phs_ph, fallback=phase_init)
    density = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, d_min=d_min
    )
    fom = free_fom(hkl, amp, phases, cell, density=density, include_shells=False)
    info.update(
        {
            "status": "ok",
            "n_phs": len(phs_ph),
            "mean_phs_fom": float(np.mean(phs_fom)),
            "free_fom": fom,
            "n_model_atoms": len(fracs),
        }
    )
    if cleanup and not keep_files:
        shutil.rmtree(wdir, ignore_errors=True)
        info["work_dir"] = None
    return phases, density, info
