"""
External SHELXS runner (not redistributed).

Locates ``shelxs`` from:
  1. explicit path / SHELXS env
  2. repo ``ShelX/shelxs`` (user-provided academic binary)
  3. SHELX_BIN / PATH

Writes job.ins + fixed-format job.hkl, runs ``shelxs job``, parses Q-peaks
from job.res → Fcalc phases for fair mapCC.

SHELX license: academic free software; do **not** commit binaries to public git.
Clear macOS quarantine if needed: ``xattr -dr com.apple.quarantine ShelX``
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

from grok_phase_solver.io.shelx_write import (
    atoms_to_fracs_elements,
    parse_shelx_res_atoms,
    write_hkl_shelx,
    write_shelxs_ins,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.solvers.free_fom import free_fom

PathLike = Union[str, Path]


def _repo_shelx_dir() -> Path:
    # solvers/ -> grok_phase_solver/ -> src/ -> repo root
    return Path(__file__).resolve().parents[3] / "ShelX"


def _ensure_executable(p: Path) -> bool:
    if not p.is_file():
        return False
    try:
        mode = p.stat().st_mode
        if not os.access(p, os.X_OK):
            p.chmod(mode | 0o111)
    except OSError:
        pass
    return os.access(p, os.X_OK)


def find_shelxs(explicit: Optional[PathLike] = None) -> Optional[Path]:
    """Locate shelxs binary."""
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    for env in ("SHELXS", "SHELXS_PATH"):
        if os.environ.get(env):
            candidates.append(Path(os.environ[env]))
    bindir = os.environ.get("SHELX_BIN") or os.environ.get("SHELXHOME")
    if bindir:
        for name in ("shelxs", "SHELXS", "shelxs.exe"):
            candidates.append(Path(bindir) / name)
            candidates.append(Path(bindir) / "bin" / name)
    # project-local academic drop-in
    for name in ("shelxs", "SHELXS"):
        candidates.append(_repo_shelx_dir() / name)
    which = shutil.which("shelxs")
    if which:
        candidates.append(Path(which))

    for p in candidates:
        if p.is_file():
            if _ensure_executable(p):
                return p.resolve()
            # still try if not chmod-able
            return p.resolve()
    return None


def shelxs_available(explicit: Optional[PathLike] = None) -> bool:
    return find_shelxs(explicit) is not None


def clear_macos_quarantine(path: Optional[PathLike] = None) -> bool:
    """Best-effort: strip com.apple.quarantine from ShelX binaries."""
    root = Path(path) if path else _repo_shelx_dir()
    if not root.exists():
        return False
    try:
        subprocess.run(
            ["xattr", "-dr", "com.apple.quarantine", str(root)],
            capture_output=True,
            check=False,
            timeout=30,
        )
        return True
    except Exception:
        return False


def shelxs_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_atoms: Optional[int] = None,
    n_try: int = 100,
    d_min: Optional[float] = None,
    lattice: int = -1,
    sfac: Optional[Sequence[str]] = None,
    work_dir: Optional[PathLike] = None,
    shelxs_path: Optional[PathLike] = None,
    timeout_s: float = 300.0,
    title: str = "gps_shelxs",
    keep_files: bool = False,
    max_peaks: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run external SHELXS; convert best peak list to phases/density.

    Returns phases, density, info.
    """
    exe = find_shelxs(shelxs_path)
    if exe is None:
        raise FileNotFoundError(
            "SHELXS not found. Place academic binary at ShelX/shelxs or set SHELXS=..."
        )
    # quarantine often blocks macOS downloads
    clear_macos_quarantine(exe.parent)

    hkl = np.asarray(hkl)
    amp = np.asarray(amplitudes, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    n_find = int(n_atoms) if n_atoms and n_atoms > 0 else int(np.clip(len(amp) // 50, 6, 40))
    n_peaks = int(max_peaks) if max_peaks else max(n_find, min(40, n_find * 3))
    sfac = list(sfac) if sfac else ["C"]

    cleanup = work_dir is None and not keep_files
    if work_dir is None:
        wdir = Path(tempfile.mkdtemp(prefix="gps_shelxs_"))
    else:
        wdir = Path(work_dir)
        wdir.mkdir(parents=True, exist_ok=True)

    job = "job"
    write_hkl_shelx(wdir / f"{job}.hkl", hkl, amplitudes=amp, fixed_format=True)
    write_shelxs_ins(
        wdir / f"{job}.ins",
        cell,
        title=title,
        n_atoms=n_find,
        n_try=n_try,
        sfac=sfac,
        unit=[float(n_find)],
        lattice=lattice,
    )

    t0 = time.time()
    cmd = [str(exe), job]
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
        raise RuntimeError(f"SHELXS timed out after {timeout_s}s") from e

    elapsed = time.time() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    res_path = wdir / f"{job}.res"
    lst_path = wdir / f"{job}.lst"
    atoms = parse_shelx_res_atoms(res_path) if res_path.exists() else []
    # Prefer Q-peaks ordered as in file; take top max_peaks
    atoms = atoms[:n_peaks]

    info: Dict = {
        "method": "shelxs",
        "shelxs_path": str(exe),
        "work_dir": str(wdir),
        "returncode": proc.returncode,
        "seconds": elapsed,
        "n_atoms_request": n_find,
        "n_try": n_try,
        "n_peaks_parsed": len(atoms),
        "stdout_tail": stdout[-2500:],
        "stderr_tail": stderr[-1000:],
        "lst_tail": lst_path.read_text(errors="replace")[-2000:] if lst_path.exists() else "",
        "res_file": str(res_path) if res_path.exists() else None,
    }

    # CFOM from listing if present
    for line in stdout.splitlines():
        if "CFOM" in line and "best" in line.lower():
            info["cfom_line"] = line.strip()
        if line.strip().startswith("RE ="):
            info.setdefault("RE_lines", []).append(line.strip())

    if not atoms:
        info["status"] = "no_atoms"
        info["files"] = sorted(p.name for p in wdir.iterdir())
        if cleanup and not keep_files:
            shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError(
            f"SHELXS produced no parseable peaks (rc={proc.returncode}). "
            f"stdout tail: {stdout[-400:]!r}"
        )

    # Use top n_find peaks as atom model (or all if fewer)
    use = atoms[: max(n_find, min(len(atoms), n_find * 2))]
    fracs, elements = atoms_to_fracs_elements(use, sfac=sfac, default_element="C")
    Fcalc = compute_structure_factors(hkl, fracs, elements, cell)
    phases = np.angle(Fcalc)
    density = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, d_min=d_min
    )
    fom = free_fom(hkl, amp, phases, cell, density=density, include_shells=False)
    ac = np.abs(Fcalc)
    scale = float(np.dot(amp, ac) / (np.dot(ac, ac) + 1e-16))
    R = float(np.sum(np.abs(amp - scale * ac)) / (np.sum(amp) + 1e-16))
    info.update(
        {
            "status": "ok",
            "fracs": fracs,
            "elements": elements,
            "free_fom": fom,
            "R_partial": R,
            "atom_labels": [a["label"] for a in use[:40]],
        }
    )
    if cleanup and not keep_files:
        shutil.rmtree(wdir, ignore_errors=True)
        info["work_dir"] = None
    return phases, density, info
