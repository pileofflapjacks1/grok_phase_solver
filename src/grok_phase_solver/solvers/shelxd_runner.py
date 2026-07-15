"""
External SHELXD runner (not bundled).

SHELX is academic free software (registration required):
  https://shelx.uni-goettingen.de/

This module:
  1. Locates ``shelxd`` on PATH or via SHELXD / SHELX_BIN env vars
  2. Writes job.ins + job.hkl in a temp/work directory
  3. Runs the binary
  4. Parses .res atom list → Fcalc phases → density for fair mapCC

If SHELXD is not installed, callers should fall back to ``dual_space_solve``
and report status ``shelxd_not_found``.
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
    write_shelxd_ins,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.solvers.free_fom import free_fom

PathLike = Union[str, Path]


def find_shelxd(explicit: Optional[PathLike] = None) -> Optional[Path]:
    """Locate shelxd executable."""
    if explicit is not None:
        p = Path(explicit)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        return None
    env = os.environ.get("SHELXD") or os.environ.get("SHELXD_PATH")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    bindir = os.environ.get("SHELX_BIN") or os.environ.get("SHELXHOME")
    if bindir:
        for name in ("shelxd", "shelxd_mp", "SHELXD", "shelxd.exe"):
            p = Path(bindir) / name
            if p.is_file() and os.access(p, os.X_OK):
                return p.resolve()
            # SHELXHOME often is the root containing bin/
            p2 = Path(bindir) / "bin" / name
            if p2.is_file() and os.access(p2, os.X_OK):
                return p2.resolve()
    which = shutil.which("shelxd") or shutil.which("shelxd_mp")
    if which:
        return Path(which).resolve()
    return None


def shelxd_available(explicit: Optional[PathLike] = None) -> bool:
    return find_shelxd(explicit) is not None


def _guess_n_find(n_atoms_hint: Optional[int], amp: np.ndarray) -> int:
    if n_atoms_hint is not None and n_atoms_hint > 0:
        return int(n_atoms_hint)
    # crude: scale with number of strong reflections
    return int(np.clip(len(amp) // 40, 6, 40))


def shelxd_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_atoms: Optional[int] = None,
    n_try: int = 50,
    seed: int = 1,
    d_min: Optional[float] = None,
    space_group_hm: Optional[str] = None,
    lattice: int = -1,
    symm: Optional[Sequence[str]] = None,
    sfac: Optional[Sequence[str]] = None,
    work_dir: Optional[PathLike] = None,
    shelxd_path: Optional[PathLike] = None,
    timeout_s: float = 300.0,
    title: str = "gps_shelxd",
    keep_files: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run external SHELXD and convert best atom model to phases/density.

    Returns
    -------
    phases, density, info

    Raises
    ------
    FileNotFoundError
        If shelxd binary is not found.
    RuntimeError
        If SHELXD runs but produces no usable atom model.
    """
    exe = find_shelxd(shelxd_path)
    if exe is None:
        raise FileNotFoundError(
            "SHELXD binary not found. Install academic SHELX from "
            "https://shelx.uni-goettingen.de/ and ensure `shelxd` is on PATH, "
            "or set SHELXD=/path/to/shelxd. "
            "For an in-repo dual-space proxy use dual_space_solve()."
        )

    hkl = np.asarray(hkl)
    amp = np.asarray(amplitudes, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    n_find = _guess_n_find(n_atoms, amp)
    sfac = list(sfac) if sfac else ["C", "H", "N", "O"]

    cleanup = work_dir is None and not keep_files
    if work_dir is None:
        tmp = tempfile.mkdtemp(prefix="gps_shelxd_")
        wdir = Path(tmp)
    else:
        wdir = Path(work_dir)
        wdir.mkdir(parents=True, exist_ok=True)

    job = "job"
    ins_path = wdir / f"{job}.ins"
    hkl_path = wdir / f"{job}.hkl"
    write_hkl_shelx(hkl_path, hkl, amplitudes=amp)
    write_shelxd_ins(
        ins_path,
        cell,
        title=title,
        n_find=n_find,
        n_try=n_try,
        seed=seed,
        sfac=sfac,
        unit=[float(n_find), 0, 0, 0][: len(sfac)],
        lattice=lattice,
        symm=symm,
        space_group_hm=space_group_hm,
        patt=True,
    )

    t0 = time.time()
    # SHELXD is typically invoked as: shelxd job   (reads job.ins / job.hkl)
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
        raise RuntimeError(f"SHELXD timed out after {timeout_s}s") from e

    elapsed = time.time() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Collect candidate result files
    res_candidates = [
        wdir / f"{job}.res",
        wdir / f"{job}.pdb",
        wdir / "shelxd.res",
    ]
    # Also any *.res written
    res_candidates.extend(sorted(wdir.glob("*.res")))
    atoms: List[dict] = []
    res_used = None
    for rp in res_candidates:
        if rp.exists() and rp.stat().st_size > 0:
            atoms = parse_shelx_res_atoms(rp)
            if atoms:
                res_used = rp
                break

    # Fallback: parse stdout for coordinates (rare)
    lst_path = wdir / f"{job}.lst"
    lst_text = lst_path.read_text(errors="replace") if lst_path.exists() else ""

    info: Dict = {
        "method": "shelxd",
        "shelxd_path": str(exe),
        "work_dir": str(wdir),
        "returncode": proc.returncode,
        "seconds": elapsed,
        "n_find": n_find,
        "n_try": n_try,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "lst_tail": lst_text[-2000:] if lst_text else "",
        "res_file": str(res_used) if res_used else None,
        "n_atoms_parsed": len(atoms),
    }

    if not atoms:
        # Keep files for debugging if failed
        info["status"] = "no_atoms"
        info["files"] = sorted(p.name for p in wdir.iterdir())
        if cleanup:
            # still keep on failure for a moment — actually clean if not keep
            if not keep_files:
                shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError(
            f"SHELXD produced no parseable atoms (rc={proc.returncode}). "
            f"work_dir={wdir} files={info.get('files')}. "
            f"stderr={stderr[:400]!r}"
        )

    fracs, elements = atoms_to_fracs_elements(atoms, sfac=sfac)
    Fcalc = compute_structure_factors(hkl, fracs, elements, cell)
    phases = np.angle(Fcalc)
    # Re-impose observed moduli for density
    density = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, d_min=d_min
    )
    fom = free_fom(hkl, amp, phases, cell, density=density, include_shells=False)
    # partial R
    R = float(
        np.sum(np.abs(amp - np.abs(Fcalc) * (np.dot(amp, np.abs(Fcalc)) /
               (np.dot(np.abs(Fcalc), np.abs(Fcalc)) + 1e-16))))
        / (np.sum(amp) + 1e-16)
    )
    info.update(
        {
            "status": "ok",
            "fracs": fracs,
            "elements": elements,
            "free_fom": fom,
            "R_partial": R,
            "atom_labels": [a["label"] for a in atoms[:50]],
        }
    )

    if cleanup and not keep_files:
        shutil.rmtree(wdir, ignore_errors=True)
        info["work_dir"] = None

    return phases, density, info


def shelxd_or_dual_space(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    prefer_shelxd: bool = True,
    shelxd_path: Optional[PathLike] = None,
    n_atoms: Optional[int] = None,
    n_try: int = 50,
    seed: int = 0,
    d_min: Optional[float] = None,
    dual_space_starts: int = 8,
    dual_space_cycles: int = 40,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prefer real SHELXD; otherwise educational dual-space baseline.

    info['backend'] is 'shelxd' or 'dual_space'.
    """
    if prefer_shelxd and shelxd_available(shelxd_path):
        try:
            ph, rho, info = shelxd_solve(
                hkl,
                amplitudes,
                cell,
                n_atoms=n_atoms,
                n_try=n_try,
                seed=seed if seed else 1,
                d_min=d_min,
                shelxd_path=shelxd_path,
                verbose=verbose,
                **{k: v for k, v in kwargs.items() if k in (
                    "space_group_hm", "lattice", "symm", "sfac",
                    "work_dir", "timeout_s", "title", "keep_files",
                )},
            )
            info["backend"] = "shelxd"
            return ph, rho, info
        except Exception as e:
            if verbose:
                print(f"  SHELXD failed ({e}); falling back to dual_space")
            fallback_err = str(e)
    else:
        fallback_err = "shelxd_not_found"

    from grok_phase_solver.solvers.dual_space import dual_space_solve

    n_find = n_atoms if n_atoms is not None else _guess_n_find(None, amplitudes)
    ph, rho, info = dual_space_solve(
        hkl,
        amplitudes,
        cell,
        n_atoms=n_find,
        n_cycles=dual_space_cycles,
        n_starts=dual_space_starts,
        seed=seed,
        d_min=d_min,
        verbose=verbose,
    )
    info["backend"] = "dual_space"
    info["shelxd_fallback_reason"] = fallback_err
    return ph, rho, info
