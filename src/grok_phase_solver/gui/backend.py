"""
GUI backend: save uploads and call the same pipeline as gps-solve.

No Streamlit dependency — unit-testable pure helpers.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from grok_phase_solver.pipeline.export import export_solution
from grok_phase_solver.pipeline.solve import SolveConfig, SolveResult, solve_structure

PathLike = Union[str, Path]

METHODS = [
    "auto",
    "ensemble",
    "charge_flipping",
    "partial_phaseed",
    "fragment_phaseed",
    "ha_phaseed",
    "strong_prior_phaseed",
    "hard_p1_phaseed",
    "phai_phaseed",
    "phai+cf_cond",
    "shelxs",
    "shelxs+shelxe",
    "raar",
    "direct_methods",
    "hio",
    "dual_space",
]


def write_upload(data: bytes, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_text_upload(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def run_solve_job(
    work_dir: PathLike,
    *,
    hkl_bytes: bytes,
    hkl_name: str = "data.hkl",
    ins_bytes: Optional[bytes] = None,
    ins_name: str = "data.ins",
    cell: Optional[str] = None,
    space_group: Optional[str] = None,
    method: str = "auto",
    d_min: Optional[float] = None,
    n_iter: int = 80,
    n_starts: int = 2,
    n_extend: int = 12,
    n_peaks: int = 40,
    seed: int = 0,
    phase_seed_csv_bytes: Optional[bytes] = None,
    phase_seed_res_bytes: Optional[bytes] = None,
    seed_peaks_csv_bytes: Optional[bytes] = None,
    seed_atoms_csv_bytes: Optional[bytes] = None,
    seed_element: str = "C",
    seed_n_atoms: Optional[int] = None,
    patterson_ha: bool = False,
    ha_element: str = "Br",
    n_ha: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a full solve into ``work_dir`` and export artifacts.

    Returns a JSON-serializable summary dict with paths and diagnostics.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hkl_path = write_upload(hkl_bytes, work / hkl_name)
    ins_path = None
    if ins_bytes:
        ins_path = write_upload(ins_bytes, work / ins_name)

    cfg = SolveConfig(
        method=method,
        d_min=d_min,
        n_iter=int(n_iter),
        n_starts=int(n_starts),
        n_extend=int(n_extend),
        n_peaks=int(n_peaks),
        seed=int(seed),
        verbose=verbose,
        seed_element=seed_element,
        seed_n_atoms=seed_n_atoms,
        patterson_ha=patterson_ha,
        ha_element=ha_element,
        n_ha=n_ha,
    )
    if phase_seed_csv_bytes:
        p = write_upload(phase_seed_csv_bytes, work / "phase_seed.csv")
        cfg.phase_seed_csv = str(p)
        cfg.export_seed_csv = str(out_dir / "mapped_seed.csv")
    if phase_seed_res_bytes:
        p = write_upload(phase_seed_res_bytes, work / "seed_model.res")
        cfg.phase_seed_res = str(p)
        if cfg.method == "auto":
            cfg.method = "partial_phaseed"
    if seed_peaks_csv_bytes:
        p = write_upload(seed_peaks_csv_bytes, work / "seed_peaks.csv")
        cfg.seed_peaks_csv = str(p)
        if cfg.method == "auto":
            cfg.method = "partial_phaseed"
    if seed_atoms_csv_bytes:
        p = write_upload(seed_atoms_csv_bytes, work / "seed_atoms.csv")
        cfg.seed_atoms_csv = str(p)
        if cfg.method == "auto":
            cfg.method = "partial_phaseed"

    result: SolveResult = solve_structure(
        hkl_path=str(hkl_path),
        ins_path=str(ins_path) if ins_path else None,
        cell=cell,
        space_group=space_group,
        config=cfg,
    )
    written = export_solution(result, out_dir)
    paths = {p.name: str(p) for p in written}

    summary = {
        "ok": True,
        "method": result.method,
        "n_reflections": len(result.hkl),
        "space_group": result.space_group_hm,
        "cell": result.cell.tolist(),
        "d_min": result.d_min,
        "n_peaks": len(result.peaks),
        "diagnostics": _jsonable(result.diagnostics),
        "warnings": list(result.warnings),
        "out_dir": str(out_dir),
        "files": paths,
        "report_md": (out_dir / "report.md").read_text()
        if (out_dir / "report.md").exists()
        else "",
        "peaks": [
            {
                "rank": p.rank,
                "x": float(p.fract[0]),
                "y": float(p.fract[1]),
                "z": float(p.fract[2]),
                "height_sigma": float(p.height_sigma),
                "height": float(p.height),
            }
            for p in result.peaks[:50]
        ],
    }
    return summary


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def zip_outdir(out_dir: PathLike, zip_path: PathLike) -> Path:
    """Zip all files in the export directory for download."""
    out_dir = Path(out_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_dir).as_posix())
    return zip_path


def demo_paths() -> Dict[str, Path]:
    """Packaged demo HKL/INS if present in the repo."""
    root = Path(__file__).resolve().parents[3]
    easy = root / "examples" / "demo_solve"
    hard = root / "examples" / "partial_seed_demo"
    return {
        "easy_hkl": easy / "demo.hkl",
        "easy_ins": easy / "demo.ins",
        "hard_hkl": hard / "demo_hard.hkl",
        "hard_ins": hard / "demo_hard.ins",
        "hard_seed_csv": hard / "known_phases_30pct.csv",
        "hard_fragment_res": hard / "fragment.res",
    }
