"""
GUI backend: save uploads and call the same pipeline as gps-solve.

No Streamlit dependency — unit-testable pure helpers.
"""

from __future__ import annotations

import io
import re
import shutil
import traceback
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

# Scientist-facing scenario → method defaults
WIZARD_SCENARIOS = {
    "auto": {
        "label": "Not sure (auto)",
        "method": "auto",
        "help": "Easy/high-res → ensemble; hard → prior/CF. Seeds force partial_phaseed.",
        "n_iter": 80,
        "n_starts": 2,
    },
    "easy": {
        "label": "Easy / high resolution",
        "method": "ensemble",
        "help": "Best open ab initio path on easy cells (CF+RAAR multistart).",
        "n_iter": 100,
        "n_starts": 3,
    },
    "hard_abinitio": {
        "label": "Hard, no partial info",
        "method": "auto",
        "help": "Honest: often unsolved. Prefer fragment/HA if map fails.",
        "n_iter": 80,
        "n_starts": 2,
    },
    "have_phases": {
        "label": "Have known phases (CSV)",
        "method": "partial_phaseed",
        "help": "Upload h,k,l,phase_deg. Oracle bar ~30% strong |E| within ~20°.",
        "n_iter": 80,
        "n_starts": 2,
    },
    "have_fragment": {
        "label": "Have fragment / SHELXS .res / model CIF",
        "method": "partial_phaseed",
        "help": "Upload .res atoms or use gps-make-seed --from-cif (AF/RF model) → Fcalc seed.",
        "n_iter": 80,
        "n_starts": 2,
    },
    "have_ha": {
        "label": "Heavy atom present (heuristic)",
        "method": "partial_phaseed",
        "help": "Patterson HA heuristic (weak). Prefer isomorphous pair offline.",
        "n_iter": 80,
        "n_starts": 2,
        "patterson_ha": True,
    },
    "shelxs": {
        "label": "External SHELXS (±E)",
        "method": "shelxs",
        "help": "Needs local ShelX/shelxs academic binary.",
        "n_iter": 60,
        "n_starts": 1,
    },
    "advanced": {
        "label": "Advanced (pick method)",
        "method": None,  # keep sidebar method
        "help": "Full method list in Advanced options.",
        "n_iter": 80,
        "n_starts": 2,
    },
}


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


def parse_cell_string(raw: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Parse unit cell from free-form scientist input.

    Accepts:
      - ``9.75,8.89,7.57,90,112.7,90``
      - ``CELL 0.71073 9.75 8.89 7.57 90 112.7 90`` (wavelength stripped)
      - whitespace / semicolon separated

    Returns (cell_csv, wavelength_or_None, notes).
    """
    notes: List[str] = []
    if not raw or not str(raw).strip():
        return None, None, notes
    s = str(raw).strip()
    # Pull trailing space group if present: ... 90  SG=P21/c
    sg_hint = None
    m_sg = re.search(
        r"(?:space\s*group|sg|symm)\s*[=:]\s*['\"]?([^'\"\n]+)['\"]?",
        s,
        flags=re.I,
    )
    if m_sg:
        sg_hint = m_sg.group(1).strip()
        s = s[: m_sg.start()] + s[m_sg.end() :]
        notes.append(f"Parsed space-group hint: {sg_hint}")

    s = re.sub(r"^(CELL|cell)\s+", "", s.strip(), flags=re.I)
    # numbers
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) < 6:
        raise ValueError(
            f"Need 6 cell parameters (a b c α β γ); found {len(nums)} number(s) in: {raw!r}"
        )
    wavelength = None
    # SHELX CELL: wavelength a b c al be ga  → 7 numbers
    if len(nums) >= 7:
        w = float(nums[0])
        # wavelength typically 0.5–2.5 Å for lab/synchrotron X-ray
        if 0.4 <= w <= 3.0:
            wavelength = f"{w}"
            a, b, c, al, be, ga = (float(x) for x in nums[1:7])
            notes.append(f"Stripped wavelength {w} Å from CELL line")
        else:
            a, b, c, al, be, ga = (float(x) for x in nums[:6])
    else:
        a, b, c, al, be, ga = (float(x) for x in nums[:6])

    # sanity
    if min(a, b, c) <= 0:
        raise ValueError("Cell edges a,b,c must be positive.")
    if not (20 <= al <= 160 and 20 <= be <= 160 and 20 <= ga <= 160):
        notes.append("Unusual cell angles — check α,β,γ are in degrees.")

    cell_csv = f"{a},{b},{c},{al},{be},{ga}"
    # stash sg in notes with marker for caller
    if sg_hint:
        notes.append(f"__SG__:{sg_hint}")
    return cell_csv, wavelength, notes


def format_user_error(exc: BaseException) -> str:
    """Short scientist-facing message for common failures."""
    msg = str(exc).strip() or type(exc).__name__
    low = msg.lower()
    if "phase seed" in low or "no phase seed source" in low:
        return (
            f"{msg}\n\n"
            "**Hint:** For partial_phaseed, upload a phase CSV, fragment .res, "
            "or peaks.csv — or pick a different scenario."
        )
    if "too few reflections" in low:
        return f"{msg}\n\n**Hint:** Check d_min cutoff and that the HKL loaded correctly."
    if "unit cell" in low or "cell" in low and "required" in low:
        return f"{msg}\n\n**Hint:** Upload a .ins with CELL, or paste a CELL line / a,b,c,α,β,γ."
    if "unknown method" in low:
        return f"{msg}\n\n**Hint:** Choose a method from the Advanced list."
    if "shelxs" in low or "shelx" in low:
        return (
            f"{msg}\n\n"
            "**Hint:** Place academic binaries under `ShelX/` "
            "(see USER_GUIDE). macOS: clear quarantine with xattr."
        )
    if "no such file" in low or "not found" in low:
        return f"{msg}\n\n**Hint:** Re-upload files; temporary work dirs are cleaned between sessions."
    return f"{msg}\n\n({type(exc).__name__})"


def format_seed_quality_panel(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """
    GUI panel payload for Carrozzini-style / partial seed quality.

    Returns a dict ready for Streamlit metrics: class, P(success), warnings, features.
    """
    d = diagnostics or {}
    sq = d.get("seed_quality") or {}
    panel: Dict[str, Any] = {
        "has_quality": bool(sq),
        "predicted_class": sq.get("predicted_class", d.get("seed_predicted_class")),
        "success_probability": sq.get(
            "success_probability", d.get("seed_success_probability")
        ),
        "predicted_mpe_deg": sq.get("predicted_mpe_deg", d.get("seed_predicted_mpe_deg")),
        "predicted_corr": sq.get("predicted_corr"),
        "warning": sq.get("warning"),
        "recommend_fallback": bool(sq.get("recommend_fallback")),
        "features": sq.get("features") or {},
        "size_meets_bar": sq.get("size_meets_bar", d.get("seed_size_meets_bar")),
        "frac_strong_seeded": sq.get("frac_strong_seeded", d.get("seed_frac_strong")),
        "method": sq.get("method"),
    }
    return panel


def map_quality_hints(summary: Dict[str, Any]) -> List[str]:
    """Truth-free post-run guidance for the GUI banner."""
    hints: List[str] = []
    d = summary.get("diagnostics") or {}
    fom = d.get("free_fom_composite")
    n_peaks = summary.get("n_peaks") or 0
    method = summary.get("method") or ""

    sq = d.get("seed_quality") or {}
    if isinstance(sq, dict) and sq.get("predicted_class") == 0:
        hints.append(
            "Seed quality Class 0 — AI-PhaSeed alone may fail. "
            "Prefer partial-φ / fragment / HA, or enable DM+AI hybrid."
        )
    elif d.get("seed_predicted_class") == 0:
        hints.append(
            "Seed quality Class 0 — consider partial-φ fallback or --ai-dm-hybrid."
        )

    if fom is not None:
        try:
            f = float(fom)
            if f < 0.45:
                hints.append(
                    "Free FOM is low — map may be unsolved. Try ensemble (easy data), "
                    "partial_phaseed with a fragment, or external shelxs."
                )
            elif f < 0.65:
                hints.append(
                    "Free FOM is moderate. Inspect density/peaks; consider more "
                    "iterations or a partial seed if chemistry looks wrong."
                )
            else:
                hints.append(
                    "Free FOM looks healthy (truth-free ranking only — not proof of solution)."
                )
        except (TypeError, ValueError):
            pass

    if n_peaks < 5:
        hints.append("Few density peaks — check resolution, SG, or try a stronger seed.")
    elif n_peaks >= 15:
        hints.append("Many peaks listed — assign elements carefully in Olex2/SHELXL.")

    if d.get("seed_size_meets_bar") is False:
        hints.append(
            "Seed set is **below** the ~30% strong-|E| size bar. "
            "Add more known φ / heavier fragment / HA sites."
        )
    if d.get("seed_size_meets_bar") is True:
        hints.append(
            "Seed size meets the ~30% strong-|E| coverage bar (correctness still unknown)."
        )

    if method in ("charge_flipping", "strong_prior_phaseed", "hard_p1_phaseed") and (
        fom is not None and float(fom) < 0.55
    ):
        hints.append(
            "Hard ab initio often fails. **Retry with peaks as seed** "
            "(partial_phaseed) if peaks look atomic."
        )
    return hints


def shelxl_handoff_snippet(out_dir: PathLike, hkl_name: str = "data.hkl") -> str:
    """Shell snippet for SHELXL after GUI export."""
    out = Path(out_dir)
    return f"""# After gps-gui: refine trial model with SHELXL (academic binary)
cd {out.parent}
mkdir -p shelxl_work && cd shelxl_work
cp "{out / "trial.res"}" work.ins
# Copy your experimental intensities (same cell/SG as the solve):
#   cp /path/to/your.hkl work.hkl
# Or if the GUI staged HKL next to out/:
cp "../{hkl_name}" work.hkl 2>/dev/null || true

# Edit work.ins: assign real elements to Q peaks, fix SFAC/UNIT
# Then:
#   ShelX/shelxl work
#   # or: shelxl work

# Open work.res / CIF in Olex2. gps-solve does not replace least-squares refinement.
"""


def resolve_wizard(
    scenario_key: str,
    advanced_method: str,
) -> Dict[str, Any]:
    """Map wizard scenario to method + option overrides."""
    sc = WIZARD_SCENARIOS.get(scenario_key) or WIZARD_SCENARIOS["auto"]
    method = sc.get("method") or advanced_method
    return {
        "method": method,
        "n_iter": sc.get("n_iter", 80),
        "n_starts": sc.get("n_starts", 2),
        "patterson_ha": bool(sc.get("patterson_ha", False)),
        "label": sc.get("label", scenario_key),
        "help": sc.get("help", ""),
    }


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
    dm_ai_weight: float = 0.0,
    low_res_path: bool = False,
    seed_quality_filter: bool = False,
    assess_seed_quality: bool = True,
    prior_weight: float = 0.30,
    verbose: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    capture_log: bool = True,
) -> Dict[str, Any]:
    """
    Run a full solve into ``work_dir`` and export artifacts.

    Returns a JSON-serializable summary dict with paths and diagnostics.
    """
    def _prog(msg: str) -> None:
        if progress:
            progress(msg)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _prog("Writing input files…")
    # Normalize extension for MTZ
    hkl_name = hkl_name or "data.hkl"
    hkl_path = write_upload(hkl_bytes, work / hkl_name)
    # Keep a stable name for handoff
    if hkl_path.suffix.lower() not in (".hkl", ".mtz", ".cif"):
        # leave as-is; experiment loader sniffs content
        pass

    ins_path = None
    if ins_bytes:
        ins_path = write_upload(ins_bytes, work / ins_name)

    # Parse cell if free-form
    cell_csv = cell
    sg = space_group
    parse_notes: List[str] = []
    if cell:
        try:
            cell_csv, _wl, parse_notes = parse_cell_string(cell)
            for n in parse_notes:
                if n.startswith("__SG__:") and not sg:
                    sg = n.split(":", 1)[1]
        except ValueError:
            # allow raw comma string through to SolveConfig if already valid
            cell_csv = cell

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
        dm_ai_weight=float(dm_ai_weight),
        low_res_path=bool(low_res_path),
        seed_quality_filter=bool(seed_quality_filter),
        assess_seed_quality=bool(assess_seed_quality),
        prior_weight=float(prior_weight),
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

    if (
        cfg.method in ("partial_phaseed", "fragment_phaseed", "ha_phaseed")
        and not any(
            [
                cfg.phase_seed_csv,
                cfg.phase_seed_res,
                cfg.seed_peaks_csv,
                cfg.seed_atoms_csv,
                cfg.patterson_ha,
            ]
        )
    ):
        raise RuntimeError(
            format_user_error(
                ValueError(
                    "partial_phaseed needs a seed: phase CSV, fragment .res, "
                    "peaks.csv, or enable Patterson HA heuristic."
                )
            )
        )

    _prog(f"Phasing with method={cfg.method}…")
    log_buf = io.StringIO()
    try:
        if capture_log:
            with redirect_stdout(log_buf):
                result: SolveResult = solve_structure(
                    hkl_path=str(hkl_path),
                    ins_path=str(ins_path) if ins_path else None,
                    cell=cell_csv,
                    space_group=sg,
                    config=cfg,
                )
        else:
            result = solve_structure(
                hkl_path=str(hkl_path),
                ins_path=str(ins_path) if ins_path else None,
                cell=cell_csv,
                space_group=sg,
                config=cfg,
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(format_user_error(e)) from e

    _prog("Exporting density, peaks, trial.res…")
    written = export_solution(result, out_dir)
    paths = {p.name: str(p) for p in written}

    # Snapshot peaks for retry-as-seed
    peaks_csv_bytes = None
    peaks_path = out_dir / "peaks.csv"
    if peaks_path.exists():
        peaks_csv_bytes = peaks_path.read_bytes()

    summary = {
        "ok": True,
        "method": result.method,
        "method_requested": method,
        "n_reflections": len(result.hkl),
        "space_group": result.space_group_hm,
        "cell": result.cell.tolist(),
        "d_min": result.d_min,
        "n_peaks": len(result.peaks),
        "diagnostics": _jsonable(result.diagnostics),
        "warnings": list(result.warnings) + [n for n in parse_notes if not n.startswith("__SG__")],
        "out_dir": str(out_dir),
        "work_dir": str(work),
        "hkl_name": hkl_name,
        "files": paths,
        "report_md": (out_dir / "report.md").read_text()
        if (out_dir / "report.md").exists()
        else "",
        "log": log_buf.getvalue() if capture_log else "",
        "hints": [],
        "shelxl_snippet": shelxl_handoff_snippet(out_dir, hkl_name=hkl_name),
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
        # For GUI retry — store inputs (may be large; peaks only for seed)
        "_retry": {
            "hkl_name": hkl_name,
            "ins_name": ins_name,
            "cell": cell_csv,
            "space_group": sg,
            "has_peaks_seed": peaks_csv_bytes is not None,
        },
    }
    # Attach peaks bytes only via side file (already in out/)
    summary["hints"] = map_quality_hints(summary)
    _prog("Done.")
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
