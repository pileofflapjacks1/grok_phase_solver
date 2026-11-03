"""
Opt-in cctbx/smtbx ab initio path (tangent formula + charge flipping).

Runs (a) cctbx.dmtbx weighted tangent (multisolution) and (b) smtbx charge
flipping with Olex2-like defaults, then keeps the candidate with the higher
*own* figure of merit. Never ranks against a known model.

cctbx is an optional dependency. On this project's shared box it is provided by
Olex2's bundled Python 3.8 (``olex2c``). See ``find_cctbx_runner`` /
docs/USER_GUIDE.md.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from grok_phase_solver.pipeline.peaks import DensityPeak

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors

PathLike = Union[str, Path]

# Fixed defaults (identical for every dataset; do not tune per entry)
E_MIN = 1.2
E_CAP_PER_ATOM = 10
E_CAP_MIN = 50
N_TANGENT_CYCLES = 12
TIME_BUDGET_S = 60.0
MIN_PEAK_DIST_A = 1.2
SIGMA2_ONLY = True
BASE_SEED = 20251102

# smtbx CF defaults matching Olex2 Method_cctbx_ChargeFlip
CF_AMPLITUDE_TYPE = "quasi-E"
CF_MAX_ATTEMPTS_PHASE_TRANSITION = 5
CF_MAX_ATTEMPTS_SHARP_CC = 5
CF_MAX_SOLVING_ITERATIONS = 500
CF_FLIPPING_INTERVAL = 60


class CctbxUnavailableError(RuntimeError):
    """Raised when the ab_initio method is requested but cctbx cannot be run."""


def e_list_cap(n_nonH: int) -> int:
    """Cap on strong |E| reflections: max(50, 10 × N_nonH)."""
    return max(E_CAP_MIN, E_CAP_PER_ATOM * max(int(n_nonH), 1))


def combined_map_fom(consistency: float, skew: float) -> float:
    """Own FOM used for ranking (model never enters).

    ``cons × (0.5 + 0.5·tanh(max(skew,0)/2)) + 0.05·max(skew,0)``.
    For charge flipping, pass consistency=1.0 after a finished run.
    """
    sk = max(float(skew), 0.0)
    cons = float(consistency)
    return cons * (0.5 + 0.5 * math.tanh(sk / 2.0)) + 0.05 * sk


def select_better_by_own_fom(
    candidates: Sequence[Dict[str, Any]],
    fom_key: str = "fom",
) -> Dict[str, Any]:
    """Return the candidate with the highest own FOM (never model-based)."""
    if not candidates:
        raise ValueError("no candidates to select")
    return max(candidates, key=lambda c: float(c.get(fom_key, float("-inf"))))


def min_distance_ok(
    site_frac: np.ndarray,
    accepted_frac: Sequence[np.ndarray],
    cell: np.ndarray,
    min_dist_A: float = MIN_PEAK_DIST_A,
) -> bool:
    """True if *site_frac* is ≥ min_dist_A from all accepted sites (orth. Å)."""
    from grok_phase_solver.io.cif import CrystalStructure

    M = CrystalStructure("t", np.asarray(cell, dtype=np.float64), "P1").orth_matrix
    xyz = M @ np.asarray(site_frac, dtype=np.float64)
    for other in accepted_frac:
        d = xyz - (M @ np.asarray(other, dtype=np.float64))
        # minimal image in fractional space along each axis (±1) is enough for
        # this plain filter; full SG symmetry is applied by cctbx peak_search.
        best = float(np.linalg.norm(d))
        for ox in (-1.0, 0.0, 1.0):
            for oy in (-1.0, 0.0, 1.0):
                for oz in (-1.0, 0.0, 1.0):
                    if ox == 0.0 and oy == 0.0 and oz == 0.0:
                        continue
                    shift = M @ np.array([ox, oy, oz], dtype=np.float64)
                    best = min(best, float(np.linalg.norm(d - shift)))
        if best < min_dist_A:
            return False
    return True


def plain_peak_pick(
    sites_frac: Sequence[Sequence[float]],
    heights: Sequence[float],
    cell: np.ndarray,
    n_keep: int,
    min_dist_A: float = MIN_PEAK_DIST_A,
) -> List["DensityPeak"]:
    """Greedy top-N keep with min distance (Å). No min_sigma gate."""
    from grok_phase_solver.pipeline.peaks import DensityPeak

    order = np.argsort(-np.asarray(heights, dtype=np.float64))
    kept_frac: List[np.ndarray] = []
    peaks: List[DensityPeak] = []
    sigma_proxy = float(np.std(heights)) + 1e-16
    for rank_i, idx in enumerate(order):
        if len(peaks) >= n_keep:
            break
        frac = np.asarray(sites_frac[int(idx)], dtype=np.float64)
        if not min_distance_ok(frac, kept_frac, cell, min_dist_A=min_dist_A):
            continue
        h = float(heights[int(idx)])
        kept_frac.append(frac)
        peaks.append(
            DensityPeak(
                rank=len(peaks),
                fract=frac,
                height=h,
                height_sigma=h / sigma_proxy,
            )
        )
    return peaks


def _default_olex2_roots() -> List[Path]:
    roots: List[Path] = []
    for key in ("OLEX2_ROOT", "OLEX2HOME", "OLEX2"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]))
    # Shared-box layout used by the prototype
    roots.append(Path("/workspace/olex2/olex2"))
    # Common local installs
    home = Path.home()
    roots.extend(
        [
            home / "olex2" / "olex2",
            home / "Olex2" / "olex2",
            Path("/usr/local/olex2"),
            Path("/opt/olex2"),
        ]
    )
    return roots


@dataclass
class CctbxRunner:
    kind: str  # "import" | "olex2c"
    olex2_root: Optional[Path] = None
    olex2c: Optional[Path] = None


def find_cctbx_runner() -> Optional[CctbxRunner]:
    """Locate a runnable cctbx (in-process import or Olex2 ``olex2c``)."""
    try:
        import cctbx  # noqa: F401
        from cctbx import dmtbx  # noqa: F401
        from smtbx.ab_initio import charge_flipping  # noqa: F401

        return CctbxRunner(kind="import")
    except Exception:
        pass

    for root in _default_olex2_roots():
        # Accept either .../olex2 (bundle root) or .../olex2/olex2
        candidates = [root]
        if (root / "olex2").is_dir():
            candidates.append(root / "olex2")
        for cand in candidates:
            exe = cand / "olex2c"
            if exe.is_file() and os.access(exe, os.X_OK):
                return CctbxRunner(kind="olex2c", olex2_root=cand, olex2c=exe)
    which = shutil.which("olex2c")
    if which:
        p = Path(which).resolve()
        return CctbxRunner(kind="olex2c", olex2_root=p.parent, olex2c=p)
    return None


def cctbx_available() -> bool:
    return find_cctbx_runner() is not None


def require_cctbx() -> CctbxRunner:
    runner = find_cctbx_runner()
    if runner is None:
        raise CctbxUnavailableError(
            "cctbx is required for --method ab_initio but was not found.\n"
            "Install options:\n"
            "  1) Use Olex2's bundled Python (recommended on this box):\n"
            "       export OLEX2_ROOT=/workspace/olex2/olex2\n"
            "       # gps-solve will call olex2c under Xvfb automatically\n"
            "  2) Activate a cctbx conda env so `import cctbx` works in this\n"
            "     interpreter, then: pip install -e '.[cctbx]'\n"
            "Prototype path: /workspace/olex2/olex2 (Python 3.8 + cctbx + smtbx)."
        )
    return runner


def _olex2_env(root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    lib_parts = [
        str(root / "lib"),
        str(root / "ilib"),
        str(root / "cctbx" / "cctbx_build" / "lib"),
        "/workspace/olex2/deps/root/usr/lib/x86_64-linux-gnu",
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in lib_parts if p)
    env["OLEX2_CCTBX_DIR"] = str(root / "cctbx")
    env["PYTHONHOME"] = str(root)
    env["BOOST_ADAPTBX_FPE_DEFAULT"] = "1"
    env["BOOST_ADAPTBX_SIGNALS_DEFAULT"] = "1"
    env["OLEX2_GL_STEREO"] = "FALSE"
    env["OLEX2_GL_MULTISAMPLE"] = "FALSE"
    env["OLEX2_GL_DEPTH_BITS"] = "16"
    env["OLEX2_GL_DEFAULT"] = "True"
    return env


def _worker_path() -> Path:
    return Path(__file__).resolve().parent / "_cctbx_ab_initio_worker.py"


def _run_via_olex2c(
    runner: CctbxRunner,
    request: Dict[str, Any],
    work_dir: Path,
    timeout_s: float,
) -> Dict[str, Any]:
    assert runner.olex2_root is not None and runner.olex2c is not None
    work_dir = Path(work_dir).resolve()
    req_path = work_dir / "request.json"
    out_path = work_dir / "result.json"
    log_path = work_dir / "olex2c.log"
    worker = _worker_path()
    shutil.copy2(worker, work_dir / "worker.py")
    req_path.write_text(json.dumps(request) + "\n")

    py = (
        "REQUEST_JSON=%r; RESULT_JSON=%r; exec(open(%r).read())"
        % (str(req_path), str(out_path), str(work_dir / "worker.py"))
    )
    py_escaped = py.replace("\\", "\\\\").replace('"', '\\"')
    cmdfile = work_dir / "cmds.txt"
    cmdfile.write_text('@py "%s"\nquit\n' % py_escaped)

    xvfb = shutil.which("xvfb-run")
    cmd: List[str]
    if xvfb:
        cmd = [xvfb, "-a", str(runner.olex2c)]
    else:
        cmd = [str(runner.olex2c)]

    with open(log_path, "w") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(runner.olex2_root),
            env=_olex2_env(runner.olex2_root),
            stdin=open(cmdfile),
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    if not out_path.exists():
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(
            "olex2c ab_initio worker produced no result.json "
            "(rc=%s). Log tail:\n%s" % (proc.returncode, tail)
        )
    result = json.loads(out_path.read_text())
    result["olex2c_returncode"] = proc.returncode
    result["olex2c_log"] = str(log_path)
    return result


def _run_via_import(request: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Execute worker body in-process when cctbx is importable."""
    req_path = work_dir / "request.json"
    out_path = work_dir / "result.json"
    req_path.write_text(json.dumps(request) + "\n")
    # Inject paths and exec worker (worker is py3.8-safe and uses only cctbx)
    g = {
        "REQUEST_JSON": str(req_path),
        "RESULT_JSON": str(out_path),
        "__name__": "__cctbx_worker__",
    }
    code = _worker_path().read_text()
    exec(compile(code, str(_worker_path()), "exec"), g)
    if not out_path.exists():
        raise RuntimeError("in-process cctbx worker wrote no result.json")
    return json.loads(out_path.read_text())


def ab_initio_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    space_group: Optional[str] = None,
    n_nonH: Optional[int] = None,
    seed: int = 0,
    time_budget_s: float = TIME_BUDGET_S,
    d_min: Optional[float] = None,
    work_dir: Optional[PathLike] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List["DensityPeak"]]:
    """
    Run tangent + smtbx CF via cctbx; keep higher own-FOM candidate.

    Returns phases, density, info, peaks (plain search, top N_nonH, 1.2 Å).
    """
    runner = require_cctbx()
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    n_atoms = int(n_nonH) if n_nonH and n_nonH > 0 else max(8, min(40, len(amp) // 40))

    cleanup = work_dir is None
    if work_dir is None:
        wdir = Path(tempfile.mkdtemp(prefix="gps_ab_initio_"))
    else:
        wdir = Path(work_dir).resolve()
        wdir.mkdir(parents=True, exist_ok=True)

    request = {
        "cell": [float(x) for x in cell.tolist()],
        "space_group": space_group or "P 1",
        "n_nonH": n_atoms,
        "seed": int(BASE_SEED + (int(seed) % 100000)),
        "time_budget_s": float(time_budget_s),
        "e_min": E_MIN,
        "e_cap": e_list_cap(n_atoms),
        "n_tangent_cycles": N_TANGENT_CYCLES,
        "min_peak_dist_A": MIN_PEAK_DIST_A,
        "sigma2_only": SIGMA2_ONLY,
        "cf": {
            "amplitude_type": CF_AMPLITUDE_TYPE,
            "max_attempts_to_get_phase_transition": CF_MAX_ATTEMPTS_PHASE_TRANSITION,
            "max_attempts_to_get_sharp_correlation_map": CF_MAX_ATTEMPTS_SHARP_CC,
            "max_solving_iterations": CF_MAX_SOLVING_ITERATIONS,
            "flipping_interval": CF_FLIPPING_INTERVAL,
        },
        "hkl": [[int(h), int(k), int(l)] for h, k, l in hkl],
        "amplitudes": [float(a) for a in amp],
    }

    t0 = time.time()
    if verbose:
        print(
            f"  ab_initio via {runner.kind}"
            + (f" ({runner.olex2_root})" if runner.olex2_root else "")
        )
    try:
        if runner.kind == "import":
            raw = _run_via_import(request, wdir)
        else:
            raw = _run_via_olex2c(
                runner, request, wdir, timeout_s=float(time_budget_s) + 180.0
            )
    except Exception:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise

    if raw.get("error"):
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError(f"ab_initio worker error: {raw['error']}")

    # Prefer worker-provided peaks; fall back to re-filtering
    peak_rows = raw.get("peaks") or []
    if peak_rows:
        sites = [p["fract"] for p in peak_rows]
        heights = [p["height"] for p in peak_rows]
        peaks = plain_peak_pick(sites, heights, cell, n_keep=n_atoms)
    else:
        peaks = []

    # Phases on input HKL if provided; else zeros (map still built from peaks via Fcalc-ish)
    phases_list = raw.get("phases_rad")
    if phases_list is not None and len(phases_list) == len(amp):
        phases = np.asarray(phases_list, dtype=np.float64)
    else:
        # Reconstruct approximate phases from peak list as C atoms (export path)
        from grok_phase_solver.physics.structure_factors import compute_structure_factors

        if peaks:
            fracs = np.array([p.fract for p in peaks], dtype=np.float64)
            elements = ["C"] * len(peaks)
            Fcalc = compute_structure_factors(hkl, fracs, elements, cell)
            phases = np.angle(Fcalc)
        else:
            phases = np.zeros(len(amp), dtype=np.float64)

    density = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, d_min=d_min
    )

    info: Dict[str, Any] = {
        "method": "ab_initio",
        "cctbx_runner": runner.kind,
        "winner": raw.get("winner"),
        "fom": raw.get("fom"),
        "candidates": raw.get("candidates"),
        "n_nonH": n_atoms,
        "defaults": {
            "E_MIN": E_MIN,
            "E_CAP": e_list_cap(n_atoms),
            "N_TANGENT_CYCLES": N_TANGENT_CYCLES,
            "TIME_BUDGET_S": time_budget_s,
            "MIN_PEAK_DIST_A": MIN_PEAK_DIST_A,
            "CF": request["cf"],
        },
        "worker": {k: raw.get(k) for k in (
            "n_trials", "n_e_used", "n_triplets", "sg", "is_centrosymmetric",
            "tangent", "charge_flip", "solver_time_s",
        )},
        "seconds": time.time() - t0,
        "work_dir": None if cleanup else str(wdir),
        "status": "ok",
    }
    if cleanup:
        shutil.rmtree(wdir, ignore_errors=True)
    return phases, density, info, peaks


__all__ = [
    "E_MIN",
    "E_CAP_PER_ATOM",
    "E_CAP_MIN",
    "N_TANGENT_CYCLES",
    "TIME_BUDGET_S",
    "MIN_PEAK_DIST_A",
    "CctbxUnavailableError",
    "e_list_cap",
    "combined_map_fom",
    "select_better_by_own_fom",
    "min_distance_ok",
    "plain_peak_pick",
    "find_cctbx_runner",
    "cctbx_available",
    "require_cctbx",
    "ab_initio_solve",
]
