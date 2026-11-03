"""
Opt-in cctbx/smtbx ab initio path (tangent formula + charge flipping).

Runs (a) cctbx.dmtbx weighted tangent (multisolution) and (b) smtbx charge
flipping with Olex2-like defaults over a fixed multi-seed list, then keeps the
candidate with the highest *method-independent E-map check* score (skewness,
peak contrast, peak-network fraction). Never ranks against a known model and
does not use CF's own FOM for selection.

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

# Fixed multi-seed CF offsets (K=5). Applied as set_random_seed(base + 17 + offset).
# Documented list — do not tune per dataset.
CF_SEED_LIST = (0, 1, 7, 42, 2025)
CF_N_SEEDS = len(CF_SEED_LIST)
# Seconds reserved inside TIME_BUDGET_S for the CF multi-seed block.
CF_TIME_RESERVE_S = 12.0

# Method-independent E-map check (selection; not CF own-FOM)
EMAP_MIN_SEP_A = 1.1
EMAP_BOND_LO_A = 1.2
EMAP_BOND_HI_A = 1.9

# Outer benchmark seed list (same offsets as CF_SEED_LIST for honesty reports)
BENCH_OUTER_SEEDS = (0, 1, 7, 42, 2025)

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
    """Return the candidate with the highest own FOM (diagnostic only; not used for selection)."""
    if not candidates:
        raise ValueError("no candidates to select")
    return max(candidates, key=lambda c: float(c.get(fom_key, float("-inf"))))


def _orth_matrix(cell: np.ndarray) -> np.ndarray:
    from grok_phase_solver.io.cif import CrystalStructure

    return CrystalStructure(
        "t", np.asarray(cell, dtype=np.float64), "P1"
    ).orth_matrix


def min_image_dist_A(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    cell: np.ndarray,
) -> float:
    """Minimum-image Cartesian distance (Å) between two fractional sites (P1 ±1)."""
    M = _orth_matrix(cell)
    a = np.asarray(frac_a, dtype=np.float64)
    b = np.asarray(frac_b, dtype=np.float64)
    best = float("inf")
    for ox in (-1.0, 0.0, 1.0):
        for oy in (-1.0, 0.0, 1.0):
            for oz in (-1.0, 0.0, 1.0):
                d = M @ (a - b + np.array([ox, oy, oz]))
                best = min(best, float(np.linalg.norm(d)))
    return best


def peak_contrast(heights: Sequence[float]) -> float:
    """Top-N height consistency: h_median / h_max in [0, 1]. Empty → 0.

    Near 1 means the strongest peaks have similar heights (atom-like E-map
    peaks). Near 0 means a single spike over a weak background — typical of
    noise / failed phasing. This is the peak-contrast term in emap_check_score.
    """
    h = np.asarray(list(heights), dtype=np.float64)
    if h.size == 0:
        return 0.0
    hmax = float(np.max(h))
    if hmax <= 0:
        return 0.0
    return float(np.median(h) / hmax)


def sensible_peak_fraction(
    sites_frac: Sequence[Sequence[float]],
    cell: np.ndarray,
    *,
    min_sep_A: float = EMAP_MIN_SEP_A,
) -> float:
    """Fraction of peaks whose nearest neighbour is farther than *min_sep_A*."""
    sites = [np.asarray(s, dtype=np.float64) for s in sites_frac]
    n = len(sites)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    ok = 0
    for i in range(n):
        nn = min(
            min_image_dist_A(sites[i], sites[j], cell)
            for j in range(n)
            if j != i
        )
        if nn > min_sep_A:
            ok += 1
    return ok / float(n)


def peak_network_fraction(
    sites_frac: Sequence[Sequence[float]],
    cell: np.ndarray,
    *,
    n_expected: Optional[int] = None,
    bond_lo_A: float = EMAP_BOND_LO_A,
    bond_hi_A: float = EMAP_BOND_HI_A,
    min_sep_A: float = EMAP_MIN_SEP_A,
) -> float:
    """Fraction of expected peaks in the largest 1.2–1.9 Å connected component.

    Only peaks with nearest-neighbour distance > *min_sep_A* are graph nodes
    (clumped peaks are dropped). Edges link pairs with distance in
    [*bond_lo_A*, *bond_hi_A*]. Score = |LCC| / max(n_expected, 1).
    """
    sites = [np.asarray(s, dtype=np.float64) for s in sites_frac]
    n_exp = int(n_expected) if n_expected is not None else len(sites)
    n_exp = max(n_exp, 1)
    # Keep peaks with sensible separation
    keep_idx: List[int] = []
    for i, s in enumerate(sites):
        if len(sites) == 1:
            keep_idx.append(i)
            continue
        nn = min(
            min_image_dist_A(s, sites[j], cell)
            for j in range(len(sites))
            if j != i
        )
        if nn > min_sep_A:
            keep_idx.append(i)
    m = len(keep_idx)
    if m == 0:
        return 0.0
    # Union-find for connected components
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(m):
        for b in range(a + 1, m):
            d = min_image_dist_A(sites[keep_idx[a]], sites[keep_idx[b]], cell)
            if bond_lo_A <= d <= bond_hi_A:
                union(a, b)
    counts: Dict[int, int] = {}
    for i in range(m):
        r = find(i)
        counts[r] = counts.get(r, 0) + 1
    lcc = max(counts.values()) if counts else 0
    return float(lcc) / float(n_exp)


def emap_check_score(
    skew: float,
    heights: Sequence[float],
    sites_frac: Sequence[Sequence[float]],
    cell: np.ndarray,
    *,
    n_nonH: Optional[int] = None,
) -> Dict[str, float]:
    """Method-independent E-map check used for candidate selection.

    Formula (documented; model never enters)::

        skew_term      = max(skew, 0)          # unsaturated; primary signal
        contrast       = h_median / h_max      # top-N height consistency
        network_frac   = |LCC| / N_nonH among peaks with NN > 1.1 Å
                         and edges for distances in [1.2, 1.9] Å
        score          = skew_term + 0.5 * contrast + 0.5 * network_frac

    Raw skew is the main term so a high-skew atomic map outranks a
    medium-skew spike/noise map even when the latter has a slightly denser
    accidental 1.2–1.9 Å peak graph. Not CF's own FOM.
    """
    sk = max(float(skew), 0.0)
    skew_term = sk
    contrast = peak_contrast(heights)
    n_exp = int(n_nonH) if n_nonH is not None else len(list(sites_frac))
    net = peak_network_fraction(
        sites_frac, cell, n_expected=n_exp
    )
    sens = sensible_peak_fraction(sites_frac, cell)
    score = float(skew_term + 0.5 * contrast + 0.5 * net)
    return {
        "score": score,
        "skew_term": float(skew_term),
        "contrast": float(contrast),
        "network_frac": float(net),
        "sensible_frac": float(sens),
        "skew": sk,
    }


def select_better_by_emap_check(
    candidates: Sequence[Dict[str, Any]],
    cell: np.ndarray,
    *,
    n_nonH: Optional[int] = None,
    score_key: str = "emap_score",
) -> Dict[str, Any]:
    """Pick the candidate with the highest E-map check score (never model-based).

    Expects each candidate to already carry *score_key* (from ``annotate_emap_scores``)
    or to carry skew/peaks for on-the-fly scoring.
    """
    if not candidates:
        raise ValueError("no candidates to select")
    annotated = annotate_emap_scores(candidates, cell, n_nonH=n_nonH)
    return max(
        annotated,
        key=lambda c: (
            float(c.get(score_key, float("-inf"))),
            float(c.get("skew", 0.0)),
            float(c.get("network_frac", 0.0)),
        ),
    )


def annotate_emap_scores(
    candidates: Sequence[Dict[str, Any]],
    cell: np.ndarray,
    *,
    n_nonH: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return copies of *candidates* with emap_check fields filled."""
    out: List[Dict[str, Any]] = []
    for c in candidates:
        cc = dict(c)
        peaks = cc.get("peaks") or []
        sites = [p["fract"] if isinstance(p, dict) else p.fract for p in peaks]
        heights = [
            p["height"] if isinstance(p, dict) else p.height for p in peaks
        ]
        detail = emap_check_score(
            float(cc.get("skew", 0.0)),
            heights,
            sites,
            cell,
            n_nonH=n_nonH if n_nonH is not None else cc.get("n_nonH"),
        )
        cc["emap_score"] = detail["score"]
        cc["skew_term"] = detail["skew_term"]
        cc["contrast"] = detail["contrast"]
        cc["network_frac"] = detail["network_frac"]
        cc["sensible_frac"] = detail["sensible_frac"]
        out.append(cc)
    return out


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
    Run tangent + multi-seed smtbx CF via cctbx; keep best E-map-check candidate.

    Returns phases, density, info, peaks (plain search, top N_nonH, 1.2 Å).
    Selection uses ``emap_check_score`` (not CF own FOM; never vs known model).
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
        "cf_seed_list": list(CF_SEED_LIST),
        "cf_time_reserve_s": float(CF_TIME_RESERVE_S),
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

    # Host-side selection by method-independent E-map check (not worker FOM).
    raw_cands = list(raw.get("candidates_full") or raw.get("candidates") or [])
    # Older workers may omit peaks inside candidates; fall back to top-level peaks.
    if raw_cands and not any(c.get("peaks") for c in raw_cands):
        if raw.get("peaks") and raw.get("winner"):
            for c in raw_cands:
                if c.get("name") == raw.get("winner"):
                    c = dict(c)
                    c["peaks"] = raw.get("peaks")
                    c["phases_rad"] = raw.get("phases_rad")
            raw_cands = [
                (
                    dict(c, peaks=raw.get("peaks"), phases_rad=raw.get("phases_rad"))
                    if c.get("name") == raw.get("winner")
                    else c
                )
                for c in raw_cands
            ]
    annotated = annotate_emap_scores(raw_cands, cell, n_nonH=n_atoms)
    if not annotated:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError("ab_initio worker returned no candidates")
    winner = select_better_by_emap_check(annotated, cell, n_nonH=n_atoms)

    peak_rows = winner.get("peaks") or []
    if peak_rows:
        sites = [p["fract"] for p in peak_rows]
        heights = [p["height"] for p in peak_rows]
        peaks = plain_peak_pick(sites, heights, cell, n_keep=n_atoms)
    else:
        peaks = []

    phases_list = winner.get("phases_rad")
    if phases_list is not None and len(phases_list) == len(amp):
        phases = np.asarray(phases_list, dtype=np.float64)
    else:
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

    # Compact candidate summary (no bulky phases); keep peaks for bench/method scores
    cand_summary = []
    for c in annotated:
        row = {
            k: c.get(k)
            for k in (
                "name", "method", "cf_seed", "fom", "cons", "skew",
                "emap_score", "skew_term", "contrast", "network_frac",
                "sensible_frac", "cc_peak_height", "had_phase_transition",
                "n_trials", "n_e_used", "n_triplets", "best_trial",
            )
            if k in c or c.get(k) is not None
        }
        row["n_peaks"] = len(c.get("peaks") or [])
        # Keep peaks for honest per-method scoring in the bench script
        row["peaks"] = c.get("peaks") or []
        cand_summary.append(row)

    info: Dict[str, Any] = {
        "method": "ab_initio",
        "cctbx_runner": runner.kind,
        "winner": winner.get("name"),
        "winner_method": winner.get("method") or (
            "charge_flip" if str(winner.get("name", "")).startswith("charge_flip") else winner.get("name")
        ),
        "emap_score": winner.get("emap_score"),
        "fom": winner.get("fom"),  # diagnostic only
        "selection": "emap_check",
        "emap_check_formula": (
            "score = skew_term + 0.5*contrast + 0.5*network_frac; "
            "skew_term = max(skew,0); "
            "contrast = h_median/h_max (top-N height consistency); "
            "network_frac = |LCC|/N_nonH over peaks with NN>1.1Å "
            "and edges in [1.2,1.9]Å"
        ),
        "candidates": cand_summary,
        "n_nonH": n_atoms,
        "defaults": {
            "E_MIN": E_MIN,
            "E_CAP": e_list_cap(n_atoms),
            "N_TANGENT_CYCLES": N_TANGENT_CYCLES,
            "TIME_BUDGET_S": time_budget_s,
            "MIN_PEAK_DIST_A": MIN_PEAK_DIST_A,
            "CF_SEED_LIST": list(CF_SEED_LIST),
            "CF_TIME_RESERVE_S": CF_TIME_RESERVE_S,
            "EMAP_MIN_SEP_A": EMAP_MIN_SEP_A,
            "EMAP_BOND_LO_A": EMAP_BOND_LO_A,
            "EMAP_BOND_HI_A": EMAP_BOND_HI_A,
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
    "CF_SEED_LIST",
    "CF_N_SEEDS",
    "CF_TIME_RESERVE_S",
    "EMAP_MIN_SEP_A",
    "EMAP_BOND_LO_A",
    "EMAP_BOND_HI_A",
    "BENCH_OUTER_SEEDS",
    "CctbxUnavailableError",
    "e_list_cap",
    "combined_map_fom",
    "select_better_by_own_fom",
    "emap_check_score",
    "annotate_emap_scores",
    "select_better_by_emap_check",
    "peak_contrast",
    "peak_network_fraction",
    "sensible_peak_fraction",
    "min_image_dist_A",
    "min_distance_ok",
    "plain_peak_pick",
    "find_cctbx_runner",
    "cctbx_available",
    "require_cctbx",
    "ab_initio_solve",
]
