"""
Opt-in cctbx/smtbx ab initio path (tangent formula + charge flipping).

Runs (a) cctbx.dmtbx weighted tangent (multisolution) and (b) smtbx charge
flipping with Olex2-like defaults over a fixed multi-seed list, then ranks
candidates. Centrosymmetric space groups keep the method-independent E-map
check (skewness, peak contrast, peak-network fraction). Non-centrosymmetric
groups rank candidates by a refined fit: top-N peaks (N = N_nonH) as isotropic
carbon, a few least-squares cycles against Fo^2 (smtbx, in the worker), lowest
wR2 wins (R1, then E-map score, break ties). If no refinement is available the
older peak-R1 check is the fallback. Never ranks against a known model and does
not use CF's own FOM for selection.

Hard wall-clock limit: the whole phasing worker (tangent + CF + refinement) runs
in a child process group that the host kills at ``time_budget_s + grace``
(minus a small host reserve); the worker checkpoints after every candidate, so
the best result so far is returned with ``timed_out`` / a ``timeout_partial``
status, or ``AbInitioTimeoutError`` if nothing had finished.

cctbx is an optional dependency. On this project's shared box it is provided by
Olex2's bundled Python 3.8 (``olex2c``). See ``find_cctbx_runner`` /
docs/USER_GUIDE.md.
"""

from __future__ import annotations

import functools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
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
# Extra wall-clock seconds beyond time_budget_s for olex2c process kill.
# Worker self-limits first; this is the hard outer cap.
CF_SUBPROCESS_GRACE_S = 30.0
# Seconds before ``time_budget_s + grace`` at which the host kills the worker,
# leaving room for the kill, reading the checkpoint and host-side selection so
# the whole ``ab_initio_solve`` call stays within budget + grace.
HOST_KILL_RESERVE_S = 3.0
# Non-centrosymmetric refinement ranking (mirrors the worker constants).
REFINE_CYCLES = 4
REFINE_U_ISO = 0.05
REFINE_TIME_RESERVE_S = 3.0
# wall - monotonic above this many seconds ⇒ the machine was probably suspended
# (or the clock stepped) while the worker ran; reported, not acted on.
CLOCK_JUMP_SUSPECT_S = 5.0

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


class AbInitioTimeoutError(RuntimeError):
    """The worker was killed at the hard wall-clock limit with no candidate yet.

    ``.info`` carries the timing/diagnostic fields (status ``timeout_no_result``).
    """

    def __init__(self, message: str, info: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.info = dict(info or {})


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


@functools.lru_cache(maxsize=32)
def _orth_matrix_cached(cell_key: Tuple[float, ...]) -> np.ndarray:
    from grok_phase_solver.io.cif import CrystalStructure

    m = CrystalStructure(
        "t", np.asarray(cell_key, dtype=np.float64), "P1"
    ).orth_matrix
    m = np.array(m, dtype=np.float64)
    m.setflags(write=False)
    return m


def _orth_matrix(cell: np.ndarray) -> np.ndarray:
    # Cached per cell: the E-map check calls min_image_dist_A O(N^2) times, and
    # rebuilding the matrix each call cost seconds of host time on larger cells.
    return _orth_matrix_cached(
        tuple(float(x) for x in np.asarray(cell, dtype=np.float64).ravel())
    )


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


def candidate_peak_r1(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    peaks: Sequence[Any],
    n_nonH: int,
) -> float:
    """R1 of top-N peaks as carbon vs observed |F| (fitted scale).

    N = n_nonH. Uses the same residual as ``metrics.success.r1_from_peaks``
    (B = 5 Å²). Does not consult a known model.
    """
    from grok_phase_solver.metrics.success import r1_from_peaks

    if not peaks or int(n_nonH) <= 0:
        return 1.0
    fracs = []
    for p in peaks:
        if isinstance(p, dict):
            fracs.append(p["fract"])
        else:
            fracs.append(p.fract)
    return float(
        r1_from_peaks(
            np.asarray(hkl, dtype=np.float64),
            np.asarray(amplitudes, dtype=np.float64),
            np.asarray(cell, dtype=np.float64),
            np.asarray(fracs, dtype=np.float64),
            n_atoms=int(n_nonH),
        )
    )


def annotate_r_scores(
    candidates: Sequence[Dict[str, Any]],
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_nonH: int,
) -> List[Dict[str, Any]]:
    """Return copies with ``peak_r1`` filled from top-N peak R1."""
    out: List[Dict[str, Any]] = []
    for c in candidates:
        cc = dict(c)
        cc["peak_r1"] = candidate_peak_r1(
            hkl, amplitudes, cell, cc.get("peaks") or [], n_nonH
        )
        out.append(cc)
    return out


def select_better_by_r_check(
    candidates: Sequence[Dict[str, Any]],
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_nonH: int,
) -> Dict[str, Any]:
    """Non-centrosymmetric selection: lowest peak R1; E-map score tiebreak.

    Centrosymmetric selection must keep using ``select_better_by_emap_check``.
    """
    if not candidates:
        raise ValueError("no candidates to select")
    annotated = annotate_emap_scores(candidates, cell, n_nonH=n_nonH)
    annotated = annotate_r_scores(
        annotated, hkl, amplitudes, cell, n_nonH=n_nonH
    )
    return min(
        annotated,
        key=lambda c: (
            float(c.get("peak_r1", 1.0)),
            -float(c.get("emap_score", float("-inf"))),
            -float(c.get("skew", 0.0)),
        ),
    )


def has_refined_fit(candidates: Sequence[Dict[str, Any]]) -> bool:
    """True if at least one candidate carries a finite ``refined_wR2``."""
    for c in candidates:
        v = c.get("refined_wR2")
        if v is not None and math.isfinite(float(v)):
            return True
    return False


def select_better_by_refined_fit(
    candidates: Sequence[Dict[str, Any]],
    cell: np.ndarray,
    *,
    n_nonH: Optional[int] = None,
) -> Dict[str, Any]:
    """Non-centrosymmetric selection: lowest refined wR2 (SHELXD-style trial ranking).

    Each candidate's top-N peaks were refined (isotropic C, a few cycles,
    Fo^2, unit weights) in the worker; ``refined_wR2`` is the residual of that
    refined model. Ties and unrefined candidates: lower ``refined_R1``, then
    higher E-map score. Candidates without a finite ``refined_wR2`` rank last.
    Never uses a known model or CF's own FOM.
    """
    if not candidates:
        raise ValueError("no candidates to select")
    annotated = annotate_emap_scores(candidates, cell, n_nonH=n_nonH)

    def _f(c: Dict[str, Any], key: str) -> float:
        v = c.get(key)
        if v is None:
            return float("inf")
        v = float(v)
        return v if math.isfinite(v) else float("inf")

    return min(
        annotated,
        key=lambda c: (
            _f(c, "refined_wR2"),
            _f(c, "refined_R1"),
            -float(c.get("emap_score", float("-inf"))),
            -float(c.get("skew", 0.0)),
        ),
    )


def select_ab_initio_candidate(
    candidates: Sequence[Dict[str, Any]],
    cell: np.ndarray,
    *,
    n_nonH: int,
    is_centrosymmetric: bool,
    hkl: Optional[np.ndarray] = None,
    amplitudes: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Dispatch selection.

    Centrosymmetric: E-map check (unchanged). Non-centrosymmetric: refined
    wR2 when the candidates carry it, else the older peak-R1 check.
    """
    if is_centrosymmetric:
        return select_better_by_emap_check(candidates, cell, n_nonH=n_nonH)
    if has_refined_fit(candidates):
        return select_better_by_refined_fit(candidates, cell, n_nonH=n_nonH)
    if hkl is None or amplitudes is None:
        raise ValueError("hkl and amplitudes required for non-centrosymmetric R-check")
    return select_better_by_r_check(
        candidates, hkl, amplitudes, cell, n_nonH=n_nonH
    )


def iterate_until_deadline(
    iterable,
    deadline: float,
    *,
    now=time.time,
):
    """Consume *iterable* until wall-clock *deadline*; return (n_items, timed_out).

    Checks the deadline before each ``next`` so a slow step is not started
    after the budget expires. Mirrored in the cctbx worker CF loop.
    """
    n = 0
    it = iter(iterable)
    while True:
        if float(now()) >= float(deadline):
            return n, True
        try:
            next(it)
        except StopIteration:
            return n, False
        n += 1


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


def run_killable_process(
    cmd: Sequence[str],
    *,
    hard_timeout_s: float,
    cwd: Optional[PathLike] = None,
    env: Optional[Dict[str, str]] = None,
    stdin_path: Optional[PathLike] = None,
    log_path: Optional[PathLike] = None,
    poll_s: float = 0.05,
) -> Dict[str, Any]:
    """Run *cmd* in its own process group; SIGKILL the whole group at the limit.

    The limit is checked against both ``time.monotonic()`` and ``time.time()``
    every *poll_s* seconds, so a stalled or hung worker (including a grandchild
    such as Xvfb or olex2c behind ``xvfb-run``) cannot outlive it. Returns
    ``returncode``, ``timed_out``, ``wall_s``, ``monotonic_s`` and
    ``clock_jump_s`` (wall minus monotonic; a large value means the machine was
    suspended or the clock stepped while the child ran).
    """
    hard = max(0.0, float(hard_timeout_s))
    posix = os.name == "posix"
    t_wall0 = time.time()
    t_mono0 = time.monotonic()
    log_fh = open(log_path, "wb") if log_path else subprocess.DEVNULL
    in_fh = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    try:
        kwargs: Dict[str, Any] = {}
        if posix:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            list(cmd),
            cwd=None if cwd is None else str(cwd),
            env=env,
            stdin=in_fh,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
        timed_out = False
        while True:
            try:
                proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            if (
                time.monotonic() - t_mono0 >= hard
                or time.time() - t_wall0 >= hard
            ):
                timed_out = True
                try:
                    if posix:
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:  # pragma: no cover - non-POSIX fallback
                        proc.kill()
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
                break
    finally:
        if log_path:
            log_fh.close()
        if stdin_path:
            in_fh.close()
    wall_s = time.time() - t_wall0
    mono_s = time.monotonic() - t_mono0
    return {
        "returncode": proc.returncode,
        "timed_out": bool(timed_out),
        "wall_s": float(wall_s),
        "monotonic_s": float(mono_s),
        "clock_jump_s": float(wall_s - mono_s),
    }


def _read_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _run_worker(
    runner: CctbxRunner,
    request: Dict[str, Any],
    work_dir: Path,
    hard_timeout_s: float,
) -> Dict[str, Any]:
    """Run the worker as a killable child; return its (maybe partial) result.

    The worker checkpoints to ``result.json`` after every candidate, so after a
    kill the best result so far is read back. The returned dict always carries
    ``_proc`` (timing / kill info) and ``_result_found``.
    """
    work_dir = Path(work_dir).resolve()
    req_path = work_dir / "request.json"
    out_path = work_dir / "result.json"
    log_path = work_dir / "olex2c.log"
    if out_path.exists():
        out_path.unlink()  # never read a stale result from a reused work_dir
    worker = _worker_path()
    shutil.copy2(worker, work_dir / "worker.py")
    request = dict(request, checkpoint_path=str(out_path))
    req_path.write_text(json.dumps(request) + "\n")

    if runner.kind == "olex2c":
        assert runner.olex2_root is not None and runner.olex2c is not None
        py = (
            "REQUEST_JSON=%r; RESULT_JSON=%r; exec(open(%r).read())"
            % (str(req_path), str(out_path), str(work_dir / "worker.py"))
        )
        py_escaped = py.replace("\\", "\\\\").replace('"', '\\"')
        cmdfile = work_dir / "cmds.txt"
        cmdfile.write_text('@py "%s"\nquit\n' % py_escaped)
        xvfb = shutil.which("xvfb-run")
        cmd = [xvfb, "-a", str(runner.olex2c)] if xvfb else [str(runner.olex2c)]
        proc_info = run_killable_process(
            cmd,
            hard_timeout_s=hard_timeout_s,
            cwd=runner.olex2_root,
            env=_olex2_env(runner.olex2_root),
            stdin_path=cmdfile,
            log_path=log_path,
        )
    else:
        # cctbx importable in this interpreter: still a separate process, so
        # the same hard kill applies.
        proc_info = run_killable_process(
            [sys.executable, str(work_dir / "worker.py"),
             str(req_path), str(out_path)],
            hard_timeout_s=hard_timeout_s,
            cwd=work_dir,
            env=os.environ.copy(),
            log_path=log_path,
        )

    result = _read_json_or_none(out_path)
    found = result is not None
    if result is None:
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-2000:]
        except Exception:
            pass
        result = {"error": "worker produced no result.json", "log_tail": tail}
    result["olex2c_returncode"] = proc_info["returncode"]
    result["olex2c_log"] = str(log_path)
    result["_proc"] = proc_info
    result["_result_found"] = found
    return result


def run_cf_deadline_probe(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    space_group: str = "P -1",
    seed: int = 0,
    time_budget_s: float = 20.0,
    work_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Run a short real smtbx CF through ``_loop_cf_with_deadline`` (test hook)."""
    runner = require_cctbx()
    cleanup = work_dir is None
    if work_dir is None:
        wdir = Path(tempfile.mkdtemp(prefix="gps_cf_probe_"))
    else:
        wdir = Path(work_dir).resolve()
        wdir.mkdir(parents=True, exist_ok=True)
    request = {
        "action": "cf_deadline_probe",
        "cell": [float(x) for x in np.asarray(cell, dtype=np.float64).tolist()],
        "space_group": space_group,
        "seed": int(BASE_SEED + (int(seed) % 100000)),
        "time_budget_s": float(time_budget_s),
        "max_solving_iterations": 40,
        "yield_solving_interval": 5,
        "hkl": [[int(h), int(k), int(l)] for h, k, l in np.asarray(hkl, dtype=int)],
        "amplitudes": [float(a) for a in np.asarray(amplitudes, dtype=np.float64)],
    }
    try:
        raw = _run_worker(
            runner, request, wdir, hard_timeout_s=float(time_budget_s) + 60.0
        )
    finally:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
    return raw


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
    grace_s: float = CF_SUBPROCESS_GRACE_S,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List["DensityPeak"]]:
    """
    Run tangent + multi-seed smtbx CF via cctbx; select best candidate.

    Returns phases, density, info, peaks (plain search, top N_nonH, 1.2 Å).
    Centrosymmetric: E-map check (unchanged). Non-centrosymmetric: lowest
    refined wR2 (top-N peaks refined a few cycles against Fo^2), peak-R1 check
    as fallback. Not CF own FOM; never vs known model.

    Hard time limit: the worker runs in a child process group killed at
    ``time_budget_s + grace_s`` (minus ``HOST_KILL_RESERVE_S``). The best result
    so far is used (``info["timed_out"]``, ``info["status"] ==
    "timeout_partial"``); if no candidate had finished,
    ``AbInitioTimeoutError`` is raised.
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
        "refine": True,
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
    hard_timeout_s = max(
        1.0,
        float(time_budget_s) + float(grace_s) - float(HOST_KILL_RESERVE_S),
    )
    if verbose:
        print(
            f"  ab_initio via {runner.kind}"
            + (f" ({runner.olex2_root})" if runner.olex2_root else "")
        )
    try:
        raw = _run_worker(runner, request, wdir, hard_timeout_s=hard_timeout_s)
    except Exception:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise
    proc_info = dict(raw.pop("_proc", {}) or {})
    host_killed = bool(proc_info.get("timed_out"))
    raw.pop("_result_found", None)
    partial = bool(raw.get("partial"))
    timing = {
        "child_wall_s": proc_info.get("wall_s"),
        "child_monotonic_s": proc_info.get("monotonic_s"),
        "clock_jump_s": proc_info.get("clock_jump_s"),
        "hard_timeout_s": hard_timeout_s,
        "host_killed": host_killed,
        "suspend_suspected": bool(
            (proc_info.get("clock_jump_s") or 0.0) > CLOCK_JUMP_SUSPECT_S
        ),
    }

    has_cands = bool(raw.get("candidates_full") or raw.get("candidates"))
    if host_killed and not has_cands:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        info_fail = dict(
            timing, status="timeout_no_result", timed_out=True,
            seconds=time.time() - t0,
        )
        raise AbInitioTimeoutError(
            "ab_initio worker killed at the hard wall-clock limit "
            "(%.1fs; budget %.1fs + grace %.1fs - reserve %.1fs) before any "
            "candidate finished" % (
                hard_timeout_s, float(time_budget_s), float(grace_s),
                HOST_KILL_RESERVE_S),
            info_fail,
        )
    if raw.get("error") and not has_cands:
        if cleanup:
            shutil.rmtree(wdir, ignore_errors=True)
        raise RuntimeError(f"ab_initio worker error: {raw['error']}")
    if host_killed:
        status = "timeout_partial"
    elif partial:
        status = "worker_exit_partial"  # worker died after some candidates
    else:
        status = "ok"
    timed_out = host_killed

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

    is_centro = raw.get("is_centrosymmetric")
    if is_centro is None:
        from grok_phase_solver.physics.symmetry import is_centrosymmetric as _is_centro
        is_centro = bool(_is_centro(space_group))
    else:
        is_centro = bool(is_centro)

    if is_centro:
        winner = select_better_by_emap_check(annotated, cell, n_nonH=n_atoms)
        selection_name = "emap_check"
    elif has_refined_fit(annotated):
        # Also keep the old peak-R1 as a diagnostic column.
        annotated = annotate_r_scores(
            annotated, hkl, amp, cell, n_nonH=n_atoms
        )
        winner = select_better_by_refined_fit(
            annotated, cell, n_nonH=n_atoms
        )
        selection_name = "refined_wR2"
    else:
        annotated = annotate_r_scores(
            annotated, hkl, amp, cell, n_nonH=n_atoms
        )
        winner = select_better_by_r_check(
            annotated, hkl, amp, cell, n_nonH=n_atoms
        )
        selection_name = "r_check"

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
                "sensible_frac", "peak_r1", "refined_wR2", "refined_R1",
                "refine_error", "cc_peak_height",
                "had_phase_transition",
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
        "peak_r1": winner.get("peak_r1"),
        "refined_wR2": winner.get("refined_wR2"),
        "refined_R1": winner.get("refined_R1"),
        "refine": raw.get("refine"),
        "fom": winner.get("fom"),  # diagnostic only
        "selection": selection_name,
        "is_centrosymmetric": bool(is_centro),
        "emap_check_formula": (
            "score = skew_term + 0.5*contrast + 0.5*network_frac; "
            "skew_term = max(skew,0); "
            "contrast = h_median/h_max (top-N height consistency); "
            "network_frac = |LCC|/N_nonH over peaks with NN>1.1Å "
            "and edges in [1.2,1.9]Å"
        ),
        "r_check_rule": (
            None if is_centro else (
                "non-centrosymmetric: lowest refined_wR2 (top-N isotropic "
                "carbon peaks, %d least-squares cycles vs Fo^2, unit weights, "
                "smtbx); refined_R1 then emap_score break ties; fallback to "
                "lowest peak_r1 if no candidate was refined; centrosymmetric "
                "unchanged (emap_check)" % REFINE_CYCLES
            )
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
            "CF_SUBPROCESS_GRACE_S": float(grace_s),
            "HOST_KILL_RESERVE_S": HOST_KILL_RESERVE_S,
            "REFINE_CYCLES": REFINE_CYCLES,
            "REFINE_U_ISO": REFINE_U_ISO,
            "REFINE_TIME_RESERVE_S": REFINE_TIME_RESERVE_S,
            "EMAP_MIN_SEP_A": EMAP_MIN_SEP_A,
            "EMAP_BOND_LO_A": EMAP_BOND_LO_A,
            "EMAP_BOND_HI_A": EMAP_BOND_HI_A,
            "CF": request["cf"],
        },
        "worker": {k: raw.get(k) for k in (
            "n_trials", "n_e_used", "n_triplets", "sg", "is_centrosymmetric",
            "tangent", "charge_flip", "solver_time_s", "timed_out",
        )},
        "seconds": time.time() - t0,
        "work_dir": None if cleanup else str(wdir),
        "timed_out": bool(
            timed_out
            or raw.get("timed_out")
            or ((raw.get("charge_flip") or {}).get("timed_out"))
        ),
        "status": status,
        **timing,
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
    "CF_SUBPROCESS_GRACE_S",
    "HOST_KILL_RESERVE_S",
    "REFINE_CYCLES",
    "REFINE_U_ISO",
    "REFINE_TIME_RESERVE_S",
    "CLOCK_JUMP_SUSPECT_S",
    "EMAP_MIN_SEP_A",
    "EMAP_BOND_LO_A",
    "EMAP_BOND_HI_A",
    "BENCH_OUTER_SEEDS",
    "CctbxUnavailableError",
    "AbInitioTimeoutError",
    "has_refined_fit",
    "select_better_by_refined_fit",
    "run_killable_process",
    "e_list_cap",
    "combined_map_fom",
    "select_better_by_own_fom",
    "emap_check_score",
    "annotate_emap_scores",
    "select_better_by_emap_check",
    "candidate_peak_r1",
    "annotate_r_scores",
    "select_better_by_r_check",
    "select_ab_initio_candidate",
    "iterate_until_deadline",
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
    "run_cf_deadline_probe",
]
