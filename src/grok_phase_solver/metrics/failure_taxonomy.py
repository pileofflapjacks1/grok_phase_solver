"""
Solvability failure taxonomy for ab initio phasing.

When a structure is **not** solved under strict SuccessThresholds, classify
*why* classical multistart failed. Operational (not philosophical) labels:

**A — Selection failure**
  Multistart *found* a good or better basin (high mapCC trial exists), but
  free-FOM ranking picked a worse trial — or free FOM ranks true phases
  below a wrong map (FOM inversion).

**B — Basin / optimization failure**
  Information looks adequate (true free FOM ≫ random; enough reflections;
  mean κ not tiny), yet **no** multistart trial reaches a good mapCC.
  Search never entered the correct basin; free FOM of trials stays below
  free FOM of truth.

**C — Information / underdetermination**
  Data+constraints insufficient to identify the true density among
  modulus-consistent maps: low reflections/atom, weak triplets (κ),
  free FOM of truth not clearly better than random, or high degeneracy
  (diverse low-CC trials with similar free FOM).

A case may be multi-label (e.g. A+B). Primary label uses priority
A > B > C when multiple flags fire (selection is actionable first).

References
----------
- Direct-methods reliability: Cochran κ, number of observations vs parameters
- Phase retrieval: non-convex landscape, multistart, FOMs as selection
- Project docs: free_fom.md, solvability_and_phai.md
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.direct_methods import (
    build_triplets,
    cochran_alpha,
    normalize_E,
)
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.iterative_retrieval import raar_solve


# --- thresholds (documented; tune via calibration script) ---
MAPCC_GOOD = 0.55          # "found a useful basin" (below strict 0.7 solve)
MAPCC_SOLVED = 0.70        # align with SuccessThresholds.mapcc_min
FOM_TRUE_MARGIN = 0.04     # true composite should beat random by this for "info ok"
REFL_PER_ATOM_LOW = 8.0    # below → information risk
MEAN_KAPPA_LOW = 0.35      # weak triplet reliability
FOM_INVERSION_MARGIN = 0.02  # free FOM prefers wrong over true by this
DEGEN_FOM_SPREAD = 0.05    # free FOMs clustered
DEGEN_MAPCC_DIVERSITY = 0.25  # pairwise mapCC of top trials low → diverse wrong basins


@dataclass
class TrialRecord:
    method: str
    seed: int
    mapcc_oi: float
    composite: float
    R_pos: float
    excess_kurtosis: float


@dataclass
class TaxonomyResult:
    """Classification of one structure / resolution experiment."""

    n_atoms: int
    d_min: float
    structure_seed: int
    n_refl: int
    refl_per_atom: float
    mean_kappa: float
    n_triplets: int

    mapcc_best_trial: float
    mapcc_fom_pick: float
    composite_best_trial: float
    composite_fom_pick: float
    composite_true: float
    composite_random: float
    R_pos_true: float
    R_pos_random: float

    solved_strict: bool
    primary: str  # "solved" | "A" | "B" | "C" | "A+B" | "unknown"
    flags: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    trials: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def information_metrics(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_atoms: int,
) -> Dict[str, float]:
    """Reflections/atom and Cochran κ statistics (truth-free on |F| only)."""
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    n_refl = len(amp)
    rpa = float(n_refl) / max(n_atoms, 1)
    E = normalize_E(hkl, amp, cell)
    _strong_idx, E_s, triples = build_triplets(
        hkl, E, e_min=1.0, max_triplets=5000, n_atoms_approx=float(n_atoms)
    )
    if triples:
        kappas = [float(t.kappa) for t in triples if t.kappa > 0]
        if not kappas:
            kappas = [
                cochran_alpha(E_s[t.i_h], E_s[t.i_k], E_s[t.i_hpk], float(n_atoms))
                for t in triples
            ]
        mean_k = float(np.mean(kappas)) if kappas else 0.0
        n_trip = len(triples)
    else:
        mean_k = 0.0
        n_trip = 0
    return {
        "n_refl": float(n_refl),
        "refl_per_atom": rpa,
        "mean_kappa": mean_k,
        "n_triplets": float(n_trip),
    }


def _mapcc(rho: np.ndarray, rho_true: np.ndarray) -> float:
    if rho.shape != rho_true.shape:
        # best effort: caller should match shapes
        pass
    cc, _ = map_correlation_origin_invariant(rho, rho_true)
    return float(cc)


def evaluate_multistart_trials(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases_true: np.ndarray,
    rho_true: np.ndarray,
    n_starts: int = 4,
    n_iter: int = 60,
    base_seed: int = 0,
    methods: Sequence[str] = ("cf", "raar"),
    d_min: Optional[float] = None,
) -> List[TrialRecord]:
    """Run multistart CF/RAAR; score each trial with mapCC and free FOM."""
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    trials: List[TrialRecord] = []
    tid = 0
    for method in methods:
        for s in range(n_starts):
            seed = base_seed + tid
            tid += 1
            if method == "cf":
                ph, rho, _ = charge_flipping_solve(
                    hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
                )
            elif method == "raar":
                ph, rho, _ = raar_solve(
                    hkl, amp, cell, n_iter=n_iter, seed=seed, d_min=d_min
                )
            else:
                raise ValueError(method)
            if rho.shape != rho_true.shape:
                rho = density_from_structure_factors(
                    hkl, amp * np.exp(1j * ph), cell, shape=rho_true.shape
                )
            fom = free_fom(hkl, amp, ph, cell, density=rho)
            trials.append(
                TrialRecord(
                    method=method,
                    seed=seed,
                    mapcc_oi=_mapcc(rho, rho_true),
                    composite=fom["composite"],
                    R_pos=fom["R_pos"],
                    excess_kurtosis=fom["excess_kurtosis"],
                )
            )
    return trials


def classify_failure(
    trials: Sequence[TrialRecord],
    composite_true: float,
    composite_random: float,
    R_pos_true: float,
    R_pos_random: float,
    n_atoms: int,
    n_refl: int,
    mean_kappa: float,
    n_triplets: int,
    d_min: float,
    structure_seed: int = 0,
    thresholds: Optional[SuccessThresholds] = None,
    mapcc_good: float = MAPCC_GOOD,
) -> TaxonomyResult:
    """
    Apply A/B/C rules given multistart trials and information metrics.
    """
    thr = thresholds or SuccessThresholds()
    rpa = float(n_refl) / max(n_atoms, 1)

    if not trials:
        return TaxonomyResult(
            n_atoms=n_atoms, d_min=d_min, structure_seed=structure_seed,
            n_refl=n_refl, refl_per_atom=rpa, mean_kappa=mean_kappa,
            n_triplets=n_triplets,
            mapcc_best_trial=0.0, mapcc_fom_pick=0.0,
            composite_best_trial=0.0, composite_fom_pick=0.0,
            composite_true=composite_true, composite_random=composite_random,
            R_pos_true=R_pos_true, R_pos_random=R_pos_random,
            solved_strict=False, primary="unknown",
            reasons=["no trials"],
        )

    mapccs = np.array([t.mapcc_oi for t in trials])
    comps = np.array([t.composite for t in trials])
    i_best_cc = int(np.argmax(mapccs))
    i_best_fom = int(np.argmax(comps))
    best_cc = float(mapccs[i_best_cc])
    fom_pick_cc = float(mapccs[i_best_fom])
    best_comp = float(comps[i_best_cc])
    fom_pick_comp = float(comps[i_best_fom])

    solved = best_cc >= thr.mapcc_min  # mapCC part of strict success; peak/R1 separate

    flags = {
        "info_low_rpa": rpa < REFL_PER_ATOM_LOW,
        "info_low_kappa": mean_kappa < MEAN_KAPPA_LOW or n_triplets < 10,
        "info_true_fom_weak": composite_true < composite_random + FOM_TRUE_MARGIN,
        "info_true_R_not_better": R_pos_true > R_pos_random - 0.02,
        "found_good_basin": best_cc >= mapcc_good,
        "fom_missed_best_cc": (
            best_cc >= mapcc_good
            and fom_pick_cc < best_cc - 0.05
            and comps[i_best_fom] >= comps[i_best_cc] - 1e-12
        ),
        "fom_inversion_vs_true": any(
            t.composite > composite_true + FOM_INVERSION_MARGIN and t.mapcc_oi < 0.45
            for t in trials
        ),
        "search_missed_true_fom": (
            best_cc < mapcc_good
            and fom_pick_comp < composite_true - 0.05
        ),
        "degeneracy": _degeneracy_flag(trials),
    }

    # Composite info flag
    flags["C_information"] = bool(
        flags["info_low_rpa"]
        or flags["info_low_kappa"]
        or flags["info_true_fom_weak"]
        or flags["degeneracy"]
    )
    flags["A_selection"] = bool(
        flags["fom_missed_best_cc"] or flags["fom_inversion_vs_true"]
    )
    flags["B_basin"] = bool(
        (not flags["found_good_basin"])
        and (not flags["C_information"] or flags["search_missed_true_fom"])
        and flags["search_missed_true_fom"]
    )
    # If no good basin and information looks OK (true FOM clearly better)
    if (
        not flags["found_good_basin"]
        and not flags["info_true_fom_weak"]
        and not flags["info_low_rpa"]
        and best_cc < mapcc_good
    ):
        flags["B_basin"] = True

    reasons: List[str] = []
    if flags["A_selection"]:
        if flags["fom_missed_best_cc"]:
            reasons.append(
                f"Selection: free-FOM pick mapCC={fom_pick_cc:.2f} but best trial "
                f"mapCC={best_cc:.2f} (FOM missed better basin)"
            )
        if flags["fom_inversion_vs_true"]:
            reasons.append(
                "Selection: free FOM ranks a wrong map above true phases (FOM inversion)"
            )
    if flags["B_basin"]:
        if fom_pick_comp > composite_true:
            reasons.append(
                f"Basin: best multistart mapCC={best_cc:.2f} < {mapcc_good}; "
                f"trial free FOM ({fom_pick_comp:.3f}) exceeds true ({composite_true:.3f}) "
                f"— wrong basins look better by free FOM (coupled selection issue)"
            )
        else:
            reasons.append(
                f"Basin: best multistart mapCC={best_cc:.2f} < {mapcc_good}; "
                f"true FOM={composite_true:.3f} > trial pick FOM={fom_pick_comp:.3f} "
                f"(search never reached truth-like free FOM)"
            )
    if flags["C_information"]:
        bits = []
        if flags["info_low_rpa"]:
            bits.append(f"refl/atom={rpa:.1f}<{REFL_PER_ATOM_LOW}")
        if flags["info_low_kappa"]:
            bits.append(f"mean_κ={mean_kappa:.2f}, n_triplets={n_triplets}")
        if flags["info_true_fom_weak"]:
            bits.append(
                f"true FOM ({composite_true:.3f}) ≲ random ({composite_random:.3f})"
            )
        if flags["degeneracy"]:
            bits.append("degenerate free-FOM landscape across diverse wrong maps")
        reasons.append("Information: " + "; ".join(bits))

    if solved and best_cc >= thr.mapcc_min:
        primary = "solved"
        if not reasons:
            reasons.append(f"Best mapCC={best_cc:.2f} ≥ {thr.mapcc_min} (map solved)")
        elif flags["fom_missed_best_cc"]:
            # Still solved, but note selection suboptimality among good trials
            reasons.insert(
                0,
                f"Solved (best mapCC={best_cc:.2f}); free-FOM pick mapCC="
                f"{fom_pick_cc:.2f} was suboptimal among trials",
            )
    else:
        a, b, c = flags["A_selection"], flags["B_basin"], flags["C_information"]
        # Near-solved: useful basin found but below strict mapCC threshold
        near = mapcc_good <= best_cc < thr.mapcc_min
        flags["near_solved"] = near

        if a and b:
            primary = "A+B"
        elif a:
            primary = "A"
        elif b and c:
            primary = "B+C"
        elif b:
            primary = "B"
        elif c:
            primary = "C"
        elif near and flags["fom_missed_best_cc"]:
            primary = "A"
            reasons.append(
                f"Near-solved / selection: best mapCC={best_cc:.2f} in "
                f"[{mapcc_good},{thr.mapcc_min}); FOM pick worse"
            )
        elif near:
            primary = "near"
            reasons.append(
                f"Near-solved: best mapCC={best_cc:.2f} in "
                f"[{mapcc_good},{thr.mapcc_min}) — useful basin, needs more "
                "iterations/polish/prior rather than more information alone"
            )
        else:
            primary = "unknown"
            reasons.append(
                f"Unsolved (best mapCC={best_cc:.2f}) but no clear A/B/C flag — "
                "borderline; inspect trials"
            )

    return TaxonomyResult(
        n_atoms=n_atoms,
        d_min=d_min,
        structure_seed=structure_seed,
        n_refl=n_refl,
        refl_per_atom=rpa,
        mean_kappa=mean_kappa,
        n_triplets=n_triplets,
        mapcc_best_trial=best_cc,
        mapcc_fom_pick=fom_pick_cc,
        composite_best_trial=best_comp,
        composite_fom_pick=fom_pick_comp,
        composite_true=composite_true,
        composite_random=composite_random,
        R_pos_true=R_pos_true,
        R_pos_random=R_pos_random,
        solved_strict=solved,
        primary=primary,
        flags=flags,
        reasons=reasons,
        trials=[
            {
                "method": t.method,
                "seed": t.seed,
                "mapcc_oi": t.mapcc_oi,
                "composite": t.composite,
                "R_pos": t.R_pos,
                "excess_kurtosis": t.excess_kurtosis,
            }
            for t in trials
        ],
    )


def _degeneracy_flag(trials: Sequence[TrialRecord]) -> bool:
    """True if several wrong maps share similar free FOM (flat selection landscape)."""
    if len(trials) < 3:
        return False
    # top half by free FOM
    order = np.argsort([-t.composite for t in trials])
    top = [trials[i] for i in order[: max(3, len(trials) // 2)]]
    comps = [t.composite for t in top]
    if max(comps) - min(comps) > DEGEN_FOM_SPREAD:
        return False
    # all wrong-ish
    if max(t.mapcc_oi for t in top) >= MAPCC_GOOD:
        return False
    return True


def diagnose_structure(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases_true: np.ndarray,
    cell: np.ndarray,
    n_atoms: int,
    d_min: float,
    structure_seed: int = 0,
    n_starts: int = 4,
    n_iter: int = 60,
    methods: Sequence[str] = ("cf", "raar"),
) -> TaxonomyResult:
    """
    Full pipeline: information metrics + multistart + free FOM of true/random
    + taxonomy classification.
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    hkl = np.asarray(hkl, dtype=int)
    ph_t = np.asarray(phases_true, dtype=np.float64)

    info = information_metrics(hkl, amp, cell, n_atoms)
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), cell, d_min=d_min
    )
    f_true = free_fom(hkl, amp, ph_t, cell, density=rho_t)
    rng = np.random.default_rng(structure_seed + 999)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    f_rand = free_fom(hkl, amp, ph_r, cell)

    trials = evaluate_multistart_trials(
        hkl, amp, cell, ph_t, rho_t,
        n_starts=n_starts, n_iter=n_iter, base_seed=structure_seed,
        methods=methods, d_min=d_min,
    )

    return classify_failure(
        trials,
        composite_true=f_true["composite"],
        composite_random=f_rand["composite"],
        R_pos_true=f_true["R_pos"],
        R_pos_random=f_rand["R_pos"],
        n_atoms=n_atoms,
        n_refl=int(info["n_refl"]),
        mean_kappa=info["mean_kappa"],
        n_triplets=int(info["n_triplets"]),
        d_min=d_min,
        structure_seed=structure_seed,
    )


def summarize_taxonomy(results: Sequence[TaxonomyResult]) -> Dict[str, Any]:
    """Aggregate counts and rates for a list of TaxonomyResult."""
    labels = ["solved", "near", "A", "B", "C", "A+B", "B+C", "unknown"]
    counts = {lab: 0 for lab in labels}
    for r in results:
        counts[r.primary] = counts.get(r.primary, 0) + 1
    n = len(results)
    by_region: Dict[str, Dict[str, int]] = {"easy": {}, "hard": {}, "other": {}}
    for r in results:
        if r.n_atoms <= 8 and r.d_min <= 1.0:
            reg = "easy"
        elif r.n_atoms >= 12 and r.d_min >= 1.5:
            reg = "hard"
        else:
            reg = "other"
        by_region[reg][r.primary] = by_region[reg].get(r.primary, 0) + 1

    return {
        "n": n,
        "counts": counts,
        "rates": {k: (v / n if n else 0.0) for k, v in counts.items()},
        "by_region": by_region,
        "mean_best_mapcc": float(np.mean([r.mapcc_best_trial for r in results])) if results else 0.0,
        "mean_refl_per_atom": float(np.mean([r.refl_per_atom for r in results])) if results else 0.0,
    }
