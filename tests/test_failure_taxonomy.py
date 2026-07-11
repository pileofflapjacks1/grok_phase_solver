"""Tests for solvability failure taxonomy."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.failure_taxonomy import (
    TrialRecord,
    classify_failure,
    diagnose_structure,
    information_metrics,
    summarize_taxonomy,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_information_metrics_positive():
    st = generate_random_organic(n_atoms=6, seed=0)
    data = structure_to_fcalc(st, d_min=1.0)
    info = information_metrics(
        data["hkl"], data["amplitudes"], st.cell, n_atoms=6
    )
    assert info["n_refl"] > 20
    assert info["refl_per_atom"] > 1
    assert info["mean_kappa"] >= 0


def test_classify_selection_A():
    """Good basin exists but FOM picks a worse trial."""
    trials = [
        TrialRecord("cf", 0, mapcc_oi=0.80, composite=0.50, R_pos=0.2, excess_kurtosis=5),
        TrialRecord("cf", 1, mapcc_oi=0.30, composite=0.70, R_pos=0.15, excess_kurtosis=10),
    ]
    res = classify_failure(
        trials,
        composite_true=0.85,
        composite_random=0.45,
        R_pos_true=0.1,
        R_pos_random=0.3,
        n_atoms=6,
        n_refl=400,
        mean_kappa=1.0,
        n_triplets=100,
        d_min=1.0,
    )
    assert res.primary in ("A", "A+B", "solved")
    # best CC high → if not solved label path still flags selection
    assert res.flags["fom_missed_best_cc"] or res.solved_strict


def test_classify_basin_B():
    """No good basin; true FOM much better than trials."""
    trials = [
        TrialRecord("cf", 0, mapcc_oi=0.30, composite=0.48, R_pos=0.28, excess_kurtosis=2),
        TrialRecord("raar", 1, mapcc_oi=0.25, composite=0.50, R_pos=0.30, excess_kurtosis=1),
    ]
    res = classify_failure(
        trials,
        composite_true=0.80,
        composite_random=0.45,
        R_pos_true=0.12,
        R_pos_random=0.30,
        n_atoms=12,
        n_refl=300,
        mean_kappa=0.8,
        n_triplets=80,
        d_min=1.5,
    )
    assert res.primary in ("B", "B+C")
    assert res.flags["B_basin"]
    assert not res.flags["found_good_basin"]


def test_classify_information_C():
    """Low refl/atom and true FOM not better than random."""
    trials = [
        TrialRecord("cf", 0, mapcc_oi=0.35, composite=0.52, R_pos=0.25, excess_kurtosis=3),
        TrialRecord("cf", 1, mapcc_oi=0.32, composite=0.51, R_pos=0.26, excess_kurtosis=3),
    ]
    res = classify_failure(
        trials,
        composite_true=0.50,
        composite_random=0.49,
        R_pos_true=0.28,
        R_pos_random=0.29,
        n_atoms=20,
        n_refl=80,  # rpa = 4 < 8
        mean_kappa=0.1,
        n_triplets=5,
        d_min=2.0,
    )
    assert res.flags["C_information"]
    assert res.primary in ("C", "B+C", "unknown")


def test_diagnose_structure_runs():
    st = generate_random_organic(n_atoms=5, seed=1)
    data = structure_to_fcalc(st, d_min=1.0)
    res = diagnose_structure(
        data["hkl"],
        data["amplitudes"],
        data["phases"],
        st.cell,
        n_atoms=5,
        d_min=1.0,
        structure_seed=1,
        n_starts=2,
        n_iter=25,
    )
    assert res.primary in (
        "solved", "near", "A", "B", "C", "A+B", "B+C", "unknown"
    )
    assert res.n_refl == len(data["amplitudes"])
    assert len(res.trials) == 4  # 2 methods × 2 starts
    assert res.composite_true >= 0


def test_summarize_taxonomy():
    st = generate_random_organic(n_atoms=4, seed=0)
    data = structure_to_fcalc(st, d_min=0.9)
    r1 = diagnose_structure(
        data["hkl"], data["amplitudes"], data["phases"], st.cell,
        n_atoms=4, d_min=0.9, structure_seed=0, n_starts=1, n_iter=20,
    )
    summary = summarize_taxonomy([r1])
    assert summary["n"] == 1
    assert sum(summary["counts"].values()) == 1


def test_diagnose_with_phase_init_includes_seed_trial():
    st = generate_random_organic(n_atoms=5, seed=2)
    data = structure_to_fcalc(st, d_min=1.0)
    # use true phases as "oracle seed" to verify seeded path
    res = diagnose_structure(
        data["hkl"],
        data["amplitudes"],
        data["phases"],
        st.cell,
        n_atoms=5,
        d_min=1.0,
        structure_seed=2,
        n_starts=1,
        n_iter=20,
        phase_init=data["phases"],
        init_label="oracle",
    )
    methods = {t["method"] for t in res.trials}
    assert "seed" in methods
    assert any(m.endswith("_seeded") or m in ("cf_seeded", "raar_seeded") for m in methods)
    seed_trial = next(t for t in res.trials if t["method"] == "seed")
    assert seed_trial["mapcc_oi"] > 0.9  # true seed
