"""Tests for strict success metrics and PhAI fair packing."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.success import evaluate_success, SuccessThresholds, peak_recovery_score
from grok_phase_solver.models.phai_fair import (
    merge_reflections_phai,
    reindex_monoclinic,
    pack_phai_amplitudes_fair,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve


def test_true_phases_count_as_solved():
    st = generate_random_organic(n_atoms=5, seed=0)
    data = structure_to_fcalc(st, d_min=1.0)
    rep = evaluate_success(
        data["hkl"],
        data["amplitudes"],
        data["phases"],
        data["phases"],
        st.cell,
        data["fracs"],
        elements=data["elements"],
        thresholds=SuccessThresholds(require_r1=True),
    )
    assert rep.mapcc_oi > 0.99
    assert rep.peak_recovery >= 0.5
    assert rep.solved


def test_random_phases_not_solved():
    st = generate_random_organic(n_atoms=5, seed=1)
    data = structure_to_fcalc(st, d_min=1.0)
    rng = np.random.default_rng(1)
    ph = rng.uniform(-np.pi, np.pi, len(data["phases"]))
    rep = evaluate_success(
        data["hkl"],
        data["amplitudes"],
        ph,
        data["phases"],
        st.cell,
        data["fracs"],
        thresholds=SuccessThresholds(),
    )
    assert not rep.solved
    assert rep.mapcc_oi < 0.7


def test_cf_often_solves_tiny():
    st = generate_random_organic(n_atoms=4, seed=2)
    data = structure_to_fcalc(st, d_min=0.9)
    ph, rho, _ = charge_flipping_solve(
        data["hkl"], data["amplitudes"], st.cell, n_iter=60, seed=2, d_min=0.9, verbose=False
    )
    rep = evaluate_success(
        data["hkl"],
        data["amplitudes"],
        ph,
        data["phases"],
        st.cell,
        data["fracs"],
        density=rho,
    )
    # Soft: mapCC should be high even if R1 peak criterion is strict
    assert rep.mapcc_oi > 0.5


def test_reindex_and_merge_max_scale():
    hkl = np.array([[1, -1, 0], [1, 1, 0], [1, 1, 0], [2, 0, 1]])
    fabs = np.array([1.0, 2.0, 4.0, 10.0])
    H, F = merge_reflections_phai(hkl, fabs)
    assert np.max(F) == 1.0  # scaled by max
    assert len(H) <= len(hkl)


def test_pack_fair_meta():
    hkl = np.array([[1, 0, 0], [0, 1, 0], [2, 1, 1], [-1, 0, 0]])
    amp = np.array([5.0, 3.0, 1.0, 5.0])
    grid, ordered, idx_map, hkl_arr, meta = pack_phai_amplitudes_fair(hkl, amp)
    assert meta["normalize"].startswith("max")
    assert meta["n_ordered"] == 1205
    assert len(idx_map) == 4
