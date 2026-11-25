"""
Charge flipping must actually solve easy synthetic cells.

Scored with the strict gate (map CC, one-to-one peak recovery, peak-model
R1) against ground truth. The R-factor phase transition is checked as the
truth-free convergence signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from grok_phase_solver.data.synthetic import generate_random_organic, simulate_diffraction
from grok_phase_solver.metrics.success import evaluate_success
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

ELEMENTS = (["C"] * 7 + ["N", "O"]) * 5


def _case(n_atoms: int, seed: int, d_min: float):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, elements=ELEMENTS[:n_atoms])
    rt = simulate_diffraction(st, d_min=d_min)
    return st, rt


def _score(st, rt, phases):
    return evaluate_success(
        rt.hkl, rt.F_meas, phases, rt.phase, st.cell,
        np.array([a.fract for a in st.atoms]),
        elements=[a.element for a in st.atoms],
    )


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_cf_solves_easy_p1_cells(seed):
    st, rt = _case(12, seed, 1.0)
    ph, rho, hist = charge_flipping_solve(rt.hkl, rt.F_meas, st.cell, n_iter=2000, seed=0)
    rep = _score(st, rt, ph)
    assert rep.solved, (rep.mapcc_oi, rep.peak_recovery, rep.r1)
    assert hist["converged"]
    assert hist["transition_iter"] is not None
    assert hist["R"][-1] < 0.4 * hist["R_random_ref"]
    assert rho.shape == tuple(hist["shape"])


def test_cf_early_stop_respects_polish_budget():
    st, rt = _case(12, 100, 1.0)
    _, _, hist = charge_flipping_solve(
        rt.hkl, rt.F_meas, st.cell, n_iter=5000, seed=0, polish_iter=50
    )
    assert hist["converged"]
    assert hist["n_iter"] == hist["transition_iter"] + 51
    assert hist["n_iter_max"] == 5000


def test_cf_without_early_stop_runs_full_budget():
    st, rt = _case(8, 3, 1.2)
    _, _, hist = charge_flipping_solve(
        rt.hkl, rt.F_meas, st.cell, n_iter=150, seed=0, stop_after_transition=False
    )
    assert hist["n_iter"] == 150
    assert len(hist["R"]) == len(hist["delta"]) == len(hist["neg_frac"]) == 150


def test_cf_centrosymmetric_phases_are_signs():
    st, rt = _case(8, 4, 1.2)
    ph, _, _ = charge_flipping_solve(
        rt.hkl, rt.F_meas, st.cell, n_iter=50, seed=0, centrosymmetric=True
    )
    assert np.all(np.isclose(ph, 0.0) | np.isclose(ph, np.pi))


def test_cf_true_phase_start_stays_solved():
    """Seeded with the answer, CF must not wander off it."""
    st, rt = _case(16, 5, 1.0)
    ph, _, _ = charge_flipping_solve(
        rt.hkl, rt.F_meas, st.cell, n_iter=100, seed=0, phase_init=rt.phase
    )
    assert _score(st, rt, ph).solved
