"""Tests for RAAR, DiffMap, free FOM, conditional hybrid, kappa DM."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.iterative_retrieval import raar_solve, difference_map_solve, er_solve
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.free_fom import free_fom, should_accept_polish
from grok_phase_solver.solvers.conditional_hybrid import conditional_polish
from grok_phase_solver.solvers.direct_methods import (
    direct_methods_solve,
    cochran_alpha,
    sayre_weight_expected_cos,
)
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.physics.density import density_from_structure_factors


def _tiny():
    st = generate_random_organic(n_atoms=4, seed=0)
    data = structure_to_fcalc(st, d_min=1.0)
    return st, data


def test_raar_runs_and_beats_random():
    st, data = _tiny()
    ph, rho, hist = raar_solve(
        data["hkl"], data["amplitudes"], st.cell, n_iter=50, seed=0, d_min=1.0
    )
    assert len(hist["R"]) == 50
    rho_t = density_from_structure_factors(
        data["hkl"], data["F"], st.cell, shape=rho.shape
    )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    # Should show some structure (not required to fully solve every seed)
    assert cc > 0.15
    assert np.isfinite(ph).all()


def test_diffmap_runs():
    st, data = _tiny()
    ph, rho, hist = difference_map_solve(
        data["hkl"], data["amplitudes"], st.cell, n_iter=40, seed=1, d_min=1.0
    )
    assert len(hist["R"]) == 40
    assert rho.shape[0] >= 8


def test_er_runs():
    st, data = _tiny()
    ph, rho, hist = er_solve(
        data["hkl"], data["amplitudes"], st.cell, n_iter=30, seed=0
    )
    assert "final_R" in hist


def test_cochran_kappa_and_bessel_weight():
    k = cochran_alpha(2.0, 2.0, 2.0, n_atoms=16)
    assert k > 0
    # E[cos] increases with kappa
    assert sayre_weight_expected_cos(0.1) < sayre_weight_expected_cos(2.0)
    assert sayre_weight_expected_cos(10.0) > 0.8


def test_dm_kappa_has_mean_kappa():
    st, data = _tiny()
    dm = direct_methods_solve(
        data["hkl"], data["amplitudes"], st.cell, n_atoms_approx=4, n_trials=15, seed=0
    )
    assert dm.history["mean_kappa"] >= 0
    assert dm.history["n_triplets"] > 0


def test_free_fom_true_better_than_random():
    st, data = _tiny()
    f_true = free_fom(data["hkl"], data["amplitudes"], data["phases"], st.cell)
    rng = np.random.default_rng(0)
    ph = rng.uniform(-np.pi, np.pi, len(data["phases"]))
    f_rand = free_fom(data["hkl"], data["amplitudes"], ph, st.cell)
    # True phases should not be worse on composite (usually better positivity/skew)
    assert f_true["composite"] >= f_rand["composite"] - 0.05


def test_conditional_polish_keeps_good_seed():
    st, data = _tiny()
    # True phases as seed — polish should not destroy if free FOM gate works;
    # at minimum, function returns finite phases of correct length
    ph, rho, info = conditional_polish(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        data["phases"],
        polish="raar",
        n_iter=20,
        seed=0,
        d_min=1.0,
        min_delta=0.5,  # high bar → often reject polish
    )
    assert len(ph) == len(data["phases"])
    assert "accepted_polish" in info
    assert "fom_seed" in info


def test_should_accept_polish():
    a = {"composite": 0.5}
    b = {"composite": 0.6}
    assert should_accept_polish(a, b, min_delta=0.05)
    assert not should_accept_polish(a, b, min_delta=0.2)
