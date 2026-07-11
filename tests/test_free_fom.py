"""Tests for free-FOM v2: residual correctness, ranking, polish gate."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.conditional_hybrid import conditional_polish
from grok_phase_solver.solvers.free_fom import (
    free_fom,
    positivity_residual,
    should_accept_polish,
    compare_fom,
    phase_displacement,
)
from grok_phase_solver.solvers.projectors import (
    density_to_F,
    project_modulus,
    project_positivity,
    r_factor_moduli,
)


def _tiny(n=6, d_min=1.0, seed=0):
    st = generate_random_organic(n_atoms=n, seed=seed)
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def test_r_pos_not_vacuous_after_modulus_bugfix():
    """Old bug: R after project_modulus was always ~0. R₊ must differ."""
    st, data = _tiny()
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    rho = density_from_structure_factors(hkl, amp * np.exp(1j * ph), st.cell)

    R_pos = positivity_residual(rho, hkl, amp, st.cell)
    # vacuous (bug) path
    F = density_to_F(project_positivity(rho), hkl, st.cell)
    R_vacuous = r_factor_moduli(project_modulus(F, amp), amp)

    assert R_vacuous < 1e-10
    assert R_pos > 0.01  # true map still has some residual after positivity
    fom = free_fom(hkl, amp, ph, st.cell, density=rho)
    assert abs(fom["R_pos"] - R_pos) < 1e-12
    assert abs(fom["R_after_ER"] - R_pos) < 1e-12  # API compat name


def test_true_beats_random_on_composite_and_R_pos():
    st, data = _tiny(n=6, seed=1)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    f_true = free_fom(hkl, amp, ph_t, st.cell)
    rng = np.random.default_rng(0)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    f_rand = free_fom(hkl, amp, ph_r, st.cell)

    assert f_true["R_pos"] < f_rand["R_pos"]
    assert f_true["composite"] > f_rand["composite"]
    assert f_true["excess_kurtosis"] > f_rand["excess_kurtosis"] - 0.5
    assert f_true.get("fom_version", 0) >= 2.1
    assert "score_afa" in f_true


def test_afa_penalizes_extreme_kurtosis_score():
    """Inverted-U: moderate kurtosis scores higher than super-spike kurtosis."""
    from grok_phase_solver.solvers.free_fom import _score_kurtosis, anti_false_atomicity_score

    assert _score_kurtosis(5.0) > _score_kurtosis(25.0)
    assert _score_kurtosis(5.0) > _score_kurtosis(0.0)
    atom_spike = {
        "excess_kurtosis": 28.0,
        "max_over_sigma": 25.0,
        "n_local_maxima": 2.0,
        "peak_second_ratio": 0.1,
        "n_strong_peaks": 1.0,
    }
    atom_ok = {
        "excess_kurtosis": 6.0,
        "max_over_sigma": 9.0,
        "n_local_maxima": 8.0,
        "peak_second_ratio": 0.6,
        "n_strong_peaks": 4.0,
    }
    assert anti_false_atomicity_score(atom_ok) > anti_false_atomicity_score(atom_spike)


def test_true_beats_random_multiple_seeds():
    wins = 0
    for seed in range(5):
        st, data = _tiny(n=6, seed=seed)
        f_t = free_fom(data["hkl"], data["amplitudes"], data["phases"], st.cell)
        ph_r = np.random.default_rng(seed + 99).uniform(
            -np.pi, np.pi, len(data["amplitudes"])
        )
        f_r = free_fom(data["hkl"], data["amplitudes"], ph_r, st.cell)
        if f_t["composite"] > f_r["composite"]:
            wins += 1
    assert wins >= 4  # should almost always rank true higher


def test_should_accept_requires_composite_and_R_pos():
    good = {"composite": 0.50, "R_pos": 0.20}
    better = {"composite": 0.55, "R_pos": 0.18}
    worse_R = {"composite": 0.60, "R_pos": 0.35}  # composite up, R₊ much worse
    assert should_accept_polish(good, better, min_delta=0.02)
    assert not should_accept_polish(good, worse_R, min_delta=0.02, max_R_pos_regression=0.03)
    assert not should_accept_polish(good, {"composite": 0.51, "R_pos": 0.19}, min_delta=0.05)


def test_rewrite_trust_region_requires_large_R_pos_drop():
    """Large phase rewrite with only mild R₊ gain must be rejected."""
    seed = {"composite": 0.55, "R_pos": 0.30}
    mild = {"composite": 0.75, "R_pos": 0.25}   # dR = 0.05 < 0.08
    strong = {"composite": 0.75, "R_pos": 0.18}  # dR = 0.12 ≥ 0.08
    assert not should_accept_polish(seed, mild, phase_disp=0.7, rewrite_min_R_improve=0.08)
    assert should_accept_polish(seed, strong, phase_disp=0.7, rewrite_min_R_improve=0.08)
    # small phase change: mild R₊ gain still ok
    assert should_accept_polish(seed, mild, phase_disp=0.1, rewrite_min_R_improve=0.08)


def test_phase_displacement_identical_and_random():
    ph = np.array([0.0, 1.0, -1.0, 2.0])
    assert phase_displacement(ph, ph) < 1e-12
    rng = np.random.default_rng(0)
    ph2 = rng.uniform(-np.pi, np.pi, 200)
    ph3 = rng.uniform(-np.pi, np.pi, 200)
    # random relative phases → mean(1-cos) ≈ 1
    d = phase_displacement(ph2, ph3)
    assert 0.7 < d < 1.3


def test_gate_rejects_destroying_good_seed():
    """True phases as seed: CF polish should often be rejected (or at least not forced)."""
    st, data = _tiny(n=6, seed=2)
    ph, rho, info = conditional_polish(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        data["phases"],
        polish="charge_flipping",
        n_iter=30,
        seed=0,
        d_min=1.0,
    )
    assert "fom_delta" in info
    assert "R_pos" in info["fom_seed"]
    # If accepted, R₊ must not have regressed badly
    if info["accepted_polish"]:
        assert info["fom_polished"]["R_pos"] <= info["fom_seed"]["R_pos"] + 0.05


def test_compare_fom_and_version():
    st, data = _tiny()
    f0 = free_fom(data["hkl"], data["amplitudes"], data["phases"], st.cell)
    assert f0.get("fom_version", 0) >= 2.1
    rng = np.random.default_rng(1)
    ph = rng.uniform(-np.pi, np.pi, len(data["phases"]))
    f1 = free_fom(data["hkl"], data["amplitudes"], ph, st.cell)
    d = compare_fom(f0, f1)
    assert "delta_composite" in d
    assert d["delta_R_pos"] == f1["R_pos"] - f0["R_pos"]


def test_cf_usually_beats_random_fom():
    st, data = _tiny(n=5, seed=3)
    hkl, amp = data["hkl"], data["amplitudes"]
    ph_cf, rho_cf, _ = charge_flipping_solve(
        hkl, amp, st.cell, n_iter=50, seed=0, d_min=1.0
    )
    f_cf = free_fom(hkl, amp, ph_cf, st.cell, density=rho_cf)
    ph_r = np.random.default_rng(0).uniform(-np.pi, np.pi, len(amp))
    f_r = free_fom(hkl, amp, ph_r, st.cell)
    assert f_cf["composite"] > f_r["composite"] - 0.02
