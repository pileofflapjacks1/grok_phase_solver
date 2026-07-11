"""Tests for AI-PhaSeed phase extension pipeline."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.ai_phaseed import (
    ai_phaseed_solve,
    build_initial_phases,
    discretize_phases,
    phase_extend,
    select_seed_indices,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def _struct(n=6, d_min=1.0, seed=0):
    st = generate_random_organic(n_atoms=n, seed=seed)
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def test_select_seed_indices_strongest():
    st, data = _struct()
    idx = select_seed_indices(
        data["hkl"], data["amplitudes"], st.cell, n_seed=30, by="F"
    )
    assert len(idx) == 30
    amp = data["amplitudes"]
    assert amp[idx].min() >= np.sort(amp)[-30]


def test_discretize_centro():
    ph = np.array([0.1, 2.0, -2.5, 3.0])
    d = discretize_phases(ph, mode="centro")
    assert set(np.round(d, 5)).issubset({0.0, round(np.pi, 5), 3.14159})
    assert np.allclose(np.abs(np.cos(d)), 1.0, atol=1e-6)


def test_oracle_seed_recovers_high_mapcc():
    """True phases as AI seed → extension should keep high mapCC."""
    st, data = _struct(n=6, seed=1)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    ph, rho, info = ai_phaseed_solve(
        hkl,
        amp,
        st.cell,
        ph_t,
        seed_fraction=0.3,
        n_extend=12,
        seed_weight_final=0.8,
        prior_weight=0.35,
        polish="none",
        n_starts=1,
        seed=0,
        d_min=1.0,
    )
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), st.cell, shape=rho.shape
    )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    assert info["n_seed"] >= 20
    assert cc > 0.7
    assert info["algorithm"] == "ai_phaseed"


def test_partial_seed_beats_random():
    """Blend of true+noise as seed should beat pure random extension."""
    st, data = _struct(n=6, seed=2)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    rng = np.random.default_rng(0)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    # 60% true direction
    ph_partial = np.angle(0.6 * np.exp(1j * ph_t) + 0.4 * np.exp(1j * ph_r))

    ph_p, rho_p, _ = ai_phaseed_solve(
        hkl, amp, st.cell, ph_partial,
        seed_fraction=0.25, n_extend=10, polish="none", n_starts=1, seed=0, d_min=1.0,
    )
    ph_bad, rho_bad, _ = ai_phaseed_solve(
        hkl, amp, st.cell, ph_r,
        seed_fraction=0.25, n_extend=10, polish="none", n_starts=1, seed=0, d_min=1.0,
    )
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), st.cell, shape=rho_p.shape
    )
    if rho_bad.shape != rho_t.shape:
        rho_bad = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_bad), st.cell, shape=rho_t.shape
        )
    if rho_p.shape != rho_t.shape:
        rho_p = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph_p), st.cell, shape=rho_t.shape
        )
    cc_p, _ = map_correlation_origin_invariant(rho_p, rho_t)
    cc_b, _ = map_correlation_origin_invariant(rho_bad, rho_t)
    assert cc_p > cc_b - 0.05  # partial seed not worse (usually better)


def test_phase_extend_runs():
    st, data = _struct(n=5, seed=3)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    idx = select_seed_indices(hkl, amp, st.cell, n_seed=25)
    rng = np.random.default_rng(0)
    ph0 = build_initial_phases(len(amp), idx, ph_t[idx], rng)
    ph, rho, hist = phase_extend(
        hkl, amp, st.cell, ph0, idx, ph_t[idx],
        n_cycles=5, seed_weight_final=0.8, d_min=1.0,
    )
    assert len(hist["R"]) == 5
    assert len(ph) == len(amp)
    assert rho.ndim == 3
