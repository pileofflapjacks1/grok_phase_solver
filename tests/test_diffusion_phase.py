"""Tests for experimental diffusion phase hybrid."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.models.diffusion_phase import (
    conditional_diffusion_complete,
    diffusion_hybrid_solve,
    reverse_diffusion_phases,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def _data(n=6, seed=0, d_min=1.0):
    st = generate_random_organic(n_atoms=n, seed=seed)
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def test_reverse_diffusion_oracle_seed_keeps_mapcc():
    st, data = _data(seed=1)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    ph, rho, hist = reverse_diffusion_phases(
        hkl,
        amp,
        st.cell,
        seed_phases=ph_t,
        n_steps=8,
        seed=0,
        d_min=1.0,
        sigma_max=0.4,
    )
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), st.cell, shape=rho.shape
    )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    assert hist["algorithm"] == "diffusion_phase_langevin"
    assert len(hist["R"]) >= 8
    assert cc > 0.6


def test_diffusion_hybrid_runs():
    st, data = _data(seed=2)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    # partial seed
    rng = np.random.default_rng(0)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    ph_p = np.angle(0.6 * np.exp(1j * ph_t) + 0.4 * np.exp(1j * ph_r))
    ph, rho, info = diffusion_hybrid_solve(
        hkl,
        amp,
        st.cell,
        seed_phases=ph_p,
        n_steps=6,
        n_starts=1,
        seed=0,
        d_min=1.0,
        polish="none",
    )
    assert info["algorithm"] == "diffusion_hybrid"
    assert info["status"] == "experimental"
    assert len(ph) == len(amp)
    assert rho.ndim == 3


def test_conditional_diffusion_complete_compat():
    st, data = _data(seed=3)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    ph, hist = conditional_diffusion_complete(
        hkl, amp, st.cell, ph_t, n_steps=5, seed=0, d_min=1.0
    )
    assert len(ph) == len(amp)
    assert "algorithm" in hist or "status" in hist
