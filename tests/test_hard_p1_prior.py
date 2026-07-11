"""Tests for hard-P1 domain-matched phase prior."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.models.hard_p1_prior import (
    hard_p1_phaseed_solve,
    predict_phases_hard_p1,
    save_hard_p1_prior,
    train_hard_p1_prior,
    load_hard_p1_prior,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_train_predict_save_load(tmp_path):
    model, meta = train_hard_p1_prior(
        n_structures=6,
        n_atoms_range=(12, 14),
        d_min_range=(1.5, 1.6),
        epochs_per=12,
        epochs_refine=4,
        hidden=32,
        seed=0,
        include_bridge=True,
        verbose=False,
    )
    assert meta["mean_train_mpe"] >= 0
    assert hasattr(model, "_feat_mu")
    path = tmp_path / "hp1.npz"
    save_hard_p1_prior(model, path, meta=meta)
    m2 = load_hard_p1_prior(path)

    st = generate_random_organic(n_atoms=12, seed=99, space_group="P1")
    data = structure_to_fcalc(st, d_min=1.5)
    ph = predict_phases_hard_p1(m2, data["hkl"], data["amplitudes"], st.cell)
    assert len(ph) == len(data["phases"])
    assert np.isfinite(ph).all()


def test_hard_p1_phaseed_runs(tmp_path):
    model, meta = train_hard_p1_prior(
        n_structures=4,
        n_atoms_range=(12, 13),
        d_min_range=(1.5, 1.55),
        epochs_per=10,
        epochs_refine=3,
        hidden=24,
        seed=1,
        include_bridge=False,
        verbose=False,
    )
    st = generate_random_organic(n_atoms=12, seed=7, space_group="P1")
    data = structure_to_fcalc(st, d_min=1.5)
    ph, rho, info = hard_p1_phaseed_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        model=model,
        n_extend=6,
        polish="none",
        n_starts=1,
        seed=0,
        d_min=1.5,
    )
    assert info["seed_source"] == "hard_p1_prior"
    assert info["algorithm"] == "ai_phaseed"
    assert len(ph) == len(data["amplitudes"])
    rho_t = density_from_structure_factors(
        data["hkl"],
        data["amplitudes"] * np.exp(1j * data["phases"]),
        st.cell,
        shape=rho.shape,
    )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    # Tiny train set: only require finite finite solve path
    assert np.isfinite(cc)
