"""Tests for GraphPhaseNet strong prior."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.models.graph_phase_net import GraphPhaseNet, prepare_graph_batch
from grok_phase_solver.models.strong_prior import (
    predict_full_phases,
    save_strong_prior,
    load_strong_prior,
    strong_prior_phaseed_solve,
    train_strong_prior,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_graph_forward_backward():
    st = generate_random_organic(n_atoms=10, seed=0)
    data = structure_to_fcalc(st, d_min=1.2)
    batch = prepare_graph_batch(
        data["hkl"], data["amplitudes"], st.cell, max_reflections=50
    )
    model = GraphPhaseNet(d_in=8, hidden=32, n_layers=2, seed=0)
    X = batch["X"]
    assert X.shape[1] == 8
    ph_s = data["phases"][batch["node_idx"]]
    loss, grads = model.loss_and_backward(
        X, batch["nbrs"], batch["wts"], ph_s, weights=batch["amp_strong"]
    )
    assert np.isfinite(loss)
    assert grads["W_in"].shape == model.W_in.shape
    model.step(grads, lr=1e-3)


def test_train_and_phaseed(tmp_path):
    model, meta = train_strong_prior(
        n_structures=5,
        n_atoms_range=(12, 14),
        d_min_range=(1.5, 1.6),
        epochs_per=8,
        epochs_refine=3,
        hidden=32,
        max_reflections=40,
        seed=1,
        verbose=False,
    )
    assert meta["architecture"] == "GraphPhaseNet"
    path = tmp_path / "sp.npz"
    save_strong_prior(model, path, meta=meta)
    m2 = load_strong_prior(path)

    st = generate_random_organic(n_atoms=12, seed=3, space_group="P1")
    data = structure_to_fcalc(st, d_min=1.5)
    ph = predict_full_phases(m2, data["hkl"], data["amplitudes"], st.cell, max_reflections=40)
    assert len(ph) == len(data["phases"])
    assert np.isfinite(ph).all()

    ph2, rho, info = strong_prior_phaseed_solve(
        data["hkl"], data["amplitudes"], st.cell, model=m2,
        n_extend=5, polish="none", n_starts=1, seed=0, d_min=1.5,
        max_reflections=40,
    )
    assert info["seed_source"] == "strong_graph_prior"
    assert rho.ndim == 3
