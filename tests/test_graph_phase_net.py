"""Tests for GraphPhaseNet strong prior (scaled)."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.models.graph_phase_net import (
    GraphPhaseNet,
    build_normalized_adj,
    prepare_graph_batch,
    triplet_cos_invariant,
    triplet_loss_and_grad,
)
from grok_phase_solver.models.strong_prior import (
    load_strong_prior,
    predict_full_phases,
    save_strong_prior,
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
    assert batch["adj"].shape == (X.shape[0], X.shape[0])
    ph_s = data["phases"][batch["node_idx"]]
    loss, grads = model.loss_and_backward(
        X,
        adj=batch["adj"],
        phase_true=ph_s,
        weights=batch["amp_strong"],
        edges=batch["edges"],
        edge_weight=batch["edge_weight"],
        triplet_weight=0.2,
    )
    assert np.isfinite(loss)
    assert grads["W_in"].shape == model.W_in.shape
    model.step(grads, lr=1e-3)


def test_vectorized_matches_list_adj():
    st = generate_random_organic(n_atoms=12, seed=1)
    data = structure_to_fcalc(st, d_min=1.3)
    batch = prepare_graph_batch(
        data["hkl"], data["amplitudes"], st.cell, max_reflections=40
    )
    model = GraphPhaseNet(d_in=8, hidden=24, n_layers=2, seed=2)
    X = batch["X"]
    out_a, _ = model.forward(X, adj=batch["adj"])
    out_b, _ = model.forward(X, nbrs=batch["nbrs"], wts=batch["wts"])
    assert np.allclose(out_a, out_b, atol=1e-8)


def test_triplet_invariant_identity():
    # Random unit phases: cos of true triplet invariant should match formula
    rng = np.random.default_rng(0)
    n = 8
    ph = rng.uniform(-np.pi, np.pi, n)
    c, s = np.cos(ph), np.sin(ph)
    edges = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    cos_p = triplet_cos_invariant(c, s, edges)
    cos_t = np.cos(ph[0] + ph[1] - ph[2]), np.cos(ph[3] + ph[4] - ph[5])
    assert np.allclose(cos_p, cos_t, atol=1e-10)
    loss, dout = triplet_loss_and_grad(
        np.column_stack([c, s]), edges, np.ones(2), phase_true=ph
    )
    assert loss < 1e-12
    assert dout.shape == (n, 2)


def test_build_adj_row_normalized():
    edges = np.array([[0, 1, 2]], dtype=np.int32)
    w = np.array([2.0])
    A = build_normalized_adj(3, edges, w)
    assert A.shape == (3, 3)
    # each row that has neighbors sums to 1
    for i in range(3):
        if A[i].sum() > 0:
            assert abs(A[i].sum() - 1.0) < 1e-10


def test_train_and_phaseed(tmp_path):
    model, meta = train_strong_prior(
        n_structures=5,
        n_atoms_range=(12, 14),
        d_min_range=(1.5, 1.6),
        epochs_per=6,
        epochs_refine=2,
        n_global_passes=1,
        hidden=32,
        n_layers=2,
        max_reflections=40,
        triplet_weight=0.1,
        curriculum=True,
        seed=1,
        verbose=False,
    )
    assert meta["architecture"] == "GraphPhaseNet"
    assert meta["scale"] in ("v2", "v3_seed_retarget")
    path = tmp_path / "sp.npz"
    save_strong_prior(model, path, meta=meta)
    m2 = load_strong_prior(path)
    assert m2.hidden == 32
    assert m2.n_layers == 2

    st = generate_random_organic(n_atoms=12, seed=3, space_group="P1")
    data = structure_to_fcalc(st, d_min=1.5)
    ph = predict_full_phases(
        m2, data["hkl"], data["amplitudes"], st.cell, max_reflections=40
    )
    assert len(ph) == len(data["phases"])
    assert np.isfinite(ph).all()

    ph2, rho, info = strong_prior_phaseed_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        model=m2,
        n_extend=5,
        polish="none",
        n_starts=1,
        seed=0,
        d_min=1.5,
        max_reflections=40,
    )
    assert info["seed_source"] == "strong_graph_prior"
    assert rho.ndim == 3
