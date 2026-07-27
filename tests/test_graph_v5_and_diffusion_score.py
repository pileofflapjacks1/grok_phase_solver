"""Tests for GraphPhaseNet v5 features and diffusion score net."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.models.diffusion_score import (
    PhaseScoreNet,
    apply_score_step,
    reflection_score_features,
    train_score_on_structures,
)
from grok_phase_solver.models.graph_phase_net import (
    node_features_from_graph,
    prepare_graph_batch,
)
from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_v5_features_dim():
    st = generate_random_organic(n_atoms=6, seed=0)
    data = structure_to_fcalc(st, d_min=1.2)
    g = reflection_graph(data["hkl"], data["amplitudes"], st.cell, max_reflections=40)
    X4 = node_features_from_graph(g, data["hkl"], data["amplitudes"], st.cell, feature_version=4)
    X5 = node_features_from_graph(g, data["hkl"], data["amplitudes"], st.cell, feature_version=5)
    assert X4.shape[1] == 10
    assert X5.shape[1] == 14
    assert X5.shape[0] == X4.shape[0]


def test_prepare_graph_batch_v5():
    st = generate_random_organic(n_atoms=6, seed=1)
    data = structure_to_fcalc(st, d_min=1.2)
    batch = prepare_graph_batch(
        data["hkl"], data["amplitudes"], st.cell, max_reflections=50, feature_version=5
    )
    assert batch["d_in"] == 14
    assert batch["adj"].shape[0] == batch["X"].shape[0]


def test_score_net_train_and_step():
    net, meta = train_score_on_structures(
        n_structures=4, epochs_per=2, hidden=32, seed=0, max_refl=40, verbose=False
    )
    assert meta["n_structures"] == 4
    st = generate_random_organic(n_atoms=5, seed=2)
    data = structure_to_fcalc(st, d_min=1.2)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    ph2 = apply_score_step(hkl, amp, st.cell, ph, net, t=0.5, step_size=0.1)
    assert len(ph2) == len(ph)
    X = reflection_score_features(hkl, amp, st.cell, ph, 0.3)
    s = net.predict_score(X)
    assert s.shape == (len(ph), 2)
