"""GraphPhaseNet v7 features, Carrozzini bin loss, Melgalvis multi-frag (v0.9)."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.synthetic_melgalvis import (
    actas2026_config,
    generate_melgalvis_structure,
)
from grok_phase_solver.models.graph_phase_net import (
    node_features_from_graph,
    phase_bin_cross_entropy,
    prepare_graph_batch,
)
from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.density_modification import estimate_solvent_fraction
from grok_phase_solver.metrics.seed_quality import extract_seed_features


def test_v7_feature_dim():
    st = generate_random_organic(n_atoms=10, seed=11)
    data = structure_to_fcalc(st, d_min=1.2)
    g = reflection_graph(data["hkl"], data["amplitudes"], st.cell, max_reflections=70)
    X6 = node_features_from_graph(
        g, data["hkl"], data["amplitudes"], st.cell, feature_version=6
    )
    X7 = node_features_from_graph(
        g, data["hkl"], data["amplitudes"], st.cell, feature_version=7
    )
    assert X6.shape[1] == 18
    assert X7.shape[1] == 22
    assert np.isfinite(X7).all()


def test_prepare_batch_v7():
    st = generate_random_organic(n_atoms=8, seed=3)
    data = structure_to_fcalc(st, d_min=1.3)
    b = prepare_graph_batch(
        data["hkl"], data["amplitudes"], st.cell, max_reflections=50, feature_version=7
    )
    assert b["d_in"] == 22
    assert b["feature_version"] == 7
    assert b["adj"].shape[0] == b["X"].shape[0]


def test_phase_bin_ce_centro_and_bins():
    n = 40
    rng = np.random.default_rng(0)
    ph_t = rng.choice([0.0, np.pi], size=n)
    # perfect (cos,sin) for true phases
    out = np.column_stack([np.cos(ph_t), np.sin(ph_t)])
    loss, dout = phase_bin_cross_entropy(out, ph_t, mode="centro")
    assert loss < 0.5
    assert dout.shape == out.shape
    ph_t2 = rng.uniform(-np.pi, np.pi, size=n)
    out2 = np.column_stack([np.cos(ph_t2), np.sin(ph_t2)])
    loss2, _ = phase_bin_cross_entropy(out2, ph_t2, n_bins=4, mode="bins")
    assert np.isfinite(loss2)


def test_seed_features_v09():
    st = generate_random_organic(n_atoms=8, seed=5)
    data = structure_to_fcalc(st, d_min=1.4)
    feats = extract_seed_features(
        data["hkl"], data["amplitudes"], st.cell, data["phases"], d_min=1.4
    )
    assert "seed_bin_entropy" in feats
    assert "seed_mean_abs_cos" in feats
    assert 0.0 <= feats["seed_bin_entropy"] <= 1.0 + 1e-6


def test_melgalvis_acta2026_and_multi_frag():
    cfg = actas2026_config(p_multi_fragment=1.0, multi_frag_n_extra=(4, 5))
    st = generate_melgalvis_structure(seed=42, cfg=cfg)
    assert len(st.atoms) >= 4
    assert st.cell is not None


def test_solvent_estimate_volume_prior():
    rng = np.random.default_rng(0)
    rho = rng.normal(size=(16, 16, 16))
    f0 = estimate_solvent_fraction(rho, protein_mode=False)
    f1 = estimate_solvent_fraction(
        rho, protein_mode=True, volume=8000.0, n_atoms_approx=100.0
    )
    assert 0.15 <= f0 <= 0.6
    assert f1 >= 0.40
