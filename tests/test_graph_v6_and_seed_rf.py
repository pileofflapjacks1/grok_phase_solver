"""Tests for GraphPhaseNet v6 features, seed RF helpers, SG aliases (v0.8)."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.models.graph_phase_net import (
    node_features_from_graph,
    prepare_graph_batch,
)
from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.physics.symmetry import normalize_space_group_name, parse_space_group
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.ai_phaseed import recommend_seed_fraction


def test_graph_v6_feature_dim():
    st = generate_random_organic(n_atoms=10, seed=3)
    data = structure_to_fcalc(st, d_min=1.2)
    g = reflection_graph(
        data["hkl"], data["amplitudes"], st.cell, max_reflections=80
    )
    X5 = node_features_from_graph(
        g, data["hkl"], data["amplitudes"], st.cell, feature_version=5
    )
    X6 = node_features_from_graph(
        g, data["hkl"], data["amplitudes"], st.cell, feature_version=6
    )
    assert X5.shape[1] == 14
    assert X6.shape[1] == 18
    assert np.isfinite(X6).all()


def test_prepare_graph_batch_v6():
    st = generate_random_organic(n_atoms=8, seed=7)
    data = structure_to_fcalc(st, d_min=1.3)
    batch = prepare_graph_batch(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        max_reflections=60,
        feature_version=6,
    )
    assert batch["d_in"] == 18
    assert batch["feature_version"] == 6
    assert batch["X"].shape[1] == 18
    assert batch["adj"].shape[0] == batch["X"].shape[0]


def test_sg_aliases():
    assert "21/c" in normalize_space_group_name("P21/c").lower().replace(" ", "")
    info = parse_space_group("P212121")
    assert info.available is True or info.hm  # gemmi or fallback
    info2 = parse_space_group("Pbca")
    assert info2.hm


def test_recommend_seed_fraction_v08():
    cell = np.array([10.0, 12.0, 14.0, 90.0, 100.0, 90.0])
    r = recommend_seed_fraction(200, cell=cell, d_min=1.6, n_asym=100.0)
    assert 0.10 <= r["seed_fraction"] <= 0.50
    assert r["method"].startswith("carrozzini_heuristic")
    r2 = recommend_seed_fraction(200, cell=cell, d_min=0.9, has_fragment_seed=True)
    assert r2["seed_fraction"] <= r["seed_fraction"] + 0.05


def test_seed_quality_rf_train_matrix_optional():
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    except Exception:
        return  # optional dep
    from grok_phase_solver.metrics.seed_quality import (
        DEFAULT_RF_FEATURE_NAMES,
        train_seed_quality_rf_from_matrix,
        save_seed_quality_rf,
        load_seed_quality_rf,
        predict_seed_quality,
    )

    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, len(DEFAULT_RF_FEATURE_NAMES)))
    # make class somewhat separable on max_W column
    y = (X[:, 0] > 0).astype(int)
    clf, meta = train_seed_quality_rf_from_matrix(X, y, seed=0)
    assert meta["accuracy"] >= 0.0
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rf.joblib"
        save_seed_quality_rf(clf, p, feature_names=DEFAULT_RF_FEATURE_NAMES, meta=meta)
        bundle = load_seed_quality_rf(p)
        assert bundle is not None
        st = generate_random_organic(n_atoms=6, seed=1)
        data = structure_to_fcalc(st, d_min=1.5)
        rep = predict_seed_quality(
            data["hkl"],
            data["amplitudes"],
            st.cell,
            data["phases"],
            model_path=p,
            use_sklearn=True,
        )
        assert rep["method"] in ("sklearn_rf", "heuristic")
        assert "success_probability" in rep


def test_trial_complete_research():
    from grok_phase_solver.pipeline.trial_complete import complete_trial_from_density
    from grok_phase_solver.physics.density import density_from_structure_factors

    st = generate_random_organic(n_atoms=6, seed=2)
    data = structure_to_fcalc(st, d_min=1.4)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    rho = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph), st.cell, d_min=1.4
    )
    fr, els, ph2, meta = complete_trial_from_density(
        hkl, amp, st.cell, rho, ph, n_peaks=12, n_cycles=1
    )
    assert meta["research_only"] is True
    assert len(els) == len(fr)
    assert len(ph2) == len(ph)


def test_se3_stub_features():
    from grok_phase_solver.models.diffusion_se3_stub import (
        reciprocal_invariant_features,
        se3_score_step_stub,
        se3_diffusion_available,
    )

    st = generate_random_organic(n_atoms=5, seed=4)
    data = structure_to_fcalc(st, d_min=1.5)
    assert se3_diffusion_available() is False
    X = reciprocal_invariant_features(
        data["hkl"], data["amplitudes"], st.cell, data["phases"], t=0.3
    )
    assert X.shape[1] == 8
    ph2, meta = se3_score_step_stub(
        data["hkl"], data["amplitudes"], st.cell, data["phases"], t=0.5
    )
    assert meta.get("status") == "research_only" or meta.get("research_only")
    assert len(ph2) == len(data["phases"])
