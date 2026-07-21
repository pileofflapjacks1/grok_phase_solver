"""Tests for Carrozzini-style seed quality predictor."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.seed_quality import (
    extract_seed_features,
    label_class_from_oracle,
    oracle_seed_metrics,
    predict_seed_quality,
)
from grok_phase_solver.solvers.ai_phaseed import select_seed_indices
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def _data(n=6, seed=0, d_min=1.0):
    st = generate_random_organic(n_atoms=n, seed=seed)
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def test_predict_oracle_seed_is_class1ish():
    st, data = _data(seed=1)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    idx = select_seed_indices(hkl, amp, st.cell, seed_fraction=0.3)
    rep = predict_seed_quality(
        hkl, amp, st.cell, ph, seed_idx=idx, d_min=1.0, n_atoms_user=6
    )
    assert "predicted_class" in rep
    assert 0.0 <= rep["success_probability"] <= 1.0
    assert "max_W" in rep["features"]
    assert "Vol" in rep["features"]
    # True phases should look better than random
    rng = np.random.default_rng(0)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    rep_r = predict_seed_quality(
        hkl, amp, st.cell, ph_r, seed_idx=idx, d_min=1.0, n_atoms_user=6
    )
    assert rep["success_probability"] >= rep_r["success_probability"] - 0.05
    assert rep["predicted_mpe_deg"] <= rep_r["predicted_mpe_deg"] + 5.0


def test_oracle_metrics_and_label():
    st, data = _data(seed=2)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    idx = select_seed_indices(hkl, amp, st.cell, n_seed=30)
    m = oracle_seed_metrics(ph, ph, idx, amplitudes=amp)
    assert m["MPE_seed_deg"] < 1.0
    assert m["CORR_seed"] > 0.99
    assert label_class_from_oracle(m["MPE_seed_deg"], m["CORR_seed"]) == 1
    assert label_class_from_oracle(80.0, 0.05) == 0


def test_extract_features_keys():
    st, data = _data(seed=3)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    feats = extract_seed_features(hkl, amp, st.cell, ph, d_min=1.0)
    for k in ("max_W", "N_asym", "Vol", "seed_fraction", "free_fom_composite"):
        assert k in feats
