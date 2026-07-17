"""Tests for strong-seed metrics and retargeted training hooks."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.strong_seed import (
    full_and_strong_metrics,
    select_strong_indices,
    strong_seed_metrics,
)
from grok_phase_solver.models.strong_prior import train_strong_prior
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_oracle_strong_seed_metrics():
    st = generate_random_organic(n_atoms=10, seed=0)
    data = structure_to_fcalc(st, d_min=1.2)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    # perfect phases
    m = strong_seed_metrics(ph, ph, hkl, amp, st.cell, fraction=0.3, within_deg=20.0)
    assert m["strong_mpe_oi"] < 1.0
    assert m["frac_within_deg"] > 0.95
    assert m["would_seed_solve"] is True


def test_random_seed_fails_bar():
    st = generate_random_organic(n_atoms=10, seed=1)
    data = structure_to_fcalc(st, d_min=1.2)
    hkl, amp, ph = data["hkl"], data["amplitudes"], data["phases"]
    rng = np.random.default_rng(0)
    pr = rng.uniform(-np.pi, np.pi, len(ph))
    m = strong_seed_metrics(pr, ph, hkl, amp, st.cell, fraction=0.3)
    assert m["frac_within_deg"] < 0.30
    assert m["would_seed_solve"] is False


def test_select_strong_count():
    st = generate_random_organic(n_atoms=8, seed=2)
    data = structure_to_fcalc(st, d_min=1.2)
    idx = select_strong_indices(
        data["hkl"], data["amplitudes"], st.cell, fraction=0.25
    )
    assert len(idx) >= 10 or len(idx) == len(data["hkl"])


def test_train_v3_smoke():
    model, meta = train_strong_prior(
        n_structures=4,
        n_atoms_range=(12, 14),
        d_min_range=(1.5, 1.6),
        epochs_per=4,
        epochs_refine=2,
        n_global_passes=1,
        hidden=32,
        n_layers=2,
        max_reflections=40,
        triplet_weight=0.1,
        wilson_match=False,
        within_weight=0.2,
        verbose=False,
        seed=0,
    )
    assert str(meta["scale"]).startswith("v")
    assert "mean_train_strong_mpe_oi" in meta or meta.get("train_strong_mpe_oi") is not None
    assert model.hidden == 32
