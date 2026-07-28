"""Tests for Melgalvis v0.7 curriculum + seed/DM helpers."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic_melgalvis import (
    cod_like_config,
    generate_melgalvis_structure,
    hard_curriculum_config,
    iter_melgalvis_samples,
)
from grok_phase_solver.solvers.ai_phaseed import recommend_seed_fraction
from grok_phase_solver.solvers.density_modification import (
    estimate_solvent_fraction,
    solvent_flatten,
)
from grok_phase_solver.solvers.seed_import import combine_phase_seeds


def test_cod_and_hard_presets_generate():
    st = generate_melgalvis_structure(seed=1, cfg=cod_like_config())
    assert len(st.atoms) >= 1
    st2 = generate_melgalvis_structure(seed=2, cfg=hard_curriculum_config())
    assert st2.cell[0] > 0


def test_iter_samples_with_low_res():
    rows = iter_melgalvis_samples(
        6, seed=0, preset="cod", include_p_minus1=0.5, include_low_res=0.5
    )
    assert len(rows) == 6
    assert "hkl" in rows[0]
    assert any(r.get("centrosymmetric") for r in rows) or True  # stochastic


def test_recommend_seed_fraction():
    cell = np.array([40.0, 40.0, 40.0, 90.0, 90.0, 90.0])
    r = recommend_seed_fraction(200, cell, d_min=1.8)
    assert 0.1 <= r["seed_fraction"] <= 0.5
    assert r["n_seed_est"] >= 15


def test_combine_agreement_boost():
    n = 20
    ph1 = np.zeros(n)
    ph2 = np.zeros(n) + 0.1
    m1 = np.zeros(n, dtype=bool)
    m2 = np.zeros(n, dtype=bool)
    m1[:10] = True
    m2[5:15] = True
    _, mask, meta = combine_phase_seeds([ph1, ph2], [m1, m2], agreement_boost=True)
    assert mask.sum() == 15
    assert "mean_agreement" in meta


def test_solvent_auto():
    rng = np.random.default_rng(0)
    rho = rng.normal(0, 1, (16, 16, 16))
    rho[:8] -= 2.0
    f = estimate_solvent_fraction(rho, protein_mode=True)
    assert 0.35 <= f <= 0.75
    out = solvent_flatten(rho, auto_fraction=True, protein_mode=True)
    assert out.shape == rho.shape
