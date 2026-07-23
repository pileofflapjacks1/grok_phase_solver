"""Tests for predicted-model seeding helpers."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.seed_import import combine_phase_seeds


def test_combine_phase_seeds_union_and_average():
    n = 40
    rng = np.random.default_rng(0)
    ph1 = rng.uniform(-np.pi, np.pi, n)
    ph2 = ph1 + rng.normal(0, 0.1, n)
    m1 = np.zeros(n, dtype=bool)
    m2 = np.zeros(n, dtype=bool)
    m1[:15] = True
    m2[10:25] = True
    comb, mask, meta = combine_phase_seeds([ph1, ph2], [m1, m2], weights=[1.0, 1.0])
    assert mask.sum() == 25  # union 0..24
    assert meta["n_sets"] == 2
    # overlap region should be near average of ph1/ph2
    overlap = m1 & m2
    assert overlap.sum() > 0
    # circular proximity
    d = np.angle(np.exp(1j * (comb[overlap] - ph1[overlap])))
    assert np.mean(np.abs(d)) < 0.2


def test_fragment_seed_from_partial_structure():
    from grok_phase_solver.solvers.seed_import import seed_from_fragment_atoms

    st = generate_random_organic(n_atoms=8, seed=1)
    data = structure_to_fcalc(st, d_min=1.0)
    fracs = np.array([a.fract for a in st.atoms[:3]], dtype=np.float64)
    els = [a.element for a in st.atoms[:3]]
    ph, mask, meta = seed_from_fragment_atoms(
        data["hkl"], data["amplitudes"], st.cell, fracs, els, seed=0
    )
    assert mask.sum() >= 5
    assert meta["n_atoms"] == 3
    assert len(ph) == len(data["amplitudes"])
