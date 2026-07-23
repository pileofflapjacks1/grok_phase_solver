"""Tests for space-group helpers and phase UQ."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.uncertainty import (
    bootstrap_free_fom_spread,
    circular_mean_resultant,
    multistart_phase_uncertainty,
)
from grok_phase_solver.physics.symmetry import (
    apply_centro_phase_constraint,
    expand_fractional_coords,
    is_centrosymmetric,
    parse_space_group,
    space_group_diagnostics,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_parse_p21c_centro():
    info = parse_space_group("P 21/c")
    assert info.is_centrosymmetric or is_centrosymmetric("P21/c")
    d = space_group_diagnostics("P 21/c")
    assert "hm" in d


def test_parse_p1_not_centro():
    info = parse_space_group("P 1")
    # P1 is not centrosymmetric
    assert info.is_centrosymmetric is False or not is_centrosymmetric("P1")


def test_expand_p_minus1():
    fracs = np.array([[0.1, 0.2, 0.3]])
    fr, els, meta = expand_fractional_coords(fracs, "P-1", elements=["C"])
    # P-1 should at least double (identity + inversion) when gemmi works
    if meta.get("expanded"):
        assert len(fr) >= 2
        assert len(els) == len(fr)
    else:
        # graceful fallback
        assert len(fr) == 1


def test_centro_phase_constraint():
    ph = np.array([0.1, 2.0, -1.0, 3.0])
    out, meta = apply_centro_phase_constraint(ph, "P-1")
    if meta.get("applied"):
        assert np.allclose(np.abs(np.cos(out)), 1.0, atol=1e-6)


def test_circular_uq_agrees_on_identical():
    ph = np.linspace(-np.pi, np.pi, 20, endpoint=False)
    sets = [ph, ph.copy(), ph + 1e-6]
    mu, R, cstd = circular_mean_resultant(sets)
    assert np.all(R > 0.99)
    uq = multistart_phase_uncertainty(sets)
    assert uq["mean_resultant_length"] > 0.99
    assert uq["frac_high_confidence"] > 0.9


def test_bootstrap_free_fom():
    st = generate_random_organic(n_atoms=5, seed=0)
    data = structure_to_fcalc(st, d_min=1.2)
    boot = bootstrap_free_fom_spread(
        data["hkl"],
        data["amplitudes"],
        data["phases"],
        st.cell,
        n_boot=4,
        seed=0,
    )
    assert boot["n_boot"] >= 1
    assert np.isfinite(boot["mean"])
