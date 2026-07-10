"""Unit tests for I/O and physics core."""

from __future__ import annotations

import numpy as np
import pytest

from grok_phase_solver.io.cif import AtomSite, CrystalStructure, expand_asymmetric_unit
from grok_phase_solver.physics.form_factors import atomic_form_factor
from grok_phase_solver.physics.reciprocal import d_spacing, generate_hkl
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.metrics.phase_error import mean_phase_error, wrap_phase
from grok_phase_solver.metrics.map_cc import map_correlation
from grok_phase_solver.data.synthetic import generate_random_organic, simulate_diffraction


def test_form_factor_carbon_at_zero():
    f0 = atomic_form_factor("C", np.array([0.0]))[0]
    # f(0) should be near Z=6
    assert 5.5 < f0 < 6.5


def test_d_spacing_cubic():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    d = d_spacing(hkl, cell)
    np.testing.assert_allclose(d[0], 10.0, rtol=1e-6)
    np.testing.assert_allclose(d[1], 10.0 / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(d[2], 10.0 / np.sqrt(3), rtol=1e-6)


def test_structure_factor_single_atom_at_origin():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    hkl = generate_hkl(cell, d_min=2.0, expand_friedel=True)
    fracs = np.array([[0.0, 0.0, 0.0]])
    F = compute_structure_factors(hkl, fracs, ["C"], cell, b_isos=np.array([0.0]))
    # Phase should be ~0 for atom at origin; |F| ~ f(s)
    assert np.allclose(np.angle(F), 0.0, atol=1e-6)
    assert np.all(np.abs(F) > 0)


def test_friedel_conjugate():
    cell = np.array([8.0, 9.0, 10.0, 90.0, 95.0, 90.0])
    hkl = np.array([[1, 2, 3], [-1, -2, -3]])
    fracs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    F = compute_structure_factors(hkl, fracs, ["C", "O"], cell)
    # F(-h) = F(h)* for real density (no anomalous)
    np.testing.assert_allclose(F[1], np.conj(F[0]), rtol=1e-5, atol=1e-5)


def test_density_roundtrip_energy():
    """Parseval-ish: dense embedding should give real density with finite energy."""
    cell = np.array([6.0, 6.0, 6.0, 90.0, 90.0, 90.0])
    hkl = generate_hkl(cell, d_min=1.5)
    fracs = np.array([[0.2, 0.3, 0.4]])
    F = compute_structure_factors(hkl, fracs, ["C"], cell)
    rho = density_from_structure_factors(hkl, F, cell, d_min=1.5, sampling=4.0)
    assert rho.shape[0] >= 8
    assert np.isfinite(rho).all()
    # Atom near 0.2,0.3,0.4 should produce a peak in that region
    assert rho.max() > rho.mean()


def test_phase_error_wrap():
    assert abs(mean_phase_error(np.array([0.0]), np.array([0.0]))) < 1e-6
    # 350 deg vs 10 deg → 20 deg error
    err = mean_phase_error(np.deg2rad([350.0]), np.deg2rad([10.0]))
    assert 19 < err < 21


def test_map_cc_identical():
    x = np.random.randn(10, 10, 10)
    assert abs(map_correlation(x, x) - 1.0) < 1e-9
    assert abs(map_correlation(x, -x) + 1.0) < 1e-9


def test_synthetic_and_simulate():
    st = generate_random_organic(n_atoms=8, seed=42)
    assert len(st.atoms) == 8
    assert st.volume > 0
    table = simulate_diffraction(st, d_min=1.5, noise_level=0.05, completeness=0.9, seed=42)
    assert len(table) > 10
    assert table.F_meas is not None
    assert np.all(table.F_meas >= 0)


def test_expand_p1():
    st = CrystalStructure(
        name="t",
        cell=np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
        space_group_hm="P 1",
        atoms=[AtomSite("C1", "C", np.array([0.1, 0.2, 0.3]))],
    )
    fracs, els, _, _ = expand_asymmetric_unit(st)
    assert len(els) == 1
    np.testing.assert_allclose(fracs[0], [0.1, 0.2, 0.3])
