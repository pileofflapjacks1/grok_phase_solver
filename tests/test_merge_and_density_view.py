"""Tests for MERGE helper and density view."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.gui.density_view import central_slices, save_slice_figure
from grok_phase_solver.physics.symmetry import merge_symmetry_equivalents
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.physics.density import density_from_structure_factors


def test_merge_p1_noop_or_compress():
    st = generate_random_organic(n_atoms=5, seed=0)
    data = structure_to_fcalc(st, d_min=1.2)
    hkl, amp = data["hkl"], data["amplitudes"]
    hm, am, meta = merge_symmetry_equivalents(hkl, amp, "P 1")
    assert meta["n_in"] == len(hkl)
    assert len(hm) == len(am)
    assert meta["n_out"] <= meta["n_in"]


def test_merge_p21c_runs():
    st = generate_random_organic(n_atoms=5, seed=1)
    data = structure_to_fcalc(st, d_min=1.2)
    hm, am, meta = merge_symmetry_equivalents(
        data["hkl"], data["amplitudes"], "P 21/c"
    )
    assert len(hm) == len(am)
    # may or may not merge depending on gemmi ASU mapping
    assert meta.get("n_out", 0) >= 1 or meta.get("merged") is False


def test_density_slices(tmp_path):
    st = generate_random_organic(n_atoms=5, seed=2)
    data = structure_to_fcalc(st, d_min=1.2)
    rho = density_from_structure_factors(
        data["hkl"],
        data["amplitudes"] * np.exp(1j * data["phases"]),
        st.cell,
        d_min=1.2,
    )
    sl = central_slices(rho)
    assert set(sl.keys()) == {"xy", "xz", "yz"}
    p = save_slice_figure(rho, tmp_path / "s.png")
    assert p.exists()
