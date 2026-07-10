"""Integration tests for physics baseline solvers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.solvers.baseline import run_physics_baseline, structure_to_fcalc


ROOT = Path(__file__).resolve().parents[1]
CIF_SMALL = ROOT / "data" / "raw" / "cod" / "2100301.cif"


def test_structure_to_fcalc_synthetic():
    st = generate_random_organic(n_atoms=6, seed=1)
    data = structure_to_fcalc(st, d_min=1.5)
    assert data["hkl"].shape[0] == data["amplitudes"].shape[0]
    assert data["n_atoms_cell"] >= 6
    assert np.all(data["amplitudes"] >= 0)


def test_random_baseline_poor_cc():
    """Random phases should not systematically recover the map."""
    st = generate_random_organic(n_atoms=6, seed=2)
    res = run_physics_baseline(
        st, method="random", d_min=1.5, n_iter=1, seed=2, verbose=False
    )
    # Origin-invariant MPE for random phases should be high (~90° average for continuous)
    assert res.mean_phase_error_origin_invariant_deg > 40.0


def test_charge_flipping_improves_over_random_tiny():
    """
    On a very small synthetic P1 structure with atomic-resolution data,
    CF should beat random phases in origin-invariant MPE *or* map CC.
    (Not guaranteed every seed; use a few atoms and fine d_min.)
    """
    st = generate_random_organic(n_atoms=4, seed=7, space_group="P1")
    # Force P1 packing (already)
    res_cf = run_physics_baseline(
        st, method="charge_flipping", d_min=0.9, n_iter=80, seed=7, verbose=False
    )
    res_rand = run_physics_baseline(
        st, method="random", d_min=0.9, n_iter=1, seed=7, verbose=False
    )
    better_mpe = (
        res_cf.mean_phase_error_origin_invariant_deg
        < res_rand.mean_phase_error_origin_invariant_deg - 5
    )
    better_cc = res_cf.map_cc > res_rand.map_cc + 0.05
    assert better_mpe or better_cc or res_cf.map_cc > 0.3, (
        f"CF failed to show signal: CF={res_cf.summary()} RAND={res_rand.summary()}"
    )


@pytest.mark.skipif(not CIF_SMALL.exists(), reason="COD sample not downloaded")
def test_load_cod_2100301_and_fcalc():
    from grok_phase_solver.io.cif import load_cif

    st = load_cif(CIF_SMALL)
    assert len(st.atoms) >= 10
    assert st.volume > 100
    data = structure_to_fcalc(st, d_min=1.5)
    # Expanded cell should have more atoms than asymmetric unit for P21/c
    assert data["n_atoms_cell"] >= len(st.atoms)
    assert len(data["amplitudes"]) > 100
