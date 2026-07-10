"""Tests for Patterson, direct methods, and experimental-phasing simulators."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.experimental_phasing import (
    simulate_mir,
    simulate_mad,
    simulate_mr,
    mir_phase_indication,
    hybrid_feature_stack_mir,
)
from grok_phase_solver.physics.patterson import (
    patterson_from_amplitudes,
    autocorrelation_density,
    find_patterson_peaks,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc, run_physics_baseline
from grok_phase_solver.solvers.direct_methods import (
    normalize_E,
    build_triplets,
    direct_methods_solve,
    figure_of_merit_triplets,
)
from grok_phase_solver.solvers.patterson import patterson_density_check
from grok_phase_solver.data.synthetic_v2 import generate_fragment_structure, write_training_shard
from grok_phase_solver.models.losses import circular_phase_loss, combined_phase_loss
from pathlib import Path


def test_patterson_matches_autocorrelation():
    st = generate_random_organic(n_atoms=4, seed=11)
    data = structure_to_fcalc(st, d_min=1.2)
    check = patterson_density_check(data["hkl"], data["F"], st.cell)
    assert check["patterson_vs_autocorr_cc"] > 0.9


def test_patterson_has_origin_peak():
    st = generate_random_organic(n_atoms=5, seed=2)
    data = structure_to_fcalc(st, d_min=1.2)
    P = patterson_from_amplitudes(
        data["hkl"], data["amplitudes"], st.cell, remove_origin=False
    )
    # Origin voxel should be among the highest
    assert P[0, 0, 0] >= np.percentile(P, 99)


def test_direct_methods_runs_and_fom():
    st = generate_random_organic(n_atoms=5, seed=5)
    data = structure_to_fcalc(st, d_min=1.0)
    dm = direct_methods_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_atoms_approx=5,
        n_trials=15,
        seed=5,
        verbose=False,
    )
    assert dm.history["n_triplets"] > 0
    assert len(dm.figures_of_merit) == 15
    # True phases on strong set should have decent FOM
    true_s = data["phases"][dm.strong_idx]
    fom_true = figure_of_merit_triplets(true_s, dm.triplets)
    assert fom_true > 0.2  # strong triplets prefer cos>0


def test_baseline_direct_methods_method():
    st = generate_random_organic(n_atoms=5, seed=8)
    res = run_physics_baseline(
        st, method="direct_methods", d_min=1.0, n_iter=50, seed=8, verbose=False
    )
    assert res.n_reflections > 0
    assert "direct methods" in " ".join(res.notes).lower() or res.method == "direct_methods"


def test_baseline_patterson_method():
    st = generate_random_organic(n_atoms=4, seed=9)
    res = run_physics_baseline(
        st, method="patterson", d_min=1.2, n_iter=1, seed=9, verbose=False
    )
    assert res.history.get("n_peaks", 0) >= 1


def test_mir_mad_mr_simulators():
    st = generate_random_organic(n_atoms=8, seed=3)
    mir = simulate_mir(st, n_heavy=1, d_min=1.5, seed=3)
    assert len(mir.F_native) == len(mir.hkl)
    phase_est, fom = mir_phase_indication(mir.F_native, mir.F_derivative, mir.F_heavy)
    assert phase_est.shape == mir.phases_true.shape
    assert np.all((fom >= 0) & (fom <= 1))
    feats = hybrid_feature_stack_mir(mir)
    assert feats.shape[1] == 6

    mad = simulate_mad(st, n_sites=1, d_min=1.5, seed=3)
    assert len(mad.wavelengths) >= 2
    wl = mad.wavelengths[0]
    assert len(mad.F_plus[wl]) == len(mad.hkl)

    mr = simulate_mr(st, d_min=2.0, seed=3)
    assert np.allclose(np.abs(mr.F_model), mr.F_obs, rtol=1e-5)


def test_fragment_synth_and_shard(tmp_path: Path):
    st = generate_fragment_structure(n_fragments=2, seed=0)
    assert len(st.atoms) >= 6
    path = write_training_shard(tmp_path / "shard0.npz", n_samples=3, seed=0, d_min=1.5)
    assert path.exists()
    z = np.load(path, allow_pickle=True)
    assert len(z["amplitudes"]) == 3


def test_losses():
    p = np.array([0.0, 0.5])
    t = np.array([0.0, 0.5])
    assert circular_phase_loss(p, t) < 1e-9
    out = combined_phase_loss(p, t + 0.1, rho=np.array([-1.0, 2.0]))
    assert "total" in out and out["positivity"] > 0
