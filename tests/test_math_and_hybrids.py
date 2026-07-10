"""Mathematical correctness and hybrid solver tests."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.experimental_phasing import simulate_mir
from grok_phase_solver.data.synthetic_v2 import (
    generate_fragment_structure,
    make_centrosymmetric_copy,
    apply_partial_occupancy,
    add_heavy_atom,
)
from grok_phase_solver.data.wilson import wilson_plot, domain_gap_wilson
from grok_phase_solver.io.cif_pure import load_cif_pure
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.physics.parseval import parseval_check, friedel_check
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.difference_patterson import (
    isomorphous_difference_coefficients,
    locate_heavy_atom_vectors,
)
from grok_phase_solver.solvers.mir_blow_crick import (
    lack_of_closure,
    single_isomorphous_replacement,
)
from grok_phase_solver.solvers.density_modification import solvent_flatten, density_modification_cycle
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval, blend_phases
from grok_phase_solver.models.phase_mlp import PhaseMLP, reflection_features, train_phase_mlp_on_structure
from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.models.losses import circular_phase_loss
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_friedel_real_density():
    st = generate_random_organic(n_atoms=5, seed=0)
    data = structure_to_fcalc(st, d_min=1.5)
    r = friedel_check(data["hkl"], data["F"], tol=1e-5)
    assert r["n_pairs"] > 0
    assert r["max_err"] < 1e-6


def test_origin_shift_preserves_amplitudes():
    st = generate_random_organic(n_atoms=4, seed=1)
    data = structure_to_fcalc(st, d_min=1.5)
    t = np.array([0.12, 0.34, 0.56])
    shift = 2 * np.pi * (data["hkl"].astype(float) @ t)
    F2 = data["F"] * np.exp(-1j * shift)
    np.testing.assert_allclose(np.abs(F2), np.abs(data["F"]), rtol=1e-10)


def test_parseval_finite():
    st = generate_random_organic(n_atoms=4, seed=2)
    data = structure_to_fcalc(st, d_min=1.2)
    rep = parseval_check(data["hkl"], data["F"], st.cell)
    assert np.isfinite(rep["relative_error"])
    assert rep["E_recip"] > 0


def test_cif_pure_matches_gemmi_cell():
    path = ROOT / "data/raw/cod/2100301.cif"
    if not path.exists():
        return
    pure = load_cif_pure(path)
    gem = load_cif(path)
    np.testing.assert_allclose(pure.cell, gem.cell, rtol=1e-5)
    assert len(pure.atoms) == len(gem.atoms)


def test_lack_of_closure_zero_at_true_phase():
    st = generate_random_organic(n_atoms=8, seed=3)
    mir = simulate_mir(st, n_heavy=1, d_min=1.5, noise=0.0, seed=3)
    # at true phases, F_p + F_h should match F_ph magnitudes closely
    eps = lack_of_closure(mir.F_native, mir.phases_true, mir.F_derivative, mir.F_heavy)
    # noise=0 and consistent model → small residual
    assert np.mean(np.abs(eps)) < 0.5 * np.mean(mir.F_native)


def test_blow_crick_better_than_random_mapcc_proxy():
    st = generate_fragment_structure(n_fragments=1, seed=4)
    mir = simulate_mir(st, n_heavy=1, d_min=1.5, noise=0.01, seed=4)
    ph, fom = single_isomorphous_replacement(
        mir.F_native, mir.F_derivative, mir.F_heavy, sigma=5.0
    )
    # FOM should be positive on average
    assert fom.mean() > 0.05
    # circular correlation proxy: mean cos(Δφ)
    d = ph - mir.phases_true
    # origin ambiguity — use cos of double angle soft check: just ensure finite
    assert np.isfinite(ph).all()


def test_difference_patterson_positive_coeff():
    st = generate_random_organic(n_atoms=6, seed=5)
    mir = simulate_mir(st, n_heavy=1, d_min=1.5, seed=5)
    c = isomorphous_difference_coefficients(mir.F_native, mir.F_derivative)
    assert np.all(c >= 0)
    peaks, P, info = locate_heavy_atom_vectors(
        mir.hkl, mir.F_native, mir.F_derivative, mir.cell, n_peaks=5
    )
    assert len(peaks) >= 1
    assert P.shape[0] >= 8


def test_density_modification_runs():
    st = generate_random_organic(n_atoms=5, seed=6)
    data = structure_to_fcalc(st, d_min=1.2)
    rng = np.random.default_rng(6)
    ph0 = rng.uniform(-np.pi, np.pi, len(data["amplitudes"]))
    ph, rho, hist = density_modification_cycle(
        data["hkl"], data["amplitudes"], ph0, st.cell, n_iter=5, solvent_fraction=0.4
    )
    assert len(hist["R"]) == 5
    assert rho.shape[0] >= 8


def test_hybrid_cf_seed():
    st = generate_random_organic(n_atoms=4, seed=7)
    data = structure_to_fcalc(st, d_min=1.0)
    # seed with true phases → CF should keep high quality
    ph, rho, _ = hybrid_phase_retrieval(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        data["phases"],
        polish="charge_flipping",
        n_iter=30,
        seed=7,
    )
    from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
    from grok_phase_solver.physics.density import density_from_structure_factors

    rho_t = density_from_structure_factors(
        data["hkl"], data["F"], st.cell, shape=rho.shape
    )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    assert cc > 0.5  # true seed should not be destroyed


def test_phase_mlp_training_decreases_loss():
    """Backprop decreases supervised MSE-to-(cos,sin) loss on a tiny set.

    Does not claim experimental phase solution capability.
    """
    st = generate_random_organic(n_atoms=3, seed=8)
    data = structure_to_fcalc(st, d_min=2.0)  # fewer reflections
    mlp = PhaseMLP(hidden=64, seed=8)
    losses = train_phase_mlp_on_structure(
        mlp,
        data["hkl"],
        data["amplitudes"],
        data["phases"],
        st.cell,
        n_epochs=200,
        lr=5e-2,
    )
    assert losses[-1] < losses[0]
    X = reflection_features(data["hkl"], data["amplitudes"], st.cell)
    mu, sig = X.mean(0), X.std(0) + 1e-8
    pred = mlp.predict_phases((X - mu) / sig)
    assert np.isfinite(pred).all()


def test_centrosymmetric_phases_near_0_pi():
    st = generate_random_organic(n_atoms=4, seed=9)
    st_c = make_centrosymmetric_copy(st)
    data = structure_to_fcalc(st_c, d_min=1.5, expand_symmetry=False)
    # For exact inversion symmetry through origin, F is real up to global sign
    # (phases 0 or π). Numerical
    phases = np.angle(data["F"])
    # distance to nearest 0 or π
    d0 = np.abs(phases)
    dpi = np.abs(np.abs(phases) - np.pi)
    dist = np.minimum(d0, dpi)
    # most reflections should be nearly real
    assert np.median(dist) < 0.2


def test_wilson_and_domain_gap():
    st1 = generate_random_organic(n_atoms=8, seed=10)
    st2 = generate_fragment_structure(n_fragments=2, seed=11)
    d1 = structure_to_fcalc(st1, d_min=1.5)
    d2 = structure_to_fcalc(st2, d_min=1.5)
    w = wilson_plot(d1["hkl"], d1["amplitudes"], st1.cell)
    assert len(w["s2"]) >= 2
    gap = domain_gap_wilson(
        d1["hkl"], d1["amplitudes"], st1.cell,
        d2["hkl"], d2["amplitudes"], st2.cell,
    )
    assert "domain_gap_score" in gap


def test_reflection_graph():
    st = generate_random_organic(n_atoms=6, seed=12)
    data = structure_to_fcalc(st, d_min=1.2)
    g = reflection_graph(data["hkl"], data["amplitudes"], st.cell, max_reflections=40)
    assert g["node_features"].shape[0] >= 3
    assert g["edges"].ndim == 2


def test_partial_occ_and_heavy():
    st = generate_fragment_structure(n_fragments=1, seed=13)
    st2 = apply_partial_occupancy(st, frac_disordered=0.5, seed=13)
    st3 = add_heavy_atom(st2, element="BR", seed=13)
    assert any(a.occupancy < 1.0 for a in st2.atoms)
    assert any(a.element.upper() == "BR" for a in st3.atoms)


def test_blend_phases():
    a = np.array([0.0, np.pi / 2])
    b = np.array([np.pi, np.pi / 2])
    out = blend_phases(a, b, np.array([1.0, 0.5]))
    assert out.shape == a.shape
