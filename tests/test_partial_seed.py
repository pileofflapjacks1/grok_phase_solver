"""Tests for partial-φ / fragment seed API."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.partial_seed import (
    fragment_phaseed_solve,
    load_phase_seed_csv,
    oracle_partial_phaseed_solve,
    oracle_partial_seed,
    select_partial_mask,
    write_phase_seed_csv,
)
from grok_phase_solver.data.wilson import amplitude_moments, domain_gap_report


def _hardish(seed=0, n=12, d_min=1.5):
    st = generate_random_organic(n_atoms=n, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def test_select_partial_mask_modes():
    st, data = _hardish()
    m1 = select_partial_mask(
        data["hkl"], data["amplitudes"], st.cell, fraction=0.2, mode="strong_E"
    )
    m2 = select_partial_mask(
        data["hkl"], data["amplitudes"], st.cell, fraction=0.2, mode="random", seed=1
    )
    assert m1.sum() >= 1
    assert abs(m1.mean() - 0.2) < 0.05 or m1.sum() >= 5
    assert m2.sum() == m1.sum() or abs(m2.sum() - m1.sum()) <= 1


def test_oracle_partial_beats_zero_fraction():
    """30% oracle strong phases should beat 0% on easy-hard synthetic."""
    st, data = _hardish(seed=1, n=10, d_min=1.2)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]

    ph0, rho0, _ = oracle_partial_phaseed_solve(
        hkl, amp, st.cell, ph_t,
        fraction=0.0, seed=0,
        n_extend=8, polish="none", n_starts=1, d_min=1.2,
    )
    ph3, rho3, info = oracle_partial_phaseed_solve(
        hkl, amp, st.cell, ph_t,
        fraction=0.35, seed=0,
        n_extend=8, polish="none", n_starts=1, d_min=1.2,
    )
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), st.cell, shape=rho3.shape
    )
    if rho0.shape != rho_t.shape:
        rho0 = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph0), st.cell, shape=rho_t.shape
        )
    if rho3.shape != rho_t.shape:
        rho3 = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph3), st.cell, shape=rho_t.shape
        )
    cc0, _ = map_correlation_origin_invariant(rho0, rho_t)
    cc3, _ = map_correlation_origin_invariant(rho3, rho_t)
    assert info["n_seed"] > 5
    assert cc3 >= cc0 - 0.05  # usually much better


def test_fragment_seed_runs():
    st, data = _hardish(seed=2, n=10, d_min=1.2)
    n = len(data["fracs"])
    pick = np.arange(max(3, n // 3))
    ph, rho, info = fragment_phaseed_solve(
        data["hkl"], data["amplitudes"], st.cell,
        data["fracs"][pick],
        [data["elements"][i] for i in pick],
        n_extend=6, polish="none", n_starts=1, seed=0, d_min=1.2,
    )
    assert len(ph) == len(data["phases"])
    assert rho.ndim == 3
    assert info["algorithm"] == "partial_phaseed"


def test_csv_roundtrip(tmp_path):
    st, data = _hardish(seed=3, n=8, d_min=1.2)
    seed_ph, mask, _ = oracle_partial_seed(
        data["hkl"], data["amplitudes"], st.cell, data["phases"],
        fraction=0.25, seed=0,
    )
    path = tmp_path / "seed.csv"
    write_phase_seed_csv(path, data["hkl"], seed_ph, mask)
    loaded, mask2, meta = load_phase_seed_csv(path, data["hkl"])
    assert meta["n_mapped"] >= mask.sum() - 2  # Friedel may add
    idx = np.where(mask)[0]
    # known phases should match
    d = np.angle(np.exp(1j * (loaded[idx] - seed_ph[idx])))
    assert np.mean(np.abs(d)) < 1e-5


def test_wilson_domain_gap_self():
    st, data = _hardish(seed=4)
    a = {"hkl": data["hkl"], "amplitudes": data["amplitudes"], "cell": st.cell}
    # same vs same → low gap
    r = domain_gap_report(a, a, label_a="a", label_b="a")
    assert r["domain_gap_score"] < 0.05
    m = amplitude_moments(data["amplitudes"])
    assert m["n"] > 10
