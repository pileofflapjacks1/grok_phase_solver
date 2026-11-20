"""
Null and sanity tests for the scoring code.

Every benchmark number in data/processed/ rests on these metrics, so they must
(1) credit a perfect answer wherever the crystallographic ambiguities put it
(origin shift, hand), and (2) refuse to credit noise. These tests pin both
ends so the old failure modes cannot come back silently:

- maps were ρ(−r) (peaks of a true-phase map sat at −x);
- peak recovery passed random peak sets (fractional tolerance, many-to-one);
- strong-phase accuracy scored perfect P1 phases at a random origin like noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from grok_phase_solver.data.synthetic import generate_random_organic, simulate_diffraction
from grok_phase_solver.metrics.phase_error import (
    best_origin_shift,
    mean_phase_error_origin_invariant,
)
from grok_phase_solver.metrics.strong_seed import strong_seed_metrics
from grok_phase_solver.metrics.success import (
    evaluate_success,
    match_peaks_to_atoms,
    peak_recovery_score,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import metric_tensor
from grok_phase_solver.physics.structure_factors import (
    compute_structure_factors,
    structure_factors_from_density,
)
from grok_phase_solver.pipeline.peaks import pick_density_peaks

ELEMENTS = (["C"] * 7 + ["N", "O"]) * 5


def _case(n_atoms: int = 16, seed: int = 0, d_min: float = 1.0):
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, elements=ELEMENTS[:n_atoms])
    rt = simulate_diffraction(st, d_min=d_min)
    fr = np.array([a.fract for a in st.atoms])
    els = [a.element for a in st.atoms]
    return st, rt, fr, els


def _nearest_angstrom(points, targets, cell):
    G = metric_tensor(cell)
    d = points[:, None, :] - targets[None, :, :]
    d -= np.round(d)
    return np.sqrt(np.einsum("...i,ij,...j->...", d, G, d)).min(axis=1)


# --------------------------------------------------------------------------- #
# Sign convention
# --------------------------------------------------------------------------- #


def test_true_phase_map_peaks_sit_at_plus_x():
    st, rt, fr, _ = _case()
    rho = density_from_structure_factors(rt.hkl, rt.F_meas * np.exp(1j * rt.phase), st.cell)
    peaks = np.array([p.fract for p in pick_density_peaks(rho, n_peaks=len(fr), min_sigma=1.5)])
    d_plus = _nearest_angstrom(peaks, fr, st.cell)
    d_minus = _nearest_angstrom(peaks, -fr, st.cell)
    assert np.median(d_plus) < 0.1, "true-phase map peaks must sit on +x"
    assert np.median(d_minus) > 0.3


def test_density_and_structure_factor_transforms_are_inverses():
    st, rt, _, _ = _case(n_atoms=8)
    F = rt.F_meas * np.exp(1j * rt.phase)
    rho = density_from_structure_factors(rt.hkl, F, st.cell)
    F_grid = structure_factors_from_density(rho, _vol(st.cell))
    nx, ny, nz = rho.shape
    h, k, l = rt.hkl.T
    back = F_grid[h % nx, k % ny, l % nz]
    np.testing.assert_allclose(back, F, rtol=1e-6, atol=1e-6 * np.abs(F).max())


def _vol(cell):
    return float(np.sqrt(np.linalg.det(metric_tensor(cell))))


def test_fft_map_matches_direct_fourier_sum():
    """ρ(r) = (1/V) Σ F e^{−2πi h·r} evaluated directly at an atom site."""
    st, rt, fr, _ = _case(n_atoms=6)
    F = compute_structure_factors(rt.hkl, np.array([a.fract for a in st.atoms]),
                                  [a.element for a in st.atoms], st.cell)
    rho = density_from_structure_factors(rt.hkl, F, st.cell)
    shape = np.array(rho.shape)
    idx = np.round(fr[0] * shape).astype(int) % shape
    r = idx / shape
    V = _vol(st.cell)
    # The grid holds each of ±h once (F(−h) = F(h)*), whether or not the list
    # already contains Friedel mates; sum over that unique set.
    terms = {}
    for hv, fv in zip(rt.hkl.tolist(), F):
        terms[tuple(hv)] = fv
        terms.setdefault(tuple(-x for x in hv), np.conj(fv))
    H = np.array(list(terms.keys()), dtype=np.float64)
    Fs = np.array(list(terms.values()))
    direct = np.sum(Fs * np.exp(-2j * np.pi * (H @ r))) / V
    np.testing.assert_allclose(rho[tuple(idx)], direct.real, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Perfect answers are credited under origin shift and inversion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1])
def test_true_phases_are_strict_solved(seed):
    st, rt, fr, els = _case(seed=seed)
    rep = evaluate_success(rt.hkl, rt.F_meas, rt.phase, rt.phase, st.cell, fr, elements=els)
    assert rep.mapcc_oi > 0.99
    assert rep.peak_recovery == 1.0
    assert rep.r1 < 0.35
    assert rep.solved


@pytest.mark.parametrize("seed", [0, 1])
def test_shifted_and_inverted_perfect_phases_are_credited(seed):
    st, rt, fr, els = _case(seed=seed)
    rng = np.random.default_rng(seed + 10)
    t = rng.random(3)
    moved = -rt.phase + 2 * np.pi * (rt.hkl @ t)
    rep = evaluate_success(rt.hkl, rt.F_meas, moved, rt.phase, st.cell, fr, elements=els)
    assert rep.solved
    assert rep.peak_recovery == 1.0
    assert rep.details["peak_hand"] == -1
    mpe, _ = mean_phase_error_origin_invariant(moved, rt.phase, rt.hkl, weights=rt.F_meas)
    assert mpe < 1.0
    sm = strong_seed_metrics(moved, rt.phase, rt.hkl, rt.F_meas, st.cell)
    assert sm["frac_within_deg"] > 0.99


def test_best_origin_shift_recovers_known_translation():
    _, rt, _, _ = _case(n_atoms=12)
    t = np.array([0.137, 0.62, 0.905])
    moved = rt.phase + 2 * np.pi * (rt.hkl @ t)
    found = best_origin_shift(moved, rt.phase, rt.hkl, weights=rt.F_meas)
    d = (found - t + 0.5) % 1.0 - 0.5
    assert np.all(np.abs(d) < 1e-4)


def test_peak_matcher_credits_shifted_inverted_atoms():
    st, _, fr, _ = _case(n_atoms=20)
    rng = np.random.default_rng(3)
    peaks = (-fr + rng.random(3)) % 1.0
    jitter = rng.normal(scale=0.01, size=peaks.shape)  # ~0.1 Å
    m = match_peaks_to_atoms((peaks + jitter) % 1.0, fr, st.cell)
    assert m["recovery"] == 1.0
    assert m["hand"] == -1
    assert m["rms_angstrom"] < 0.3


# --------------------------------------------------------------------------- #
# Noise is not credited
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_atoms", [12, 20, 40])
def test_random_peaks_fail_peak_recovery(n_atoms):
    st, _, fr, _ = _case(n_atoms=n_atoms)
    rng = np.random.default_rng(n_atoms)
    rec = [
        match_peaks_to_atoms(rng.random((2 * n_atoms, 3)), fr, st.cell)["recovery"]
        for _ in range(3)
    ]
    assert max(rec) < 0.5, rec


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_phases_are_not_solved(seed):
    st, rt, fr, els = _case(seed=seed)
    rng = np.random.default_rng(seed + 100)
    ph = rng.uniform(-np.pi, np.pi, len(rt.phase))
    rep = evaluate_success(rt.hkl, rt.F_meas, ph, rt.phase, st.cell, fr, elements=els)
    assert not rep.solved
    assert rep.mapcc_oi < 0.4
    assert rep.r1 > 0.45


def test_random_phases_strong_seed_floor():
    """
    Origin-free scoring has a noise floor above 20/180 ≈ 0.11; it must stay
    well below the 30% seed bar and far below a perfect answer.
    """
    rng = np.random.default_rng(7)
    fr20 = []
    for seed in range(4):
        st, rt, _, _ = _case(n_atoms=20, seed=seed)
        ph = rng.uniform(-np.pi, np.pi, len(rt.phase))
        fr20.append(strong_seed_metrics(ph, rt.phase, rt.hkl, rt.F_meas, st.cell)["frac_within_deg"])
    assert np.mean(fr20) < 0.24, fr20
    assert max(fr20) < 0.30, fr20


def test_legacy_peak_recovery_without_cell_still_runs():
    st, _, fr, _ = _case(n_atoms=8)
    frac, shift = peak_recovery_score(fr, fr, tol=0.15, n_origin_shifts=4)
    assert frac == 1.0 and shift.shape == (3,)
    frac2, _ = peak_recovery_score(fr, fr, cell=st.cell)
    assert frac2 == 1.0
