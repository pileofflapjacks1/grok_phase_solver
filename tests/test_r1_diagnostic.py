"""Typed-peak R1 is a side calculation. It does not feed the strict bar."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.metrics.r1_diagnostic import r1_from_typed_peaks
from grok_phase_solver.metrics.success import r1_from_peaks


def test_matched_peaks_take_deposited_elements_and_strays_stay_carbon():
    peaks = np.array(
        [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60], [0.90, 0.10, 0.20]],
        dtype=np.float64,
    )
    atoms = np.array([[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]], dtype=np.float64)
    hkl = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 1]], dtype=np.int32)
    cell = np.array([10.0, 12.0, 14.0, 90.0, 90.0, 90.0])
    f_obs = np.array([10.0, 8.0, 6.0, 4.0])
    out = r1_from_typed_peaks(
        hkl,
        f_obs,
        cell,
        peaks,
        atoms,
        ["N", "O"],
        n_atoms=3,
        origin_shift=np.zeros(3),
        tol=0.05,
    )
    assert out.elements == ["N", "O", "C"]
    assert out.n_matched == 2
    assert out.n_peaks_used == 3
    assert 0.0 <= out.r1 <= 2.0


def test_origin_shift_is_applied_before_matching():
    peaks = np.array([[0.25, 0.00, 0.00]], dtype=np.float64)
    atoms = np.array([[0.05, 0.00, 0.00]], dtype=np.float64)
    hkl = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int32)
    cell = np.array([8.0, 8.0, 8.0, 90.0, 90.0, 90.0])
    f_obs = np.ones(2)
    missed = r1_from_typed_peaks(
        hkl, f_obs, cell, peaks, atoms, ["O"],
        n_atoms=1, origin_shift=np.zeros(3), tol=0.05,
    )
    hit = r1_from_typed_peaks(
        hkl, f_obs, cell, peaks, atoms, ["O"],
        n_atoms=1, origin_shift=np.array([0.20, 0.0, 0.0]), tol=0.05,
    )
    assert missed.elements == ["C"]
    assert hit.elements == ["O"]
    assert hit.n_matched == 1


def test_typed_r1_differs_from_all_carbon_when_a_heavier_match_exists():
    peaks = np.array([[0.10, 0.20, 0.30], [0.60, 0.10, 0.40]], dtype=np.float64)
    hkl = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [2, 1, 0]],
        dtype=np.int32,
    )
    cell = np.array([10.0, 11.0, 12.0, 90.0, 100.0, 90.0])
    f_obs = np.array([12.0, 9.0, 8.0, 5.0, 4.0, 3.0])
    carbon = r1_from_peaks(hkl, f_obs, cell, peaks, n_atoms=2)
    typed = r1_from_typed_peaks(
        hkl, f_obs, cell, peaks, peaks, ["C", "S"],
        n_atoms=2, origin_shift=np.zeros(3), tol=0.05,
    )
    assert typed.elements == ["C", "S"]
    assert typed.r1 != carbon


def test_periodic_min_image_matches_without_an_extra_shift():
    peaks = np.array([[0.99, 0.50, 0.50]], dtype=np.float64)
    atoms = np.array([[0.01, 0.50, 0.50]], dtype=np.float64)
    hkl = np.array([[1, 0, 0]], dtype=np.int32)
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    out = r1_from_typed_peaks(
        hkl, np.array([1.0]), cell, peaks, atoms, ["F"],
        n_atoms=1, origin_shift=np.zeros(3), tol=0.05,
    )
    assert out.elements == ["F"]
