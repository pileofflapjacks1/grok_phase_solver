"""Solver reload and the COD benches share one d_min slack."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.io.hkl import (
    D_MIN_SLACK_A,
    ReflectionTable,
    d_min_keep_mask,
)


def test_slack_keeps_an_ulp_and_drops_a_wider_window():
    d = np.array([1.0, 1.0 - 0.5 * D_MIN_SLACK_A, 1.0 - 5 * D_MIN_SLACK_A, 0.9])
    mask = d_min_keep_mask(d, 1.0)
    assert mask.tolist() == [True, True, False, False]


def test_filter_resolution_uses_the_same_slack():
    # Cubic a = d(100). Pick a so (100) sits inside the slack under 1.0 Å.
    a = 1.0 - 0.5 * D_MIN_SLACK_A
    table = ReflectionTable(
        hkl=np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int32),
        F_meas=np.array([0.0, 4.0]),
        cell=np.array([a, a, a, 90.0, 90.0, 90.0]),
    )
    kept = table.filter_resolution(d_min=1.0)
    assert len(kept) == 1
    assert kept.hkl.tolist() == [[1, 0, 0]]
