"""Unit tests for ab_initio helpers; live cctbx path skipped when unavailable."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# Import pipeline.peaks before solvers package to avoid a pre-existing
# solvers ↔ pipeline circular import during collection.
import grok_phase_solver.pipeline.peaks  # noqa: F401

from grok_phase_solver.solvers.ab_initio_cctbx import (
    BASE_SEED,
    E_CAP_MIN,
    E_CAP_PER_ATOM,
    E_MIN,
    MIN_PEAK_DIST_A,
    N_TANGENT_CYCLES,
    TIME_BUDGET_S,
    CctbxUnavailableError,
    cctbx_available,
    combined_map_fom,
    e_list_cap,
    find_cctbx_runner,
    min_distance_ok,
    plain_peak_pick,
    require_cctbx,
    select_better_by_own_fom,
)


def test_e_list_cap_matches_prototype():
    assert e_list_cap(22) == max(E_CAP_MIN, E_CAP_PER_ATOM * 22)
    assert e_list_cap(1) == E_CAP_MIN
    assert e_list_cap(0) == E_CAP_MIN
    assert e_list_cap(34) == 340


def test_budget_and_tangent_defaults_frozen():
    assert E_MIN == 1.2
    assert N_TANGENT_CYCLES == 12
    assert TIME_BUDGET_S == 60.0
    assert MIN_PEAK_DIST_A == 1.2
    assert BASE_SEED == 20251102


def test_combined_map_fom_monotonic_in_skew():
    f0 = combined_map_fom(1.0, 0.0)
    f1 = combined_map_fom(1.0, 2.0)
    f2 = combined_map_fom(1.0, 5.0)
    assert f0 < f1 < f2
    # consistency scales the tanh term
    assert combined_map_fom(0.5, 2.0) < combined_map_fom(1.0, 2.0)


def test_select_better_by_own_fom_ignores_model_fields():
    a = {"name": "tangent", "fom": 0.9, "true_phase_mean_cos": 0.1}
    b = {"name": "charge_flip", "fom": 1.1, "true_phase_mean_cos": 0.99}
    # Must pick by fom only — even if b has better true-phase match that field is ignored
    assert select_better_by_own_fom([a, b])["name"] == "charge_flip"
    assert select_better_by_own_fom([b, a])["name"] == "charge_flip"
    c = {"name": "tangent", "fom": 1.2, "true_phase_mean_cos": 0.0}
    assert select_better_by_own_fom([a, b, c])["name"] == "tangent"


def test_select_better_empty_raises():
    with pytest.raises(ValueError):
        select_better_by_own_fom([])


def test_plain_peak_search_min_distance():
    # Cubic 10 Å cell: 0.2 frac = 2 Å. With min_dist 1.2 Å, second peak at 0.05 frac
    # (0.5 Å) must be rejected; peak at 0.2 frac kept.
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    sites = [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],  # too close
        [0.25, 0.0, 0.0],  # ok (~2.5 Å)
        [0.50, 0.50, 0.50],
    ]
    heights = [10.0, 9.0, 8.0, 7.0]
    peaks = plain_peak_pick(sites, heights, cell, n_keep=3, min_dist_A=1.2)
    assert len(peaks) == 3
    fracs = [tuple(np.round(p.fract, 5)) for p in peaks]
    assert (0.05, 0.0, 0.0) not in fracs
    assert peaks[0].rank == 0
    assert math.isclose(peaks[0].height, 10.0)


def test_min_distance_ok_rejects_close():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    assert min_distance_ok(
        np.array([0.0, 0.0, 0.0]), [], cell, min_dist_A=1.2
    )
    assert not min_distance_ok(
        np.array([0.05, 0.0, 0.0]),
        [np.array([0.0, 0.0, 0.0])],
        cell,
        min_dist_A=1.2,
    )


def test_require_cctbx_error_message_when_forced(monkeypatch):
    monkeypatch.setattr(
        "grok_phase_solver.solvers.ab_initio_cctbx.find_cctbx_runner",
        lambda: None,
    )
    with pytest.raises(CctbxUnavailableError) as ei:
        require_cctbx()
    msg = str(ei.value)
    assert "cctbx is required" in msg
    assert "OLEX2_ROOT" in msg or "olex2" in msg.lower()


@pytest.mark.skipif(not cctbx_available(), reason="cctbx / olex2c not available")
def test_cctbx_runner_discovers_something():
    r = find_cctbx_runner()
    assert r is not None
    assert r.kind in ("import", "olex2c")


@pytest.mark.skipif(not cctbx_available(), reason="cctbx / olex2c not available")
def test_ab_initio_live_smoke():
    """Short-budget smoke on a tiny synthetic cell (not a COD claim)."""
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.solvers.ab_initio_cctbx import ab_initio_solve
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    st = generate_random_organic(n_atoms=6, seed=3, space_group="P-1")
    data = structure_to_fcalc(st, d_min=0.9)
    phases, density, info, peaks = ab_initio_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        space_group="P-1",
        n_nonH=6,
        seed=3,
        time_budget_s=8.0,
        verbose=False,
    )
    assert info["status"] == "ok"
    assert density.ndim == 3
    assert len(phases) == len(data["amplitudes"])
    assert info.get("winner") in ("tangent", "charge_flip")
    assert len(peaks) <= 6
