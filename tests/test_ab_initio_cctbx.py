"""Unit tests for ab_initio helpers; live cctbx path skipped when unavailable."""

from __future__ import annotations

import math

import numpy as np
import pytest

# Import pipeline.peaks before solvers package to avoid a pre-existing
# solvers ↔ pipeline circular import during collection.
import grok_phase_solver.pipeline.peaks  # noqa: F401

from grok_phase_solver.solvers.ab_initio_cctbx import (
    BASE_SEED,
    BENCH_OUTER_SEEDS,
    CF_N_SEEDS,
    CF_SEED_LIST,
    CF_TIME_RESERVE_S,
    E_CAP_MIN,
    E_CAP_PER_ATOM,
    E_MIN,
    EMAP_BOND_HI_A,
    EMAP_BOND_LO_A,
    EMAP_MIN_SEP_A,
    MIN_PEAK_DIST_A,
    N_TANGENT_CYCLES,
    TIME_BUDGET_S,
    CctbxUnavailableError,
    annotate_emap_scores,
    cctbx_available,
    combined_map_fom,
    e_list_cap,
    emap_check_score,
    find_cctbx_runner,
    min_distance_ok,
    peak_contrast,
    peak_network_fraction,
    plain_peak_pick,
    require_cctbx,
    select_better_by_emap_check,
    select_better_by_own_fom,
    sensible_peak_fraction,
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
    assert CF_SEED_LIST == (0, 1, 7, 42, 2025)
    assert CF_N_SEEDS == 5
    assert CF_TIME_RESERVE_S == 12.0
    assert BENCH_OUTER_SEEDS == (0, 1, 7, 42, 2025)
    assert EMAP_MIN_SEP_A == 1.1
    assert EMAP_BOND_LO_A == 1.2
    assert EMAP_BOND_HI_A == 1.9


def test_combined_map_fom_monotonic_in_skew():
    f0 = combined_map_fom(1.0, 0.0)
    f1 = combined_map_fom(1.0, 2.0)
    f2 = combined_map_fom(1.0, 5.0)
    assert f0 < f1 < f2
    assert combined_map_fom(0.5, 2.0) < combined_map_fom(1.0, 2.0)


def test_select_better_by_own_fom_ignores_model_fields():
    a = {"name": "tangent", "fom": 0.9, "true_phase_mean_cos": 0.1}
    b = {"name": "charge_flip", "fom": 1.1, "true_phase_mean_cos": 0.99}
    assert select_better_by_own_fom([a, b])["name"] == "charge_flip"
    c = {"name": "tangent", "fom": 1.2, "true_phase_mean_cos": 0.0}
    assert select_better_by_own_fom([a, b, c])["name"] == "tangent"


def test_select_better_empty_raises():
    with pytest.raises(ValueError):
        select_better_by_own_fom([])
    with pytest.raises(ValueError):
        select_better_by_emap_check([], np.array([10.0, 10.0, 10.0, 90, 90, 90]))


def test_plain_peak_search_min_distance():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    sites = [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.25, 0.0, 0.0],
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
    assert min_distance_ok(np.array([0.0, 0.0, 0.0]), [], cell, min_dist_A=1.2)
    assert not min_distance_ok(
        np.array([0.05, 0.0, 0.0]),
        [np.array([0.0, 0.0, 0.0])],
        cell,
        min_dist_A=1.2,
    )


def test_peak_contrast_and_sensible_fraction():
    assert peak_contrast([]) == 0.0
    assert peak_contrast([5.0, 5.0, 5.0]) == pytest.approx(1.0)
    # Spike map: median 2 / max 10 = 0.2
    assert peak_contrast([10.0, 2.0, 2.0]) == pytest.approx(0.2)
    # Atom-like: median 8 / max 10 = 0.8
    assert peak_contrast([10.0, 9.0, 8.0, 7.0]) == pytest.approx(0.85)
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    # Two peaks 0.5 Å apart → sensible_frac 0
    assert sensible_peak_fraction([[0, 0, 0], [0.05, 0, 0]], cell) == 0.0
    # Two peaks 2.5 Å apart → both sensible
    assert sensible_peak_fraction([[0, 0, 0], [0.25, 0, 0]], cell) == 1.0


def test_peak_network_fraction_connected_chain():
    # Cubic 10 Å: place 4 peaks along x at 1.5 Å spacing (0.15 frac)
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    sites = [[i * 0.15, 0.0, 0.0] for i in range(4)]
    # bonds 1.5 Å in [1.2, 1.9] → one connected component of 4
    assert peak_network_fraction(sites, cell, n_expected=4) == pytest.approx(1.0)
    # Far-apart peaks (5 Å) → no edges → LCC=1 → 1/4
    sites_far = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]]
    assert peak_network_fraction(sites_far, cell, n_expected=4) == pytest.approx(0.25)


def test_emap_check_score_prefers_connected_high_skew():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    good_sites = [[i * 0.15, 0.0, 0.0] for i in range(4)]
    good_h = [10.0, 8.0, 7.0, 6.0]
    bad_sites = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]]
    bad_h = [3.0, 2.9, 2.8, 2.7]
    good = emap_check_score(4.0, good_h, good_sites, cell, n_nonH=4)
    bad = emap_check_score(0.9, bad_h, bad_sites, cell, n_nonH=4)
    assert good["score"] > bad["score"]
    assert good["network_frac"] > bad["network_frac"]


def test_select_better_by_emap_check_not_own_fom():
    """Regression: own-FOM can prefer a failing CF; E-map check must not."""
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    # Fake "CF FAIL": inflated own-FOM (cons=1), medium skew, sparse peaks,
    # spike-ish heights — matches audit seeds where CF FOM > tan FOM but FAIL.
    cf_fail = {
        "name": "charge_flip_seed_42",
        "method": "charge_flip",
        "fom": 1.05,
        "skew": 1.5,
        "peaks": [
            {"fract": [0.0, 0.0, 0.0], "height": 10.0},
            {"fract": [0.5, 0.0, 0.0], "height": 3.0},
            {"fract": [0.0, 0.5, 0.0], "height": 2.5},
            {"fract": [0.5, 0.5, 0.0], "height": 2.0},
        ],
    }
    # Fake "tangent PASS": lower own-FOM, similar skew, connected 1.5 Å chain
    tan_pass = {
        "name": "tangent",
        "method": "tangent",
        "fom": 0.92,
        "skew": 1.6,
        "peaks": [
            {"fract": [i * 0.15, 0.0, 0.0], "height": h}
            for i, h in enumerate([10.0, 9.0, 8.0, 7.0])
        ],
    }
    # High-skew CF PASS should beat both
    cf_pass = {
        "name": "charge_flip_seed_7",
        "method": "charge_flip",
        "fom": 1.19,
        "skew": 4.1,
        "peaks": [
            {"fract": [i * 0.15, 0.0, 0.0], "height": h}
            for i, h in enumerate([10.0, 9.0, 8.5, 8.0])
        ],
    }
    assert select_better_by_own_fom([cf_fail, tan_pass])["name"].startswith(
        "charge_flip"
    )
    assert (
        select_better_by_emap_check([cf_fail, tan_pass], cell, n_nonH=4)["name"]
        == "tangent"
    )
    assert (
        select_better_by_emap_check(
            [cf_fail, tan_pass, cf_pass], cell, n_nonH=4
        )["name"]
        == "charge_flip_seed_7"
    )


def test_annotate_emap_scores_fills_fields():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    cands = [
        {
            "name": "tangent",
            "skew": 2.0,
            "peaks": [
                {"fract": [0.0, 0.0, 0.0], "height": 10.0},
                {"fract": [0.15, 0.0, 0.0], "height": 8.0},
            ],
        }
    ]
    out = annotate_emap_scores(cands, cell, n_nonH=2)
    assert "emap_score" in out[0]
    assert "network_frac" in out[0]
    assert "contrast" in out[0]


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
    assert info.get("selection") == "emap_check"
    assert density.ndim == 3
    assert len(phases) == len(data["amplitudes"])
    assert info.get("winner_method") in ("tangent", "charge_flip")
    assert len(peaks) <= 6
    # Multi-seed CF should yield multiple CF candidates when budget allows
    names = [c.get("name") for c in (info.get("candidates") or [])]
    assert any(n == "tangent" or (n or "").startswith("charge_flip") for n in names)
