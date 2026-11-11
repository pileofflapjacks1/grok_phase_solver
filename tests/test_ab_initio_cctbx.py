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
    CF_SUBPROCESS_GRACE_S,
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
    annotate_r_scores,
    cctbx_available,
    combined_map_fom,
    e_list_cap,
    emap_check_score,
    find_cctbx_runner,
    iterate_until_deadline,
    min_distance_ok,
    peak_contrast,
    peak_network_fraction,
    plain_peak_pick,
    require_cctbx,
    select_ab_initio_candidate,
    select_better_by_emap_check,
    select_better_by_own_fom,
    select_better_by_r_check,
    sensible_peak_fraction,
    run_cf_deadline_probe,
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
    assert CF_SUBPROCESS_GRACE_S == 30.0
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


def test_select_better_by_r_check_prefers_lower_r1():
    """Non-centro rule: lowest peak R1; emap is only a tiebreak."""
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    # Two reflections; fabricate amplitudes and peak sets so R1 ordering is known.
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=float)
    # Place a "good" 4-atom chain; compute Fo from it, then a scrambled bad set.
    from grok_phase_solver.physics.structure_factors import compute_structure_factors

    good_fracs = np.array([[i * 0.15, 0.0, 0.0] for i in range(4)], dtype=float)
    bad_fracs = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
        dtype=float,
    )
    Fo = np.abs(
        compute_structure_factors(
            hkl, good_fracs, ["C"] * 4, cell, b_isos=np.full(4, 5.0)
        )
    )
    good = {
        "name": "good_r",
        "skew": 1.0,  # deliberately lower skew than bad
        "peaks": [
            {"fract": list(f), "height": h}
            for f, h in zip(good_fracs, [10.0, 9.0, 8.0, 7.0])
        ],
    }
    bad = {
        "name": "bad_r",
        "skew": 5.0,  # would win emap_check
        "peaks": [
            {"fract": list(f), "height": h}
            for f, h in zip(bad_fracs, [10.0, 9.5, 9.0, 8.5])
        ],
    }
    assert select_better_by_emap_check([good, bad], cell, n_nonH=4)["name"] == "bad_r"
    assert (
        select_better_by_r_check([good, bad], hkl, Fo, cell, n_nonH=4)["name"]
        == "good_r"
    )


def test_select_ab_initio_candidate_centro_keeps_emap():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    amp = np.array([10.0, 8.0, 6.0])
    high_emap = {
        "name": "high_emap",
        "skew": 4.0,
        "peaks": [
            {"fract": [i * 0.15, 0.0, 0.0], "height": h}
            for i, h in enumerate([10.0, 9.0, 8.0, 7.0])
        ],
    }
    low_emap = {
        "name": "low_emap",
        "skew": 0.5,
        "peaks": [
            {"fract": [0.0, 0.0, 0.0], "height": 10.0},
            {"fract": [0.5, 0.0, 0.0], "height": 2.0},
            {"fract": [0.0, 0.5, 0.0], "height": 2.0},
            {"fract": [0.5, 0.5, 0.0], "height": 2.0},
        ],
    }
    winner = select_ab_initio_candidate(
        [high_emap, low_emap],
        cell,
        n_nonH=4,
        is_centrosymmetric=True,
        hkl=hkl,
        amplitudes=amp,
    )
    assert winner["name"] == "high_emap"


def test_select_ab_initio_candidate_noncentro_uses_r():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=float)
    from grok_phase_solver.physics.structure_factors import compute_structure_factors

    good_fracs = np.array([[i * 0.15, 0.0, 0.0] for i in range(4)], dtype=float)
    bad_fracs = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
        dtype=float,
    )
    Fo = np.abs(
        compute_structure_factors(
            hkl, good_fracs, ["C"] * 4, cell, b_isos=np.full(4, 5.0)
        )
    )
    good = {
        "name": "good_r",
        "skew": 1.0,
        "peaks": [
            {"fract": list(f), "height": h}
            for f, h in zip(good_fracs, [10.0, 9.0, 8.0, 7.0])
        ],
    }
    bad = {
        "name": "bad_r",
        "skew": 5.0,
        "peaks": [
            {"fract": list(f), "height": h}
            for f, h in zip(bad_fracs, [10.0, 9.5, 9.0, 8.5])
        ],
    }
    winner = select_ab_initio_candidate(
        [good, bad],
        cell,
        n_nonH=4,
        is_centrosymmetric=False,
        hkl=hkl,
        amplitudes=Fo,
    )
    assert winner["name"] == "good_r"


def test_iterate_until_deadline_stops_near_limit():
    """Hard wall-clock abort: stop near the deadline, not far past it."""
    import time as _time

    def slow():
        while True:
            _time.sleep(0.05)
            yield "tick"

    limit_s = 0.25
    t0 = _time.time()
    n, timed_out = iterate_until_deadline(slow(), t0 + limit_s)
    elapsed = _time.time() - t0
    assert timed_out
    assert n >= 1
    # Must stop near the limit (allow a small sleep overshoot, not multi-second).
    assert elapsed < limit_s + 0.35
    assert elapsed >= limit_s


def test_annotate_r_scores_fills_peak_r1():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    amp = np.array([5.0, 4.0, 3.0])
    cands = [
        {
            "name": "t",
            "skew": 1.0,
            "peaks": [
                {"fract": [0.0, 0.0, 0.0], "height": 10.0},
                {"fract": [0.2, 0.0, 0.0], "height": 8.0},
            ],
        }
    ]
    out = annotate_r_scores(cands, hkl, amp, cell, n_nonH=2)
    assert "peak_r1" in out[0]
    assert 0.0 <= out[0]["peak_r1"] <= 2.0


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


@pytest.mark.skipif(not cctbx_available(), reason="cctbx / olex2c not available")
def test_loop_cf_with_deadline_real_smtbx_iterator():
    """Regression: smtbx solving_iterator must be driven via iter()+next().

    Runs a real CF solving_iterator through ``_loop_cf_with_deadline`` under
    olex2c/cctbx. Must not raise TypeError; must complete a few cycles.
    """
    from pathlib import Path

    from grok_phase_solver.io.experiment import load_experiment
    from grok_phase_solver.metrics.qpeak_benchmark import read_cif_nonh

    root = Path(__file__).resolve().parents[1]
    cif = root / "data" / "raw" / "cod_qpeak" / "2200239.cif"
    hkl_path = root / "data" / "raw" / "cod_qpeak" / "2200239.hkl"
    cell, sg, _atoms = read_cif_nonh(cif)
    table, _ins = load_experiment(str(hkl_path))
    raw = run_cf_deadline_probe(
        table.hkl,
        table.amplitudes,
        np.asarray(cell, dtype=np.float64),
        space_group=sg or "P21/c",
        seed=0,
        time_budget_s=30.0,
    )
    assert raw.get("error") is None, raw
    assert raw.get("first_yield_ok") is True
    # Either finished inside budget or timed out cleanly — never TypeError crash.
    assert "timed_out" in raw
    assert float(raw.get("solver_time_s") or 0) > 0


# ---------------------------------------------------------------------------
# Hard time limit: killable worker process group
# ---------------------------------------------------------------------------
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402

from grok_phase_solver.solvers import ab_initio_cctbx as _ab  # noqa: E402
from grok_phase_solver.solvers.ab_initio_cctbx import (  # noqa: E402
    HOST_KILL_RESERVE_S,
    REFINE_CYCLES,
    REFINE_TIME_RESERVE_S,
    REFINE_U_ISO,
    AbInitioTimeoutError,
    CctbxRunner,
    ab_initio_solve,
    has_refined_fit,
    run_killable_process,
    select_better_by_refined_fit,
)

SLACK_S = 3.0  # generous CI slack on top of the kill limit


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie children of init are reaped quickly; treat /proc state Z as dead
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except Exception:
        return True


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_run_killable_process_kills_hung_child_within_limit():
    t0 = time.monotonic()
    info = run_killable_process(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        hard_timeout_s=1.0,
    )
    dt = time.monotonic() - t0
    assert info["timed_out"] is True
    assert dt < 1.0 + SLACK_S
    assert info["returncode"] is not None and info["returncode"] != 0


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_run_killable_process_kills_grandchildren(tmp_path):
    """xvfb-run -> olex2c style: the whole process group must die."""
    pidfile = tmp_path / "grandchild.pid"
    code = textwrap.dedent(
        f"""
        import subprocess, sys, time
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        open({str(pidfile)!r}, "w").write(str(p.pid))
        time.sleep(300)
        """
    )
    info = run_killable_process([sys.executable, "-c", code], hard_timeout_s=1.5)
    assert info["timed_out"] is True
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 3.0
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(pid), "grandchild survived the group kill"


def test_run_killable_process_normal_exit_not_timed_out(tmp_path):
    log = tmp_path / "log.txt"
    info = run_killable_process(
        [sys.executable, "-c", "print('hello')"],
        hard_timeout_s=30.0,
        log_path=log,
    )
    assert info["timed_out"] is False
    assert info["returncode"] == 0
    assert "hello" in log.read_text()
    assert info["wall_s"] < 10.0
    assert abs(info["clock_jump_s"]) < 1.0


_FAKE_WORKER = '''
import json, os, subprocess, sys, time
req = json.load(open(sys.argv[1]))
out = sys.argv[2]
MODE = {mode!r}
if MODE == "partial_then_hang":
    cand = {{
        "name": "tangent", "method": "tangent", "fom": 1.0, "cons": 1.0,
        "skew": 2.0,
        "peaks": [{{"fract": [0.1, 0.2, 0.3], "height": 5.0}},
                  {{"fract": [0.6, 0.2, 0.3], "height": 4.0}}],
        "phases_rad": [0.0] * len(req["hkl"]),
    }}
    res = {{"error": None, "partial": True, "winner": "tangent", "fom": 1.0,
            "peaks": cand["peaks"], "phases_rad": cand["phases_rad"],
            "candidates_full": [cand], "candidates": [],
            "is_centrosymmetric": True, "tangent": {{}}, "charge_flip": {{}}}}
    tmp = out + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, out)
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
time.sleep(300)
'''


def _patch_fake_worker(monkeypatch, tmp_path, mode):
    fake = tmp_path / "fake_worker.py"
    fake.write_text(_FAKE_WORKER.format(mode=mode))
    monkeypatch.setattr(_ab, "_worker_path", lambda: fake)
    monkeypatch.setattr(_ab, "find_cctbx_runner", lambda: CctbxRunner(kind="import"))


_SMALL_CELL = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
_SMALL_HKL = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=int)
_SMALL_AMP = np.array([10.0, 8.0, 6.0, 4.0])


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_hung_worker_killed_within_budget_returns_partial(monkeypatch, tmp_path):
    _patch_fake_worker(monkeypatch, tmp_path, "partial_then_hang")
    budget, grace = 1.0, 4.0  # hard limit = 1 + 4 - 3 = 2 s
    t0 = time.monotonic()
    phases, density, info, peaks = ab_initio_solve(
        _SMALL_HKL, _SMALL_AMP, _SMALL_CELL,
        space_group="P1", n_nonH=2, seed=0,
        time_budget_s=budget, grace_s=grace, work_dir=tmp_path / "w",
    )
    dt = time.monotonic() - t0
    assert dt < budget + grace + SLACK_S  # nothing exceeds budget + grace
    assert dt < budget + grace  # ...and in practice not even the full grace
    assert info["status"] == "timeout_partial"
    assert info["timed_out"] is True and info["host_killed"] is True
    assert info["winner"] == "tangent"
    assert len(phases) == len(_SMALL_AMP)
    assert info["child_wall_s"] <= budget + grace - HOST_KILL_RESERVE_S + SLACK_S


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_hung_worker_without_result_raises_timeout(monkeypatch, tmp_path):
    _patch_fake_worker(monkeypatch, tmp_path, "hang")
    budget, grace = 1.0, 4.0
    t0 = time.monotonic()
    with pytest.raises(AbInitioTimeoutError) as ei:
        ab_initio_solve(
            _SMALL_HKL, _SMALL_AMP, _SMALL_CELL,
            space_group="P1", n_nonH=2, seed=0,
            time_budget_s=budget, grace_s=grace, work_dir=tmp_path / "w",
        )
    assert time.monotonic() - t0 < budget + grace
    assert ei.value.info["status"] == "timeout_no_result"
    assert ei.value.info["timed_out"] is True


def test_stale_result_json_is_not_reused(monkeypatch, tmp_path):
    """A leftover result.json in a reused work_dir must not leak into a new run."""
    w = tmp_path / "w"
    w.mkdir()
    (w / "result.json").write_text(json.dumps({"candidates_full": [{"name": "stale"}]}))
    _patch_fake_worker(monkeypatch, tmp_path, "hang")
    with pytest.raises(AbInitioTimeoutError):
        ab_initio_solve(
            _SMALL_HKL, _SMALL_AMP, _SMALL_CELL,
            space_group="P1", n_nonH=2, seed=0,
            time_budget_s=1.0, grace_s=4.0, work_dir=w,
        )


# ---------------------------------------------------------------------------
# Non-centrosymmetric refined-fit ranking
# ---------------------------------------------------------------------------
def _cand(name, wr2, r1=None, emap_skew=1.0):
    return {
        "name": name,
        "skew": emap_skew,
        "refined_wR2": wr2,
        "refined_R1": r1,
        "peaks": [
            {"fract": [0.1 * i, 0.0, 0.0], "height": 10.0 - i} for i in range(3)
        ],
    }


def test_select_better_by_refined_fit_lowest_wr2_wins_over_emap():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    best_fit = _cand("low_wr2_low_skew", 0.10, 0.20, emap_skew=0.1)
    pretty_map = _cand("high_skew_bad_fit", 0.60, 0.50, emap_skew=9.0)
    w = select_better_by_refined_fit([pretty_map, best_fit], cell, n_nonH=3)
    assert w["name"] == "low_wr2_low_skew"


def test_select_better_by_refined_fit_ties_and_unrefined_rank_last():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    a = _cand("a", 0.30, 0.40)
    b = _cand("b", 0.30, 0.35)  # same wR2, lower R1
    none = _cand("unrefined", None, None, emap_skew=50.0)
    nan = _cand("nan", float("nan"), None, emap_skew=50.0)
    assert select_better_by_refined_fit([a, b, none, nan], cell, n_nonH=3)["name"] == "b"
    assert select_better_by_refined_fit([none, nan, a], cell, n_nonH=3)["name"] == "a"


def test_has_refined_fit():
    assert not has_refined_fit([{"name": "x"}, {"name": "y", "refined_wR2": None}])
    assert not has_refined_fit([{"refined_wR2": float("inf")}])
    assert has_refined_fit([{"refined_wR2": 0.2}])


def test_dispatch_noncentro_uses_refined_centro_ignores_it():
    cell = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    amp = np.array([5.0, 4.0, 3.0])
    hi_skew = _cand("hi_skew", 0.50, 0.50, emap_skew=9.0)
    lo_wr2 = _cand("lo_wr2", 0.10, 0.20, emap_skew=0.1)
    nc = select_ab_initio_candidate(
        [hi_skew, lo_wr2], cell, n_nonH=3, is_centrosymmetric=False,
        hkl=hkl, amplitudes=amp,
    )
    assert nc["name"] == "lo_wr2"
    # Centrosymmetric selection is unchanged: E-map check, refined fields ignored.
    c = select_ab_initio_candidate(
        [hi_skew, lo_wr2], cell, n_nonH=3, is_centrosymmetric=True,
        hkl=hkl, amplitudes=amp,
    )
    assert c["name"] == select_better_by_emap_check(
        [hi_skew, lo_wr2], cell, n_nonH=3
    )["name"] == "hi_skew"


def test_orth_matrix_cache_matches_uncached():
    from grok_phase_solver.io.cif import CrystalStructure

    cell = np.array([7.6, 5.9, 17.0, 90.0, 99.4, 90.0])
    ref = CrystalStructure("t", cell, "P1").orth_matrix
    got = _ab._orth_matrix(cell)
    assert np.allclose(ref, got)
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.9, 0.1, 0.35])
    d1 = _ab.min_image_dist_A(a, b, cell)
    d2 = _ab.min_image_dist_A(a, b, cell.copy())
    assert d1 == d2


def test_worker_and_host_refine_constants_agree():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gps_worker_for_test", _ab._worker_path()
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.REFINE_CYCLES == REFINE_CYCLES == 4
    assert mod.REFINE_U_ISO == REFINE_U_ISO == 0.05
    assert mod.REFINE_TIME_RESERVE_S == REFINE_TIME_RESERVE_S == 3.0
    assert 3 <= REFINE_CYCLES <= 5


@pytest.mark.skipif(not cctbx_available(), reason="cctbx / olex2c not available")
def test_ab_initio_live_noncentro_refined_selection():
    """Synthetic P21 smoke: refinement ranks candidates (not a COD claim)."""
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    st = generate_random_organic(n_atoms=8, seed=5, space_group="P21")
    data = structure_to_fcalc(st, d_min=0.9)
    t0 = time.time()
    phases, density, info, peaks = ab_initio_solve(
        data["hkl"], data["amplitudes"], st.cell,
        space_group="P21", n_nonH=8, seed=5, time_budget_s=15.0,
    )
    assert time.time() - t0 < 15.0 + 30.0
    assert info["is_centrosymmetric"] is False
    assert info["selection"] == "refined_wR2"
    assert info["refined_wR2"] is not None and np.isfinite(info["refined_wR2"])
    refined = [c for c in info["candidates"] if c.get("refined_wR2") is not None]
    assert refined
    assert info["refined_wR2"] == min(c["refined_wR2"] for c in refined)
    assert info["refine"]["n_cycles"] == REFINE_CYCLES
