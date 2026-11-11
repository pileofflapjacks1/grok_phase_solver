"""Fast checks for the Q-peak benchmark scorer (full run: scripts/run_qpeak_benchmark.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.metrics.qpeak_benchmark import (
    planted_selftest,
    read_cif_nonh,
    score_q_peaks,
    space_group_ops,
    verdict,
)

COD = Path(__file__).resolve().parents[1] / "data" / "raw" / "cod_qpeak"
PANEL = {"2200239": 22, "2200001": 34, "2230859": 17}
N_RANDOM_FAST = 20


def _write_res(path: Path, q: np.ndarray) -> Path:
    lines = ["TITL test", "SFAC C"]
    lines += [f"Q{i + 1} 1 {x:.6f} {y:.6f} {z:.6f} 11.0 0.05" for i, (x, y, z) in enumerate(q)]
    path.write_text("\n".join(lines + ["END"]) + "\n")
    return path


@pytest.mark.parametrize("cid,n", PANEL.items())
def test_planted_atoms_are_recovered(cid, n):
    out = planted_selftest(COD / f"{cid}.cif")
    assert out["N_nonH"] == n
    assert out["exact"]["hit05"] == n
    assert out["exact"]["ghost"] == 0
    assert out["jitter"]["hit05"] >= n - 2


def test_planted_trial_res_passes_and_beats_reduced_baseline(tmp_path):
    cif = COD / "2230859.cif"
    _cell, hm, atoms = read_cif_nonh(cif)
    ops = space_group_ops(hm)
    af = np.array([a[2] for a in atoms])
    # Symmetry mate + inversion + origin shift: the alignment search must undo all three.
    m, t = ops[1]
    q = -(af @ m.T + t) + np.array([0.5, 0.5, 0.0])
    res = _write_res(tmp_path / "trial.res", q % 1.0)
    out = score_q_peaks("2230859", cif, res, None, n_random=N_RANDOM_FAST)
    assert out["strength"].startswith("res-order")
    assert out["real"]["hit05"] == len(af)
    assert out["ratio05"] >= 3
    assert out["verdict"] == "PASS"
    assert out["rand"]["n"] == N_RANDOM_FAST
    assert 0 < out["rand"]["hit05_mean"] < 0.25 * len(af)


def test_random_trial_res_fails_and_baseline_is_seeded(tmp_path):
    cif = COD / "2200239.cif"
    q = np.random.default_rng(7).random((22, 3))
    res = _write_res(tmp_path / "trial.res", q)
    a = score_q_peaks("2200239", cif, res, None, n_random=N_RANDOM_FAST, seed=3)
    b = score_q_peaks("2200239", cif, res, None, n_random=N_RANDOM_FAST, seed=3)
    assert a["verdict"] == "FAIL"
    assert a["rand"] == b["rand"]


def test_verdict_thresholds():
    assert verdict(0.50, 3.0, 0.40) == "PASS"
    assert verdict(0.60, 3.0, 0.41) == "FAIL"
    assert verdict(0.60, 2.9, 0.10) == "FAIL"
    assert verdict(0.30, 3.1, 0.90) == "PARTIAL"
    assert verdict(0.30, 3.0, 0.10) == "FAIL"
    assert verdict(0.24, 9.0, 0.10) == "FAIL"


# ---------------------------------------------------------------------------
# Polar-axis rule (grader polar-shift-v1)
# ---------------------------------------------------------------------------
from grok_phase_solver.metrics.qpeak_benchmark import (  # noqa: E402
    GRADER_VERSION,
    evaluate,
    orbit,
    orth_matrix,
    polar_axes,
)

CELL_MONO = [7.6, 5.9, 17.0, 90.0, 99.4, 90.0]


def _spread_atoms(n, seed, cell=CELL_MONO, min_sep=1.4):
    """Random fractional sites with a minimum P1 separation (Angstrom)."""
    from grok_phase_solver.metrics.qpeak_benchmark import min_dist

    rng = np.random.default_rng(seed)
    orth = orth_matrix(cell)
    pts = []
    while len(pts) < n:
        c = rng.random(3)
        if all(min_dist(c[None], np.array([p]), orth)[0, 0] > min_sep for p in pts):
            pts.append(c)
    return np.array(pts)


def _write_cif(path: Path, hm: str, atoms, cell=CELL_MONO) -> Path:
    a, b, c, al, be, ga = cell
    lines = [
        "data_t",
        f"_cell_length_a {a}", f"_cell_length_b {b}", f"_cell_length_c {c}",
        f"_cell_angle_alpha {al}", f"_cell_angle_beta {be}", f"_cell_angle_gamma {ga}",
        f"_symmetry_space_group_name_H-M '{hm}'",
        "loop_", "_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y",
        "_atom_site_fract_z", "_atom_site_type_symbol",
    ]
    lines += [f"C{i + 1} {x:.6f} {y:.6f} {z:.6f} C" for i, (x, y, z) in enumerate(atoms)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_polar_axes_derived_from_operators():
    assert polar_axes(space_group_ops("P 1 21 1")) == [1]   # P2_1: b
    assert polar_axes(space_group_ops("P n a 21")) == [2]   # Pna2_1: c
    assert polar_axes(space_group_ops("P 1")) == [0, 1, 2]  # P1: every axis
    assert polar_axes(space_group_ops("P 1 c 1")) == [0, 2]  # Pc: a and c
    # non-polar groups: no polar axis -> discrete search kept
    for hm in ("P 1 21/c 1", "P -1", "P 21 21 21", "P b c a", "C 1 2/c 1"):
        assert polar_axes(space_group_ops(hm)) == [], hm
    assert GRADER_VERSION == "polar-shift-v1"


def _planted_p21(shift_b, inv=1, seed=3, n=12):
    ops = space_group_ops("P 1 21 1")
    af = _spread_atoms(n, seed)
    rng = np.random.default_rng(seed + 1)
    idx = [int(rng.integers(len(ops))) for _ in af]
    q = np.array([ops[k][0] @ a + ops[k][1] for k, a in zip(idx, af)])
    q = inv * q + np.array([0.0, shift_b, 0.0])
    return ops, af, q


@pytest.mark.parametrize("shift_b,inv", [(0.137, 1), (0.61, 1), (0.29, -1)])
def test_p21_shifted_correct_solution_found_new_not_old(shift_b, inv):
    ops, af, q = _planted_p21(shift_b, inv)
    orth = orth_matrix(CELL_MONO)
    a_orb = orbit(af, ops)
    new = evaluate(af, a_orb, q, ops, orth)
    old = evaluate(af, a_orb, q, ops, orth, polar_shift=False)
    assert new["hit05"] == len(af) and new["ghost"] == 0
    assert new["polar_axes"] == [1]
    assert old["hit05"] < len(af)  # generic y-shift is invisible to {0,1/2}^3


def test_p21_scored_trial_res_passes_new_fails_old_and_baseline_uses_same_search(tmp_path):
    ops, af, q = _planted_p21(0.137)
    cif = _write_cif(tmp_path / "x.cif", "P 1 21 1", af)
    res = _write_res(tmp_path / "trial.res", q % 1.0)
    new = score_q_peaks("x", cif, res, None, n_random=30)
    old = score_q_peaks("x", cif, res, None, n_random=30, polar_shift=False)
    assert new["grader"] == "polar-shift-v1" and old["grader"] == "discrete-v0"
    assert new["verdict"] == "PASS" and old["verdict"] == "FAIL"
    assert new["real"]["hit05"] == len(af)
    # The random baseline gets the identical shift search: it can only rise.
    assert new["rand"]["hit05_mean"] > old["rand"]["hit05_mean"]
    # ...and the planted solution still clears 3x the (raised) baseline.
    assert new["ratio05"] >= 3


def test_random_baseline_is_scored_with_the_same_polar_evaluate(tmp_path):
    """Recompute the baseline by hand with evaluate(): must equal score_q_peaks."""
    ops = space_group_ops("P 1 21 1")
    af = _spread_atoms(10, 9)
    cif = _write_cif(tmp_path / "x.cif", "P 1 21 1", af)
    res = _write_res(tmp_path / "trial.res", np.random.default_rng(1).random((10, 3)))
    out = score_q_peaks("x", cif, res, None, n_random=15, seed=5)
    orth = orth_matrix(CELL_MONO)
    a_orb = orbit(af, ops)
    rng = np.random.default_rng(5)
    hand = [evaluate(af, a_orb, rng.random((10, 3)), ops, orth)["hit05"] for _ in range(15)]
    assert out["rand"]["hit05_mean"] == pytest.approx(float(np.mean(hand)))


@pytest.mark.parametrize("cid", list(PANEL))
def test_non_polar_groups_unchanged_exactly(cid):
    cell, hm, atoms = read_cif_nonh(COD / f"{cid}.cif")
    ops = space_group_ops(hm)
    assert polar_axes(ops) == []
    orth = orth_matrix(cell)
    af = np.array([a[2] for a in atoms])
    a_orb = orbit(af, ops)
    rng = np.random.default_rng(11)
    for q in (rng.random((len(af), 3)), planted_q := -af + np.array([0.5, 0.0, 0.5])):
        assert evaluate(af, a_orb, q, ops, orth) == evaluate(
            af, a_orb, q, ops, orth, polar_shift=False
        )


def test_exact_polar_search_is_at_least_a_fine_grid():
    ops = space_group_ops("P 1 21 1")
    orth = orth_matrix(CELL_MONO)
    af = _spread_atoms(10, 21)
    a_orb = orbit(af, ops)
    rng = np.random.default_rng(2)
    from grok_phase_solver.metrics.qpeak_benchmark import min_dist

    for k in range(3):
        q = rng.random((10, 3)) if k == 0 else af + rng.normal(0, 0.05, af.shape)
        exact = evaluate(af, a_orb, q, ops, orth)["hit05"]
        grid = 0
        for inv in (1, -1):
            for ox in (0.0, 0.5):
                for oz in (0.0, 0.5):
                    for y in np.arange(0, 1, 0.01):
                        qa = inv * q + np.array([ox, y, oz])
                        d = min_dist(af, orbit(qa, ops), orth).min(1)
                        grid = max(grid, int((d <= 0.5).sum()))
        assert exact >= grid


def test_two_polar_axes_pc_shift_in_a_and_c_recovered():
    ops = space_group_ops("P 1 c 1")
    assert polar_axes(ops) == [0, 2]
    orth = orth_matrix(CELL_MONO)
    af = _spread_atoms(8, 4)
    a_orb = orbit(af, ops)
    q = af + np.array([0.171, 0.0, 0.343])
    new = evaluate(af, a_orb, q, ops, orth)
    assert new["hit05"] == len(af)
    assert evaluate(af, a_orb, q, ops, orth, polar_shift=False)["hit05"] < len(af)
