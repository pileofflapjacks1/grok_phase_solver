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
