"""Peak budget after unique-ASU fold (Mark/Bragg COD 2200001 n=34)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grok_phase_solver.physics.unique_asu import (
    DEFAULT_N_NON_H_BUDGET,
    budget_peaks,
    unique_peaks,
)
from grok_phase_solver.pipeline.export import write_shelxl_res
from grok_phase_solver.pipeline.peaks import DensityPeak

_CELL = np.array([8.920, 16.282, 18.504, 90.0, 90.0, 90.0])
_BRAGG = (
    "SYMM 0.5-X, -Y, 0.5+Z",
    "SYMM -X, 0.5+Y, 0.5-Z",
    "SYMM 0.5+X, 0.5-Y, -Z",
)


def _atom_lines(res: str):
    skip = {
        "TITL",
        "CELL",
        "ZERR",
        "LATT",
        "SYMM",
        "SFAC",
        "UNIT",
        "FVAR",
        "REM",
        "HKLF",
        "END",
    }
    out = []
    for line in res.splitlines():
        if not line.strip():
            continue
        if line.split()[0] in skip:
            continue
        out.append(line)
    return out


def _make_peaks(n: int, *, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    peaks = []
    for i in range(n):
        # Spread well apart so P1 unique_asu does not fold them together
        fract = np.array(
            [
                (0.05 + 0.9 * (i % 8) / 7.0) % 1.0,
                (0.07 + 0.85 * ((i // 8) % 5) / 4.0) % 1.0,
                (0.09 + 0.8 * (i / max(n - 1, 1))) % 1.0,
            ],
            dtype=np.float64,
        )
        # Distinct heights: higher index → lower sigma so top-34 are 0..33
        sig = float(100.0 - i)
        peaks.append(
            DensityPeak(
                rank=i,
                fract=fract,
                height=sig,
                height_sigma=sig,
            )
        )
    return peaks


def test_default_budget_constant():
    assert DEFAULT_N_NON_H_BUDGET == 34


def test_budget_keeps_top_34_of_40():
    peaks = _make_peaks(40)
    kept, meta = budget_peaks(peaks, n_non_h_budget=34)
    assert meta["n_in"] == 40
    assert meta["n_out"] == 34
    assert meta["budgeted"] is True
    assert len(kept) == 34
    sigs = [p.height_sigma for p in kept]
    assert sigs == sorted(sigs, reverse=True)
    assert min(sigs) >= 100.0 - 33  # ranks 0..33


def test_budget_noop_when_under_cap():
    peaks = _make_peaks(10)
    kept, meta = budget_peaks(peaks, n_non_h_budget=34)
    assert meta["n_out"] == 10
    assert meta["budgeted"] is False
    assert len(kept) == 10


def test_write_shelxl_res_budgets_after_fold_p212121():
    """40 independent ASU peaks → fold keeps 40 → budget keeps 34; SYMM present."""
    peaks = _make_peaks(40)
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 21 21 21",
        method="test",
        diagnostics={},
        peaks=peaks,
    )
    res = write_shelxl_res(result, element="Q")
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res
    atoms = _atom_lines(res)
    assert len(atoms) == 34
    assert "REM peak_budget n=34" in res
    assert "SFAC C H N O" in res
    # Q labels only — no Br/Cl/S (or H) element typing
    for line in atoms:
        lab = line.split()[0]
        assert lab.startswith("Q"), lab
        assert not any(lab.startswith(x) for x in ("Br", "Cl", "S", "H"))


def test_p1_path_budgets_too():
    peaks = _make_peaks(40)
    # unique fold is a no-op in P1; budget still applies
    folded, umeta = unique_peaks(peaks, _CELL, "P 1")
    assert umeta["n_ops"] == 1
    assert len(folded) == 40
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 1",
        method="test",
        diagnostics={},
        peaks=peaks,
    )
    res = write_shelxl_res(result)
    atoms = _atom_lines(res)
    assert len(atoms) == 34
    assert "REM peak_budget n=34" in res
    # P1: LATT -1, no SYMM cards
    assert "LATT -1" in res
    assert "SYMM" not in res


def test_no_heavy_element_labels_in_trial_res():
    peaks = _make_peaks(40)
    # Make a few very strong so old CrystalX would have labeled Br/Cl/S
    for i in range(3):
        peaks[i] = DensityPeak(
            rank=i,
            fract=peaks[i].fract,
            height=50.0,
            height_sigma=20.0 + i,
        )
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 21 21 21",
        method="test",
        diagnostics={},
        peaks=peaks,
    )
    res = write_shelxl_res(result, element="Q", n_non_h_budget=34)
    body = "\n".join(_atom_lines(res))
    for bad in ("Br", "Cl", " S", "S1", "S2", "Cl1", "Br1"):
        assert bad not in body or body.startswith("Q")
    for line in _atom_lines(res):
        assert line.split()[0].startswith("Q")
