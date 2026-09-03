"""trial.res LATT/SYMM from space group (SYMM-only ticket)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grok_phase_solver.physics.shelx_cards import shelx_latt_symm
from grok_phase_solver.pipeline.crystalx_typing import TypedAtom, typed_atoms_to_shelxl_res
from grok_phase_solver.pipeline.export import write_shelxl_res
from grok_phase_solver.pipeline.peaks import DensityPeak

_BRAGG = (
    "SYMM 0.5-X, -Y, 0.5+Z",
    "SYMM -X, 0.5+Y, 0.5-Z",
    "SYMM 0.5+X, 0.5-Y, -Z",
)
_CELL = np.array([8.920, 16.282, 18.504, 90.0, 90.0, 90.0])


def test_shelx_latt_symm_p212121_bragg_spec():
    latt, cards = shelx_latt_symm("P 21 21 21")
    assert latt == -1
    assert cards == [
        "0.5-X, -Y, 0.5+Z",
        "-X, 0.5+Y, 0.5-Z",
        "0.5+X, 0.5-Y, -Z",
    ]
    latt2, cards2 = shelx_latt_symm("P212121")
    assert (latt2, cards2) == (latt, cards)


def test_shelx_latt_symm_p1_and_pminus1():
    latt, cards = shelx_latt_symm("P 1")
    assert latt == -1
    assert cards == []
    latt, cards = shelx_latt_symm("P-1")
    assert latt == 1
    assert cards == []


def test_ins_symm_passthrough():
    latt, cards = shelx_latt_symm(
        "P 1",
        lattice=-1,
        symm=["0.5-X, -Y, 0.5+Z", "-X, 0.5+Y, 0.5-Z", "0.5+X, 0.5-Y, -Z"],
    )
    assert latt == -1
    assert len(cards) == 3


def test_typed_atoms_res_writes_p212121_symm():
    atom = TypedAtom(
        label="C1",
        element="C",
        fract=np.array([0.1, 0.2, 0.3]),
        height_sigma=3.0,
    )
    res = typed_atoms_to_shelxl_res([atom], _CELL, space_group="P 21 21 21")
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res
    assert "REM space_group_hint=P 21 21 21" in res
    header = res.split("SFAC")[0]
    assert header.count("SYMM") == 3


def test_typed_atoms_res_p1_has_no_symm():
    atom = TypedAtom(
        label="C1",
        element="C",
        fract=np.array([0.1, 0.2, 0.3]),
        height_sigma=3.0,
    )
    res = typed_atoms_to_shelxl_res([atom], _CELL, space_group="P1")
    assert "LATT -1" in res
    assert "SYMM" not in res.split("SFAC")[0]


def test_write_shelxl_res_p212121_symm():
    peak = DensityPeak(
        rank=0,
        fract=np.array([0.1, 0.2, 0.3]),
        height=10.0,
        height_sigma=4.0,
    )
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 21 21 21",
        method="test",
        diagnostics={},
        peaks=[peak],
    )
    res = write_shelxl_res(result)
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res
    assert "REM space_group_hint=P 21 21 21" in res
