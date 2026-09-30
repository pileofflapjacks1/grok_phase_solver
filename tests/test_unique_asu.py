"""Fold P1 peak dumps under 2₁ ops (unique-ASU ticket). No SFAC retune."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grok_phase_solver.physics.unique_asu import unique_asu_fracs, space_group_ops
from grok_phase_solver.pipeline.crystalx_typing import TypedAtom, typed_atoms_to_shelxl_res
from grok_phase_solver.pipeline.export import write_shelxl_res
from grok_phase_solver.pipeline.peaks import DensityPeak

_CELL = np.array([8.920, 16.282, 18.504, 90.0, 90.0, 90.0])
_BRAGG = (
    "SYMM 0.5-X, -Y, 0.5+Z",
    "SYMM -X, 0.5+Y, 0.5-Z",
    "SYMM 0.5+X, 0.5-Y, -Z",
)


def _p212121_images(xyz):
    x, y, z = xyz
    return np.array(
        [
            [x, y, z],
            [0.5 - x, -y, 0.5 + z],
            [-x, 0.5 + y, 0.5 - z],
            [0.5 + x, 0.5 - y, -z],
        ],
        dtype=np.float64,
    ) % 1.0


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


def test_p212121_four_images_collapse_to_one():
    xyz = np.array([0.10, 0.20, 0.30])
    fracs = _p212121_images(xyz)
    weights = np.array([4.0, 3.0, 2.0, 1.0])
    out, idx, meta = unique_asu_fracs(fracs, "P 21 21 21", _CELL, weights=weights)
    assert meta["n_ops"] == 4
    assert len(out) == 1
    assert meta["n_in"] == 4 and meta["n_out"] == 1
    assert int(idx[0]) == 0  # highest weight kept


def test_p1_does_not_fold():
    fracs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.1, 0.2]])
    out, idx, meta = unique_asu_fracs(fracs, "P 1", _CELL)
    assert meta["n_ops"] == 1
    assert len(out) == 3
    assert list(idx) == [0, 1, 2]


def test_pminus1_folds_inversion():
    xyz = np.array([0.10, 0.20, 0.30])
    fracs = np.vstack([xyz, (-xyz) % 1.0])
    out, idx, meta = unique_asu_fracs(fracs, "P-1", _CELL, weights=np.array([2.0, 1.0]))
    assert meta["n_ops"] == 2
    assert len(out) == 1


def test_two_independent_orbits_stay_two():
    a = np.array([0.10, 0.20, 0.30])
    b = np.array([0.22, 0.41, 0.18])
    fracs = np.vstack([_p212121_images(a), _p212121_images(b)])
    out, idx, meta = unique_asu_fracs(fracs, "P212121", _CELL)
    assert len(out) == 2


def test_typed_writer_folds_and_keeps_symm():
    xyz = np.array([0.10, 0.20, 0.30])
    atoms = []
    for i, f in enumerate(_p212121_images(xyz)):
        atoms.append(
            TypedAtom(
                label=f"C{i+1}",
                element="C",
                fract=f,
                height_sigma=5.0 - i,
            )
        )
    res = typed_atoms_to_shelxl_res(atoms, _CELL, space_group="P 21 21 21")
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res
    atoms_out = _atom_lines(res)
    assert len(atoms_out) == 1
    assert "REM unique_asu n_in=4 n_out=1" in res
    # SFAC algorithm unchanged: still derived from remaining elements, UNIT all 1s
    assert "SFAC C" in res
    assert "UNIT 1" in res


def test_write_shelxl_res_folds_peaks():
    xyz = np.array([0.10, 0.20, 0.30])
    peaks = []
    for i, f in enumerate(_p212121_images(xyz)):
        peaks.append(
            DensityPeak(rank=i, fract=f, height=10.0 - i, height_sigma=4.0 - 0.1 * i)
        )
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 21 21 21",
        method="test",
        diagnostics={},
        peaks=peaks,
    )
    res = write_shelxl_res(result)
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res
    assert len(_atom_lines(res)) == 1
    assert "SFAC C" in res
    assert "SFAC C H N O" not in res
    assert "UNIT 1" in res
    assert "hand-build peaks" in res


def test_space_group_ops_p212121_count():
    ops = space_group_ops("P 21 21 21")
    assert len(ops) == 4
