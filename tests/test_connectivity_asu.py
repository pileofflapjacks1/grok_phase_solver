"""Connectivity/ASU packing after fold+budget (Mark/Bragg)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import io
from contextlib import redirect_stdout

from grok_phase_solver.physics.connectivity_asu import (
    COVALENT_C_C_CUTOFF_A,
    ConnectivityAsuError,
    format_trial_res_gate,
    pack_discrete_asu,
)
from grok_phase_solver.pipeline.export import write_shelxl_res
from grok_phase_solver.pipeline.peaks import DensityPeak

# Orthorhombic cell large enough that only intentional short contacts bond.
_CELL = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
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


def _peak(fract, sig: float, rank: int = 0) -> DensityPeak:
    f = np.asarray(fract, dtype=np.float64)
    return DensityPeak(rank=rank, fract=f, height=float(sig), height_sigma=float(sig))


def _chain_fracs(n: int, *, start, step_cart, cell=_CELL):
    """n sites along a cartesian step (Å), returned as fractional coords."""
    a, b, c = float(cell[0]), float(cell[1]), float(cell[2])
    out = []
    x0, y0, z0 = start
    dx, dy, dz = step_cart
    for i in range(n):
        out.append(
            np.array(
                [
                    (x0 + i * dx / a) % 1.0,
                    (y0 + i * dy / b) % 1.0,
                    (z0 + i * dz / c) % 1.0,
                ],
                dtype=np.float64,
            )
        )
    return out


def _assert_raises_polymer(fn):
    try:
        fn()
    except ConnectivityAsuError as exc:
        assert "polymer" in str(exc).lower()
        return
    raise AssertionError("expected ConnectivityAsuError")


def test_cutoff_constant_documented():
    assert COVALENT_C_C_CUTOFF_A == 1.8


def test_discrete_molecule_kept():
    """Finite C–C chain (~8 atoms) packs; no polymer error."""
    fracs = _chain_fracs(8, start=(0.10, 0.20, 0.30), step_cart=(1.5, 0.0, 0.0))
    peaks = [_peak(f, 50.0 - i, rank=i) for i, f in enumerate(fracs)]
    kept, meta = pack_discrete_asu(
        peaks, _CELL, "P 1", n_non_h_budget=34, covalent_cutoff=1.8
    )
    assert meta["n_in"] == 8
    assert meta["n_out"] == 8
    assert meta["n_components"] == 1
    assert meta["n_infinite"] == 0
    assert len(kept) == 8


def test_infinite_polymer_lattice_fails_closed():
    """Short-a P1 chain across the cell boundary → infinite polymer."""
    cell = np.array([1.6, 20.0, 20.0, 90.0, 90.0, 90.0])
    peaks = [
        _peak([0.10, 0.20, 0.20], 40.0, 0),
        _peak([0.60, 0.20, 0.20], 39.0, 1),
    ]

    def _go():
        pack_discrete_asu(
            peaks, cell, "P 1", n_non_h_budget=34, covalent_cutoff=1.8
        )

    _assert_raises_polymer(_go)


def test_infinite_under_21_screw_fails_closed():
    """Explicit 2₁ along b: site bonds through screw into next cell."""
    cell = np.array([20.0, 3.0, 20.0, 90.0, 90.0, 90.0])
    symm = ["-X, 0.5+Y, -Z"]
    peaks = [_peak([0.0, 0.05, 0.0], 50.0, 0)]

    def _go():
        pack_discrete_asu(
            peaks,
            cell,
            None,
            lattice=-1,
            symm=symm,
            n_non_h_budget=34,
            covalent_cutoff=1.8,
        )

    _assert_raises_polymer(_go)


def test_writer_rem_and_no_crystalx_discrete():
    fracs = _chain_fracs(6, start=(0.12, 0.22, 0.32), step_cart=(1.5, 0.0, 0.0))
    peaks = [_peak(f, 40.0 - i, rank=i) for i, f in enumerate(fracs)]
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 21 21 21",
        method="test",
        diagnostics={},
        peaks=peaks,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = write_shelxl_res(result, element="Q", n_non_h_budget=34)
    gate = format_trial_res_gate(
        sg="P 21 21 21", non_h=6, finite=True, pass_=True
    )
    assert gate in buf.getvalue()
    assert f"REM gate sg=P 21 21 21 non_h=6 finite=yes pass=yes" in res
    assert "REM connectivity_asu n_in=" in res
    assert "n_components=" in res
    assert "SFAC C H N O" in res
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res
    atoms = _atom_lines(res)
    assert len(atoms) == 6
    for line in atoms:
        assert line.split()[0].startswith("Q")
    body = "\n".join(atoms)
    for bad in ("Br", "Cl", "S1", "Br1", "Cl1"):
        assert bad not in body


def test_writer_polymer_raises_no_fake_res():
    cell = np.array([1.6, 20.0, 20.0, 90.0, 90.0, 90.0])
    peaks = [
        _peak([0.10, 0.20, 0.20], 40.0, 0),
        _peak([0.60, 0.20, 0.20], 39.0, 1),
    ]
    result = SimpleNamespace(
        cell=cell,
        space_group_hm="P 1",
        method="test",
        diagnostics={},
        peaks=peaks,
    )

    buf = io.StringIO()

    def _go():
        with redirect_stdout(buf):
            write_shelxl_res(result, element="Q", n_non_h_budget=34)

    _assert_raises_polymer(_go)
    gate = format_trial_res_gate(sg="P 1", non_h=0, finite=False, pass_=False)
    assert gate in buf.getvalue()
    assert buf.getvalue().count("GATE ") == 1


def test_format_trial_res_gate_sample():
    assert (
        format_trial_res_gate(sg="P 21 21 21", non_h=34, finite=True, pass_=True)
        == "GATE sg=P 21 21 21 non_h=34 finite=yes pass=yes"
    )
    assert (
        format_trial_res_gate(sg="P 1", non_h=0, finite=False, pass_=False)
        == "GATE sg=P 1 non_h=0 finite=no pass=no"
    )


def test_writer_discrete_gate_stdout_pass_yes():
    fracs = _chain_fracs(6, start=(0.12, 0.22, 0.32), step_cart=(1.5, 0.0, 0.0))
    peaks = [_peak(f, 40.0 - i, rank=i) for i, f in enumerate(fracs)]
    result = SimpleNamespace(
        cell=_CELL,
        space_group_hm="P 21 21 21",
        method="test",
        diagnostics={},
        peaks=peaks,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = write_shelxl_res(result, element="Q", n_non_h_budget=34)
    assert "REM gate sg=P 21 21 21 non_h=6 finite=yes pass=yes" in res
    assert "GATE sg=P 21 21 21 non_h=6 finite=yes pass=yes" in buf.getvalue()
