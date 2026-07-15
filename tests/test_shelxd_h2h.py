"""Tests for SHELX I/O, dual-space baseline, and SHELXD runner hooks."""

from __future__ import annotations

import numpy as np
import pytest

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.io.shelx_write import (
    parse_shelx_res_atoms,
    write_hkl_shelx,
    write_shelxd_ins,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.dual_space import dual_space_solve
from grok_phase_solver.solvers.shelxd_runner import (
    find_shelxd,
    shelxd_available,
    shelxd_or_dual_space,
)


def test_write_shelx_hkl_ins(tmp_path):
    st = generate_random_organic(n_atoms=8, seed=0)
    data = structure_to_fcalc(st, d_min=1.2)
    hkl_path = tmp_path / "job.hkl"
    ins_path = tmp_path / "job.ins"
    write_hkl_shelx(hkl_path, data["hkl"], amplitudes=data["amplitudes"])
    write_shelxd_ins(ins_path, st.cell, n_find=8, n_try=10, seed=1, title="test")
    text = hkl_path.read_text()
    assert "0    0    0" in text or text.strip().endswith("0")
    ins = ins_path.read_text()
    assert "FIND" in ins
    assert "HKLF 4" in ins
    assert "CELL" in ins


def test_parse_res_atoms(tmp_path):
    res = tmp_path / "job.res"
    res.write_text(
        "TITL test\n"
        "CELL 0.71 10 10 10 90 90 90\n"
        "SFAC C H\n"
        "C1 1 0.1 0.2 0.3 11.000 0.05\n"
        "C2 1 0.4 0.5 0.6 11.000 0.05\n"
        "HKLF 4\n"
        "END\n"
    )
    atoms = parse_shelx_res_atoms(res)
    assert len(atoms) == 2
    assert abs(atoms[0]["fract"][0] - 0.1) < 1e-9


def test_dual_space_runs():
    st = generate_random_organic(n_atoms=8, seed=1)
    data = structure_to_fcalc(st, d_min=1.1)
    ph, rho, info = dual_space_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_atoms=8,
        n_cycles=12,
        n_starts=3,
        seed=0,
        d_min=1.1,
        polish_cf_iters=10,
    )
    assert len(ph) == len(data["phases"])
    assert rho.ndim == 3
    assert info["method"] == "dual_space"
    assert "not SHELXD" in info["note"].lower() or "Not SHELXD" in info["note"]


def test_shelxd_or_dual_fallback():
    """Without shelxd binary, must use dual_space backend."""
    if shelxd_available():
        pytest.skip("real SHELXD present — skip fallback test")
    st = generate_random_organic(n_atoms=7, seed=2)
    data = structure_to_fcalc(st, d_min=1.2)
    ph, rho, info = shelxd_or_dual_space(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_atoms=7,
        dual_space_starts=2,
        dual_space_cycles=8,
        seed=0,
        d_min=1.2,
    )
    assert info["backend"] == "dual_space"
    assert len(ph) == len(data["phases"])


def test_find_shelxd_missing():
    assert find_shelxd("/nonexistent/path/to/shelxd") is None


def test_pipeline_dual_space_method():
    from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
    from grok_phase_solver.io.hkl import ReflectionTable, write_hkl_simple
    import tempfile
    from pathlib import Path

    st = generate_random_organic(n_atoms=6, seed=3)
    data = structure_to_fcalc(st, d_min=1.0)
    with tempfile.TemporaryDirectory() as td:
        hkl_path = Path(td) / "t.hkl"
        table = ReflectionTable(
            hkl=data["hkl"], F_meas=data["amplitudes"], cell=st.cell
        )
        write_hkl_simple(hkl_path, table)
        cell = ",".join(str(x) for x in st.cell)
        res = solve_structure(
            str(hkl_path),
            cell=cell,
            space_group="P1",
            config=SolveConfig(
                method="dual_space", n_iter=30, n_starts=2, n_peaks=10, verbose=False
            ),
        )
        assert res.method == "dual_space"
        assert len(res.phases) == len(data["phases"])
