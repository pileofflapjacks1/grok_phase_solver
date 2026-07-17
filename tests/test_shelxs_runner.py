"""Tests for SHELXS detection, ins/hkl writers, and optional live run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.io.shelx_write import (
    parse_shelx_res_atoms,
    write_hkl_shelx,
    write_shelxs_ins,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.shelxs_runner import find_shelxs, shelxs_available, shelxs_solve


def test_write_shelxs_ins_and_fixed_hkl(tmp_path):
    st = generate_random_organic(n_atoms=8, seed=0)
    data = structure_to_fcalc(st, d_min=1.0)
    hkl_path = tmp_path / "job.hkl"
    ins_path = tmp_path / "job.ins"
    write_hkl_shelx(hkl_path, data["hkl"], amplitudes=data["amplitudes"], fixed_format=True)
    write_shelxs_ins(ins_path, st.cell, n_atoms=8, n_try=50, title="test")
    text = hkl_path.read_text().splitlines()[0]
    # fixed format: no spaces between hkl fields (positions 0:4,4:8,8:12)
    assert len(text) >= 20
    ins = ins_path.read_text()
    assert "TREF" in ins
    assert "HKLF 4" in ins


def test_parse_q_peaks():
    res = (
        "TITL t\nCELL 0.71 10 10 10 90 90 90\nSFAC C\n"
        "Q1 1 0.1 0.2 0.3 11.00000 0.05\n"
        "Q2 1 0.4 0.5 0.6 11.00000 0.05\n"
        "HKLF 4\nEND\n"
    )
    atoms = parse_shelx_res_atoms_from_text(res)
    assert len(atoms) == 2


def parse_shelx_res_atoms_from_text(text: str):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".res", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return parse_shelx_res_atoms(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_find_shelxs_repo_or_none():
    # Should find ShelX/shelxs if user installed it; else None — both OK
    p = find_shelxs()
    if p is not None:
        assert p.name.lower().startswith("shelxs") or "shelxs" in str(p).lower()


@pytest.mark.skipif(not shelxs_available(), reason="SHELXS binary not installed")
def test_shelxs_live_easy():
    st = generate_random_organic(n_atoms=8, seed=42, space_group="P1")
    data = structure_to_fcalc(st, d_min=0.9)
    ph, rho, info = shelxs_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_atoms=data["n_atoms_cell"],
        n_try=80,
        d_min=0.9,
        timeout_s=120,
    )
    assert info["status"] == "ok"
    assert info["n_peaks_parsed"] >= 1
    assert len(ph) == len(data["phases"])
    assert rho.ndim == 3
