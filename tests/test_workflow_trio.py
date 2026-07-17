"""Tests for auto policy, partial-φ demo, SHELXE helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.pipeline.solve import resolve_method
from grok_phase_solver.solvers.shelxe_runner import (
    find_shelxe,
    map_phs_to_reflections,
    parse_shelxe_phs,
    shelxe_available,
    write_model_ins,
)
from grok_phase_solver.solvers.workflow import workflow_decision_tree_md

ROOT = Path(__file__).resolve().parents[1]


def test_auto_easy_is_ensemble():
    m, r = resolve_method("auto", "P-1", data_dmin=1.0, n_refl=300)
    assert m == "ensemble"


def test_auto_hard_not_ensemble():
    m, r = resolve_method("auto", "P1", data_dmin=1.8, n_refl=150)
    assert m != "ensemble"


def test_partial_seed_demo_files_exist():
    d = ROOT / "examples" / "partial_seed_demo"
    assert (d / "demo_hard.hkl").exists()
    assert (d / "demo_hard.ins").exists()
    assert (d / "known_phases_30pct.csv").exists()
    assert (d / "README.md").exists()
    csv = (d / "known_phases_30pct.csv").read_text()
    assert "phase" in csv.lower()
    assert len(csv.strip().splitlines()) > 20


def test_decision_tree_md():
    md = workflow_decision_tree_md()
    assert "partial_phaseed" in md
    assert "ensemble" in md


def test_write_model_ins(tmp_path):
    cell = np.array([10.0, 11.0, 12.0, 90.0, 95.0, 90.0])
    fracs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    p = write_model_ins(tmp_path / "job.ins", cell, fracs, ["C", "C"])
    text = p.read_text()
    assert "SFAC" in text
    assert "C1" in text or "C2" in text


def test_map_phs_friedel():
    hkl = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    phs_h = np.array([[1, 0, 0]])
    phs_p = np.array([0.5])
    out = map_phs_to_reflections(hkl, phs_h, phs_p, fallback=np.zeros(3))
    assert abs(out[0] - 0.5) < 1e-9
    assert abs(out[1] - (-0.5)) < 1e-9


@pytest.mark.skipif(not shelxe_available(), reason="SHELXE not installed")
def test_shelxe_live_after_model():
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.solvers.baseline import structure_to_fcalc
    from grok_phase_solver.solvers.shelxe_runner import shelxe_polish

    st = generate_random_organic(n_atoms=8, seed=42)
    data = structure_to_fcalc(st, d_min=0.9)
    # partial truth model
    n = min(5, len(data["fracs"]))
    ph, rho, info = shelxe_polish(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        data["fracs"][:n],
        data["elements"][:n],
        n_mod_cycles=8,
        solvent_fraction=0.4,
        d_min=0.9,
        timeout_s=90,
    )
    assert info["status"] == "ok"
    assert len(ph) == len(data["phases"])
    assert rho.ndim == 3
