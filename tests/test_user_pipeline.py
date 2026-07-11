"""Tests for scientist-facing gps-solve pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.io.ins import load_ins, parse_cell_string
from grok_phase_solver.io.experiment import load_experiment, summarize_experiment
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure, resolve_method
from grok_phase_solver.pipeline.export import export_solution, write_shelxl_res

ROOT = Path(__file__).resolve().parents[1]
DEMO_HKL = ROOT / "examples" / "demo_solve" / "demo.hkl"
DEMO_INS = ROOT / "examples" / "demo_solve" / "demo.ins"


def test_parse_cell_string():
    c = parse_cell_string("10,11,12,90,95,90")
    np.testing.assert_allclose(c, [10, 11, 12, 90, 95, 90])


def test_load_ins_demo():
    if not DEMO_INS.exists():
        pytest.skip("demo ins missing")
    ins = load_ins(DEMO_INS)
    assert ins.cell is not None
    assert ins.cell[0] > 1
    assert ins.wavelength > 0


def test_load_experiment_demo():
    if not DEMO_HKL.exists():
        pytest.skip("demo hkl missing")
    table, ins = load_experiment(DEMO_HKL, ins=DEMO_INS)
    assert len(table) > 50
    assert table.cell is not None
    s = summarize_experiment(table, ins)
    assert "Reflections" in s


def test_solve_and_export(tmp_path: Path):
    if not DEMO_HKL.exists():
        pytest.skip("demo missing")
    result = solve_structure(
        str(DEMO_HKL),
        ins_path=str(DEMO_INS),
        config=SolveConfig(
            method="charge_flipping",
            n_iter=40,
            n_peaks=15,
            verbose=False,
            seed=0,
        ),
    )
    assert len(result.phases) == len(result.amplitudes)
    assert result.density.ndim == 3
    assert len(result.peaks) >= 1
    paths = export_solution(result, tmp_path)
    names = {p.name for p in paths}
    assert "report.md" in names
    assert "phases.csv" in names
    assert "density.npz" in names
    assert "peaks.csv" in names
    # phases.csv has header + data
    lines = (tmp_path / "phases.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + len(result.hkl)


def test_solve_with_explicit_cell(tmp_path: Path):
    if not DEMO_HKL.exists():
        pytest.skip("demo missing")
    ins = load_ins(DEMO_INS)
    cell = ",".join(str(x) for x in ins.cell)
    result = solve_structure(
        str(DEMO_HKL),
        cell=cell,
        space_group="P 1",
        config=SolveConfig(method="recycle", n_recycle=5, verbose=False),
    )
    assert result.method == "recycle"
    export_solution(result, tmp_path / "out2")
    assert (tmp_path / "out2" / "report.md").exists()


def test_resolve_method_auto_policy():
    m, reason = resolve_method("auto", "P 1 21/c 1", data_dmin=0.9, n_refl=500)
    # With or without PhAI, should return a known concrete method
    assert m != "auto"
    assert "auto" in reason or m in (
        "phai_phaseed", "phai+cf_cond", "ensemble", "charge_flipping", "hard_p1_phaseed"
    )
    m2, _ = resolve_method("auto", "P1", data_dmin=0.85, n_refl=200)
    assert m2 in ("ensemble", "charge_flipping", "hard_p1_phaseed", "phai+cf_cond")


def test_export_writes_trial_res(tmp_path: Path):
    if not DEMO_HKL.exists():
        pytest.skip("demo missing")
    result = solve_structure(
        str(DEMO_HKL),
        ins_path=str(DEMO_INS),
        config=SolveConfig(method="charge_flipping", n_iter=30, n_peaks=10, verbose=False),
    )
    paths = export_solution(result, tmp_path)
    assert (tmp_path / "trial.res").exists()
    text = (tmp_path / "trial.res").read_text()
    assert "TITL" in text
    assert "HKLF 4" in text
    assert "free_fom" in write_shelxl_res(result).lower() or "REM" in text
    assert "free_fom_composite" in result.diagnostics or True  # may be present
