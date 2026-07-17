"""GUI backend tests (no Streamlit required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from grok_phase_solver.gui.backend import (
    demo_paths,
    format_user_error,
    map_quality_hints,
    parse_cell_string,
    resolve_wizard,
    run_solve_job,
    shelxl_handoff_snippet,
    zip_outdir,
)

ROOT = Path(__file__).resolve().parents[1]


def test_demo_paths_exist():
    d = demo_paths()
    assert d["easy_hkl"].exists()
    assert d["easy_ins"].exists()


def test_parse_cell_comma():
    cell, wl, notes = parse_cell_string("9.75,8.89,7.57,90,112.7,90")
    assert cell.startswith("9.75")
    assert wl is None


def test_parse_cell_shelx_line():
    cell, wl, notes = parse_cell_string(
        "CELL 0.71073 9.748 8.890 7.566 90 112.74 90"
    )
    assert wl == "0.71073"
    parts = [float(x) for x in cell.split(",")]
    assert abs(parts[0] - 9.748) < 1e-6
    assert abs(parts[4] - 112.74) < 1e-6
    assert any("wavelength" in n.lower() for n in notes)


def test_parse_cell_bad():
    with pytest.raises(ValueError):
        parse_cell_string("1 2 3")


def test_resolve_wizard_easy():
    w = resolve_wizard("easy", "auto")
    assert w["method"] == "ensemble"
    assert w["n_starts"] >= 2


def test_resolve_wizard_fragment():
    w = resolve_wizard("have_fragment", "ensemble")
    assert w["method"] == "partial_phaseed"


def test_resolve_wizard_advanced():
    w = resolve_wizard("advanced", "raar")
    assert w["method"] == "raar"


def test_format_user_error_seed():
    msg = format_user_error(ValueError("No phase seed source provided"))
    assert "Hint" in msg


def test_map_quality_hints_low_fom():
    hints = map_quality_hints(
        {"diagnostics": {"free_fom_composite": 0.3}, "n_peaks": 2, "method": "charge_flipping"}
    )
    assert any("low" in h.lower() or "unsolved" in h.lower() for h in hints)


def test_shelxl_snippet():
    s = shelxl_handoff_snippet("/tmp/out", hkl_name="data.hkl")
    assert "trial.res" in s
    assert "shelxl" in s.lower()


def test_run_solve_job_demo(tmp_path):
    demos = demo_paths()
    steps = []
    summary = run_solve_job(
        tmp_path,
        hkl_bytes=demos["easy_hkl"].read_bytes(),
        hkl_name="demo.hkl",
        ins_bytes=demos["easy_ins"].read_bytes(),
        ins_name="demo.ins",
        method="charge_flipping",
        n_iter=25,
        n_starts=1,
        n_extend=4,
        n_peaks=15,
        verbose=False,
        progress=steps.append,
        capture_log=True,
    )
    assert summary["ok"] is True
    assert summary["method"] == "charge_flipping"
    assert summary["n_reflections"] > 50
    assert summary["hints"]
    assert "shelxl_snippet" in summary
    assert steps  # progress called
    out = Path(summary["out_dir"])
    assert (out / "report.md").exists()
    assert (out / "phases.csv").exists()
    assert (out / "trial.res").exists() or summary["n_peaks"] == 0
    z = zip_outdir(out, tmp_path / "out.zip")
    assert z.exists() and z.stat().st_size > 100


def test_run_solve_job_with_parsed_cell(tmp_path):
    demos = demo_paths()
    # Force cell path: use HKL without relying on ins cell — still pass ins
    cell, _, _ = parse_cell_string("CELL 0.71 10 10 10 90 90 90")
    summary = run_solve_job(
        tmp_path,
        hkl_bytes=demos["easy_hkl"].read_bytes(),
        ins_bytes=demos["easy_ins"].read_bytes(),
        cell=cell,
        method="charge_flipping",
        n_iter=15,
        n_starts=1,
        n_extend=3,
        verbose=False,
    )
    assert summary["ok"]


def test_partial_seed_demo_job(tmp_path):
    demos = demo_paths()
    if not demos["hard_seed_csv"].exists():
        pytest.skip("partial seed demo missing")
    summary = run_solve_job(
        tmp_path,
        hkl_bytes=demos["hard_hkl"].read_bytes(),
        ins_bytes=demos["hard_ins"].read_bytes(),
        method="partial_phaseed",
        phase_seed_csv_bytes=demos["hard_seed_csv"].read_bytes(),
        n_iter=30,
        n_starts=1,
        n_extend=6,
        verbose=False,
    )
    assert summary["method"] == "partial_phaseed"
    assert (Path(summary["out_dir"]) / "report.md").exists()


def test_partial_without_seed_raises(tmp_path):
    demos = demo_paths()
    with pytest.raises(RuntimeError, match="seed"):
        run_solve_job(
            tmp_path,
            hkl_bytes=demos["easy_hkl"].read_bytes(),
            ins_bytes=demos["easy_ins"].read_bytes(),
            method="partial_phaseed",
            n_iter=10,
            n_starts=1,
            verbose=False,
        )
