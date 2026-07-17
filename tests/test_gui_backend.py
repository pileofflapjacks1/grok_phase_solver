"""GUI backend tests (no Streamlit required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from grok_phase_solver.gui.backend import demo_paths, run_solve_job, zip_outdir

ROOT = Path(__file__).resolve().parents[1]


def test_demo_paths_exist():
    d = demo_paths()
    assert d["easy_hkl"].exists()
    assert d["easy_ins"].exists()


def test_run_solve_job_demo(tmp_path):
    demos = demo_paths()
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
    )
    assert summary["ok"] is True
    assert summary["method"] == "charge_flipping"
    assert summary["n_reflections"] > 50
    out = Path(summary["out_dir"])
    assert (out / "report.md").exists()
    assert (out / "phases.csv").exists()
    assert (out / "trial.res").exists() or summary["n_peaks"] == 0
    z = zip_outdir(out, tmp_path / "out.zip")
    assert z.exists() and z.stat().st_size > 100


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
