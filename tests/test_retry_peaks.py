"""CLI/GUI-parity peaks retry (second-pass partial_phaseed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grok_phase_solver.pipeline.peaks import DensityPeak
from grok_phase_solver.pipeline.retry import (
    retry_config,
    should_retry_with_peaks,
)
from grok_phase_solver.pipeline.solve import SolveConfig, SolveResult


def _result(**kwargs) -> SolveResult:
    peaks = kwargs.pop("peaks", None)
    if peaks is None:
        peaks = [
            DensityPeak(rank=i, fract=np.array([0.1 * i, 0.2, 0.3]), height=4.0, height_sigma=3.0)
            for i in range(6)
        ]
    return SolveResult(
        hkl=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        amplitudes=np.ones(3),
        phases=np.zeros(3),
        density=np.zeros((8, 8, 8)),
        cell=np.array([10.0, 11.0, 12.0, 90.0, 90.0, 90.0]),
        space_group_hm="P 1",
        d_min=1.2,
        peaks=peaks,
        **kwargs,
    )


def test_retry_when_cf_weak():
    ok, reason = should_retry_with_peaks(
        _result(method="charge_flipping", diagnostics={"free_fom_composite": 0.31})
    )
    assert ok is True
    assert "partial_phaseed" in reason


def test_skip_when_already_seeded():
    ok, reason = should_retry_with_peaks(
        _result(
            method="partial_phaseed",
            diagnostics={"free_fom_composite": 0.31, "seed_kind": "fragment"},
        )
    )
    assert ok is False
    assert "seed" in reason.lower()


def test_skip_when_healthy_fom():
    ok, reason = should_retry_with_peaks(
        _result(method="ensemble", diagnostics={"free_fom_composite": 0.82})
    )
    assert ok is False
    assert "healthy" in reason.lower()


def test_skip_few_peaks():
    ok, _ = should_retry_with_peaks(
        _result(
            method="charge_flipping",
            diagnostics={"free_fom_composite": 0.2},
            peaks=[],
        )
    )
    assert ok is False


def test_retry_config_clears_other_seeds():
    cfg = SolveConfig(
        method="auto",
        phase_seed_csv="old.csv",
        patterson_ha=True,
        n_iter=40,
    )
    cfg2 = retry_config(cfg, Path("peaks.csv"))
    assert cfg2.method == "partial_phaseed"
    assert cfg2.seed_peaks_csv.endswith("peaks.csv")
    assert cfg2.phase_seed_csv is None
    assert cfg2.patterson_ha is False
    assert cfg.method == "auto"  # original unchanged


def test_cli_retry_flag_and_demo_second_pass(tmp_path: Path):
    from grok_phase_solver.cli import solve_main

    root = Path(__file__).resolve().parents[1]
    hkl = root / "examples" / "demo_solve" / "demo.hkl"
    ins = root / "examples" / "demo_solve" / "demo.ins"
    if not hkl.exists():
        return
    out = tmp_path / "out"
    solve_main(
        [
            "--hkl",
            str(hkl),
            "--ins",
            str(ins),
            "--method",
            "charge_flipping",
            "--n-iter",
            "25",
            "--n-peaks",
            "10",
            "--quiet",
            "--retry-with-peaks",
            "--out",
            str(out),
        ]
    )
    assert (out / "report.md").exists()
    assert (out / "peaks.csv").exists()
    # Easy demo may skip retry if FOM looks healthy; either path is valid.
    report = (out / "report.md").read_text()
    retry_dir = out / "retry_peaks"
    if retry_dir.is_dir():
        assert (retry_dir / "report.md").exists()
        assert "Retry with peaks" in report
        assert (retry_dir / "solve_summary.json").exists()
    else:
        # skipped: still a successful first pass
        assert "peaks.csv" in report or "Next action" in report
