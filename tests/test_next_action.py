"""Vol-band next-action chooser (report.md / GUI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grok_phase_solver.pipeline.export import export_solution, _render_report
from grok_phase_solver.pipeline.next_action import (
    classify_vol_band,
    format_next_action_md,
    recommend_next_action,
)
from grok_phase_solver.pipeline.peaks import DensityPeak
from grok_phase_solver.pipeline.solve import SolveResult


def test_classify_vol_band():
    assert classify_vol_band(605) == "vol_lt_1000"
    assert classify_vol_band(1000) == "vol_1000_3500"
    assert classify_vol_band(1027) == "vol_1000_3500"
    assert classify_vol_band(3500) == "vol_1000_3500"
    assert classify_vol_band(4676) == "vol_gt_3500"


def test_mid_band_unsolved_points_at_fragment():
    rec = recommend_next_action(
        cell=[12.0, 12.0, 12.0, 90.0, 90.0, 90.0],  # 1728 Å³
        d_min=1.2,
        method="ensemble",
        n_reflections=400,
        n_peaks=6,
        diagnostics={"free_fom_composite": 0.32},
    )
    assert rec["vol_band"] == "vol_1000_3500"
    assert rec["primary_id"] == "fragment_or_predicted"
    assert rec["map_outlook"] == "likely_unsolved"
    md = format_next_action_md(rec)
    assert "partial_phaseed" in md
    assert any("retry-with-peaks" in a for a in rec["alternatives"])
    assert "0.71" in rec["why"] or "0.71" in md


def test_large_cell_wants_ha_or_big_fragment():
    rec = recommend_next_action(
        cell=[20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
        d_min=1.5,
        method="charge_flipping",
        n_peaks=3,
        diagnostics={"free_fom_composite": 0.25},
    )
    assert rec["vol_band"] == "vol_gt_3500"
    assert rec["primary_id"] == "large_fragment_or_ha"
    assert "ha_phaseed" in " ".join(rec["commands"])


def test_small_highres_cf_suggests_ensemble():
    rec = recommend_next_action(
        cell=[8.0, 8.0, 8.0, 90.0, 90.0, 90.0],
        d_min=0.95,
        method="charge_flipping",
        n_peaks=4,
        diagnostics={"free_fom_composite": 0.30},
    )
    assert rec["vol_band"] == "vol_lt_1000"
    assert rec["primary_id"] == "try_ensemble"


def test_healthy_outlook_is_refine():
    rec = recommend_next_action(
        cell=[8.0, 8.0, 8.0, 90.0, 90.0, 90.0],
        d_min=0.9,
        method="ensemble",
        n_peaks=12,
        diagnostics={"free_fom_composite": 0.82},
    )
    assert rec["primary_id"] == "refine_shelxl"
    assert rec["map_outlook"] == "looks_healthy"


def test_undersized_seed_says_enlarge():
    rec = recommend_next_action(
        cell=[12.0, 12.0, 12.0, 90.0, 90.0, 90.0],
        d_min=1.2,
        method="partial_phaseed",
        n_peaks=5,
        diagnostics={
            "free_fom_composite": 0.40,
            "seed_quality": {"size_meets_bar": False, "n_seed": 10, "frac_strong_seeded": 0.1},
        },
    )
    assert rec["already_seeded"] is True
    assert rec["primary_id"] == "enlarge_seed"


def test_seeded_but_weak_says_better_source():
    rec = recommend_next_action(
        cell=[12.0, 12.0, 12.0, 90.0, 90.0, 90.0],
        d_min=1.2,
        method="partial_phaseed",
        n_peaks=5,
        diagnostics={
            "free_fom_composite": 0.38,
            "seed_kind": "fragment",
            "seed_quality": {"size_meets_bar": True, "n_seed": 80, "frac_strong_seeded": 0.35},
        },
    )
    assert rec["primary_id"] == "better_seed"


def test_report_includes_next_action_section(tmp_path: Path):
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    amp = np.ones(4)
    phases = np.zeros(4)
    cell = np.array([12.0, 12.0, 12.0, 90.0, 90.0, 90.0])
    rho = np.zeros((8, 8, 8))
    rho[2, 2, 2] = 5.0
    result = SolveResult(
        hkl=hkl,
        amplitudes=amp,
        phases=phases,
        density=rho,
        cell=cell,
        space_group_hm="P 1",
        method="ensemble",
        d_min=1.2,
        peaks=[
            DensityPeak(
                rank=1,
                fract=np.array([0.25, 0.25, 0.25]),
                height=5.0,
                height_sigma=3.0,
            )
        ],
        diagnostics={"free_fom_composite": 0.31},
    )
    md = _render_report(result)
    assert "## Next action" in md
    assert "Vol 1000" in md
    assert "partial_phaseed" in md
    written = export_solution(result, tmp_path)
    names = {p.name for p in written}
    assert "report.md" in names
    assert "solve_summary.json" in names
    import json

    summary = json.loads((tmp_path / "solve_summary.json").read_text())
    assert summary["next_action"]["primary_id"] == "fragment_or_predicted"
    assert summary["next_action"]["vol_band"] == "vol_1000_3500"
