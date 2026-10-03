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
    assert "fragment_half mean mapCC ~0.72" in rec["why"]
    assert "auto ~0.16" in rec["why"]
    assert "n=4" in rec["why"]
    assert "0.72" in md


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


def test_measured_phi_above_bar_extends():
    rec = recommend_next_action(
        cell=[12.0, 12.0, 12.0, 90.0, 90.0, 90.0],
        d_min=1.5,
        method="partial_phaseed",
        n_peaks=4,
        diagnostics={
            "free_fom_composite": 0.40,
            "information_source": "measured-φ",
            "seed_quality": {
                "frac_le_20_all_strong": 0.42,
                "frac_strong_seeded": 0.42,
                "size_meets_bar": True,
            },
        },
    )
    assert rec["information_source"] == "measured-φ"
    assert rec["primary_id"] == "measured_phi_extend"
    assert "partial_phaseed" in rec["primary"]


def test_measured_phi_below_bar_does_not_polish():
    rec = recommend_next_action(
        cell=[12.0, 12.0, 12.0, 90.0, 90.0, 90.0],
        d_min=1.5,
        method="partial_phaseed",
        n_peaks=3,
        diagnostics={
            "free_fom_composite": 0.30,
            "information_source": "measured-φ",
            "seed_quality": {
                "frac_le_20_all_strong": 0.12,
                "frac_strong_seeded": 0.10,
                "size_meets_bar": False,
            },
        },
    )
    assert rec["primary_id"] == "measured_phi_improve"
    assert "do not polish" in rec["primary"].lower()


def test_retry_with_peaks_labeled_not_a_fragment():
    rec = recommend_next_action(
        cell=[12.0, 12.0, 12.0, 90.0, 90.0, 90.0],
        d_min=1.2,
        method="ensemble",
        n_peaks=6,
        diagnostics={"free_fom_composite": 0.32},
    )
    blob = " ".join(rec.get("alternatives") or [])
    assert "peaks-as-carbon, not a fragment" in blob


def test_healthy_outlook_is_olex2_handbuild():
    rec = recommend_next_action(
        cell=[8.0, 8.0, 8.0, 90.0, 90.0, 90.0],
        d_min=0.9,
        method="ensemble",
        n_peaks=12,
        diagnostics={"free_fom_composite": 0.82},
    )
    assert rec["primary_id"] == "olex2_handbuild"
    assert rec["map_outlook"] == "looks_healthy"
    assert "Do not SHELXL" in rec["primary"]
    assert any("View → Work → Info" in c for c in rec["commands"])
    assert any("built.res" in a for a in rec["alternatives"])


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
    assert "carbon-peak R1" not in md
    written = export_solution(result, tmp_path)
    names = {p.name for p in written}
    assert "report.md" in names
    assert "solve_summary.json" in names
    assert "olex2_handbuild.md" in names
    hand = (tmp_path / "olex2_handbuild.md").read_text()
    assert "View → Work → Info" in hand
    assert "Do **not** run SHELXL on the raw Q list" in hand
    import json

    summary = json.loads((tmp_path / "solve_summary.json").read_text())
    assert summary["next_action"]["primary_id"] == "fragment_or_predicted"
    assert summary["next_action"]["vol_band"] == "vol_1000_3500"


def test_fragment_report_names_the_three_gates_separately():
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    result = SolveResult(
        hkl=hkl,
        amplitudes=np.ones(4),
        phases=np.zeros(4),
        density=np.zeros((8, 8, 8)),
        cell=np.array([12.0, 12.0, 12.0, 90.0, 90.0, 90.0]),
        space_group_hm="P 1 21 1",
        method="partial_phaseed",
        d_min=1.0,
        peaks=[],
        diagnostics={
            "free_fom_composite": 0.77,
            "seed_kind": "fragment_fcalc",
            "seed_source": "predicted_model",
            "information_source": "fragment",
        },
    )
    md = _render_report(result)
    map_at = md.index("**mapCC_OI**")
    peak_at = md.index("**peak recovery**")
    r1_at = md.index("**carbon-peak R1**")
    assert map_at < peak_at < r1_at
    assert "not a SHELXL residual" in md
    assert "B = 5" in md
    assert "does not compute them" in md

    peaks_only = SolveResult(
        hkl=hkl,
        amplitudes=np.ones(4),
        phases=np.zeros(4),
        density=np.zeros((8, 8, 8)),
        cell=np.array([12.0, 12.0, 12.0, 90.0, 90.0, 90.0]),
        space_group_hm="P 1",
        method="partial_phaseed",
        d_min=1.0,
        peaks=[],
        diagnostics={"seed_kind": "seed_peaks_csv", "seed_source": "seed_peaks_csv"},
    )
    assert "carbon-peak R1" not in _render_report(peaks_only)
