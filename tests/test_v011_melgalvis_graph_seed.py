"""v0.11: Melgalvis large-cell, GraphPhaseNet v8 features, seed filters."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic_melgalvis import (
    actas2026_config,
    build_ring_scaffold,
    generate_melgalvis_structure,
    ha_heavy_config,
    large_cell_config,
    iter_melgalvis_samples,
    sample_volume,
)
from grok_phase_solver.models.graph_phase_net import (
    node_features_from_graph,
    prepare_graph_batch,
)
from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.solvers.ai_phaseed import (
    filter_seed_by_bin_quality,
    recommend_seed_fraction,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.projectors import unit_cell_volume
from grok_phase_solver.metrics.seed_quality import format_seed_class_diagnostics
from grok_phase_solver.metrics.stratified_prior import stratify_by_volume


def test_large_cell_preset_volume_band():
    cfg = large_cell_config()
    rng = np.random.default_rng(0)
    vols = [sample_volume(rng, cfg) for _ in range(40)]
    assert all(cfg.v_min <= v <= cfg.v_max for v in vols)
    # Median should sit in the AI-PhaSeed-friendly band
    assert np.median(vols) >= 1000.0
    st = generate_melgalvis_structure(seed=11, cfg=cfg)
    vol = float(unit_cell_volume(np.asarray(st.cell, dtype=np.float64)))
    assert 700.0 <= vol <= 4000.0  # packing may clip slightly


def test_ring_scaffold_and_ha_z19():
    rng = np.random.default_rng(2)
    els, xyz = build_ring_scaffold(rng, kind="phenyl")
    assert len(els) == 6
    assert xyz.shape == (6, 3)
    # HA-heavy should often inject Br/I
    max_zs = []
    for s in range(20):
        st = generate_melgalvis_structure(seed=s, cfg=ha_heavy_config())
        zmap = {"Br": 35, "BR": 35, "I": 53, "Cl": 17, "CL": 17, "S": 16, "P": 15}
        mz = max(zmap.get(a.element, zmap.get(a.element.upper(), 6)) for a in st.atoms)
        max_zs.append(mz)
    assert max(max_zs) >= 17
    assert any(z >= 19 for z in max_zs)


def test_iter_large_preset():
    rows = iter_melgalvis_samples(4, seed=3, preset="large", d_min=1.5)
    assert len(rows) == 4
    assert rows[0]["generator"] == "melgalvis2026"
    assert "cell_volume" in rows[0]


def test_graph_features_v8_d_in_26():
    st = generate_melgalvis_structure(seed=5, cfg=actas2026_config())
    data = structure_to_fcalc(st, d_min=1.5)
    batch = prepare_graph_batch(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        max_reflections=80,
        feature_version=8,
    )
    assert batch["d_in"] == 26
    assert batch["X"].shape[1] == 26
    assert batch["feature_version"] == 8


def test_recommend_seed_vol_band():
    # ~30³ orthogonal ≈ 27000? use ~12³ ≈ 1728
    cell = np.array([12.0, 12.0, 12.0, 90.0, 90.0, 90.0])
    r = recommend_seed_fraction(200, cell, d_min=1.2)
    assert r["vol_band"] == "vol_1000_3500"
    assert r["seed_fraction"] >= 0.25
    assert "practical_bar_note" in r
    assert r["method"].startswith("carrozzini_heuristic_v11")


def test_filter_seed_e_floor_and_entropy():
    n = 40
    ph = np.linspace(-np.pi, np.pi, n)
    idx = np.arange(n)
    E = np.linspace(0.5, 3.0, n)
    out, meta = filter_seed_by_bin_quality(
        ph, idx, e_values=E, e_min=1.2, max_entropy=0.5, multi_bin=True
    )
    assert len(out) >= 8
    assert "seed_bin_entropy" in meta


def test_seed_diagnostics_md():
    md = format_seed_class_diagnostics(
        {
            "predicted_class": 0,
            "success_probability": 0.2,
            "predicted_mpe_deg": 70.0,
            "predicted_corr": 0.1,
            "method": "heuristic",
            "features": {"Vol": 1500.0, "seed_fraction": 0.25},
            "warning": "low quality",
        }
    )
    assert "Class" in md
    assert "30%" in md or "partial" in md.lower()


def test_stratify_by_volume():
    rows = [
        {"cell_volume": 500, "frac_within_20": 0.1, "seedOK": False, "strong_mpe_oi": 80},
        {"cell_volume": 2000, "frac_within_20": 0.25, "seedOK": True, "strong_mpe_oi": 50},
        {"cell_volume": 4000, "frac_within_20": 0.15, "seedOK": False, "strong_mpe_oi": 70},
    ]
    s = stratify_by_volume(rows)
    assert s["vol_lt_1000"]["n"] == 1
    assert s["vol_1000_3500"]["n"] == 1
    assert s["vol_gt_3500"]["n"] == 1
