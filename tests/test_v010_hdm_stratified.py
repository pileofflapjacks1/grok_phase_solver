"""v0.10: HDM projector, seed bin filter, HA config, stratified metrics."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.synthetic_melgalvis import ha_heavy_config, generate_melgalvis_structure
from grok_phase_solver.metrics.stratified_prior import (
    max_Z_from_elements,
    is_ha_bearing,
    stratify_holdout_rows,
)
from grok_phase_solver.solvers.ai_phaseed import (
    filter_seed_by_bin_quality,
    recommend_seed_fraction,
    select_seed_indices,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.iterative_retrieval import hybrid_difference_map_solve


def test_hdm_runs_on_small_organic():
    st = generate_random_organic(n_atoms=6, seed=9)
    data = structure_to_fcalc(st, d_min=1.4)
    ph, rho, hist = hybrid_difference_map_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_iter=12,
        protein_mode=True,
        seed=0,
        d_min=1.4,
        verbose=False,
    )
    assert hist.get("algorithm") == "hybrid_difference_map"
    assert hist.get("research_only") is True
    assert len(ph) == len(data["amplitudes"])
    assert rho.ndim == 3


def test_filter_seed_bin_quality():
    st = generate_random_organic(n_atoms=8, seed=2)
    data = structure_to_fcalc(st, d_min=1.3)
    idx = select_seed_indices(
        data["hkl"], data["amplitudes"], st.cell, seed_fraction=0.25
    )
    # random phases → high entropy → thinned
    rng = np.random.default_rng(0)
    ph = rng.uniform(-np.pi, np.pi, size=len(data["amplitudes"]))
    out, meta = filter_seed_by_bin_quality(ph, idx, max_entropy=0.5)
    assert len(out) >= 8
    assert "seed_bin_entropy" in meta


def test_recommend_seed_fraction_v10():
    cell = np.array([12.0, 14.0, 16.0, 90.0, 105.0, 90.0])
    r = recommend_seed_fraction(300, cell=cell, d_min=1.2)
    assert r["method"].startswith("carrozzini_heuristic")
    assert 0.10 <= r["seed_fraction"] <= 0.50


def test_ha_heavy_config_and_stratify():
    cfg = ha_heavy_config(p_heavy_atom=1.0)
    st = generate_melgalvis_structure(seed=7, cfg=cfg)
    els = [a.element for a in st.atoms]
    assert max_Z_from_elements(els) >= 1
    rows = [
        {"frac_within_20": 0.3, "seedOK": True, "strong_mpe_oi": 50.0, "max_Z": 35, "ha_bearing": True, "space_group": "P1"},
        {"frac_within_20": 0.15, "seedOK": False, "strong_mpe_oi": 70.0, "max_Z": 8, "ha_bearing": False, "space_group": "P1"},
    ]
    strat = stratify_holdout_rows(rows)
    assert strat["all"]["n"] == 2
    assert strat["ha_bearing_Zge17"]["n"] == 1
