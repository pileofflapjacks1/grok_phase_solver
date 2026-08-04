"""v0.12: GraphPhaseNet v10 features + generative structure proposal."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic_melgalvis import (
    MelgalvisGenConfig,
    generate_melgalvis_structure,
    large_cell_config,
)
from grok_phase_solver.models.generative_structure import (
    generative_structure_available,
    generative_structure_propose,
    estimate_composition_from_volume,
)
from grok_phase_solver.models.graph_phase_net import prepare_graph_batch
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.ai_phaseed import recommend_seed_fraction


def test_graph_features_v9_d_in_30():
    st = generate_melgalvis_structure(seed=7, cfg=large_cell_config())
    data = structure_to_fcalc(st, d_min=1.5)
    batch = prepare_graph_batch(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        max_reflections=80,
        feature_version=9,
    )
    assert batch["d_in"] == 30
    assert batch["X"].shape[1] == 30
    assert batch["feature_version"] == 9
    assert np.isfinite(batch["X"]).all()


def test_b_factor_inflate_config():
    cfg = MelgalvisGenConfig(mode="cluster", p_b_factor_inflate=1.0, b_inflate_lo=2.0, b_inflate_hi=2.0)
    st = generate_melgalvis_structure(seed=3, cfg=cfg)
    us = [a.u_iso for a in st.atoms if a.element.upper() not in ("H", "D")]
    assert len(us) >= 1
    # at least some elevated U after forced inflate
    assert max(us) >= 0.02


def test_generative_structure_propose_runs():
    st = generate_melgalvis_structure(seed=1, cfg=MelgalvisGenConfig(mode="cluster", n_nonh_lo=8, n_nonh_hi=12))
    data = structure_to_fcalc(st, d_min=1.2)
    assert generative_structure_available()
    ph, rho, meta = generative_structure_propose(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_atoms=8,
        d_min=1.2,
        polish="none",
        seed=0,
    )
    assert len(ph) == len(data["amplitudes"])
    assert rho.ndim == 3
    assert meta.get("research_only") is True
    assert "generative" in meta.get("algorithm", "")


def test_composition_estimate():
    cell = np.array([15.0, 15.0, 15.0, 90.0, 90.0, 90.0])
    els, n = estimate_composition_from_volume(cell, ha_element="Br")
    assert n >= 4
    assert els[0] == "Br"


def test_seed_fraction_still_vol_band():
    cell = np.array([12.0, 12.0, 12.0, 90.0, 90.0, 90.0])
    r = recommend_seed_fraction(200, cell, d_min=1.2)
    assert r["vol_band"] == "vol_1000_3500"
    assert r["seed_fraction"] >= 0.25
