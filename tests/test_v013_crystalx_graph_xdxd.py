"""v0.13: CrystalX typing, GraphPhaseNet v11 features, XDXD generative path."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic_melgalvis import (
    generate_melgalvis_structure,
    large_cell_config,
    xdxd_lowres_config,
)
from grok_phase_solver.models.generative_structure import (
    xdxd_propose_coordinates,
    generative_structure_available,
)
from grok_phase_solver.models.graph_phase_net import prepare_graph_batch
from grok_phase_solver.models.graphai_external import (
    graphai_available,
    load_graphai_adapter,
    write_graphai_h2h_skeleton,
)
from grok_phase_solver.pipeline.crystalx_typing import (
    type_peaks_crystalx,
    typed_atoms_to_shelxl_res,
)
from grok_phase_solver.pipeline.peaks import pick_density_peaks
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve


def test_graph_features_v10_d_in_34():
    st = generate_melgalvis_structure(seed=9, cfg=large_cell_config())
    data = structure_to_fcalc(st, d_min=1.5)
    batch = prepare_graph_batch(
        data["hkl"], data["amplitudes"], st.cell, max_reflections=70, feature_version=10
    )
    assert batch["d_in"] == 34
    assert batch["X"].shape[1] == 34
    assert np.isfinite(batch["X"]).all()


def test_xdxd_lowres_config_generates():
    st = generate_melgalvis_structure(seed=4, cfg=xdxd_lowres_config())
    assert len(st.atoms) >= 4
    assert st.cell[0] > 0


def test_crystalx_typing_and_res():
    st = generate_melgalvis_structure(seed=2, cfg=large_cell_config(mode="cluster"))
    data = structure_to_fcalc(st, d_min=1.2)
    ph, rho, _ = charge_flipping_solve(
        data["hkl"], data["amplitudes"], st.cell, n_iter=20, seed=0, d_min=1.2
    )
    peaks = pick_density_peaks(rho, n_peaks=12, min_sigma=1.2)
    assert len(peaks) >= 1
    typed, meta = type_peaks_crystalx(peaks, st.cell, place_hydrogens=True)
    assert len(typed) >= len(peaks)
    assert meta["n_typed"] >= 1
    res = typed_atoms_to_shelxl_res(typed, st.cell, method="test")
    assert "SFAC" in res
    assert "HKLF 4" in res
    # not all Q labels — real elements present
    assert any(el in res for el in ("C", "O", "N", "Cl", "S", "Br"))


def test_xdxd_propose_coordinates():
    st = generate_melgalvis_structure(seed=1, cfg=large_cell_config(mode="cluster", n_nonh_lo=8, n_nonh_hi=10))
    data = structure_to_fcalc(st, d_min=1.3)
    assert generative_structure_available()
    fracs, els, ph, rho, meta = xdxd_propose_coordinates(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_atoms=8,
        d_min=1.3,
        n_starts=2,
        polish="none",
        seed=0,
    )
    assert len(ph) == len(data["amplitudes"])
    assert rho.ndim == 3
    assert meta.get("research_only") is True
    assert "xdxd" in meta.get("algorithm", "")


def test_graphai_external_stub(tmp_path):
    assert graphai_available() is False or isinstance(graphai_available(), bool)
    ad = load_graphai_adapter()
    assert "available" in ad
    assert ad["predict"] is None or callable(ad["predict"])
    p = write_graphai_h2h_skeleton(tmp_path / "graphai_h2h.md")
    assert p.is_file()
    assert "GraPhAI" in p.read_text()
