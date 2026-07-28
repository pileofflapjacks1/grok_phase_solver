"""Tests for improved fragment / predicted-model seeding (SG expand + strong |E|)."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.metrics.phase_error import mean_phase_error
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.direct_methods import normalize_E
from grok_phase_solver.solvers.partial_seed import fragment_seed_phases
from grok_phase_solver.solvers.seed_import import (
    select_fragment_atoms,
    seed_from_fragment_atoms,
)
from grok_phase_solver.physics.symmetry import expand_fractional_coords


def test_select_heaviest_cluster():
    fracs = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.12, 0.11, 0.1],
            [0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9],
        ]
    )
    els = ["C", "C", "Br", "O"]
    fr, el, meta = select_fragment_atoms(
        fracs, els, max_atoms=2, mode="heaviest_cluster"
    )
    assert len(fr) == 2
    assert "Br" in el  # heaviest seed
    assert meta["n_selected"] == 2


def test_fragment_expand_lowers_mpe_on_cod_like():
    """ASU-only half fragment is weak; expanded cluster should be much better."""
    st = load_cif("data/raw/cod/2016452.cif")
    data = structure_to_fcalc(st, d_min=1.0)
    hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
    fracs = np.array(
        [a.fract for a in st.atoms if a.element.upper() not in ("H", "D")],
        dtype=np.float64,
    )
    els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
    n_half = max(3, len(els) // 2)

    # Legacy-style: first half, no expand, no full prior
    sph0, m0, _ = seed_from_fragment_atoms(
        hkl,
        amp,
        st.cell,
        fracs[:n_half],
        els[:n_half],
        expand_symmetry=False,
        prefer_strong_E=False,
        fcalc_min_rel=0.15,
        atom_mode="all",
        full_fcalc_prior=False,
        auto_b_iso=False,
    )
    # Improved: heaviest cluster + SG expand + strong E + full Fcalc prior
    sph1, m1, meta1 = seed_from_fragment_atoms(
        hkl,
        amp,
        st.cell,
        fracs,
        els,
        expand_symmetry=True,
        space_group=st.space_group_hm,
        atom_mode="heaviest_cluster",
        max_atoms=n_half,
        prefer_strong_E=True,
        fcalc_min_rel=0.06,
        target_strong_frac=0.30,
        full_fcalc_prior=True,
        auto_b_iso=True,
    )
    E = normalize_E(hkl, amp, st.cell)
    strong = np.argsort(-E)[: max(1, int(0.3 * len(E)))]

    def mpe_strong(sph, mask):
        sm = mask[strong]
        if sm.sum() < 5:
            return 90.0
        return float(mean_phase_error(sph[strong][sm], ph_t[strong][sm]))

    mpe0 = mpe_strong(sph0, m0)
    mpe1 = mpe_strong(sph1, m1)
    assert meta1.get("expand", {}).get("expanded") is True
    assert meta1.get("full_fcalc_prior") is True
    assert mpe1 < mpe0 - 15.0 or mpe1 < 25.0
    assert m1[strong].mean() >= 0.25


def test_full_fcalc_prior_fills_all_reflections():
    """seed_phases must carry Fcalc φ everywhere (soft prior), not random off-mask."""
    st = load_cif("data/raw/cod/2016452.cif")
    data = structure_to_fcalc(st, d_min=1.0)
    hkl, amp = data["hkl"], data["amplitudes"]
    fracs = np.array(
        [a.fract for a in st.atoms if a.element.upper() not in ("H", "D")],
        dtype=np.float64,
    )
    els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
    n_half = max(3, len(els) // 2)
    fr_sel, el_sel, _ = select_fragment_atoms(
        fracs, els, max_atoms=n_half, mode="heaviest_cluster", seed=0
    )
    sph, mask, meta = seed_from_fragment_atoms(
        hkl,
        amp,
        st.cell,
        fr_sel,
        el_sel,
        expand_symmetry=True,
        space_group=st.space_group_hm,
        full_fcalc_prior=True,
        auto_b_iso=False,
        b_iso=8.0,
        seed=0,
    )
    fr2, el2, _ = expand_fractional_coords(
        fr_sel, st.space_group_hm, elements=el_sel
    )
    ph_fc, _ = fragment_seed_phases(hkl, fr2, el2, st.cell, b_iso=8.0)
    # All reflections should match Fcalc phases (within float noise)
    d = np.angle(np.exp(1j * (sph - ph_fc)))
    assert float(np.max(np.abs(d))) < 1e-6
    assert meta["full_fcalc_prior"] is True
    assert int(mask.sum()) >= 8
    # Legacy path still random-fills outside mask
    sph_leg, mask_leg, meta_leg = seed_from_fragment_atoms(
        hkl,
        amp,
        st.cell,
        fr_sel,
        el_sel,
        expand_symmetry=True,
        space_group=st.space_group_hm,
        full_fcalc_prior=False,
        auto_b_iso=False,
        b_iso=8.0,
        seed=0,
    )
    assert meta_leg["full_fcalc_prior"] is False
    off = ~mask_leg
    if off.sum() > 10:
        d_off = np.angle(np.exp(1j * (sph_leg[off] - ph_fc[off])))
        assert float(np.mean(np.abs(d_off))) > 0.5  # not all equal to Fcalc
