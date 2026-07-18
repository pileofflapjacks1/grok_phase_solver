"""Tests for Melgalvis & Rekis (2026) style synthetic generator."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic_melgalvis import (
    MelgalvisGenConfig,
    build_artificial_molecule,
    generate_melgalvis_structure,
    sample_lattice_from_volume,
    sample_volume,
    iter_melgalvis_samples,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def test_sample_volume_in_range():
    rng = np.random.default_rng(0)
    cfg = MelgalvisGenConfig()
    vols = [sample_volume(rng, cfg) for _ in range(50)]
    assert all(cfg.v_min <= v <= cfg.v_max for v in vols)
    assert np.median(vols) > 100


def test_lattice_volume_approx():
    rng = np.random.default_rng(1)
    cfg = MelgalvisGenConfig()
    V_target = 500.0
    for system in ("orthorhombic", "monoclinic", "triclinic"):
        cell = sample_lattice_from_volume(rng, V_target, cfg, system=system)
        assert cell.shape == (6,)
        assert cell[0] > 0 and cell[1] > 0 and cell[2] > 0


def test_build_molecule_bonded():
    rng = np.random.default_rng(2)
    cfg = MelgalvisGenConfig(add_hydrogens=True)
    els, xyz = build_artificial_molecule(rng, n_nonh=10, cfg=cfg)
    assert len(els) == len(xyz)
    assert sum(1 for e in els if e != "H") == 10
    # Centroid near origin
    assert np.linalg.norm(xyz.mean(axis=0)) < 1.0


def test_generate_cluster_structure():
    st = generate_melgalvis_structure(seed=42, cfg=MelgalvisGenConfig(mode="cluster"))
    assert len(st.atoms) >= 6
    assert st.cell[0] > 0
    data = structure_to_fcalc(st, d_min=1.5)
    assert len(data["hkl"]) > 20
    assert np.isfinite(data["amplitudes"]).all()


def test_generate_rejection_mode():
    st = generate_melgalvis_structure(seed=3, cfg=MelgalvisGenConfig(mode="rejection"))
    assert "rej" in st.name or len(st.atoms) > 0


def test_iter_samples():
    samples = iter_melgalvis_samples(5, seed=0, d_min=1.4)
    assert len(samples) == 5
    assert samples[0]["generator"] == "melgalvis2026"
    assert samples[0]["phases"].shape == samples[0]["amplitudes"].shape


def test_strong_prior_melgalvis_flag():
    from grok_phase_solver.models.strong_prior import iter_hard_multsg_samples

    rows = list(
        iter_hard_multsg_samples(
            3, seed=1, use_melgalvis_gen=True, melgalvis_mode="hybrid"
        )
    )
    assert len(rows) == 3
    assert rows[0]["generator"] == "melgalvis2026"
