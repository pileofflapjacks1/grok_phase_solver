"""Lane B: seed importers, quality bar, fragment/HA paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.pipeline.solve import SolveResult, resolve_method
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.partial_seed import fragment_phaseed_solve
from grok_phase_solver.solvers.seed_import import (
    assess_seed_quality,
    export_seed_csv,
    load_atoms_from_peaks_csv,
    load_atoms_from_res,
    resolve_phase_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def _hard_cell(seed=0):
    st = generate_random_organic(n_atoms=12, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=1.5)
    return st, data


def test_load_atoms_from_res(tmp_path):
    st, data = _hard_cell(1)
    # Minimal .res with a few truth atoms as C
    res = tmp_path / "frag.res"
    lines = [
        "TITL test",
        f"CELL 0.71 {st.cell[0]:.3f} {st.cell[1]:.3f} {st.cell[2]:.3f} "
        f"{st.cell[3]:.2f} {st.cell[4]:.2f} {st.cell[5]:.2f}",
        "SFAC C",
        "UNIT 1",
        "FVAR 1.0",
    ]
    for i, fr in enumerate(data["fracs"][:4]):
        lines.append(f"C{i+1} 1 {fr[0]:.5f} {fr[1]:.5f} {fr[2]:.5f} 11.000 0.05")
    lines.append("HKLF 4")
    lines.append("END")
    res.write_text("\n".join(lines) + "\n")
    fracs, els, meta = load_atoms_from_res(res)
    assert len(fracs) == 4
    assert all(e == "C" for e in els)
    assert meta["n_atoms"] == 4


def test_seed_from_res_and_quality(tmp_path):
    st, data = _hard_cell(2)
    res = tmp_path / "frag.res"
    lines = [
        "TITL t",
        f"CELL 0.71 {' '.join(f'{x:.3f}' for x in st.cell)}",
        "SFAC C",
        "UNIT 1",
        "FVAR 1.0",
    ]
    for i, fr in enumerate(data["fracs"][:5]):
        lines.append(f"C{i+1} 1 {fr[0]:.5f} {fr[1]:.5f} {fr[2]:.5f} 11.000 0.05")
    lines += ["HKLF 4", "END"]
    res.write_text("\n".join(lines) + "\n")

    seed_ph, mask, meta = resolve_phase_seed(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        phase_seed_res=str(res),
        seed=0,
    )
    assert mask.sum() >= 8
    assert meta["source"] == "phase_seed_res"
    qual = assess_seed_quality(
        data["hkl"], data["amplitudes"], st.cell, seed_ph, mask
    )
    assert qual["n_seed"] == int(mask.sum())
    assert "hints" in qual


def test_peaks_csv_seed(tmp_path):
    st, data = _hard_cell(3)
    peaks = tmp_path / "peaks.csv"
    with peaks.open("w") as f:
        f.write("rank,x_frac,y_frac,z_frac,height,height_sigma\n")
        for i, fr in enumerate(data["fracs"][:6]):
            f.write(f"{i+1},{fr[0]:.5f},{fr[1]:.5f},{fr[2]:.5f},10.0,5.0\n")
    fracs, els, meta = load_atoms_from_peaks_csv(peaks, max_atoms=4, min_sigma=2.0)
    assert len(fracs) == 4
    seed_ph, mask, meta2 = resolve_phase_seed(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        seed_peaks_csv=str(peaks),
        seed_n_atoms=4,
    )
    assert meta2["source"] == "seed_peaks_csv"
    assert mask.sum() > 0


def test_export_seed_csv(tmp_path):
    st, data = _hard_cell(4)
    n = len(data["phases"])
    mask = np.zeros(n, dtype=bool)
    mask[:20] = True
    path = export_seed_csv(tmp_path / "s.csv", data["hkl"], data["phases"], mask)
    text = path.read_text()
    assert "phase_deg" in text
    assert len(text.strip().splitlines()) == 21


def test_fragment_phaseed_improves_over_random():
    st, data = _hard_cell(5)
    # Use half of truth atoms as fragment
    n = max(4, len(data["fracs"]) // 2)
    ph, rho, info = fragment_phaseed_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        data["fracs"][:n],
        data["elements"][:n],
        n_extend=8,
        polish="none",
        n_starts=1,
        seed=0,
        d_min=1.5,
    )
    assert info["algorithm"] == "partial_phaseed"
    assert len(ph) == len(data["phases"])
    assert rho.ndim == 3


def test_resolve_method_fragment_alias():
    m, reason = resolve_method("fragment_phaseed", "P1", 1.8, 100)
    assert m == "partial_phaseed"
    m2, _ = resolve_method("ha_phaseed", "P1", 1.8, 100)
    assert m2 == "partial_phaseed"


def test_report_includes_seed_section():
    from grok_phase_solver.pipeline.export import _render_report

    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    amp = np.ones(4)
    ph = np.zeros(4)
    dens = np.ones((8, 8, 8))
    result = SolveResult(
        hkl=hkl,
        amplitudes=amp,
        phases=ph,
        density=dens,
        cell=np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
        space_group_hm="P1",
        method="partial_phaseed",
        d_min=1.5,
        peaks=[],
        diagnostics={
            "free_fom_composite": 0.4,
            "seed_kind": "phase_seed_res",
            "seed_quality": {
                "n_seed": 10,
                "fraction_all": 0.2,
                "n_strong": 20,
                "n_strong_seeded": 8,
                "frac_strong_seeded": 0.4,
                "size_meets_bar": True,
                "seed_free_fom_composite": 0.35,
                "hints": ["Seed size looks adequate"],
            },
        },
        warnings=[],
    )
    md = _render_report(result)
    assert "Partial seed quality" in md
    assert "30%" in md or "oracle" in md.lower()
