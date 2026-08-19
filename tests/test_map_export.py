"""CCP4 map + PyMOL/Coot handoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grok_phase_solver.pipeline.export import export_solution
from grok_phase_solver.pipeline.map_export import (
    map_isolevel,
    write_ccp4_map,
    write_map_handoff,
    write_peaks_pdb,
)
from grok_phase_solver.pipeline.peaks import DensityPeak
from grok_phase_solver.pipeline.solve import SolveResult


def test_ccp4_roundtrip_preserves_peak(tmp_path: Path):
    rho = np.zeros((8, 10, 12), dtype=np.float32)
    rho[2, 3, 4] = 5.0
    cell = [10.0, 12.0, 14.0, 90.0, 90.0, 90.0]
    path = write_ccp4_map(tmp_path / "density.map", rho, cell, space_group="P 1")
    assert path.is_file()
    raw = path.read_bytes()
    assert raw[208:212] == b"MAP "
    import gemmi

    m = gemmi.read_ccp4_map(str(path))
    arr = np.array(m.grid)
    assert arr.shape == (8, 10, 12)
    assert float(arr[2, 3, 4]) == 5.0
    assert tuple(int(x) for x in np.unravel_index(int(arr.argmax()), arr.shape)) == (2, 3, 4)


def test_peaks_pdb_has_cryst1(tmp_path: Path):
    peaks = [
        DensityPeak(rank=0, fract=np.array([0.1, 0.2, 0.3]), height=4.0, height_sigma=3.0)
    ]
    p = write_peaks_pdb(tmp_path / "peaks.pdb", peaks, [10, 11, 12, 90, 95, 90], "P21/c")
    text = p.read_text()
    assert "CRYST1" in text
    assert "HETATM" in text


def test_isolevel_positive():
    rng = np.random.default_rng(0)
    rho = rng.normal(0.0, 1.0, size=(16, 16, 16))
    lvl = map_isolevel(rho, n_sigma=1.5)
    assert lvl > float(rho.mean())


def test_export_writes_map_and_scripts(tmp_path: Path):
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    rho = np.zeros((8, 8, 8))
    rho[2, 2, 2] = 5.0
    result = SolveResult(
        hkl=hkl,
        amplitudes=np.ones(4),
        phases=np.zeros(4),
        density=rho,
        cell=np.array([10.0, 11.0, 12.0, 90.0, 90.0, 90.0]),
        space_group_hm="P 1",
        method="ensemble",
        d_min=1.0,
        peaks=[
            DensityPeak(
                rank=1,
                fract=np.array([0.25, 0.25, 0.25]),
                height=5.0,
                height_sigma=3.0,
            )
        ],
        diagnostics={"free_fom_composite": 0.6},
    )
    written = export_solution(result, tmp_path)
    names = {p.name for p in written}
    assert "density.map" in names
    assert "open_in_pymol.pml" in names
    assert "open_in_coot.sh" in names
    assert "peaks.pdb" in names
    pml = (tmp_path / "open_in_pymol.pml").read_text()
    assert "load density.map" in pml
    report = (tmp_path / "report.md").read_text()
    assert "density.map" in report
    assert "open_in_pymol.pml" in report


def test_write_map_handoff_skips_empty(tmp_path: Path):
    result = SolveResult(
        hkl=np.zeros((1, 3)),
        amplitudes=np.ones(1),
        phases=np.zeros(1),
        density=np.zeros((0, 0, 0)),
        cell=np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
        space_group_hm="P 1",
        method="ensemble",
        d_min=1.0,
    )
    assert write_map_handoff(result, tmp_path) == []
