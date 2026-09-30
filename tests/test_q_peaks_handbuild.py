"""Q-peaks .res for Olex2 hand-build — Mark/Bragg ticket."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grok_phase_solver.pipeline.export import export_solution, write_shelxl_res
from grok_phase_solver.pipeline.peaks import DensityPeak

_CELL = np.array([8.920, 16.282, 18.504, 90.0, 90.0, 90.0])
_BRAGG = (
    "SYMM 0.5-X, -Y, 0.5+Z",
    "SYMM -X, 0.5+Y, 0.5-Z",
    "SYMM 0.5+X, 0.5-Y, -Z",
)


def _atom_lines(res: str):
    skip = {
        "TITL",
        "CELL",
        "ZERR",
        "LATT",
        "SYMM",
        "SFAC",
        "UNIT",
        "FVAR",
        "REM",
        "HKLF",
        "END",
    }
    out = []
    for line in res.splitlines():
        if not line.strip():
            continue
        if line.split()[0] in skip:
            continue
        out.append(line)
    return out


def _make_peaks(n: int) -> list:
    peaks = []
    for i in range(n):
        fract = np.array(
            [
                (0.05 + 0.9 * (i % 8) / 7.0) % 1.0,
                (0.07 + 0.85 * ((i // 8) % 5) / 4.0) % 1.0,
                (0.09 + 0.8 * (i / max(n - 1, 1))) % 1.0,
            ],
            dtype=np.float64,
        )
        sig = float(100.0 - i)
        peaks.append(
            DensityPeak(rank=i, fract=fract, height=sig, height_sigma=sig)
        )
    return peaks


def _result(peaks, sg="P 21 21 21"):
    return SimpleNamespace(
        cell=_CELL,
        space_group_hm=sg,
        method="test",
        diagnostics={"free_fom_composite": 0.5},
        peaks=peaks,
        hkl=np.zeros((0, 3), dtype=int),
        amplitudes=np.zeros(0),
        phases=np.zeros(0),
        density=np.zeros((4, 4, 4)),
        d_min=0.8,
        warnings=[],
    )


def test_sfac_c_only_and_dummy_unit():
    res = write_shelxl_res(_result(_make_peaks(40)), element="Q")
    assert "SFAC C" in res
    assert "SFAC C H" not in res
    assert "SFAC C H N O" not in res
    unit_lines = [l for l in res.splitlines() if l.startswith("UNIT")]
    assert unit_lines == ["UNIT 1"]
    assert "FVAR 1.0" in res
    assert "HKLF 4" in res
    assert res.strip().endswith("END")


def test_titl_rem_hand_build_not_shelxl_start():
    res = write_shelxl_res(_result(_make_peaks(10)), element="Q")
    assert "TITL gps-solve hand-build peaks (not a SHELXL start)" in res
    assert "REM hand_build_peaks n=34" in res
    assert "REM unique_asu" in res
    assert "REM peak_budget" in res
    assert "REM space_group_hint=" in res
    low = res.lower()
    assert "starting model" not in low
    assert "trial model" not in low
    assert "refinable" not in low
    assert "crystalx" not in low
    assert "shelxl starting" not in low


def test_q_lines_budget_soft_u_and_sof():
    res = write_shelxl_res(_result(_make_peaks(40)), element="Q", n_non_h_budget=34)
    atoms = _atom_lines(res)
    assert len(atoms) == 34
    for i, line in enumerate(atoms):
        parts = line.split()
        assert parts[0] == f"Q{i+1}"
        assert parts[1] == "1"
        assert parts[5] == "11.00000"
        u = float(parts[6])
        assert abs(u - 0.05) < 1e-6
        assert not any(parts[0].startswith(x) for x in ("Br", "Cl", "S", "H", "N", "O"))


def test_latt_symm_present_for_p212121():
    res = write_shelxl_res(_result(_make_peaks(5)), element="Q")
    assert "LATT -1" in res
    for line in _BRAGG:
        assert line in res


def test_no_crystalx_elements_in_header_or_atoms():
    res = write_shelxl_res(_result(_make_peaks(20)), element="Q")
    sfac_lines = [l for l in res.splitlines() if l.startswith("SFAC")]
    assert sfac_lines == ["SFAC C"]
    assert "CrystalX" not in res
    assert "Br" not in "".join(_atom_lines(res))
    assert "Cl" not in "".join(_atom_lines(res))
    for line in _atom_lines(res):
        assert line.split()[0].startswith("Q")


def test_export_always_writes_trial_res_no_packing_gate(tmp_path=None):
    """Packing GATE is out of this path — always write Q list when peaks exist."""
    import tempfile
    from pathlib import Path

    result = _result(_make_peaks(40))
    # Ensure no ConnectivityAsuError path is required
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        written = export_solution(result, out)
        names = {p.name for p in written}
        assert "trial.res" in names
        text = (out / "trial.res").read_text()
        assert "hand-build peaks" in text
        assert "SFAC C" in text
        assert len(_atom_lines(text)) == 34
