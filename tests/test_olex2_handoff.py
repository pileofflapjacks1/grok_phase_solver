"""Olex2 hand-build checklist next to trial.res."""

from __future__ import annotations

from pathlib import Path

from grok_phase_solver.pipeline.olex2_handoff import (
    olex2_handbuild_markdown,
    write_olex2_handbuild_md,
)


def test_markdown_points_at_work_info_and_forbids_raw_shelxl():
    md = olex2_handbuild_markdown(
        space_group="P 1 21 1",
        n_peaks=34,
        diagnostics={"free_fom_composite": 0.776},
    )
    assert "File → Open" in md
    assert "View → Work → Info" in md
    assert "Z′" in md or "Z'" in md
    assert "not a SHELXL starting model" in md.lower() or "not** a SHELXL" in md
    assert "Do **not** run SHELXL on the raw Q list" in md
    assert "peaks-as-carbon, not a fragment" in md
    assert "cp built.res work.ins" in md
    assert "P 1 21 1" in md
    assert "34" in md
    assert "0.776" in md


def test_write_olex2_handbuild_md(tmp_path: Path):
    path = write_olex2_handbuild_md(
        tmp_path,
        space_group="P 21 21 21",
        n_peaks=12,
        diagnostics={"free_fom_composite": 0.81},
    )
    assert path.name == "olex2_handbuild.md"
    assert path.exists()
    text = path.read_text()
    assert "P 21 21 21" in text
    assert "12" in text


def test_midband_trail_writes_half_cif(tmp_path: Path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    cif = root / "data" / "raw" / "cod" / "2012000.cif"
    script = root / "scripts" / "run_cod_midband_trail.py"
    if not cif.exists() or not script.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "run_cod_midband_trail",
        script,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    out = tmp_path / "2012000_half.cif"
    meta = mod.write_fragment_half_cif(cif, out)
    assert out.exists()
    assert meta["n_nonh"] == 27
    assert meta["n_frag"] == 13
    assert "21" in str(meta["space_group"])
    text = out.read_text()
    assert "_atom_site_fract_x" in text
    assert text.count("\n") >= 13
