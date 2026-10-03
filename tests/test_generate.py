"""CPU tests for the coordinate-generation track. No network and no large weights."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from grok_phase_solver.generate.cohort import mark_holdout, record_from_cif
from grok_phase_solver.generate.geometry import (
    expanded_min_distance,
    greedy_match_fraction,
    pairwise_min_distance,
    space_group_consistent,
)
from grok_phase_solver.generate.noise import sigma_schedule
from grok_phase_solver.generate.report import (
    METHOD_DENOISER,
    METHOD_DIFFUSION,
    METHOD_RANDOM,
    STABILITY,
    render_scoreboard_md,
)
from grok_phase_solver.generate.score import reflection_list, score_sites

ROOT = Path(__file__).resolve().parents[1]
MINIMAL = ROOT / "examples" / "generate" / "minimal.cif"
CELL = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])


def test_package_modules_do_not_import_torch_at_top_level():
    generate = ROOT / "src" / "grok_phase_solver" / "generate"
    for path in generate.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                mods = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]]
            else:
                continue
            assert "torch" not in mods, path.name


def test_clash_and_periodic_image():
    close = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    assert pairwise_min_distance(close, CELL) == pytest.approx(0.5, abs=1e-6)
    wrapped = np.array([[0.01, 0.0, 0.0], [0.99, 0.0, 0.0]])
    assert pairwise_min_distance(wrapped, CELL) == pytest.approx(0.2, abs=1e-6)
    score = score_sites(close, ["C", "C"], ["C", "C"], CELL, "P1")
    assert score["clash"] is True
    assert score["composition_match"] is True
    assert score["valid"] is False


def test_composition_mismatch_is_invalid_without_a_clash():
    fracs = np.array([[0.1, 0.2, 0.3], [0.7, 0.2, 0.3]])
    assert expanded_min_distance(fracs, ["C", "N"], CELL, "P1") > 1.0
    score = score_sites(fracs, ["C", "N"], ["C", "C"], CELL, "P1")
    assert score["clash"] is False
    assert score["composition_match"] is False
    assert score["valid"] is False


def test_bad_space_group_is_not_consistent():
    fracs = np.array([[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]])
    assert space_group_consistent(fracs, ["C", "C"], CELL, "P1") is True
    assert space_group_consistent(fracs, ["C", "C"], CELL, "not-a-space-group") is False


def test_true_coordinates_match_and_have_a_small_residual():
    record = record_from_cif(MINIMAL)
    hkl = reflection_list(record["cell"], record["space_group"])
    from grok_phase_solver.generate.score import proposal_amplitudes

    amp = proposal_amplitudes(
        record["fracs"], record["elements"], record["cell"], record["space_group"], hkl
    )
    score = score_sites(
        record["fracs"],
        record["elements"],
        record["elements"],
        record["cell"],
        record["space_group"],
        reference_fracs=record["fracs"],
        reference_elements=record["elements"],
        hkl=hkl,
        amplitudes=amp,
        count_reconstruction=True,
    )
    assert score["valid"] is True
    assert score["match_fraction"] == pytest.approx(1.0)
    assert score["reconstructed"] is True
    assert score["r_factor"] is not None and score["r_factor"] < 1e-6


def test_origin_grid_recovers_a_quarter_cell_shift():
    record = record_from_cif(MINIMAL)
    shifted = np.mod(record["fracs"] + np.array([0.25, 0.0, 0.0]), 1.0)
    fraction = greedy_match_fraction(
        shifted, record["elements"], record["fracs"], record["elements"], record["cell"]
    )
    assert fraction == pytest.approx(1.0)
    # 0.10 fractional is 1.0 Å on this 10 Å cell, outside the 0.75 Å tolerance
    # and not on the 0.25 grid.
    off = np.mod(record["fracs"] + np.array([0.10, 0.0, 0.0]), 1.0)
    off_fraction = greedy_match_fraction(
        off, record["elements"], record["fracs"], record["elements"], record["cell"]
    )
    assert off_fraction < 0.5


def test_uniqueness_flags_a_duplicate_and_keeps_the_first():
    from grok_phase_solver.generate.score import assign_uniqueness

    fracs = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.4]])
    other = np.array([[0.15, 0.2, 0.3], [0.7, 0.2, 0.8]])
    samples = []
    for coords in (fracs, fracs.copy(), other):
        samples.append(
            {
                "fracs": coords,
                "elements": ["C", "O"],
                "score": score_sites(coords, ["C", "O"], ["C", "O"], CELL, "P1"),
            }
        )
    assign_uniqueness(samples)
    assert samples[0]["score"]["unique"] is True
    assert samples[1]["score"]["unique"] is False
    assert samples[2]["score"]["unique"] is True


def test_holdout_is_the_largest_structure():
    small = {"name": "a", "elements": ["C"] * 4, "holdout": False}
    large = {"name": "b", "elements": ["C"] * 8, "holdout": False}
    tie = {"name": "c", "elements": ["C"] * 8, "holdout": False}
    mark_holdout([small, large, tie])
    assert small["holdout"] is False
    assert large["holdout"] is False
    assert tie["holdout"] is True
    only = {"name": "solo", "elements": ["C"] * 4, "holdout": True}
    mark_holdout([only])
    assert only["holdout"] is False


def test_sigma_schedule_is_descending():
    sigma = sigma_schedule(5)
    assert sigma[0] > sigma[-1]
    assert len(sigma) == 5


def test_scoreboard_table_has_the_baseline_and_the_fixed_columns():
    md = render_scoreboard_md(
        rows=[
            {
                "method": METHOD_RANDOM,
                "n": 4,
                "validity": "4/4 (1.000)",
                "uniqueness": "4/4 (1.000)",
                "reconstruction": "0/1 (0.000)",
                "note": "baseline",
            },
            {
                "method": METHOD_DENOISER,
                "n": 4,
                "validity": "1/4 (0.250)",
                "uniqueness": "1/1 (1.000)",
                "reconstruction": "0/1 (0.000)",
                "note": "denoiser",
            },
            {
                "method": METHOD_DIFFUSION,
                "n": 1,
                "validity": "0/1 (0.000)",
                "uniqueness": "0/0",
                "reconstruction": "0/1 (0.000)",
                "note": "experimental oracle-|F|",
            },
        ],
        per_cell=[],
        protocol_lines=["seed 0"],
        comparison="The coordinate denoiser validity is 0.250, below the random-plus-clash baseline at 1.000.",
        included=["tiny (train)"],
        excluded=["big (non-H count 30 outside 4–20)"],
    )
    assert METHOD_RANDOM in md
    assert METHOD_DENOISER in md
    assert "experimental" in md
    assert "not measured" in md
    assert STABILITY in md
    assert "below the random-plus-clash baseline" in md
    assert "Do not edit the rates by hand." in md


def test_random_propose_is_deterministic_and_scored():
    from grok_phase_solver.generate.propose import propose

    kwargs = dict(
        elements=["C", "N", "O", "C"],
        cell=CELL,
        space_group="P1",
        z=1,
        n=3,
        method="random",
        seed=0,
    )
    a = propose(**kwargs)
    b = propose(**kwargs)
    assert len(a) == 3
    for left, right in zip(a, b):
        np.testing.assert_allclose(left["fracs"], right["fracs"])
        assert set(left["score"]) >= {
            "valid",
            "clash",
            "composition_match",
            "min_distance_A",
            "sg_consistent",
            "r_factor",
            "match_fraction",
            "reconstructed",
            "unique",
        }
        assert left["score"]["composition_match"] is True


def test_denoiser_samples_repeat_on_cpu():
    pytest.importorskip("torch")
    from grok_phase_solver.generate.model import sample_denoiser, train_denoiser

    example = {
        "fracs": np.array(
            [[0.10, 0.20, 0.30], [0.45, 0.20, 0.30], [0.70, 0.55, 0.40], [0.20, 0.75, 0.65]]
        ),
        "elements": ["C", "N", "O", "C"],
        "cell": CELL,
    }
    first = sample_denoiser(
        train_denoiser([example], steps=3, seed=0),
        example["elements"],
        CELL,
        n=2,
        seed=0,
        n_steps=3,
    )
    second = sample_denoiser(
        train_denoiser([example], steps=3, seed=0),
        example["elements"],
        CELL,
        n=2,
        seed=0,
        n_steps=3,
    )
    np.testing.assert_allclose(first, second, atol=1e-6)
    assert first.shape == (2, 4, 3)


def test_diffusion_hybrid_does_not_pad_and_is_labeled():
    """One cheap experimental call. It must not invent atoms to fill the composition."""
    pytest.importorskip("torch")
    from grok_phase_solver.generate.propose import propose
    from grok_phase_solver.generate.score import proposal_amplitudes

    record = record_from_cif(MINIMAL)
    hkl = reflection_list(record["cell"], record["space_group"], d_min=2.0)
    amp = proposal_amplitudes(
        record["fracs"], record["elements"], record["cell"], record["space_group"], hkl
    )
    samples = propose(
        record["elements"],
        record["cell"],
        record["space_group"],
        int(record["z"]),
        n=1,
        method="diffusion_hybrid",
        seed=0,
        hkl=hkl,
        amplitudes=amp,
        diffusion_steps=1,
        diffusion_starts=1,
        diffusion_polish=1,
        count_reconstruction=True,
        reference_fracs=record["fracs"],
        reference_elements=record["elements"],
    )
    sample = samples[0]
    assert sample["method"] == "diffusion_hybrid"
    assert sample["experimental"] is True
    assert sample["oracle_amplitudes"] is True
    assert len(sample["elements"]) <= len(record["elements"])
    if len(sample["elements"]) < len(record["elements"]):
        assert sample["score"]["composition_match"] is False
        assert sample["score"]["valid"] is False


def test_gps_generate_writes_samples_and_report(tmp_path):
    pytest.importorskip("torch")
    from grok_phase_solver.generate.cli import main

    out = tmp_path / "gen_out"
    main(
        [
            "--cif",
            str(MINIMAL),
            "--n",
            "2",
            "--steps",
            "2",
            "--sample-steps",
            "2",
            "--out",
            str(out),
            "--seed",
            "0",
        ]
    )
    cif_text = (out / "samples.cif").read_text()
    report = (out / "report.md").read_text()
    assert cif_text.count("data_gen_") == 2
    assert METHOD_RANDOM in report
    assert METHOD_DENOISER in report
    assert "this CIF only" in report
    assert "not measured — no energy model" in report
    assert "diffusion_hybrid" in report
    import gemmi

    doc = gemmi.cif.read(str(out / "samples.cif"))
    assert len(doc) == 2
