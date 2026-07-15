"""Tests for Wilson domain-gap closing."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.wilson import domain_gap_report
from grok_phase_solver.data.wilson_match import (
    WilsonMatchConfig,
    close_wilson_gap,
    match_wilson_slope,
    match_amplitude_quantiles,
    save_reference_template,
    load_reference_template,
    apply_wilson_match_if_template,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def _pack(n=10, d_min=1.2, seed=0, B_extra=0.0):
    st = generate_random_organic(n_atoms=n, seed=seed)
    data = structure_to_fcalc(st, d_min=d_min)
    amp = data["amplitudes"].copy()
    if B_extra != 0:
        from grok_phase_solver.data.wilson_match import apply_overall_b

        amp = apply_overall_b(data["hkl"], amp, st.cell, B_extra)
    return {
        "name": f"s{seed}",
        "hkl": data["hkl"],
        "amplitudes": amp,
        "phases": data["phases"],
        "cell": st.cell,
    }


def test_match_wilson_slope_moves_toward_target():
    p = _pack(seed=1, B_extra=15.0)  # artificially damped
    from grok_phase_solver.data.wilson import wilson_plot

    w0 = wilson_plot(p["hkl"], p["amplitudes"], p["cell"])
    target = w0["slope"] + 8.0  # want flatter
    amp2, info = match_wilson_slope(p["hkl"], p["amplitudes"], p["cell"], target)
    w1 = wilson_plot(p["hkl"], amp2, p["cell"])
    assert abs(w1["slope"] - target) < abs(w0["slope"] - target)
    assert abs(w1["slope"] - target) < 2.0  # close


def test_close_wilson_gap_reduces_score():
    # raw vs damped version of same structure family
    ref = _pack(n=12, d_min=1.3, seed=2, B_extra=0.0)
    synth = _pack(n=14, d_min=1.7, seed=3, B_extra=20.0)
    gap0 = domain_gap_report(synth, ref)["domain_gap_score"]
    matched, rep = close_wilson_gap(
        synth, ref,
        WilsonMatchConfig(
            match_slope=True, match_shells=True, match_quantiles=True,
            noise_level=0.0, seed=0,
        ),
    )
    assert rep["gap_after"] < gap0
    assert rep["gap_reduction"] > 0
    # phases not required for match; amplitudes changed
    assert not np.allclose(matched["amplitudes"], synth["amplitudes"])


def test_quantile_match_preserves_rank():
    rng = np.random.default_rng(0)
    a = rng.exponential(1.0, 200)
    ref = rng.normal(5.0, 1.0, 300)
    ref = np.abs(ref) + 0.1
    m, info = match_amplitude_quantiles(a, ref)
    assert info["applied"]
    # rank order preserved (monotone map)
    assert np.array_equal(np.argsort(a), np.argsort(m))


def test_template_roundtrip(tmp_path):
    ref = _pack(seed=5)
    path = tmp_path / "tpl.npz"
    save_reference_template(ref, path)
    loaded = load_reference_template(path)
    assert loaded is not None
    assert len(loaded["amplitudes"]) == len(ref["amplitudes"])
    amp2, meta = apply_wilson_match_if_template(
        ref["hkl"], ref["amplitudes"] * 1.5, ref["cell"], template=loaded,
    )
    assert meta["matched"]
