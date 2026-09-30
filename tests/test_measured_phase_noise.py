"""CPU-only tests for the measured-phase noise simulator."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.physics.measured_phase_noise import (
    C4_FRAC_LE_20,
    WINDOW_DEG,
    add_measured_phase_noise,
    apply_measured_phase_to_seed,
    make_measured_phase_mask,
    seed_quality_from_noisy_phases,
    sigma_phi_from_E,
    wrap_angle_deg,
    write_measured_phase_seed_csv,
)
from grok_phase_solver.solvers.partial_seed import load_phase_seed_csv


def test_wrap_idempotent_and_interval():
    rng = np.random.default_rng(0)
    raw = rng.uniform(-720, 720, size=400)
    raw = np.append(raw, [0.0, 180.0, -180.0, 360.0, -360.0, 181.0, -181.0])
    w = wrap_angle_deg(raw)
    assert np.all(w > -180.0 - 1e-12)
    assert np.all(w <= 180.0 + 1e-12)
    w2 = wrap_angle_deg(w)
    np.testing.assert_allclose(w2, w, atol=1e-10)
    # 180 stays 180; -180 maps onto 180 (closed on the right)
    assert wrap_angle_deg(180.0) == 180.0
    assert wrap_angle_deg(-180.0) == 180.0


def test_sigma_zero_recovers_true_phase():
    rng = np.random.default_rng(1)
    phi = rng.uniform(-180, 180, size=80)
    for model in ("von_mises", "wrapped_normal"):
        out = add_measured_phase_noise(phi, 0.0, rng, model=model)
        d = np.abs(wrap_angle_deg(out - phi))
        assert float(np.max(d)) < 1.0, model


def test_tiny_sigma_within_one_degree():
    rng = np.random.default_rng(2)
    phi = np.linspace(-170, 170, 60)
    out = add_measured_phase_noise(phi, 1e-6, rng, model="von_mises")
    d = np.abs(wrap_angle_deg(out - phi))
    assert float(np.max(d)) < 1.0


def test_large_sigma_frac_le_20_falls_toward_chance():
    rng = np.random.default_rng(3)
    n = 4000
    phi = rng.uniform(-180, 180, size=n)
    E = np.ones(n)
    out = add_measured_phase_noise(phi, 90.0, rng, model="von_mises")
    q = seed_quality_from_noisy_phases(phi, out, E, mask=np.ones(n, dtype=bool))
    # Uniform chance for |Δφ| ≤ 20° is 40/360 ≈ 0.111. At σ=90° we should
    # be well below the C4 bar and approaching chance.
    assert q["frac_le_20"] < 0.28
    assert q["frac_le_20"] > 0.02
    assert q["meets_c4_bar"] is False
    assert q["c4_bar"] == C4_FRAC_LE_20
    assert q["window_deg"] == WINDOW_DEG


def test_mask_fraction_within_tolerance():
    rng = np.random.default_rng(4)
    E = np.linspace(0.1, 3.0, 200)
    # strong_fraction=1 → candidates are all reflections; frac is of all
    for frac in (0.10, 0.30, 0.50, 1.00):
        mask = make_measured_phase_mask(E, frac, rng, strong_fraction=1.0)
        assert abs(float(mask.mean()) - frac) < 0.03
        assert int(mask.sum()) == int(round(frac * len(E)))


def test_mask_prefers_high_E():
    rng = np.random.default_rng(5)
    E = np.linspace(0.05, 4.0, 150)
    mask = make_measured_phase_mask(E, 0.30, rng, strong_fraction=1.0)
    assert float(np.mean(E[mask])) > float(np.mean(E[~mask]))
    # default strong set is top 30%; measuring all of it should be the top 30%
    mask_all_strong = make_measured_phase_mask(E, 1.0, rng)
    assert float(np.min(E[mask_all_strong])) >= float(np.max(E[~mask_all_strong])) - 1e-12


def test_sigma_phi_from_E_stronger_is_narrower():
    E = np.array([0.5, 1.0, 2.0])
    sig = sigma_phi_from_E(E, sigma0_deg=20.0, E_ref=1.0, floor_deg=1.0)
    assert sig[0] > sig[1] > sig[2]
    assert sig[2] >= 1.0
    # floor
    sig_f = sigma_phi_from_E(np.array([100.0]), 20.0, E_ref=1.0, floor_deg=5.0)
    assert float(sig_f[0]) == 5.0


def test_apply_and_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    n = 40
    hkl = np.column_stack(
        [np.arange(n), np.zeros(n, dtype=int), np.zeros(n, dtype=int)]
    )
    phi_true = rng.uniform(-np.pi, np.pi, size=n)
    E = np.linspace(0.2, 3.0, n)
    mask0 = np.ones(n, dtype=bool)
    seed_ph, mask, meta = apply_measured_phase_to_seed(
        phi_true,
        mask0,
        E,
        sigma_phi_deg=5.0,
        frac_strong=0.50,
        rng=rng,
        model="von_mises",
    )
    assert meta["kind"] == "measured_phase"
    assert int(mask.sum()) >= 1
    path = tmp_path / "meas.csv"
    write_measured_phase_seed_csv(path, hkl, seed_ph, mask)
    loaded, mask2, load_meta = load_phase_seed_csv(path, hkl)
    assert load_meta["n_mapped"] >= int(mask.sum()) - 1
    idx = np.where(mask)[0]
    d = np.angle(np.exp(1j * (loaded[idx] - seed_ph[idx])))
    assert float(np.mean(np.abs(d))) < np.deg2rad(1.0)


def test_frac_le_20_unmeasured_count_as_fail():
    n = 100
    E = np.linspace(0.1, 5.0, n)
    phi = np.zeros(n)
    # perfect measurement on 10% of the strong set
    rng = np.random.default_rng(7)
    mask = make_measured_phase_mask(E, 0.10, rng)
    q = seed_quality_from_noisy_phases(phi, phi, E, mask=mask)
    # among measured strong: all within 20°
    assert q["frac_le_20"] == 1.0
    # among all strong: only the measured 10%
    assert abs(q["frac_le_20_all_strong"] - 0.10) < 0.05
    assert q["meets_c4_bar"] is False
