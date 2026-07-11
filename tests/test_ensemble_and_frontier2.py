"""Tests for ensemble, DiffMap retune, recycle net, COD hybrid helpers."""

from __future__ import annotations

import numpy as np

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.ensemble import ensemble_cf_raar, ensemble_solve
from grok_phase_solver.solvers.free_fom import free_fom
from grok_phase_solver.solvers.iterative_retrieval import (
    difference_map_solve,
    retune_difference_map,
)
from grok_phase_solver.solvers.recycle_net import (
    make_phase_fn,
    recycle_net_solve,
    train_recycle_net_hard,
)


def _tiny(n=4, d_min=1.0, seed=0):
    st = generate_random_organic(n_atoms=n, seed=seed)
    data = structure_to_fcalc(st, d_min=d_min)
    return st, data


def test_ensemble_cf_raar_runs():
    st, data = _tiny()
    ph, rho, info = ensemble_cf_raar(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_starts=2,
        n_iter=25,
        base_seed=0,
        d_min=1.0,
    )
    assert len(ph) == len(data["amplitudes"])
    assert info["n_trials"] == 4  # 2 methods × 2 starts
    assert info["best_method"] in ("cf", "raar")
    assert "composite" in info["best_fom"]
    assert np.isfinite(ph).all()
    assert rho.ndim == 3


def test_ensemble_selects_highest_fom():
    st, data = _tiny(seed=1)
    ph, rho, info = ensemble_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        methods=("cf",),
        n_starts=3,
        n_iter=20,
        base_seed=0,
        d_min=1.0,
    )
    composites = [t["composite"] for t in info["all_foms"] if "error" not in t]
    assert info["best_fom"]["composite"] == max(composites)


def test_diffmap_charge_flip_projector():
    st, data = _tiny()
    ph, rho, hist = difference_map_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        n_iter=20,
        beta=0.7,
        real_proj="charge_flip",
        delta_sigma=1.0,
        seed=0,
        d_min=1.0,
    )
    assert hist["real_proj"] == "charge_flip"
    assert hist["delta_sigma"] == 1.0
    assert len(hist["R"]) == 20
    assert np.isfinite(ph).all()


def test_retune_difference_map_small_grid():
    st, data = _tiny()
    ret = retune_difference_map(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        beta_grid=(0.7, 1.0),
        real_proj_options=("positivity", "charge_flip"),
        delta_sigma_grid=(1.0,),
        n_iter=15,
        seeds=(0,),
        d_min=1.0,
    )
    assert "beta" in ret["best_params"]
    assert "real_proj" in ret["best_params"]
    assert len(ret["grid_results"]) >= 3
    assert ret["best_fom"]["composite"] >= 0


def test_recycle_net_train_and_solve():
    model, meta = train_recycle_net_hard(
        n_structures=2,
        n_atoms_range=(12, 14),
        d_min_range=(1.5, 1.6),
        epochs_per=15,
        hidden=32,
        seed=0,
        verbose=False,
    )
    assert meta["mean_mpe"] >= 0
    assert hasattr(model, "_feat_mu")

    st, data = _tiny(n=12, d_min=1.5, seed=2)
    ph, rho, hist = recycle_net_solve(
        data["hkl"],
        data["amplitudes"],
        st.cell,
        model,
        n_cycles=3,
        seed=0,
        d_min=1.5,
    )
    assert hist["algorithm"] == "recycle_net"
    assert len(ph) == len(data["amplitudes"])
    fom = free_fom(data["hkl"], data["amplitudes"], ph, st.cell, density=rho)
    assert 0 <= fom["composite"] <= 1.5


def test_make_phase_fn_blend():
    model, _ = train_recycle_net_hard(
        n_structures=1,
        n_atoms_range=(12, 12),
        d_min_range=(1.5, 1.5),
        epochs_per=5,
        hidden=16,
        seed=1,
        verbose=False,
    )
    st, data = _tiny(n=12, d_min=1.5, seed=3)
    fn0 = make_phase_fn(model, st.cell, blend=0.0)
    ph_in = data["phases"] * 0.0 + 0.5
    out0 = fn0(data["hkl"], data["amplitudes"], ph_in)
    np.testing.assert_allclose(out0, ph_in)
