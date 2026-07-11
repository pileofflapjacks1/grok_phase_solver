"""
Domain-matched phase prior for hard P1 synthetic cells.

Motivation
----------
PhAI (COD / P2₁/c–biased training) shows a **domain gap** on random P1
hard cells (n≥12, d_min≥1.5). This module trains a PhaseMLP **only** on
that domain so AI-PhaSeed can be seeded with an in-distribution prior.

Training protocol
-----------------
1. Sample many P1 organics with n_atoms ∈ [n_lo, n_hi], d_min ∈ [d_lo, d_hi].
2. Compute global feature mean/std over all reflections (cross-structure).
3. Supervised (cos φ, sin φ) MSE with amplitude weights using **fixed** global
   normalization (not per-structure stats that leak resolution).
4. Optional second pass at lower LR.

Inference: φ = atan2(sin, cos) from PhaseMLP(features); feed into
``ai_phaseed_solve``.

Honest scope: synthetic P1 hard region only — not a general experimental
or multi-SG solver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from grok_phase_solver.metrics.phase_error import mean_phase_error
from grok_phase_solver.models.phase_mlp import PhaseMLP, reflection_features


def iter_hard_p1_samples(
    n_samples: int,
    seed: int = 0,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    include_bridge: bool = True,
) -> Iterator[Dict]:
    """
    Yield hard (and optional bridge) P1 synthetic samples.

    Bridge samples (n∈[8,12), d_min∈[1.2,1.5)) help the net see slightly
    easier but related cells without leaving P1.
    """
    # Lazy imports avoid cycle: models → data.synthetic → solvers → models
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    rng = np.random.default_rng(seed)
    n_lo, n_hi = n_atoms_range
    d_lo, d_hi = d_min_range
    for i in range(n_samples):
        s = int(rng.integers(0, 2**31 - 1))
        if include_bridge and (i % 4 == 0):
            n_atoms = int(rng.integers(8, 12))
            d_min = float(rng.uniform(1.2, 1.5))
            region = "bridge"
        else:
            n_atoms = int(rng.integers(n_lo, n_hi + 1))
            d_min = float(rng.uniform(d_lo, d_hi))
            region = "hard"
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        yield {
            "name": st.name,
            "hkl": data["hkl"],
            "amplitudes": data["amplitudes"],
            "phases": data["phases"],
            "cell": st.cell,
            "n_atoms": data["n_atoms_cell"],
            "d_min": d_min,
            "region": region,
            "structure_seed": s,
            "fracs": data["fracs"],
            "elements": data["elements"],
        }


def accumulate_feature_stats(
    samples: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Global feature mean / std over all reflections in the sample list."""
    feat_sum = None
    feat_sq = None
    n = 0
    for s in samples:
        X = reflection_features(s["hkl"], s["amplitudes"], s["cell"])
        if feat_sum is None:
            feat_sum = X.sum(0)
            feat_sq = (X ** 2).sum(0)
        else:
            feat_sum = feat_sum + X.sum(0)
            feat_sq = feat_sq + (X ** 2).sum(0)
        n += len(X)
    mu = feat_sum / max(n, 1)
    var = feat_sq / max(n, 1) - mu ** 2
    sig = np.sqrt(np.maximum(var, 1e-12))
    return mu, sig, n


def origin_phase_candidates(
    hkl: np.ndarray,
    phases: np.ndarray,
    n_grid: int = 4,
    include_enantiomorph: bool = True,
) -> List[np.ndarray]:
    """
    Discrete origin shifts (and optional enantiomorph) of a phase set.

    φ'(h) = φ(h) − 2π h·t,  t ∈ {0, 1/n, …}^3
    Enantiomorph: φ → −φ (then origin-shifted).
    """
    hkl = np.asarray(hkl, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    cands: List[np.ndarray] = []
    bases = [ph]
    if include_enantiomorph:
        bases.append(-ph)
    for base in bases:
        for tx in grid:
            for ty in grid:
                for tz in grid:
                    t = np.array([tx, ty, tz], dtype=np.float64)
                    shift = 2.0 * np.pi * (hkl @ t)
                    cands.append(base - shift)
    return cands


def train_phase_mlp_global(
    model: PhaseMLP,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    n_epochs: int = 80,
    lr: float = 3e-3,
    batch_frac: float = 1.0,
    seed: int = 0,
    origin_invariant: bool = True,
    n_origin_grid: int = 4,
    strong_frac: float = 0.6,
) -> List[float]:
    """
    Train one structure with fixed global feature normalization.

    If origin_invariant, each step picks the origin/enantiomorph of *true*
    phases that best matches the current prediction (weighted MSE on unit
    circle), then backprops to that target. Absolute phases are meaningless
    without a fixed origin; this is essential for hard-P1 domain training.
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    X = reflection_features(hkl, amplitudes, cell)
    X = (X - mu) / sig
    y = np.asarray(phases, dtype=np.float64)
    w = np.asarray(amplitudes, dtype=np.float64)
    # Focus training weight on strong reflections (seed set for PhaSeed)
    order = np.argsort(-w)
    n_strong = max(8, int(strong_frac * len(w)))
    strong_mask = np.zeros(len(w), dtype=bool)
    strong_mask[order[:n_strong]] = True
    w_train = w.copy()
    w_train[~strong_mask] *= 0.25

    if origin_invariant:
        # Precompute candidates once (cheap for n_grid=4 → 128 cands)
        cands = origin_phase_candidates(
            hkl, y, n_grid=n_origin_grid, include_enantiomorph=True
        )
        cand_ut = [
            np.column_stack([np.cos(c), np.sin(c)]) for c in cands
        ]
    else:
        cand_ut = None

    losses: List[float] = []
    n = len(X)
    for _ in range(n_epochs):
        idx = np.arange(n)
        if batch_frac < 1.0:
            m = max(8, int(batch_frac * n))
            idx = rng.choice(n, size=m, replace=False)

        if origin_invariant and cand_ut is not None:
            z, _ = model.forward(X)
            # pick best origin target by weighted MSE on full set (stable)
            best_i = 0
            best_l = 1e99
            wn = w_train / (w_train.mean() + 1e-16)
            for i, ut in enumerate(cand_ut):
                diff = z - ut
                l = 0.5 * float(np.mean(wn * np.sum(diff ** 2, axis=1)))
                if l < best_l:
                    best_l = l
                    best_i = i
            y_tgt = cands[best_i]
            loss, grads = model.loss_and_backward(
                X[idx], y_tgt[idx], weights=w_train[idx]
            )
        else:
            loss, grads = model.loss_and_backward(
                X[idx], y[idx], weights=w_train[idx]
            )
        model.step(grads, lr=lr)
        losses.append(loss)
    return losses


def predict_phases_hard_p1(
    model: PhaseMLP,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    origin_fom_search: bool = True,
    n_origin_grid: int = 3,
) -> np.ndarray:
    """
    Predict phases with global feature norms.

    If origin_fom_search, try discrete origin shifts of the raw prediction
    and keep the one with best free-FOM composite (truth-free).
    """
    X = reflection_features(hkl, amplitudes, cell)
    mu = getattr(model, "_feat_mu", None)
    sig = getattr(model, "_feat_sig", None)
    if mu is not None and sig is not None:
        X = (X - mu) / sig
    ph0 = model.predict_phases(X)
    if not origin_fom_search:
        return ph0

    from grok_phase_solver.solvers.free_fom import free_fom

    best_c = -1.0
    best_ph = ph0
    for ph in origin_phase_candidates(
        hkl, ph0, n_grid=n_origin_grid, include_enantiomorph=True
    ):
        fom = free_fom(hkl, amplitudes, ph, cell, include_shells=False, include_sayre=False)
        if fom["composite"] > best_c:
            best_c = fom["composite"]
            best_ph = ph
    return best_ph


def train_hard_p1_prior(
    n_structures: int = 80,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    epochs_per: int = 50,
    epochs_refine: int = 20,
    hidden: int = 128,
    seed: int = 0,
    lr: float = 3e-3,
    lr_refine: float = 1e-3,
    include_bridge: bool = True,
    verbose: bool = True,
) -> Tuple[PhaseMLP, Dict]:
    """
    Train domain-matched PhaseMLP on hard (+bridge) P1 synthetic cells.

    Returns (model, meta) with global feature norms stored on the model.
    """
    samples = list(
        iter_hard_p1_samples(
            n_structures,
            seed=seed,
            n_atoms_range=n_atoms_range,
            d_min_range=d_min_range,
            include_bridge=include_bridge,
        )
    )
    mu, sig, n_feat = accumulate_feature_stats(samples)
    model = PhaseMLP(hidden=hidden, seed=seed)
    model._feat_mu = mu  # type: ignore[attr-defined]
    model._feat_sig = sig  # type: ignore[attr-defined]

    all_losses: List[float] = []
    all_mpe: List[float] = []
    if verbose:
        print(
            f"Hard-P1 prior: {len(samples)} structures, hidden={hidden}, "
            f"epochs={epochs_per}+{epochs_refine}, N_feat_refl={n_feat}"
        )

    # Pass 1: main training with global norm
    for i, s in enumerate(samples):
        losses = train_phase_mlp_global(
            model,
            s["hkl"],
            s["amplitudes"],
            s["phases"],
            s["cell"],
            mu,
            sig,
            n_epochs=epochs_per,
            lr=lr,
            seed=seed + i,
            origin_invariant=True,
        )
        all_losses.append(float(np.mean(losses[-10:])))
        pred = predict_phases_hard_p1(
            model, s["hkl"], s["amplitudes"], s["cell"], origin_fom_search=False
        )
        # report origin-invariant MPE when available
        try:
            from grok_phase_solver.metrics.phase_error import (
                mean_phase_error_origin_invariant,
            )
            mpe, _ = mean_phase_error_origin_invariant(
                pred, s["phases"], s["hkl"], weights=s["amplitudes"]
            )
            mpe = float(mpe)
        except Exception:
            mpe = float(mean_phase_error(pred, s["phases"], weights=s["amplitudes"]))
        all_mpe.append(mpe)
        if verbose and (i < 3 or i == len(samples) - 1 or (i + 1) % 20 == 0):
            print(
                f"  [{i+1}/{len(samples)}] {s['region']} n={s['n_atoms']} "
                f"d={s['d_min']:.2f} loss≈{all_losses[-1]:.4f} MPE_OI={mpe:.1f}°"
            )

    # Pass 2: refine at lower LR (shuffle order)
    rng = np.random.default_rng(seed + 999)
    order = rng.permutation(len(samples))
    refine_mpe: List[float] = []
    for j, i in enumerate(order):
        s = samples[i]
        train_phase_mlp_global(
            model,
            s["hkl"],
            s["amplitudes"],
            s["phases"],
            s["cell"],
            mu,
            sig,
            n_epochs=epochs_refine,
            lr=lr_refine,
            seed=seed + 1000 + j,
            origin_invariant=True,
        )
        if j < 10 or j == len(samples) - 1:
            pred = predict_phases_hard_p1(
                model, s["hkl"], s["amplitudes"], s["cell"], origin_fom_search=False
            )
            try:
                from grok_phase_solver.metrics.phase_error import (
                    mean_phase_error_origin_invariant,
                )
                mpe_oi, _ = mean_phase_error_origin_invariant(
                    pred, s["phases"], s["hkl"], weights=s["amplitudes"]
                )
                refine_mpe.append(float(mpe_oi))
            except Exception:
                refine_mpe.append(
                    float(mean_phase_error(pred, s["phases"], weights=s["amplitudes"]))
                )
    if verbose and refine_mpe:
        print(f"  refine spot MPE≈{np.mean(refine_mpe):.1f}°")

    # Hold-out MPE on last 15% regenerated with different seeds
    hold_mpe = _holdout_mpe(
        model, n_hold=max(4, n_structures // 8), seed=seed + 50000,
        n_atoms_range=n_atoms_range, d_min_range=d_min_range,
    )
    if verbose:
        print(f"  hold-out hard MPE≈{np.mean(hold_mpe):.1f}° (n={len(hold_mpe)})")

    meta = {
        "domain": "hard_P1",
        "n_structures": n_structures,
        "n_atoms_range": list(n_atoms_range),
        "d_min_range": list(d_min_range),
        "include_bridge": include_bridge,
        "epochs_per": epochs_per,
        "epochs_refine": epochs_refine,
        "hidden": hidden,
        "seed": seed,
        "n_feature_reflections": n_feat,
        "train_final_losses": all_losses,
        "train_mpe_deg": all_mpe,
        "mean_train_mpe": float(np.mean(all_mpe)),
        "mean_holdout_mpe": float(np.mean(hold_mpe)) if hold_mpe else None,
        "holdout_mpe": hold_mpe,
        "feat_mu": mu.tolist(),
        "feat_sig": sig.tolist(),
        "note": (
            "Domain-matched PhaseMLP for synthetic P1 hard cells. "
            "Use as AI-PhaSeed prior via hard_p1_phaseed_solve. "
            "Not a claimed multi-SG or experimental general solver."
        ),
    }
    return model, meta


def _holdout_mpe(
    model: PhaseMLP,
    n_hold: int,
    seed: int,
    n_atoms_range: Tuple[int, int],
    d_min_range: Tuple[float, float],
) -> List[float]:
    from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant

    mpes = []
    for s in iter_hard_p1_samples(
        n_hold, seed=seed, n_atoms_range=n_atoms_range,
        d_min_range=d_min_range, include_bridge=False,
    ):
        pred = predict_phases_hard_p1(
            model, s["hkl"], s["amplitudes"], s["cell"], origin_fom_search=True
        )
        mpe_oi, _ = mean_phase_error_origin_invariant(
            pred, s["phases"], s["hkl"], weights=s["amplitudes"]
        )
        mpes.append(float(mpe_oi))
    return mpes


def save_hard_p1_prior(
    model: PhaseMLP,
    path: Path,
    meta: Optional[Dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mu = getattr(model, "_feat_mu", None)
    sig = getattr(model, "_feat_sig", None)
    np.savez(
        path,
        W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
        W3=model.W3, b3=model.b3,
        d_in=model.d_in, hidden=model.hidden, seed=model.seed,
        feat_mu=np.asarray(mu) if mu is not None else np.zeros(model.d_in),
        feat_sig=np.asarray(sig) if sig is not None else np.ones(model.d_in),
        domain=np.array("hard_P1"),
    )
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    return path


def load_hard_p1_prior(path: Path) -> PhaseMLP:
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    m = PhaseMLP(d_in=int(z["d_in"]), hidden=int(z["hidden"]), seed=int(z["seed"]))
    m.W1, m.b1 = z["W1"], z["b1"]
    m.W2, m.b2 = z["W2"], z["b2"]
    m.W3, m.b3 = z["W3"], z["b3"]
    if "feat_mu" in z.files:
        m._feat_mu = z["feat_mu"]  # type: ignore[attr-defined]
        m._feat_sig = z["feat_sig"]  # type: ignore[attr-defined]
    return m


def default_hard_p1_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "hard_p1_prior.npz"


def hard_p1_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    model: Optional[PhaseMLP] = None,
    model_path: Optional[Path] = None,
    n_seed: Optional[int] = None,
    seed_fraction: float = 0.25,
    n_extend: int = 15,
    polish: str = "charge_flipping",
    n_polish: int = 60,
    n_starts: int = 2,
    seed: int = 0,
    d_min: Optional[float] = None,
    prior_weight: float = 0.35,
    use_free_fom_gate: bool = True,
    verbose: bool = False,
):
    """
    Hard-P1 PhaseMLP seed → AI-PhaSeed extension → free-FOM polish.

    Loads default weights from data/processed/hard_p1_prior.npz if model unset.
    """
    from grok_phase_solver.physics.density import density_from_structure_factors
    from grok_phase_solver.solvers.ai_phaseed import ai_phaseed_solve
    from grok_phase_solver.solvers.free_fom import free_fom

    if model is None:
        path = Path(model_path) if model_path else default_hard_p1_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Hard-P1 prior not found at {path}. "
                "Train with: python scripts/train_hard_p1_prior.py"
            )
        model = load_hard_p1_prior(path)
        if verbose:
            print(f"  loaded hard-P1 prior from {path}")

    ph_ai = predict_phases_hard_p1(model, hkl, amplitudes, cell)
    ph, rho, info = ai_phaseed_solve(
        hkl,
        amplitudes,
        cell,
        ph_ai,
        n_seed=n_seed,
        seed_fraction=seed_fraction,
        n_extend=n_extend,
        polish=polish,
        n_polish=n_polish,
        n_starts=n_starts,
        seed=seed,
        d_min=d_min,
        prior_weight=prior_weight,
        use_free_fom_gate=use_free_fom_gate,
        verbose=verbose,
    )
    info["seed_source"] = "hard_p1_prior"
    rho_ai = density_from_structure_factors(
        hkl, amplitudes * np.exp(1j * ph_ai), cell, d_min=d_min
    )
    info["fom_prior_only"] = free_fom(hkl, amplitudes, ph_ai, cell, density=rho_ai)
    info["phases_prior"] = ph_ai
    return ph, rho, info
