"""
Physics-recycle net: PhaseMLP phase predictor inside Fourier modulus projection.

Training focuses on the **hard** solvability region (n≥12 atoms, d_min≥1.5 Å)
where pure CF/RAAR fail under strict SuccessThresholds.

At inference:
  phase_fn(hkl, amp, φ_in) → PhaseMLP(features) → φ_pred
  then one or more ER-style modulus projections (phase_recycle).

Honest scope: supervised synthetic learning + physics polish — not a claimed
general experimental solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.models.phase_mlp import (
    PhaseMLP,
    reflection_features,
    train_phase_mlp_on_structure,
)
from grok_phase_solver.solvers.phase_recycle import phase_recycle


def standardize_features(
    X: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sig: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mu is None:
        mu = X.mean(0)
    if sig is None:
        sig = X.std(0) + 1e-8
    return (X - mu) / sig, mu, sig


def make_phase_fn(
    model: PhaseMLP,
    cell: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sig: Optional[np.ndarray] = None,
    blend: float = 1.0,
):
    """
    Build phase_fn(hkl, amplitudes, phases_in) → phases_out for phase_recycle.

    blend ∈ [0,1]: 1 = pure network; 0 = keep input phases;
    intermediate = circular blend toward network prediction.
    """
    mu = getattr(model, "_feat_mu", mu)
    sig = getattr(model, "_feat_sig", sig)

    def phase_fn(hkl, amplitudes, phases_in):
        X = reflection_features(hkl, amplitudes, cell)
        if mu is not None and sig is not None:
            X = (X - mu) / sig
        ph_pred = model.predict_phases(X)
        if blend >= 1.0 - 1e-12:
            return ph_pred
        if blend <= 1e-12:
            return np.asarray(phases_in, dtype=np.float64)
        # circular interpolation via unit vectors
        c = (1.0 - blend) * np.cos(phases_in) + blend * np.cos(ph_pred)
        s = (1.0 - blend) * np.sin(phases_in) + blend * np.sin(ph_pred)
        return np.arctan2(s, c)

    return phase_fn


def recycle_net_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    model: PhaseMLP,
    n_cycles: int = 8,
    seed: int = 0,
    d_min: Optional[float] = None,
    blend: float = 1.0,
    use_positivity: bool = True,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Run PhaseMLP-seeded / PhaseMLP-in-loop physics recycle."""
    phase_fn = make_phase_fn(model, cell, blend=blend)
    phases, rho, hist = phase_recycle(
        hkl,
        amplitudes,
        cell,
        n_cycles=n_cycles,
        phase_init=phase_init,
        phase_fn=phase_fn,
        use_positivity=use_positivity,
        seed=seed,
        d_min=d_min,
        verbose=verbose,
    )
    hist = dict(hist)
    hist["algorithm"] = "recycle_net"
    hist["blend"] = blend
    return phases, rho, hist


def train_recycle_net_hard(
    n_structures: int = 24,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    epochs_per: int = 60,
    hidden: int = 96,
    seed: int = 0,
    lr: float = 3e-3,
    verbose: bool = True,
) -> Tuple[PhaseMLP, Dict]:
    """
    Train PhaseMLP on hard-region synthetic cells (n≥12, d_min≥1.5).

    Returns (model, training_meta).
    """
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.metrics.phase_error import mean_phase_error
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    rng = np.random.default_rng(seed)
    model = PhaseMLP(hidden=hidden, seed=seed)
    all_losses: List[float] = []
    all_mpe: List[float] = []
    # Running feature stats across structures (approximate global norm)
    feat_sum = None
    feat_sq = None
    feat_n = 0

    n_lo, n_hi = n_atoms_range
    d_lo, d_hi = d_min_range

    for i in range(n_structures):
        n_atoms = int(rng.integers(n_lo, n_hi + 1))
        d_min = float(rng.uniform(d_lo, d_hi))
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, phases = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell

        X_raw = reflection_features(hkl, amp, cell)
        if feat_sum is None:
            feat_sum = X_raw.sum(0)
            feat_sq = (X_raw ** 2).sum(0)
        else:
            feat_sum = feat_sum + X_raw.sum(0)
            feat_sq = feat_sq + (X_raw ** 2).sum(0)
        feat_n += len(X_raw)

        losses = train_phase_mlp_on_structure(
            model,
            hkl,
            amp,
            phases,
            cell,
            n_epochs=epochs_per,
            lr=lr,
            seed=seed + i,
            verbose=(verbose and i == 0),
        )
        all_losses.append(float(np.mean(losses[-10:])))

        # eval with structure-local norm (as training used)
        X = reflection_features(hkl, amp, cell)
        mu, sig = X.mean(0), X.std(0) + 1e-8
        pred = model.predict_phases((X - mu) / sig)
        mpe = mean_phase_error(pred, phases, weights=amp)
        all_mpe.append(float(mpe))
        if verbose:
            print(
                f"  [{i+1}/{n_structures}] n={n_atoms} d={d_min:.2f} "
                f"Nrefl={len(amp)} loss≈{all_losses[-1]:.4f} MPE={mpe:.1f}°"
            )

    # Store global feature stats for cross-structure inference
    mu_g = feat_sum / max(feat_n, 1)
    var_g = feat_sq / max(feat_n, 1) - mu_g ** 2
    sig_g = np.sqrt(np.maximum(var_g, 1e-12))
    model._feat_mu = mu_g  # type: ignore[attr-defined]
    model._feat_sig = sig_g  # type: ignore[attr-defined]

    meta = {
        "n_structures": n_structures,
        "n_atoms_range": list(n_atoms_range),
        "d_min_range": list(d_min_range),
        "epochs_per": epochs_per,
        "hidden": hidden,
        "seed": seed,
        "final_losses": all_losses,
        "mean_mpe_deg": all_mpe,
        "mean_final_loss": float(np.mean(all_losses)),
        "mean_mpe": float(np.mean(all_mpe)),
        "feat_mu": mu_g.tolist(),
        "feat_sig": sig_g.tolist(),
        "note": (
            "Trained on hard synthetic region (n≥12, d_min≥1.5). "
            "Use via recycle_net_solve / phase_fn inside phase_recycle. "
            "Not a claimed general experimental solver."
        ),
    }
    return model, meta


def save_recycle_net(model: PhaseMLP, path: Path, meta: Optional[Dict] = None) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    # also persist feature norms
    extra = {}
    if hasattr(model, "_feat_mu"):
        extra["feat_mu"] = np.asarray(model._feat_mu)
        extra["feat_sig"] = np.asarray(model._feat_sig)
        np.savez(
            path,
            W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
            W3=model.W3, b3=model.b3,
            d_in=model.d_in, hidden=model.hidden, seed=model.seed,
            feat_mu=extra["feat_mu"], feat_sig=extra["feat_sig"],
        )
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    return path


def load_recycle_net(path: Path) -> PhaseMLP:
    path = Path(path)
    z = np.load(path)
    m = PhaseMLP(d_in=int(z["d_in"]), hidden=int(z["hidden"]), seed=int(z["seed"]))
    m.W1, m.b1 = z["W1"], z["b1"]
    m.W2, m.b2 = z["W2"], z["b2"]
    m.W3, m.b3 = z["W3"], z["b3"]
    if "feat_mu" in z.files:
        m._feat_mu = z["feat_mu"]  # type: ignore[attr-defined]
        m._feat_sig = z["feat_sig"]  # type: ignore[attr-defined]
    return m
