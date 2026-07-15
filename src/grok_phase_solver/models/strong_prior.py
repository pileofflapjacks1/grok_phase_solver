"""
Strong phase prior: GraphPhaseNet trained on hard multi-SG synthetic data.

Upgrades vs hard_p1 PhaseMLP:
- Triplet-graph message passing (direct-methods connectivity)
- Multi-SG: P1 + P-1 (centrosymmetric expansion)
- Origin/enantiomorph-invariant training
- Free-FOM origin search at inference
- Seeds AI-PhaSeed for full structure solution

Honest scope: improved synthetic hard-region prior — not a general
experimental multi-SG production solver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from grok_phase_solver.models.graph_phase_net import (
    GraphPhaseNet,
    build_undirected_adj,
    node_features_from_graph,
    prepare_graph_batch,
)
from grok_phase_solver.models.hard_p1_prior import origin_phase_candidates
from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.metrics.phase_error import (
    mean_phase_error,
    mean_phase_error_origin_invariant,
)


def iter_hard_multsg_samples(
    n_samples: int,
    seed: int = 0,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    include_bridge: bool = True,
    p_minus1_frac: float = 0.25,
) -> Iterator[Dict]:
    """Yield hard-region samples in P1 and P-1."""
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.data.synthetic_v2 import make_centrosymmetric_copy
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    rng = np.random.default_rng(seed)
    n_lo, n_hi = n_atoms_range
    d_lo, d_hi = d_min_range
    for i in range(n_samples):
        s = int(rng.integers(0, 2**31 - 1))
        if include_bridge and (i % 5 == 0):
            n_atoms = int(rng.integers(8, 12))
            d_min = float(rng.uniform(1.2, 1.5))
            region = "bridge"
        else:
            n_atoms = int(rng.integers(n_lo, n_hi + 1))
            d_min = float(rng.uniform(d_lo, d_hi))
            region = "hard"
        st = generate_random_organic(n_atoms=n_atoms, seed=s, space_group="P1")
        sg = "P1"
        if rng.random() < p_minus1_frac:
            try:
                st = make_centrosymmetric_copy(st)
                sg = "P-1"
            except Exception:
                pass
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
            "space_group": sg,
            "structure_seed": s,
            "fracs": data["fracs"],
            "elements": data["elements"],
        }


def _standardize(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu) / sig


def train_graph_on_sample(
    model: GraphPhaseNet,
    sample: Dict,
    mu: np.ndarray,
    sig: np.ndarray,
    n_epochs: int = 40,
    lr: float = 2e-3,
    seed: int = 0,
    max_reflections: int = 100,
    n_origin_grid: int = 3,
) -> List[float]:
    """Train model on one structure with OI targets on strong graph nodes."""
    hkl = sample["hkl"]
    amp = sample["amplitudes"]
    ph = sample["phases"]
    cell = sample["cell"]
    batch = prepare_graph_batch(
        hkl, amp, cell, max_reflections=max_reflections, e_min=0.9
    )
    X = _standardize(batch["X"], mu, sig)
    nbrs, wts = batch["nbrs"], batch["wts"]
    idx = batch["node_idx"]
    ph_s = ph[idx]
    w_s = amp[idx].astype(np.float64)

    if X.shape[0] < 4:
        return [0.0]

    cands = origin_phase_candidates(
        batch["hkl_strong"], ph_s, n_grid=n_origin_grid, include_enantiomorph=True
    )
    # Restrict cand list for speed: sample subset of origins if huge
    if len(cands) > 80:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(cands), size=80, replace=False)
        cands = [cands[i] for i in pick]

    losses = []
    for ep in range(n_epochs):
        # pick best origin target
        z, _ = model.forward(X, nbrs, wts)
        best_ph = cands[0]
        best_l = 1e99
        wn = w_s / (w_s.mean() + 1e-16)
        for c in cands:
            ut = np.column_stack([np.cos(c), np.sin(c)])
            diff = z - ut
            l = 0.5 * float(np.mean(wn * np.sum(diff ** 2, axis=1)))
            if l < best_l:
                best_l = l
                best_ph = c
        loss, grads = model.loss_and_backward(X, nbrs, wts, best_ph, weights=w_s)
        model.step(grads, lr=lr)
        losses.append(loss)
    return losses


def accumulate_feature_stats(samples: List[Dict], max_reflections: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    feat_sum = None
    feat_sq = None
    n = 0
    for s in samples:
        batch = prepare_graph_batch(
            s["hkl"], s["amplitudes"], s["cell"], max_reflections=max_reflections
        )
        X = batch["X"]
        if len(X) == 0:
            continue
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
    return mu, sig


def predict_strong_phases(
    model: GraphPhaseNet,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    max_reflections: int = 120,
    origin_fom_search: bool = True,
    n_origin_grid: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict phases on strong reflections.

    Returns (node_idx, phases_strong).
    """
    batch = prepare_graph_batch(
        hkl, amplitudes, cell, max_reflections=max_reflections, e_min=0.9
    )
    X = batch["X"]
    if hasattr(model, "_feat_mu"):
        X = _standardize(X, model._feat_mu, model._feat_sig)
    idx = batch["node_idx"]
    if len(X) == 0:
        return idx, np.array([])
    ph = model.predict_phases(X, batch["nbrs"], batch["wts"])

    if not origin_fom_search or len(ph) < 3:
        return idx, ph

    from grok_phase_solver.solvers.free_fom import free_fom

    # Build full phase vector for FOM: strong predicted, weak random fixed seed
    rng = np.random.default_rng(0)
    full0 = rng.uniform(-np.pi, np.pi, size=len(amplitudes))
    best_c = -1.0
    best_ph = ph
    for cand in origin_phase_candidates(
        batch["hkl_strong"], ph, n_grid=n_origin_grid, include_enantiomorph=True
    ):
        full = full0.copy()
        full[idx] = cand
        fom = free_fom(
            hkl, amplitudes, full, cell, include_shells=False, include_sayre=False
        )
        if fom["composite"] > best_c:
            best_c = fom["composite"]
            best_ph = cand
    return idx, best_ph


def predict_full_phases(
    model: GraphPhaseNet,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    max_reflections: int = 120,
    origin_fom_search: bool = True,
) -> np.ndarray:
    """Full phase array: graph net on strong nodes, weak = soft from strong NN."""
    idx, ph_s = predict_strong_phases(
        model, hkl, amplitudes, cell,
        max_reflections=max_reflections,
        origin_fom_search=origin_fom_search,
    )
    phases = np.zeros(len(amplitudes), dtype=np.float64)
    if len(idx) == 0:
        return phases
    phases[idx] = ph_s
    # Weak reflections: nearest strong in reciprocal-space (index space)
    hkl = np.asarray(hkl, dtype=np.float64)
    strong_h = hkl[idx]
    weak = np.setdiff1d(np.arange(len(amplitudes)), idx)
    if len(weak) and len(idx):
        for i in weak:
            d2 = np.sum((strong_h - hkl[i]) ** 2, axis=1)
            j = int(np.argmin(d2))
            phases[i] = ph_s[j]  # crude transfer
    return phases


def train_strong_prior(
    n_structures: int = 60,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    epochs_per: int = 35,
    epochs_refine: int = 12,
    hidden: int = 80,
    n_layers: int = 2,
    max_reflections: int = 100,
    seed: int = 0,
    lr: float = 2e-3,
    lr_refine: float = 8e-4,
    verbose: bool = True,
) -> Tuple[GraphPhaseNet, Dict]:
    samples = list(
        iter_hard_multsg_samples(
            n_structures,
            seed=seed,
            n_atoms_range=n_atoms_range,
            d_min_range=d_min_range,
        )
    )
    mu, sig = accumulate_feature_stats(samples, max_reflections=max_reflections)
    model = GraphPhaseNet(d_in=8, hidden=hidden, n_layers=n_layers, seed=seed)
    model._feat_mu = mu  # type: ignore[attr-defined]
    model._feat_sig = sig  # type: ignore[attr-defined]

    all_losses: List[float] = []
    all_mpe: List[float] = []
    if verbose:
        print(
            f"Strong prior (GraphPhaseNet): {len(samples)} structs, "
            f"hidden={hidden}, layers={n_layers}, max_refl={max_reflections}"
        )

    for i, s in enumerate(samples):
        losses = train_graph_on_sample(
            model, s, mu, sig,
            n_epochs=epochs_per, lr=lr, seed=seed + i,
            max_reflections=max_reflections,
        )
        all_losses.append(float(np.mean(losses[-5:])) if losses else 0.0)
        idx, ph_pred = predict_strong_phases(
            model, s["hkl"], s["amplitudes"], s["cell"],
            max_reflections=max_reflections, origin_fom_search=False,
        )
        if len(idx):
            mpe, _ = mean_phase_error_origin_invariant(
                ph_pred, s["phases"][idx], s["hkl"][idx],
                weights=s["amplitudes"][idx],
            )
            all_mpe.append(float(mpe))
        if verbose and (i < 3 or (i + 1) % 15 == 0 or i == len(samples) - 1):
            mpe_s = all_mpe[-1] if all_mpe else float("nan")
            print(
                f"  [{i+1}/{len(samples)}] {s['region']} {s['space_group']} "
                f"n={s['n_atoms']} d={s['d_min']:.2f} loss≈{all_losses[-1]:.4f} "
                f"MPE_OI={mpe_s:.1f}° nodes={len(idx)}"
            )

    # refine pass
    rng = np.random.default_rng(seed + 7)
    for j, i in enumerate(rng.permutation(len(samples))):
        train_graph_on_sample(
            model, samples[i], mu, sig,
            n_epochs=epochs_refine, lr=lr_refine, seed=seed + 2000 + j,
            max_reflections=max_reflections,
        )

    hold = _holdout_eval(
        model, n_hold=max(4, n_structures // 10), seed=seed + 90000,
        n_atoms_range=n_atoms_range, d_min_range=d_min_range,
        max_reflections=max_reflections,
    )
    meta = {
        "architecture": "GraphPhaseNet",
        "domain": "hard_multi_SG",
        "n_structures": n_structures,
        "n_atoms_range": list(n_atoms_range),
        "d_min_range": list(d_min_range),
        "hidden": hidden,
        "n_layers": n_layers,
        "max_reflections": max_reflections,
        "epochs_per": epochs_per,
        "epochs_refine": epochs_refine,
        "seed": seed,
        "train_losses": all_losses,
        "train_mpe_oi": all_mpe,
        "mean_train_mpe_oi": float(np.mean(all_mpe)) if all_mpe else None,
        "holdout": hold,
        "mean_holdout_mpe_oi": float(np.mean([h["mpe_oi"] for h in hold])) if hold else None,
        "mean_holdout_mapcc_prior": float(np.mean([h["mapcc_prior"] for h in hold])) if hold else None,
        "feat_mu": mu.tolist(),
        "feat_sig": sig.tolist(),
        "note": (
            "Triplet-graph phase prior for hard multi-SG synthetic. "
            "Use strong_prior_phaseed_solve. Not a claimed general experimental solver."
        ),
    }
    if verbose and hold:
        print(
            f"  hold-out mean MPE_OI={meta['mean_holdout_mpe_oi']:.1f}° "
            f"prior mapCC={meta['mean_holdout_mapcc_prior']:.3f}"
        )
    return model, meta


def _holdout_eval(
    model: GraphPhaseNet,
    n_hold: int,
    seed: int,
    n_atoms_range,
    d_min_range,
    max_reflections: int,
) -> List[Dict]:
    from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
    from grok_phase_solver.physics.density import density_from_structure_factors

    rows = []
    for s in iter_hard_multsg_samples(
        n_hold, seed=seed, n_atoms_range=n_atoms_range, d_min_range=d_min_range,
        include_bridge=False, p_minus1_frac=0.2,
    ):
        ph = predict_full_phases(
            model, s["hkl"], s["amplitudes"], s["cell"],
            max_reflections=max_reflections, origin_fom_search=True,
        )
        mpe, _ = mean_phase_error_origin_invariant(
            ph, s["phases"], s["hkl"], weights=s["amplitudes"]
        )
        rho = density_from_structure_factors(
            s["hkl"], s["amplitudes"] * np.exp(1j * ph), s["cell"], d_min=s["d_min"]
        )
        rho_t = density_from_structure_factors(
            s["hkl"], s["amplitudes"] * np.exp(1j * s["phases"]), s["cell"],
            shape=rho.shape,
        )
        cc, _ = map_correlation_origin_invariant(rho, rho_t)
        rows.append({
            "n_atoms": s["n_atoms"],
            "d_min": s["d_min"],
            "space_group": s["space_group"],
            "mpe_oi": float(mpe),
            "mapcc_prior": float(cc),
        })
    return rows


def save_strong_prior(model: GraphPhaseNet, path: Path, meta: Optional[Dict] = None) -> Path:
    path = Path(path)
    model.save(path)
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    return path


def load_strong_prior(path: Path) -> GraphPhaseNet:
    return GraphPhaseNet.load(path)


def default_strong_prior_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "strong_prior.npz"


def strong_prior_phaseed_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    model: Optional[GraphPhaseNet] = None,
    model_path: Optional[Path] = None,
    n_seed: Optional[int] = None,
    seed_fraction: float = 0.30,
    n_extend: int = 15,
    polish: str = "charge_flipping",
    n_polish: int = 60,
    n_starts: int = 2,
    seed: int = 0,
    d_min: Optional[float] = None,
    prior_weight: float = 0.40,
    max_reflections: int = 120,
    use_free_fom_gate: bool = True,
    verbose: bool = False,
):
    """GraphPhaseNet seed → AI-PhaSeed → free-FOM polish."""
    from grok_phase_solver.physics.density import density_from_structure_factors
    from grok_phase_solver.solvers.ai_phaseed import ai_phaseed_solve
    from grok_phase_solver.solvers.free_fom import free_fom

    if model is None:
        path = Path(model_path) if model_path else default_strong_prior_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Strong prior not found at {path}. "
                "Train with: python scripts/train_strong_prior.py"
            )
        model = load_strong_prior(path)
        if verbose:
            print(f"  loaded strong prior from {path}")

    ph_ai = predict_full_phases(
        model, hkl, amplitudes, cell,
        max_reflections=max_reflections, origin_fom_search=True,
    )
    ph, rho, info = ai_phaseed_solve(
        hkl, amplitudes, cell, ph_ai,
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
    info["seed_source"] = "strong_graph_prior"
    rho_ai = density_from_structure_factors(
        hkl, amplitudes * np.exp(1j * ph_ai), cell, d_min=d_min
    )
    info["fom_prior_only"] = free_fom(hkl, amplitudes, ph_ai, cell, density=rho_ai)
    info["phases_prior"] = ph_ai
    return ph, rho, info
