"""
Strong phase prior: GraphPhaseNet trained on hard multi-SG synthetic data.

Scale upgrades vs first GraphPhaseNet pass:
- Vectorized adj message passing
- Triplet-consistency auxiliary loss (origin-invariant)
- Curriculum: bridge → hard
- Multi-pass global epochs over larger synthetic sets
- Larger default capacity (hidden / layers / reflections)

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
    prepare_graph_batch,
)
from grok_phase_solver.models.hard_p1_prior import origin_phase_candidates
from grok_phase_solver.metrics.phase_error import (
    mean_phase_error_origin_invariant,
)


def iter_hard_multsg_samples(
    n_samples: int,
    seed: int = 0,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    include_bridge: bool = True,
    p_minus1_frac: float = 0.25,
    bridge_frac: float = 0.25,
    wilson_match: bool = False,
    wilson_template: Optional[Dict] = None,
) -> Iterator[Dict]:
    """Yield hard-region samples in P1 and P-1 (optional bridge easy cells).

    If ``wilson_match`` is True, |F| are transformed toward an experimental
    Wilson template (phases from Fcalc unchanged).
    """
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.data.synthetic_v2 import make_centrosymmetric_copy
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    rng = np.random.default_rng(seed)
    n_lo, n_hi = n_atoms_range
    d_lo, d_hi = d_min_range
    template = wilson_template
    if wilson_match and template is None:
        try:
            from grok_phase_solver.data.wilson_match import load_reference_template

            template = load_reference_template()
        except Exception:
            template = None
    for i in range(n_samples):
        s = int(rng.integers(0, 2**31 - 1))
        if include_bridge and (rng.random() < bridge_frac):
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
        amp = data["amplitudes"]
        wmeta = {"matched": False}
        if wilson_match and template is not None:
            from grok_phase_solver.data.wilson_match import apply_wilson_match_if_template

            amp, wmeta = apply_wilson_match_if_template(
                data["hkl"], amp, st.cell, phases=data["phases"], template=template,
                config=None,
            )
            wmeta["matched"] = bool(wmeta.get("matched", True))
        # curriculum score: lower = easier
        difficulty = float(n_atoms) * float(d_min)
        yield {
            "name": st.name,
            "hkl": data["hkl"],
            "amplitudes": amp,
            "phases": data["phases"],
            "cell": st.cell,
            "n_atoms": data["n_atoms_cell"],
            "d_min": d_min,
            "region": region,
            "space_group": sg,
            "structure_seed": s,
            "fracs": data["fracs"],
            "elements": data["elements"],
            "difficulty": difficulty,
            "wilson_match": wmeta,
        }


def _standardize(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu) / sig


def _prebuild_packed(
    sample: Dict,
    mu: np.ndarray,
    sig: np.ndarray,
    max_reflections: int,
    n_origin_grid: int,
    seed: int,
    max_cands: int = 54,
) -> Optional[Dict]:
    """Precompute graph, standardized features, origin candidate targets."""
    batch = prepare_graph_batch(
        sample["hkl"],
        sample["amplitudes"],
        sample["cell"],
        max_reflections=max_reflections,
        e_min=0.9,
    )
    X = _standardize(batch["X"], mu, sig)
    idx = batch["node_idx"]
    if X.shape[0] < 4:
        return None
    ph_s = sample["phases"][idx]
    w_s = sample["amplitudes"][idx].astype(np.float64)
    cands = origin_phase_candidates(
        batch["hkl_strong"], ph_s, n_grid=n_origin_grid, include_enantiomorph=True
    )
    if len(cands) > max_cands:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(cands), size=max_cands, replace=False)
        cands = [cands[i] for i in pick]
    cand_ut = [np.column_stack([np.cos(c), np.sin(c)]) for c in cands]
    return {
        "X": X,
        "adj": batch["adj"],
        "edges": batch["edges"],
        "edge_weight": batch["edge_weight"],
        "nbrs": batch["nbrs"],
        "wts": batch["wts"],
        "node_idx": idx,
        "w_s": w_s,
        "cands": cands,
        "cand_ut": cand_ut,
        "hkl_strong": batch["hkl_strong"],
        "sample": sample,
    }


def _pick_best_origin(z: np.ndarray, packed: Dict) -> np.ndarray:
    wn = packed["w_s"] / (packed["w_s"].mean() + 1e-16)
    best_ph = packed["cands"][0]
    best_l = 1e99
    for c, ut in zip(packed["cands"], packed["cand_ut"]):
        diff = z - ut
        l = 0.5 * float(np.mean(wn * np.sum(diff ** 2, axis=1)))
        if l < best_l:
            best_l = l
            best_ph = c
    return best_ph


def train_graph_on_packed(
    model: GraphPhaseNet,
    packed: Dict,
    n_epochs: int = 20,
    lr: float = 2e-3,
    triplet_weight: float = 0.15,
    origin_every: int = 3,
) -> List[float]:
    """Train on one prebuilt structure pack with OI + triplet aux."""
    X = packed["X"]
    adj = packed["adj"]
    edges = packed["edges"]
    ewt = packed["edge_weight"]
    w_s = packed["w_s"]
    losses: List[float] = []
    best_ph = packed["cands"][0]
    for ep in range(n_epochs):
        if ep % max(origin_every, 1) == 0:
            z, _ = model.forward(X, adj=adj)
            best_ph = _pick_best_origin(z, packed)
        loss, grads = model.loss_and_backward(
            X,
            adj=adj,
            phase_true=best_ph,
            weights=w_s,
            edges=edges,
            edge_weight=ewt,
            triplet_weight=triplet_weight,
        )
        model.step(grads, lr=lr)
        losses.append(loss)
    return losses


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
    triplet_weight: float = 0.15,
) -> List[float]:
    """Train model on one structure with OI targets on strong graph nodes."""
    packed = _prebuild_packed(
        sample, mu, sig, max_reflections, n_origin_grid, seed
    )
    if packed is None:
        return [0.0]
    return train_graph_on_packed(
        model, packed, n_epochs=n_epochs, lr=lr, triplet_weight=triplet_weight
    )


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
    ph = model.predict_phases(X, adj=batch["adj"])

    if not origin_fom_search or len(ph) < 3:
        return idx, ph

    from grok_phase_solver.solvers.free_fom import free_fom

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
    """Full phase array: graph net on strong nodes, weak = nearest strong."""
    idx, ph_s = predict_strong_phases(
        model, hkl, amplitudes, cell,
        max_reflections=max_reflections,
        origin_fom_search=origin_fom_search,
    )
    phases = np.zeros(len(amplitudes), dtype=np.float64)
    if len(idx) == 0:
        return phases
    phases[idx] = ph_s
    hkl = np.asarray(hkl, dtype=np.float64)
    strong_h = hkl[idx]
    weak = np.setdiff1d(np.arange(len(amplitudes)), idx)
    if len(weak) and len(idx):
        for i in weak:
            d2 = np.sum((strong_h - hkl[i]) ** 2, axis=1)
            j = int(np.argmin(d2))
            phases[i] = ph_s[j]
    return phases


def train_strong_prior(
    n_structures: int = 250,
    n_atoms_range: Tuple[int, int] = (12, 20),
    d_min_range: Tuple[float, float] = (1.5, 2.0),
    epochs_per: int = 18,
    n_global_passes: int = 3,
    epochs_refine: int = 8,
    hidden: int = 128,
    n_layers: int = 3,
    max_reflections: int = 120,
    seed: int = 0,
    lr: float = 2e-3,
    lr_refine: float = 6e-4,
    triplet_weight: float = 0.15,
    curriculum: bool = True,
    wilson_match: bool = False,
    verbose: bool = True,
) -> Tuple[GraphPhaseNet, Dict]:
    """
    Scale-ready GraphPhaseNet trainer.

    Defaults target ~5× prior budget (structures × capacity) with curriculum
    multi-pass training and triplet auxiliary loss.

    wilson_match: if True and a Wilson template exists, match |F| statistics
    to experimental before training (phases unchanged).
    """
    samples = list(
        iter_hard_multsg_samples(
            n_structures,
            seed=seed,
            n_atoms_range=n_atoms_range,
            d_min_range=d_min_range,
            include_bridge=True,
            bridge_frac=0.30,
            p_minus1_frac=0.25,
            wilson_match=wilson_match,
        )
    )
    if curriculum:
        samples = sorted(samples, key=lambda s: s["difficulty"])

    mu, sig = accumulate_feature_stats(samples, max_reflections=max_reflections)
    model = GraphPhaseNet(d_in=8, hidden=hidden, n_layers=n_layers, seed=seed)
    model._feat_mu = mu  # type: ignore[attr-defined]
    model._feat_sig = sig  # type: ignore[attr-defined]

    if verbose:
        n_bridge = sum(1 for s in samples if s["region"] == "bridge")
        n_p1 = sum(1 for s in samples if s["space_group"] == "P1")
        print(
            f"Strong prior (scaled GraphPhaseNet): {len(samples)} structs "
            f"(bridge={n_bridge}, P1={n_p1}, P-1={len(samples)-n_p1}), "
            f"hidden={hidden}, layers={n_layers}, max_refl={max_reflections}, "
            f"triplet_w={triplet_weight}, passes={n_global_passes}, "
            f"epochs/struct={epochs_per}, curriculum={curriculum}"
        )

    # Prebuild packs once
    packs: List[Dict] = []
    for i, s in enumerate(samples):
        p = _prebuild_packed(
            s, mu, sig, max_reflections, n_origin_grid=3, seed=seed + i
        )
        if p is not None:
            packs.append(p)
    if verbose:
        print(f"  prebuilt {len(packs)}/{len(samples)} graph packs")

    all_losses: List[float] = []
    all_mpe: List[float] = []
    pass_mpe: List[float] = []

    for gpass in range(n_global_passes):
        # anneal LR and increase hard focus
        lr_p = lr * (0.7 ** gpass)
        # later passes: slightly more epochs on hard half
        order = list(range(len(packs)))
        if curriculum and gpass > 0:
            # reverse curriculum emphasis: hard structures first on later passes
            order = list(reversed(order))
        rng = np.random.default_rng(seed + 1000 * gpass)
        # mild shuffle within curriculum halves for SGD noise
        if len(order) > 4:
            mid = len(order) // 2
            first, second = order[:mid], order[mid:]
            rng.shuffle(first)
            rng.shuffle(second)
            order = first + second if gpass == 0 else second + first

        for j, pi in enumerate(order):
            packed = packs[pi]
            ep = epochs_per if gpass < n_global_passes - 1 else max(epochs_per // 2, 6)
            losses = train_graph_on_packed(
                model,
                packed,
                n_epochs=ep,
                lr=lr_p,
                triplet_weight=triplet_weight,
                origin_every=3,
            )
            all_losses.append(float(np.mean(losses[-5:])) if losses else 0.0)

            if j < 2 or (j + 1) % max(25, len(packs) // 6) == 0 or j == len(order) - 1:
                s = packed["sample"]
                idx, ph_pred = predict_strong_phases(
                    model,
                    s["hkl"],
                    s["amplitudes"],
                    s["cell"],
                    max_reflections=max_reflections,
                    origin_fom_search=False,
                )
                mpe_s = float("nan")
                if len(idx):
                    mpe, _ = mean_phase_error_origin_invariant(
                        ph_pred,
                        s["phases"][idx],
                        s["hkl"][idx],
                        weights=s["amplitudes"][idx],
                    )
                    mpe_s = float(mpe)
                    all_mpe.append(mpe_s)
                if verbose:
                    print(
                        f"  pass {gpass+1}/{n_global_passes} "
                        f"[{j+1}/{len(order)}] {s['region']} {s['space_group']} "
                        f"n={s['n_atoms']} d={s['d_min']:.2f} "
                        f"loss≈{all_losses[-1]:.4f} MPE_OI={mpe_s:.1f}° "
                        f"nodes={len(idx)} edges={packed['edges'].shape[0]}"
                    )

        # pass-level hold-out probe (cheap)
        if all_mpe:
            pass_mpe.append(float(np.mean(all_mpe[-min(20, len(all_mpe)):])))
            if verbose:
                print(f"  → pass {gpass+1} recent train MPE_OI≈{pass_mpe[-1]:.1f}°")

    # refine pass: hard-only packs
    hard_packs = [p for p in packs if p["sample"]["region"] == "hard"]
    if not hard_packs:
        hard_packs = packs
    rng = np.random.default_rng(seed + 7)
    for j, pi in enumerate(rng.permutation(len(hard_packs))):
        train_graph_on_packed(
            model,
            hard_packs[pi],
            n_epochs=epochs_refine,
            lr=lr_refine,
            triplet_weight=triplet_weight * 1.2,
            origin_every=2,
        )

    hold = _holdout_eval(
        model,
        n_hold=max(6, n_structures // 12),
        seed=seed + 90000,
        n_atoms_range=n_atoms_range,
        d_min_range=d_min_range,
        max_reflections=max_reflections,
    )
    meta = {
        "architecture": "GraphPhaseNet",
        "scale": "v2",
        "domain": "hard_multi_SG",
        "n_structures": n_structures,
        "n_packs": len(packs),
        "n_atoms_range": list(n_atoms_range),
        "d_min_range": list(d_min_range),
        "hidden": hidden,
        "n_layers": n_layers,
        "max_reflections": max_reflections,
        "epochs_per": epochs_per,
        "n_global_passes": n_global_passes,
        "epochs_refine": epochs_refine,
        "triplet_weight": triplet_weight,
        "curriculum": curriculum,
        "wilson_match": wilson_match,
        "seed": seed,
        "train_losses": all_losses,
        "train_mpe_oi": all_mpe,
        "pass_mpe_oi": pass_mpe,
        "mean_train_mpe_oi": float(np.mean(all_mpe)) if all_mpe else None,
        "holdout": hold,
        "mean_holdout_mpe_oi": float(np.mean([h["mpe_oi"] for h in hold])) if hold else None,
        "mean_holdout_mapcc_prior": float(np.mean([h["mapcc_prior"] for h in hold])) if hold else None,
        "feat_mu": mu.tolist(),
        "feat_sig": sig.tolist(),
        "note": (
            "Scaled triplet-graph phase prior (curriculum multi-pass + triplet aux). "
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
        n_hold,
        seed=seed,
        n_atoms_range=n_atoms_range,
        d_min_range=d_min_range,
        include_bridge=False,
        p_minus1_frac=0.2,
    ):
        ph = predict_full_phases(
            model,
            s["hkl"],
            s["amplitudes"],
            s["cell"],
            max_reflections=max_reflections,
            origin_fom_search=True,
        )
        mpe, _ = mean_phase_error_origin_invariant(
            ph, s["phases"], s["hkl"], weights=s["amplitudes"]
        )
        rho = density_from_structure_factors(
            s["hkl"], s["amplitudes"] * np.exp(1j * ph), s["cell"], d_min=s["d_min"]
        )
        rho_t = density_from_structure_factors(
            s["hkl"],
            s["amplitudes"] * np.exp(1j * s["phases"]),
            s["cell"],
            shape=rho.shape,
        )
        cc, _ = map_correlation_origin_invariant(rho, rho_t)
        rows.append(
            {
                "n_atoms": s["n_atoms"],
                "d_min": s["d_min"],
                "space_group": s["space_group"],
                "mpe_oi": float(mpe),
                "mapcc_prior": float(cc),
            }
        )
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
                "Train with: python scripts/train_strong_prior.py --scale"
            )
        model = load_strong_prior(path)
        if verbose:
            print(f"  loaded strong prior from {path}")

    ph_ai = predict_full_phases(
        model,
        hkl,
        amplitudes,
        cell,
        max_reflections=max_reflections,
        origin_fom_search=True,
    )
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
    info["seed_source"] = "strong_graph_prior"
    rho_ai = density_from_structure_factors(
        hkl, amplitudes * np.exp(1j * ph_ai), cell, d_min=d_min
    )
    info["fom_prior_only"] = free_fom(hkl, amplitudes, ph_ai, cell, density=rho_ai)
    info["phases_prior"] = ph_ai
    return ph, rho, info
