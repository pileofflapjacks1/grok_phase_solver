"""
Classical direct methods: normalized E-values, triplet invariants, tangent formula.

Pedagogy (Cowtan 2001; Cochran 1952):
  When three strong reflections h, k, −h−k have large |E|, positivity + atomicity
  imply the three-phase structure invariant

      Φ_hk = φ(h) + φ(k) + φ(−h−k)  ≈  0   (mod 2π)

  with reliability increasing with |E_h E_k E_{h+k}| (Cochran distribution).

This module implements a transparent multi-start tangent-formula phaser for
benchmarking against charge flipping / NN methods — not a full SHELXD clone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.reciprocal import d_spacing


def normalize_E(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_atoms_approx: Optional[int] = None,
    n_shells: int = 10,
) -> np.ndarray:
    """
    Approximate normalized structure factors |E|.

    Wilson-plot style: within resolution shells, ⟨|F|²⟩ ≈ Σ f² · ⟨E²⟩,
    so |E| ≈ |F| / sqrt(⟨|F|²⟩_shell).

    For equal-atom models at high resolution this recovers the usual E-scale
    up to a global constant (harmless for triplet ranking).
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    d = d_spacing(hkl, cell)
    # shells by equal count in 1/d
    order = np.argsort(1.0 / np.maximum(d, 1e-6))
    E = np.zeros_like(amp)
    n = len(amp)
    edges = np.linspace(0, n, n_shells + 1, dtype=int)
    for s in range(n_shells):
        idx = order[edges[s] : edges[s + 1]]
        if len(idx) == 0:
            continue
        mean_I = np.mean(amp[idx] ** 2) + 1e-16
        E[idx] = amp[idx] / np.sqrt(mean_I)
    # Optional equal-atom rescaling: ⟨E²⟩ → 1
    scale = np.sqrt(np.mean(E**2) + 1e-16)
    E = E / scale
    return E


@dataclass
class Triplet:
    """Three-phase structure invariant indices into a reflection list."""

    i_h: int
    i_k: int
    i_hpk: int  # index of h+k
    weight: float  # |E_h E_k E_{h+k}|
    kappa: float = 0.0  # Cochran reliability κ


def build_triplets(
    hkl: np.ndarray,
    E: np.ndarray,
    e_min: float = 1.2,
    max_reflections: int = 200,
    max_triplets: int = 5000,
    n_atoms_approx: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, List[Triplet]]:
    """
    Enumerate strong-reflection triplets among the top |E| reflections.

    Weight = |E_h E_k E_{h+k}|; kappa = 2 N^{-1/2} |EEE| (equal-atom Cochran).
    """
    hkl = np.asarray(hkl, dtype=int)
    E = np.asarray(E, dtype=np.float64)
    order = np.argsort(-E)
    strong_idx = order[:max_reflections]
    strong_idx = strong_idx[E[strong_idx] >= e_min]
    if len(strong_idx) < 3:
        strong_idx = order[: min(max_reflections, len(order))]

    hkl_s = hkl[strong_idx]
    E_s = E[strong_idx]
    lookup = {}
    for i, (h, k, l) in enumerate(hkl_s):
        lookup[(int(h), int(k), int(l))] = i

    triplets: List[Triplet] = []
    n = len(hkl_s)
    n_at = max(float(n_atoms_approx), 1.0)
    for i in range(n):
        hi = hkl_s[i]
        for j in range(i, n):
            hj = hkl_s[j]
            hs = hi + hj
            key = (int(hs[0]), int(hs[1]), int(hs[2]))
            m = lookup.get(key)
            if m is None:
                key2 = (-key[0], -key[1], -key[2])
                m = lookup.get(key2)
                if m is None:
                    continue
            eee = float(abs(E_s[i] * E_s[j] * E_s[m]))
            if eee < e_min**3 * 0.5:
                continue
            kappa = cochran_alpha(E_s[i], E_s[j], E_s[m], n_at)
            # Use κ as tangent weight (probabilistic reliability)
            w = max(kappa, eee * (n_at**-0.5))
            triplets.append(
                Triplet(i_h=i, i_k=j, i_hpk=m, weight=w, kappa=kappa)
            )
            if len(triplets) >= max_triplets:
                triplets.sort(key=lambda t: -t.kappa)
                return strong_idx, E_s, triplets
    triplets.sort(key=lambda t: -t.kappa)
    return strong_idx, E_s, triplets


def cochran_alpha(E_h: float, E_k: float, E_hpk: float, n_atoms: float) -> float:
    """
    Cochran reliability parameter κ ≈ 2 N^{-1/2} |E_h E_k E_{h+k}|
    for equal atoms (σ₃/σ₂^{3/2} ≈ N^{-1/2}).

    P(Φ) ∝ exp(κ cos Φ)  ⇒  E[cos Φ] = I₁(κ)/I₀(κ).
    """
    if n_atoms <= 0:
        n_atoms = 1.0
    return 2.0 * (n_atoms**-0.5) * abs(float(E_h) * float(E_k) * float(E_hpk))


def sayre_weight_expected_cos(kappa: float) -> float:
    """
    E[cos Φ] = I₁(κ)/I₀(κ) for von Mises. Small-κ: ≈ κ/2; large-κ: → 1.
    """
    k = abs(float(kappa))
    if k < 1e-8:
        return 0.0
    if k < 0.5:
        # series: I1/I0 ≈ k/2 * (1 - k²/8 + ...)
        return 0.5 * k * (1.0 - 0.125 * k * k)
    # asymptotic + mid: use stable ratio via modified Bessel from scipy if available
    try:
        from scipy.special import i0, i1

        return float(i1(k) / (i0(k) + 1e-16))
    except Exception:
        return float(min(1.0, 0.5 * k / (1.0 + 0.25 * k)))


def tangent_formula_iteration(
    phases: np.ndarray,
    E: np.ndarray,
    triplets: Sequence[Triplet],
    n_iter: int = 20,
    use_kappa: bool = True,
    ai_phases: Optional[np.ndarray] = None,
    ai_weight: float = 0.0,
    ai_reliability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    κ-weighted Karle–Hauptman tangent formula.

    Accumulator for reflection h:
      Σ_t w_t exp(i(φ_k + φ_{h-k})) with w_t = κ_t (or |EEE|).

    Optional AI prior (Carrozzini et al. 2025 modified tangent)::

        A_h ← Σ_triplets … + λ · r_h · |E_h| · exp(i φ_h^{AI})

    where ``ai_weight`` is λ ∈ [0, 1+] scaling a priori AI phases and
    ``ai_reliability`` r_h ∈ [0, 1] is a per-reflection weight (default 1 on
    all strong indices). When ``ai_weight=0`` (default), behaviour is unchanged.
    """
    phases = np.asarray(phases, dtype=np.float64).copy()
    n = len(phases)
    ai_w = float(max(ai_weight, 0.0))
    ai_ph = None if ai_phases is None else np.asarray(ai_phases, dtype=np.float64)
    if ai_ph is not None and len(ai_ph) != n:
        raise ValueError("ai_phases length must match strong-phase vector")
    if ai_reliability is not None:
        r_ai = np.asarray(ai_reliability, dtype=np.float64)
        if len(r_ai) != n:
            raise ValueError("ai_reliability length mismatch")
    else:
        r_ai = np.ones(n, dtype=np.float64)

    E = np.asarray(E, dtype=np.float64)
    for _ in range(n_iter):
        acc = np.zeros(n, dtype=np.complex128)
        for t in triplets:
            w = t.kappa if (use_kappa and t.kappa > 0) else t.weight
            # reliability reweight by expected cos (soft)
            w = w * (0.5 + 0.5 * sayre_weight_expected_cos(t.kappa if t.kappa > 0 else w))
            ph, pk, pm = phases[t.i_h], phases[t.i_k], phases[t.i_hpk]
            acc[t.i_h] += w * np.exp(1j * (pm - pk))
            acc[t.i_k] += w * np.exp(1j * (pm - ph))
            acc[t.i_hpk] += w * np.exp(1j * (ph + pk))
        # Modified tangent: inject AI phases as a priori complex weights
        if ai_w > 0.0 and ai_ph is not None:
            # Scale relative to mean triplet weight so λ~0.5 is meaningful
            scale = float(np.mean(np.abs(acc)) + 1e-12)
            acc += ai_w * scale * r_ai * np.maximum(E, 0.0) * np.exp(1j * ai_ph)
        mag = np.abs(acc)
        mask = mag > 1e-12
        phases[mask] = np.angle(acc[mask])
    return phases


def dm_ai_hybrid_refine(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases_full: np.ndarray,
    ai_phases: np.ndarray,
    *,
    ai_weight: float = 0.5,
    seed_idx: Optional[np.ndarray] = None,
    n_atoms_approx: float = 30.0,
    e_min: float = 1.1,
    max_reflections: int = 150,
    tangent_iter: int = 8,
    ai_reliability: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    One-shot DM+AI hybrid: κ-tangent on strong set with AI prior (modified TF).

    Aligns with Carrozzini et al. (2025) eqs. 3–5 style: AI phases enter the
    tangent accumulator with reliability weights rather than only as hard seeds
    after density modification.

    Parameters
    ----------
    phases_full : current full phase list (length M)
    ai_phases : AI prior phases (length M)
    ai_weight : λ for AI a priori term (0 = classical tangent only)
    seed_idx : if given, boost reliability on seed reflections

    Returns
    -------
    phases_full_updated, info
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases_full, dtype=np.float64).copy()
    ai = np.asarray(ai_phases, dtype=np.float64)
    E = normalize_E(hkl, amp, cell, n_atoms_approx=int(n_atoms_approx))
    strong_idx, E_s, triplets = build_triplets(
        hkl,
        E,
        e_min=e_min,
        max_reflections=max_reflections,
        n_atoms_approx=float(n_atoms_approx),
    )
    if len(strong_idx) < 3 or not triplets:
        return ph, {
            "algorithm": "dm_ai_hybrid",
            "applied": False,
            "reason": "insufficient strong reflections/triplets",
            "n_strong": int(len(strong_idx)),
            "n_triplets": len(triplets),
        }

    # Map full → strong
    ph_s = ph[strong_idx].copy()
    ai_s = ai[strong_idx].copy()
    r_s = np.ones(len(strong_idx), dtype=np.float64)
    if ai_reliability is not None:
        r_full = np.asarray(ai_reliability, dtype=np.float64)
        r_s = r_full[strong_idx]
    elif seed_idx is not None:
        # Higher reliability on explicit seed set
        seed_set = set(int(i) for i in np.asarray(seed_idx, dtype=int))
        for j, gi in enumerate(strong_idx):
            r_s[j] = 1.0 if int(gi) in seed_set else 0.55

    ph_s = tangent_formula_iteration(
        ph_s,
        E_s,
        triplets,
        n_iter=tangent_iter,
        use_kappa=True,
        ai_phases=ai_s,
        ai_weight=float(ai_weight),
        ai_reliability=r_s,
    )
    ph[strong_idx] = ph_s
    fom = figure_of_merit_triplets(ph_s, triplets, use_kappa=True)
    return ph, {
        "algorithm": "dm_ai_hybrid",
        "applied": True,
        "ai_weight": float(ai_weight),
        "n_strong": int(len(strong_idx)),
        "n_triplets": len(triplets),
        "triplet_fom": float(fom),
        "mean_kappa": float(np.mean([t.kappa for t in triplets])) if triplets else 0.0,
    }


@dataclass
class DirectMethodsResult:
    phases_full: np.ndarray  # phases for all input reflections (weak random/zero)
    phases_strong: np.ndarray
    strong_idx: np.ndarray
    E: np.ndarray
    triplets: List[Triplet]
    n_trials: int
    best_trial: int
    figures_of_merit: List[float] = field(default_factory=list)
    history: Dict = field(default_factory=dict)


def figure_of_merit_triplets(
    phases: np.ndarray,
    triplets: Sequence[Triplet],
    use_kappa: bool = True,
) -> float:
    """
    Weighted mean cos(φ_h + φ_k − φ_{h+k}).

    Weights prefer high-κ triplets (more reliable).
    """
    if not triplets:
        return 0.0
    num = 0.0
    den = 0.0
    for t in triplets:
        w = t.kappa if (use_kappa and t.kappa > 0) else t.weight
        phi = phases[t.i_h] + phases[t.i_k] - phases[t.i_hpk]
        num += w * np.cos(phi)
        den += w
    return float(num / (den + 1e-16))


def direct_methods_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_atoms_approx: int = 20,
    n_trials: int = 30,
    e_min: float = 1.1,
    max_reflections: int = 150,
    tangent_iter: int = 25,
    seed: int = 0,
    verbose: bool = False,
    select_by: str = "triplet_fom",
) -> DirectMethodsResult:
    """
    Multi-start κ-weighted direct methods.

    1. Compute |E| (Wilson shells)
    2. Strong reflections + triplets with Cochran κ
    3. Multistart random phases → κ-tangent expansion
    4. Rank trials by triplet FOM (or free density FOM if select_by='free_fom')
    5. Expand strong phases to full list (weak remain random)

    select_by:
      - ``triplet_fom``: classical Σ κ cos Φ
      - ``free_fom``: truth-free density composite (needs more CPU)
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    E = normalize_E(hkl, amp, cell, n_atoms_approx=n_atoms_approx)
    strong_idx, E_s, triplets = build_triplets(
        hkl,
        E,
        e_min=e_min,
        max_reflections=max_reflections,
        n_atoms_approx=float(n_atoms_approx),
    )
    n_s = len(strong_idx)
    if verbose:
        mean_k = np.mean([t.kappa for t in triplets]) if triplets else 0.0
        print(f"  DM: {n_s} strong, {len(triplets)} triplets, ⟨κ⟩={mean_k:.3f}")

    best_score = -np.inf
    best_phases = np.zeros(n_s)
    best_trial = 0
    foms: List[float] = []

    for trial in range(n_trials):
        phases = rng.uniform(-np.pi, np.pi, size=n_s)
        if n_s > 0:
            phases[0] = 0.0  # partial origin fix
        if n_s > 3:
            # second reflection random sign for enantiomorph exploration
            phases[1] = 0.0 if rng.random() < 0.5 else np.pi
        phases = tangent_formula_iteration(
            phases, E_s, triplets, n_iter=tangent_iter, use_kappa=True
        )
        trip_fom = figure_of_merit_triplets(phases, triplets, use_kappa=True)
        foms.append(trip_fom)
        score = trip_fom
        if select_by == "free_fom":
            phases_full_try = rng.uniform(-np.pi, np.pi, size=len(amp))
            phases_full_try[strong_idx] = phases
            try:
                from grok_phase_solver.solvers.free_fom import free_fom

                score = free_fom(hkl, amp, phases_full_try, cell)["composite"]
            except Exception:
                score = trip_fom
        if score > best_score:
            best_score = score
            best_phases = phases
            best_trial = trial
        if verbose and (trial % 10 == 0 or trial == n_trials - 1):
            print(
                f"  DM trial {trial:3d}  tripFOM={trip_fom:.4f}  "
                f"score={score:.4f}  best={best_score:.4f}"
            )

    phases_full = rng.uniform(-np.pi, np.pi, size=len(amp))
    phases_full[strong_idx] = best_phases

    return DirectMethodsResult(
        phases_full=phases_full,
        phases_strong=best_phases,
        strong_idx=strong_idx,
        E=E,
        triplets=triplets,
        n_trials=n_trials,
        best_trial=best_trial,
        figures_of_merit=foms,
        history={
            "best_fom": float(best_score),
            "best_triplet_fom": float(foms[best_trial]) if foms else 0.0,
            "n_triplets": len(triplets),
            "n_strong": n_s,
            "mean_kappa": float(np.mean([t.kappa for t in triplets])) if triplets else 0.0,
            "select_by": select_by,
        },
    )


def triplet_invariant_true(
    phases: np.ndarray,
    hkl: np.ndarray,
    triplets: Sequence[Triplet],
    strong_idx: np.ndarray,
) -> np.ndarray:
    """True Φ = φ_h + φ_k − φ_{h+k} for each triplet (using full phase list)."""
    out = []
    for t in triplets:
        ih = strong_idx[t.i_h]
        ik = strong_idx[t.i_k]
        im = strong_idx[t.i_hpk]
        out.append(phases[ih] + phases[ik] - phases[im])
    return np.asarray(out)
