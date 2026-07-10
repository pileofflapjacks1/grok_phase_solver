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
    i_hpk: int  # index of h+k (stored as reflection for −(−h−k) via Friedel)
    weight: float  # ~ |E_h E_k E_{h+k}|


def build_triplets(
    hkl: np.ndarray,
    E: np.ndarray,
    e_min: float = 1.2,
    max_reflections: int = 200,
    max_triplets: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, List[Triplet]]:
    """
    Enumerate strong-reflection triplets among the top |E| reflections.

    Returns
    -------
    strong_idx : indices into original arrays
    hkl_s, E_s mapped through strong set
    triplets : list of Triplet with indices into strong set
    """
    hkl = np.asarray(hkl, dtype=int)
    E = np.asarray(E, dtype=np.float64)
    order = np.argsort(-E)
    strong_idx = order[:max_reflections]
    strong_idx = strong_idx[E[strong_idx] >= e_min]
    if len(strong_idx) < 3:
        # relax
        strong_idx = order[: min(max_reflections, len(order))]

    hkl_s = hkl[strong_idx]
    E_s = E[strong_idx]
    # Map Miller triple → index in strong set (and Friedel mates)
    lookup = {}
    for i, (h, k, l) in enumerate(hkl_s):
        lookup[(int(h), int(k), int(l))] = i

    triplets: List[Triplet] = []
    n = len(hkl_s)
    for i in range(n):
        hi = hkl_s[i]
        for j in range(i, n):  # include j=i for 2h-type
            hj = hkl_s[j]
            hs = hi + hj
            key = (int(hs[0]), int(hs[1]), int(hs[2]))
            # try h+k or Friedel of −(h+k) → we need reflection h+k in list
            m = lookup.get(key)
            if m is None:
                # try −(h+k) and note phase conjugation later in tangent
                key2 = (-key[0], -key[1], -key[2])
                m = lookup.get(key2)
                if m is None:
                    continue
                # store with negative marker via weight sign? handle in tangent
            w = float(E_s[i] * E_s[j] * E_s[m])
            if w < e_min**3 * 0.5:
                continue
            triplets.append(Triplet(i_h=i, i_k=j, i_hpk=m, weight=w))
            if len(triplets) >= max_triplets:
                return strong_idx, E_s, triplets
    # Sort by reliability
    triplets.sort(key=lambda t: -t.weight)
    return strong_idx, E_s, triplets


def cochran_alpha(E_h: float, E_k: float, E_hpk: float, n_atoms: float) -> float:
    """
    Cochran reliability parameter κ ≈ 2 σ₃ σ₂^{−3/2} |E_h E_k E_{h+k}|
    with σ_n ≈ N * ⟨Z^n⟩; for equal atoms σ₃/σ₂^{3/2} ≈ N^{−1/2}.

    Returns approximate concentration parameter for von Mises(Φ; 0, κ).
    """
    if n_atoms <= 0:
        n_atoms = 1.0
    return 2.0 * (n_atoms ** -0.5) * abs(E_h * E_k * E_hpk)


def tangent_formula_iteration(
    phases: np.ndarray,
    E: np.ndarray,
    triplets: Sequence[Triplet],
    n_iter: int = 20,
) -> np.ndarray:
    """
    Karle–Hauptman tangent formula (simplified).

    For each reflection h:
        tan φ_h  ∝  Σ_k |E_k E_{h−k}| sin(φ_k + φ_{h−k})
        (and cosine analog)

    Here triplets are stored as (h, k, h+k) so:
        φ_{h+k} ≈ −(φ_h + φ_k)  →  φ_h ≈ −φ_k − φ_{h+k}  for new estimates.
    We accumulate complex indicators exp(iφ) weighted by |E| products.
    """
    phases = np.asarray(phases, dtype=np.float64).copy()
    n = len(phases)
    for _ in range(n_iter):
        acc = np.zeros(n, dtype=np.complex128)
        for t in triplets:
            # Invariant: φ_h + φ_k − φ_{h+k} ≈ 0  if hpk indexes h+k
            # → estimate φ_h ≈ φ_{h+k} − φ_k
            # estimate φ_k ≈ φ_{h+k} − φ_h
            # estimate φ_{h+k} ≈ φ_h + φ_k
            w = t.weight
            ph, pk, pm = phases[t.i_h], phases[t.i_k], phases[t.i_hpk]
            acc[t.i_h] += w * np.exp(1j * (pm - pk))
            acc[t.i_k] += w * np.exp(1j * (pm - ph))
            acc[t.i_hpk] += w * np.exp(1j * (ph + pk))
        # Update where we have signal
        mag = np.abs(acc)
        mask = mag > 1e-12
        phases[mask] = np.angle(acc[mask])
    return phases


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
) -> float:
    """Mean cos(φ_h + φ_k − φ_{h+k}) weighted — closer to 1 is better."""
    if not triplets:
        return 0.0
    num = 0.0
    den = 0.0
    for t in triplets:
        phi = phases[t.i_h] + phases[t.i_k] - phases[t.i_hpk]
        num += t.weight * np.cos(phi)
        den += t.weight
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
) -> DirectMethodsResult:
    """
    Multi-start direct methods (Cowtan's outline):

    1. Compute |E|
    2. Select strong reflections and triplets
    3. Random phases for a few strong reflections → expand via tangent formula
    4. Keep trial with best triplet FOM
    5. Map phases back to full reflection list
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    E = normalize_E(hkl, amp, cell, n_atoms_approx=n_atoms_approx)
    strong_idx, E_s, triplets = build_triplets(
        hkl, E, e_min=e_min, max_reflections=max_reflections
    )
    n_s = len(strong_idx)
    if verbose:
        print(f"  DM: {n_s} strong refl, {len(triplets)} triplets")

    best_fom = -np.inf
    best_phases = np.zeros(n_s)
    best_trial = 0
    foms: List[float] = []

    # Origin-fixing reflections: leave a few free; randomize all for multi-solution
    for trial in range(n_trials):
        phases = rng.uniform(-np.pi, np.pi, size=n_s)
        # Optionally fix one phase to 0 for origin (P1 partial)
        if n_s > 0:
            phases[0] = 0.0
        phases = tangent_formula_iteration(phases, E_s, triplets, n_iter=tangent_iter)
        fom = figure_of_merit_triplets(phases, triplets)
        foms.append(fom)
        if fom > best_fom:
            best_fom = fom
            best_phases = phases
            best_trial = trial
        if verbose and (trial % 10 == 0 or trial == n_trials - 1):
            print(f"  DM trial {trial:3d}  FOM={fom:.4f}  best={best_fom:.4f}")

    # Expand to full list: strong get solved phases; weak get random or 0
    phases_full = rng.uniform(-np.pi, np.pi, size=len(amp))
    phases_full[strong_idx] = best_phases
    # Weak: set phase of Friedel pairs consistently is hard without pairing —
    # leave random for weak; density will be dominated by strong E

    return DirectMethodsResult(
        phases_full=phases_full,
        phases_strong=best_phases,
        strong_idx=strong_idx,
        E=E,
        triplets=triplets,
        n_trials=n_trials,
        best_trial=best_trial,
        figures_of_merit=foms,
        history={"best_fom": best_fom, "n_triplets": len(triplets), "n_strong": n_s},
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
