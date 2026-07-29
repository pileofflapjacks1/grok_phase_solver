"""
Research-only SE(3)-aware / equivariant-inspired diffusion helpers (v0.8).

**Not production.** Documents a path toward equivariant atomic / density
denoisers conditioned on |F(hkl)| and symmetry (PXRDnet / XRDSol / XDXD
conceptual lineage).

v0.8 additions
--------------
- Reciprocal-space **rotation-invariant** score features (E, s, |h| shells)
  that any future equivariant net can consume without breaking physics paths.
- Optional soft SE(3)-inspired residual step that nudges phases toward a
  low-resolution Fcalc envelope when a trial atom list is supplied — still
  research-only and off the default auto path.

Honest non-claims
-----------------
- This is **not** full SE(3) equivariance on atomic coordinates.
- No trained XDXD/PXRDnet weights; no default pipeline routing.
- Prefer ``diffusion_phase`` Langevin + optional ``PhaseScoreNet`` for experiments.
- Physics fallback: classical ensemble / AI-PhaSeed / partial_phaseed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.direct_methods import normalize_E


def se3_diffusion_available() -> bool:
    """True only when a real equivariant checkpoint ships (currently False)."""
    return False


def reciprocal_invariant_features(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: np.ndarray,
    *,
    t: float = 0.5,
) -> np.ndarray:
    """
    Rotation-invariant per-reflection features for future SE(3)/E(3) nets.

    Shape (N, 8): E, s_n, amp_n, |h|_n, cos, sin, t, shell_rank.
    Invariant under global reciprocal-frame rotation of indexing conventions
    only in the scalar channels; phase channels transform as U(1).
    """
    hkl = np.asarray(hkl, dtype=float)
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64)
    E = normalize_E(np.asarray(hkl, dtype=int), amp, cell)
    d = d_spacing(np.asarray(hkl, dtype=int), cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-6))
    s_n = s / (s.max() + 1e-16)
    amp_n = amp / (amp.std() + 1e-16)
    hn = np.linalg.norm(hkl, axis=1)
    hn = hn / (hn.max() + 1e-16)
    order = np.argsort(s)
    shell_rank = np.zeros(len(amp), dtype=np.float64)
    n_shells = max(4, min(10, len(amp) // 10 + 1))
    edges = np.linspace(0, len(amp), n_shells + 1, dtype=int)
    for si in range(n_shells):
        sl = order[edges[si] : edges[si + 1]]
        if len(sl) == 0:
            continue
        r = np.argsort(np.argsort(E[sl])).astype(np.float64)
        shell_rank[sl] = r / max(len(sl) - 1, 1)
    t_col = np.full(len(amp), float(t), dtype=np.float64)
    return np.column_stack(
        [E, s_n, amp_n, hn, np.cos(ph), np.sin(ph), t_col, shell_rank]
    ).astype(np.float64)


def se3_score_step_stub(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: np.ndarray,
    *,
    t: float = 0.5,
    trial_fracs: Optional[np.ndarray] = None,
    trial_elements: Optional[Sequence[str]] = None,
    blend: float = 0.15,
) -> Tuple[np.ndarray, Dict]:
    """
    Research residual step.

    Without trial atoms: return phases unchanged + feature diagnostics.
    With trial atoms: softly blend toward Fcalc phases (MR-lite style), still
    marked research_only.
    """
    ph = np.asarray(phases, dtype=np.float64).copy()
    meta: Dict = {
        "algorithm": "diffusion_se3_stub",
        "status": "research_only",
        "trained": False,
        "available": se3_diffusion_available(),
        "t": float(t),
        "n_features": 8,
        "note": (
            "SE(3) equivariant atomic diffusion not fully implemented. "
            "Use diffusion_hybrid / PhaseScoreNet or classical methods."
        ),
    }
    feats = reciprocal_invariant_features(hkl, amplitudes, cell, ph, t=t)
    meta["feature_mean"] = feats.mean(axis=0).tolist()

    if trial_fracs is not None and len(trial_fracs) > 0:
        try:
            from grok_phase_solver.physics.structure_factors import (
                compute_structure_factors,
            )

            fr = np.asarray(trial_fracs, dtype=np.float64).reshape(-1, 3)
            els = list(trial_elements) if trial_elements is not None else ["C"] * len(fr)
            if len(els) < len(fr):
                els = (els + ["C"] * len(fr))[: len(fr)]
            F = compute_structure_factors(hkl, fr, els, cell)
            ph_fc = np.angle(F)
            w = float(np.clip(blend, 0.0, 0.5)) * (1.0 - 0.5 * float(t))
            ph = np.angle((1.0 - w) * np.exp(1j * ph) + w * np.exp(1j * ph_fc))
            meta["trial_blend"] = w
            meta["n_trial_atoms"] = len(fr)
            meta["note"] = (
                "Research soft blend toward trial Fcalc; not full SE(3) diffusion."
            )
        except Exception as e:
            meta["trial_error"] = str(e)

    return ph, meta
