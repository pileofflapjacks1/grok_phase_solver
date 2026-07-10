"""
Physics-informed loss functions for Phase-2 neural phase retrieval.

Each term maps to a Cowtan/physics constraint:
- phase circular loss ↔ correct φ
- positivity ↔ ρ ≥ 0
- Fourier modulus ↔ |F| consistency
- triplet auxiliary ↔ direct-methods prior
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def circular_phase_loss(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """1 − cos(Δφ), weighted mean (in [0, 2])."""
    d = np.asarray(phase_pred) - np.asarray(phase_true)
    val = 1.0 - np.cos(d)
    if weights is None:
        return float(np.mean(val))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * val) / (np.sum(w) + 1e-16))


def positivity_loss(rho: np.ndarray) -> float:
    """Mean squared negative density."""
    neg = np.minimum(np.asarray(rho, dtype=np.float64), 0.0)
    return float(np.mean(neg**2))


def fourier_modulus_loss(
    F_pred: np.ndarray,
    amplitudes: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """|| |F_pred| − |F_obs| ||² (optionally weighted)."""
    err = np.abs(np.asarray(F_pred)) - np.asarray(amplitudes, dtype=np.float64)
    if weights is None:
        return float(np.mean(err**2))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * err**2) / (np.sum(w) + 1e-16))


def triplet_fom_loss(
    phases: np.ndarray,
    triplet_indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Encourage cos(φ_h + φ_k − φ_{h+k}) → +1 for strong triplets.

    triplet_indices: (T, 3) int indices into phases array.
    Returns 1 − mean cos (lower is better).
    """
    idx = np.asarray(triplet_indices, dtype=int)
    if idx.size == 0:
        return 0.0
    phi = (
        phases[idx[:, 0]] + phases[idx[:, 1]] - phases[idx[:, 2]]
    )
    c = np.cos(phi)
    if weights is None:
        return float(1.0 - np.mean(c))
    w = np.asarray(weights, dtype=np.float64)
    return float(1.0 - np.sum(w * c) / (np.sum(w) + 1e-16))


def combined_phase_loss(
    phase_pred: np.ndarray,
    phase_true: np.ndarray,
    rho: Optional[np.ndarray] = None,
    F_pred: Optional[np.ndarray] = None,
    amplitudes: Optional[np.ndarray] = None,
    amp_weights: Optional[np.ndarray] = None,
    lambda_pos: float = 0.1,
    lambda_mod: float = 0.1,
) -> dict:
    """Bundle losses for logging / training loops (NumPy reference)."""
    out = {
        "phase": circular_phase_loss(phase_pred, phase_true, weights=amp_weights),
    }
    total = out["phase"]
    if rho is not None and lambda_pos > 0:
        out["positivity"] = positivity_loss(rho)
        total = total + lambda_pos * out["positivity"]
    if F_pred is not None and amplitudes is not None and lambda_mod > 0:
        out["modulus"] = fourier_modulus_loss(F_pred, amplitudes, weights=amp_weights)
        total = total + lambda_mod * out["modulus"]
    out["total"] = float(total)
    return out
