"""
Multi-derivative MIR phase combination (Blow–Crick style).

Classic lack-of-closure for a single derivative (centric simplified + general):

    |F_PH|  ≈  | F_P + F_H |

With known F_H (complex, from heavy-atom model), the residual

    ε(φ) = |F_PH| − | |F_P| e^{iφ} + F_H |

is minimized for the correct phase φ of F_P (two-fold ambiguity in acentric
SIR; resolved by multiple derivatives).

Blow & Crick (1959) model phase probability via Gaussian lack-of-closure:

    P(φ) ∝ exp( −ε(φ)² / (2 σ²) )

Combined FOMs and centroid phases from product of independent derivative
likelihoods (assuming independent errors — standard first-order treatment).

This implementation is numerically correct for the pedagogical model; it is
not a full MLPHARE/SHARP replacement (no correlated error models, no density
modification inside).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DerivativeData:
    """One isomorphous derivative."""

    F_ph: np.ndarray  # measured |F_PH|
    F_h: np.ndarray  # complex calculated heavy-atom SF
    sigma: float = 1.0  # lack-of-closure σ (same scale as F)


def lack_of_closure(
    F_p_mag: np.ndarray,
    phase: np.ndarray,
    F_ph: np.ndarray,
    F_h: np.ndarray,
) -> np.ndarray:
    """ε = |F_PH| − | |F_P| e^{iφ} + F_H |."""
    Fp = np.asarray(F_p_mag, dtype=np.float64) * np.exp(1j * np.asarray(phase))
    return np.abs(F_ph) - np.abs(Fp + F_h)


def phase_probability_1d(
    F_p_mag: float,
    F_ph: float,
    F_h: complex,
    sigma: float,
    n_steps: int = 72,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    P(φ) on a uniform grid for a single reflection / single derivative.

    Returns (phases, unnormalized densities).
    """
    phis = np.linspace(-np.pi, np.pi, n_steps, endpoint=False)
    Fp = F_p_mag * np.exp(1j * phis)
    eps = F_ph - np.abs(Fp + F_h)
    # Gaussian Blow–Crick
    logp = -0.5 * (eps / (sigma + 1e-16)) ** 2
    logp -= logp.max()  # stabilize
    p = np.exp(logp)
    return phis, p


def combine_mir_phases(
    F_native: np.ndarray,
    derivatives: Sequence[DerivativeData],
    n_steps: int = 72,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine multiple derivatives into centroid phases and FOMs.

    For each reflection:
      P(φ) ∝ Π_d exp(−ε_d(φ)² / 2σ_d²)
      φ_centroid = arg( ∫ e^{iφ} P(φ) dφ )
      FOM = |∫ e^{iφ} P(φ) dφ|   ∈ [0, 1]

    Returns
    -------
    phase_centroid : (N,) radians
    fom : (N,) in [0, 1]
    best_phase_map : (N,) phase at max P (MAP estimate)
    """
    Fp = np.asarray(F_native, dtype=np.float64)
    n = len(Fp)
    phase_c = np.zeros(n)
    fom = np.zeros(n)
    phase_map = np.zeros(n)
    phis = np.linspace(-np.pi, np.pi, n_steps, endpoint=False)
    dphi = 2 * np.pi / n_steps

    for i in range(n):
        logp = np.zeros(n_steps)
        for der in derivatives:
            Fp_c = Fp[i] * np.exp(1j * phis)
            eps = der.F_ph[i] - np.abs(Fp_c + der.F_h[i])
            logp += -0.5 * (eps / (der.sigma + 1e-16)) ** 2
        logp -= logp.max()
        p = np.exp(logp)
        p /= p.sum() * dphi + 1e-16  # normalize roughly
        # centroid
        z = np.sum(p * np.exp(1j * phis)) * dphi
        # re-normalize integral of p
        mass = np.sum(p) * dphi
        z = z / (mass + 1e-16)
        phase_c[i] = np.angle(z)
        fom[i] = float(np.clip(np.abs(z), 0, 1))
        phase_map[i] = phis[int(np.argmax(p))]

    return phase_c, fom, phase_map


def single_isomorphous_replacement(
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    F_heavy: np.ndarray,
    sigma: float = 1.0,
    n_steps: int = 72,
) -> Tuple[np.ndarray, np.ndarray]:
    """SIR centroid phase + FOM (two-fold ambiguity → lower FOM)."""
    der = DerivativeData(F_ph=F_derivative, F_h=F_heavy, sigma=sigma)
    phase, fom, _ = combine_mir_phases(F_native, [der], n_steps=n_steps)
    return phase, fom
