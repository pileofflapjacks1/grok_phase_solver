"""
Iterative projection algorithms for crystallographic phase retrieval.

Implemented update rules (real-space iterate x ≈ ρ):

**Error reduction (ER)** — optional baseline:
  x ← P_+ P_M x

**HIO** (Fienup): already in hio.py; here for unified API.

**RAAR** (Luke 2005) — Relaxed Averaged Alternating Reflections:
  x ← ½ β (R_+ R_M + I) x + (1 − β) P_M x
  where R_S = 2 P_S − I (reflector).

**Difference Map** (Elser 2003), β-parameterized form used in practice:
  x ← x + β (P_+ ((1+γ_M) P_M − γ_M I) − P_M ((1+γ_S) P_+ − γ_S I)) x
  with γ_S = 1/β, γ_M = −1/β commonly (simplifies implementation).

We use density as the iterate and enforce modulus via FFT each step.
Charge-flip can replace positivity as the real-space projector.

References
----------
Elser, V. (2003). J. Opt. Soc. Am. A 20, 40–55.
Luke, D. R. (2005). Inverse Problems 21, 37–50.
Marchesini, S. (2007). Rev. Sci. Instrum. 78, 011301 (review).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.projectors import (
    density_from_phases,
    density_to_F,
    project_charge_flip,
    project_modulus,
    project_positivity,
    r_factor_moduli,
    setup_grid,
)


def _real_space_projector(
    rho: np.ndarray,
    kind: str = "positivity",
    delta_sigma: float = 0.0,
) -> np.ndarray:
    if kind == "positivity":
        return project_positivity(rho)
    if kind == "charge_flip":
        sigma = float(rho.std()) + 1e-16
        delta = 0.1 * delta_sigma * sigma
        return project_charge_flip(rho, delta=delta)
    if kind == "none":
        return rho
    raise ValueError(kind)


def _P_M_density(
    rho: np.ndarray,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modulus projection in density space: ρ → F → |F_obs| phase keep → ρ'.

    Returns (rho_M, phases, F_after_mod).
    """
    F = density_to_F(rho, hkl, cell)
    F_m = project_modulus(F, amplitudes)
    phases = np.angle(F_m)
    # rebuild density from constrained F
    shape = rho.shape
    rho_m = density_from_phases(hkl, amplitudes, phases, cell, shape)
    return rho_m, phases, F_m


def raar_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 200,
    beta: float = 0.9,
    real_proj: str = "positivity",
    seed: int = 0,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    RAAR phase retrieval.

    Luke (2005): x_{n+1} = ½ β (R_S R_M + I) x_n + (1−β) P_M x_n
    with S = positivity (or charge flip).
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    shape, d_min = setup_grid(hkl, cell, d_min=d_min, sampling=sampling)

    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()

    x = density_from_phases(hkl, amp, phases, cell, shape)
    history = {"R": [], "neg_frac": [], "beta": beta, "algorithm": "RAAR"}

    for it in range(n_iter):
        # P_M x
        Pm_x, phases, F_m = _P_M_density(x, hkl, amp, cell)
        # R_M x = 2 P_M x − x
        Rm_x = 2.0 * Pm_x - x
        # P_S R_M x
        Ps_Rm = _real_space_projector(Rm_x, kind=real_proj)
        # R_S R_M x = 2 P_S R_M x − R_M x
        Rs_Rm = 2.0 * Ps_Rm - Rm_x
        # RAAR update
        x = 0.5 * beta * (Rs_Rm + x) + (1.0 - beta) * Pm_x

        R = r_factor_moduli(F_m, amp)
        history["R"].append(R)
        history["neg_frac"].append(float((x < 0).mean()))
        if verbose and (it % 40 == 0 or it == n_iter - 1):
            print(f"  RAAR iter {it:4d}  R={R:.4f}  neg={history['neg_frac'][-1]:.3f}")

    # Final: enforce modulus on last density
    _, phases, F_m = _P_M_density(x, hkl, amp, cell)
    rho = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, shape=shape
    )
    history["final_R"] = r_factor_moduli(F_m, amp)
    history["n_iter"] = n_iter
    history["shape"] = shape
    return phases, rho, history


def difference_map_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 200,
    beta: float = 1.0,
    real_proj: str = "positivity",
    seed: int = 0,
    d_min: Optional[float] = None,
    sampling: float = 3.0,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Difference-map iteration (Elser).

    Practical form with γ_S = 1/β, γ_M = −1/β:
      f_M = (1+γ_M) P_M − γ_M I
      f_S = (1+γ_S) P_S − γ_S I
      x ← x + β (P_S f_M − P_M f_S) x

    For β=1, γ_S=1, γ_M=-1 this reduces to a common DM specialization.
    """
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    shape, d_min = setup_grid(hkl, cell, d_min=d_min, sampling=sampling)

    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()

    x = density_from_phases(hkl, amp, phases, cell, shape)
    # Avoid division by zero beta
    beta = float(beta) if abs(beta) > 1e-8 else 1.0
    gamma_S = 1.0 / beta
    gamma_M = -1.0 / beta
    history = {
        "R": [],
        "neg_frac": [],
        "beta": beta,
        "algorithm": "difference_map",
        "gamma_S": gamma_S,
        "gamma_M": gamma_M,
    }

    for it in range(n_iter):
        Pm_x, phases, F_m = _P_M_density(x, hkl, amp, cell)
        Ps_x = _real_space_projector(x, kind=real_proj)

        # f_M(x) = (1+γ_M) P_M x − γ_M x
        f_M = (1.0 + gamma_M) * Pm_x - gamma_M * x
        # f_S(x) = (1+γ_S) P_S x − γ_S x
        f_S = (1.0 + gamma_S) * Ps_x - gamma_S * x

        # P_S f_M and P_M f_S
        Ps_fM = _real_space_projector(f_M, kind=real_proj)
        Pm_fS, _, _ = _P_M_density(f_S, hkl, amp, cell)

        x = x + beta * (Ps_fM - Pm_fS)

        R = r_factor_moduli(F_m, amp)
        history["R"].append(R)
        history["neg_frac"].append(float((x < 0).mean()))
        if verbose and (it % 40 == 0 or it == n_iter - 1):
            print(f"  DM iter {it:4d}  R={R:.4f}  neg={history['neg_frac'][-1]:.3f}")

    _, phases, F_m = _P_M_density(x, hkl, amp, cell)
    rho = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, shape=shape
    )
    history["final_R"] = r_factor_moduli(F_m, amp)
    history["n_iter"] = n_iter
    history["shape"] = shape
    return phases, rho, history


def er_solve(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    n_iter: int = 100,
    real_proj: str = "positivity",
    seed: int = 0,
    d_min: Optional[float] = None,
    phase_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Error reduction: alternate modulus and positivity (can stagnate)."""
    rng = np.random.default_rng(seed)
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    shape, d_min = setup_grid(hkl, cell, d_min=d_min)
    if phase_init is None:
        phases = rng.uniform(-np.pi, np.pi, size=len(amp))
    else:
        phases = np.asarray(phase_init, dtype=np.float64).copy()
    x = density_from_phases(hkl, amp, phases, cell, shape)
    history = {"R": [], "algorithm": "ER"}
    for it in range(n_iter):
        Pm_x, phases, F_m = _P_M_density(x, hkl, amp, cell)
        x = _real_space_projector(Pm_x, kind=real_proj)
        R = r_factor_moduli(F_m, amp)
        history["R"].append(R)
        if verbose and it % 40 == 0:
            print(f"  ER iter {it:4d}  R={R:.4f}")
    _, phases, F_m = _P_M_density(x, hkl, amp, cell)
    rho = density_from_structure_factors(
        hkl, amp * np.exp(1j * phases), cell, shape=shape
    )
    history["final_R"] = r_factor_moduli(F_m, amp)
    return phases, rho, history
