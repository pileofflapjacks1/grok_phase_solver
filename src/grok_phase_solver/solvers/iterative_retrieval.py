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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.free_fom import free_fom
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
    delta_sigma: float = 1.0,
) -> np.ndarray:
    """
    Real-space constraint projector.

    charge_flip: flip density below δ = delta_sigma * σ(ρ)
    (delta_sigma=1.0 is a typical Oszlányi–Sütő scale; 0 → flip all negative).
    """
    if kind == "positivity":
        return project_positivity(rho)
    if kind == "charge_flip":
        sigma = float(rho.std()) + 1e-16
        delta = float(delta_sigma) * sigma
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
    delta_sigma: float = 1.0,
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
    history = {
        "R": [],
        "neg_frac": [],
        "beta": beta,
        "algorithm": "RAAR",
        "real_proj": real_proj,
        "delta_sigma": delta_sigma,
    }

    for it in range(n_iter):
        # P_M x
        Pm_x, phases, F_m = _P_M_density(x, hkl, amp, cell)
        # R_M x = 2 P_M x − x
        Rm_x = 2.0 * Pm_x - x
        # P_S R_M x
        Ps_Rm = _real_space_projector(Rm_x, kind=real_proj, delta_sigma=delta_sigma)
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
    delta_sigma: float = 1.0,
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

    Parameters
    ----------
    beta : step size / DM parameter (try 0.5–1.5 in retunes)
    real_proj : ``"positivity"`` | ``"charge_flip"`` | ``"none"``
    delta_sigma : flip threshold in units of density σ (charge_flip only)
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
        "real_proj": real_proj,
        "delta_sigma": delta_sigma,
        "gamma_S": gamma_S,
        "gamma_M": gamma_M,
    }

    for it in range(n_iter):
        Pm_x, phases, F_m = _P_M_density(x, hkl, amp, cell)
        Ps_x = _real_space_projector(x, kind=real_proj, delta_sigma=delta_sigma)

        # f_M(x) = (1+γ_M) P_M x − γ_M x
        f_M = (1.0 + gamma_M) * Pm_x - gamma_M * x
        # f_S(x) = (1+γ_S) P_S x − γ_S x
        f_S = (1.0 + gamma_S) * Ps_x - gamma_S * x

        # P_S f_M and P_M f_S
        Ps_fM = _real_space_projector(f_M, kind=real_proj, delta_sigma=delta_sigma)
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


def retune_difference_map(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    beta_grid: Optional[Sequence[float]] = None,
    real_proj_options: Optional[Sequence[str]] = None,
    delta_sigma_grid: Optional[Sequence[float]] = None,
    n_iter: int = 80,
    seeds: Optional[Sequence[int]] = None,
    d_min: Optional[float] = None,
    verbose: bool = False,
) -> Dict:
    """
    Grid-search DiffMap hyperparameters; rank by free-FOM composite (no truth).

    Returns dict with ``best_params``, ``best_fom``, ``grid_results``.
    """
    if beta_grid is None:
        beta_grid = (0.5, 0.7, 1.0, 1.2)
    if real_proj_options is None:
        real_proj_options = ("positivity", "charge_flip")
    if delta_sigma_grid is None:
        delta_sigma_grid = (0.0, 0.5, 1.0)
    if seeds is None:
        seeds = (0, 1)

    grid_results = []
    best = None  # (composite, params, phases, rho, fom, hist)

    for beta in beta_grid:
        for real_proj in real_proj_options:
            deltas = delta_sigma_grid if real_proj == "charge_flip" else (1.0,)
            for delta_sigma in deltas:
                composites = []
                for seed in seeds:
                    ph, rho, hist = difference_map_solve(
                        hkl,
                        amplitudes,
                        cell,
                        n_iter=n_iter,
                        beta=float(beta),
                        real_proj=real_proj,
                        delta_sigma=float(delta_sigma),
                        seed=int(seed),
                        d_min=d_min,
                        verbose=False,
                    )
                    fom = free_fom(hkl, amplitudes, ph, cell, density=rho)
                    composites.append(fom["composite"])
                    entry = {
                        "beta": float(beta),
                        "real_proj": real_proj,
                        "delta_sigma": float(delta_sigma),
                        "seed": int(seed),
                        "composite": fom["composite"],
                        "pos_frac": fom["pos_frac"],
                        "R_after_ER": fom["R_after_ER"],
                        "final_R": hist.get("final_R"),
                    }
                    grid_results.append(entry)
                    if best is None or fom["composite"] > best[0]:
                        best = (fom["composite"], entry, ph, rho, fom, hist)
                mean_c = float(np.mean(composites))
                if verbose:
                    print(
                        f"  β={beta:.2f} Ps={real_proj:12s} δσ={delta_sigma:.1f} "
                        f"mean_composite={mean_c:.3f}"
                    )

    assert best is not None
    _, params, ph, rho, fom, hist = best
    return {
        "best_params": {
            "beta": params["beta"],
            "real_proj": params["real_proj"],
            "delta_sigma": params["delta_sigma"],
        },
        "best_seed": params["seed"],
        "best_fom": fom,
        "grid_results": grid_results,
        "phases": ph,
        "density": rho,
        "history": hist,
    }



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
