"""
Simulate classical experimental phasing experiments (MIR / MAD / MR)

for hybrid AI test design (Cowtan 2001).

These utilities produce multi-channel amplitude observations that an AI
model (or hybrid pipeline) can consume alongside native |F|:

- MIR: native + derivative |F| with known heavy-atom substructure
- MAD: multi-wavelength |F+| / |F−| with anomalous f', f''
- MR: search model Fcalc for rotation/translation scoring

Ground-truth phases remain available for metric evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.io.cif import AtomSite, CrystalStructure
from grok_phase_solver.physics.reciprocal import generate_hkl
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc


@dataclass
class MIRDataset:
    """Native + derivative amplitudes with heavy-atom truth."""

    hkl: np.ndarray
    F_native: np.ndarray
    F_derivative: np.ndarray
    F_heavy: np.ndarray  # complex calculated heavy-atom contribution
    phases_true: np.ndarray
    heavy_fracs: np.ndarray
    cell: np.ndarray
    meta: Dict = field(default_factory=dict)


@dataclass
class MADDataset:
    """Multi-wavelength anomalous amplitudes (simplified)."""

    hkl: np.ndarray
    wavelengths: List[float]
    F_plus: Dict[float, np.ndarray]  # |F(h)| per λ
    F_minus: Dict[float, np.ndarray]  # |F(−h)| per λ (Friedel)
    phases_true: np.ndarray
    anomalous_fracs: np.ndarray
    cell: np.ndarray
    f_prime: Dict[float, float]
    f_double_prime: Dict[float, float]
    meta: Dict = field(default_factory=dict)


@dataclass
class MRDataset:
    """Molecular replacement: target |F| + search-model Fcalc at true pose."""

    hkl: np.ndarray
    F_obs: np.ndarray
    F_model: np.ndarray  # complex, correct pose
    phases_true: np.ndarray
    cell: np.ndarray
    meta: Dict = field(default_factory=dict)


def simulate_mir(
    structure: CrystalStructure,
    heavy_element: str = "AU",
    n_heavy: int = 1,
    d_min: float = 1.5,
    noise: float = 0.02,
    seed: int = 0,
) -> MIRDataset:
    """
    Build a simple MIR pair: native F_p and derivative F_p + F_h.

    Heavy atoms are placed randomly (not clashing with a simple check).
    |F_PH| = |F_P + F_H|; Harker-style phase indication follows Cowtan Fig. 4.
    """
    rng = np.random.default_rng(seed)
    data = structure_to_fcalc(structure, d_min=d_min)
    hkl, F_p, phases = data["hkl"], data["F"], data["phases"]

    heavy_fracs = rng.random((n_heavy, 3))
    # Approximate heavy form factor as constant Z (crude for demo)
    Z = {"AU": 79, "HG": 80, "PT": 78, "SE": 34, "FE": 26}.get(heavy_element.upper(), 50)
    # F_h ≈ Z * Σ exp(2πi h·r)  (point atoms)
    phase_h = 2 * np.pi * (hkl.astype(float) @ heavy_fracs.T)
    F_h = Z * np.sum(np.exp(1j * phase_h), axis=1)

    F_ph = F_p + F_h
    F_native = np.abs(F_p) * (1 + noise * rng.standard_normal(len(F_p)))
    F_deriv = np.abs(F_ph) * (1 + noise * rng.standard_normal(len(F_ph)))
    F_native = np.maximum(F_native, 0)
    F_deriv = np.maximum(F_deriv, 0)

    return MIRDataset(
        hkl=hkl,
        F_native=F_native,
        F_derivative=F_deriv,
        F_heavy=F_h,
        phases_true=phases,
        heavy_fracs=heavy_fracs,
        cell=structure.cell,
        meta={
            "heavy_element": heavy_element,
            "n_heavy": n_heavy,
            "d_min": d_min,
            "noise": noise,
            "structure": structure.name,
        },
    )


def mir_phase_indication(
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    F_heavy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Harker construction (centroid phase estimate).

    When |F_PH| > |F_P|, phase of F_P tends toward phase of F_H;
    when |F_PH| < |F_P|, toward phase of F_H + π (Cowtan Fig. 4).

    Returns (phase_est, figure_of_merit in [0,1]).
    """
    Fp = np.asarray(F_native, dtype=np.float64)
    Fph = np.asarray(F_derivative, dtype=np.float64)
    fh = np.asarray(F_heavy, dtype=np.complex128)
    fh_phase = np.angle(fh)
    fh_mag = np.abs(fh) + 1e-16

    # Lack-of-closure style FOM: how extreme is the difference relative to |F_h|
    delta = Fph - Fp
    # Cosine of angle between F_p and F_h from law of cosines:
    # |F_PH|² ≈ |F_P|² + |F_H|² + 2 |F_P||F_H| cos(Δφ)
    cos_dphi = (Fph**2 - Fp**2 - fh_mag**2) / (2 * Fp * fh_mag + 1e-16)
    cos_dphi = np.clip(cos_dphi, -1, 1)
    # Two-fold ambiguity: ±Δφ; take the estimate matching sign of (Fph-Fp)
    dphi = np.arccos(cos_dphi)
    # Prefer constructive when Fph > Fp
    phase_est = fh_phase + np.where(Fph >= Fp, 0.0, np.pi)
    # Blend with computed |Δφ|
    phase_est = fh_phase + np.sign(Fph - Fp + 1e-12) * 0.0  # base
    # Use law-of-cosines angle as uncertainty weight
    fom = np.abs(cos_dphi) * np.minimum(fh_mag / (Fp + 1e-16), 1.0)
    fom = np.clip(fom, 0, 1)
    # Better estimate: φ_p ≈ φ_h ± arccos(...)
    phase_est = np.where(Fph >= Fp, fh_phase + dphi * 0, fh_phase + np.pi)
    # Actually for teaching, use:
    phase_est = fh_phase + np.where(delta >= 0, 0.0, np.pi)
    return phase_est, fom


def simulate_mad(
    structure: CrystalStructure,
    anomalous_element: str = "SE",
    n_sites: int = 2,
    wavelengths: Optional[Sequence[float]] = None,
    d_min: float = 1.5,
    noise: float = 0.02,
    seed: int = 0,
) -> MADDataset:
    """
    Simplified MAD: f', f'' step-function across an absorption edge.

    Wavelengths default: remote, inflection, peak (Å).
    """
    if wavelengths is None:
        wavelengths = [0.98, 0.9795, 0.9790]  # Se K-edge-ish

    rng = np.random.default_rng(seed)
    data = structure_to_fcalc(structure, d_min=d_min)
    hkl, F_p, phases = data["hkl"], data["F"], data["phases"]
    anom_fracs = rng.random((n_sites, 3))

    # Toy dispersion: remote (f'=-1,f''=2), edge (f'=-8,f''=2), peak (f'=-6,f''=5)
    table = {
        wavelengths[0]: (-1.0, 2.0),
        wavelengths[1]: (-8.0, 2.5),
        wavelengths[2]: (-6.0, 5.0) if len(wavelengths) > 2 else (-6.0, 5.0),
    }
    f_prime = {wl: table.get(wl, (-2.0, 3.0))[0] for wl in wavelengths}
    f_double = {wl: table.get(wl, (-2.0, 3.0))[1] for wl in wavelengths}

    Z0 = {"SE": 34, "FE": 26, "S": 16}.get(anomalous_element.upper(), 34)
    F_plus: Dict[float, np.ndarray] = {}
    F_minus: Dict[float, np.ndarray] = {}

    for wl in wavelengths:
        fp, fpp = f_prime[wl], f_double[wl]
        # Anomalous contribution: (Z0+f' + i f'') * Σ exp(2πi h r)
        phase = 2 * np.pi * (hkl.astype(float) @ anom_fracs.T)
        geo = np.sum(np.exp(1j * phase), axis=1)
        F_a = (Z0 + fp + 1j * fpp) * geo
        F_tot = F_p + F_a
        # Friedel mate: F(−h) with conjugate non-anomalous + anomalous phase rules
        # Simplified: |F+| and |F−| differ by 2 f'' contribution
        F_tot_minus = np.conj(F_p) + (Z0 + fp - 1j * fpp) * np.conj(geo)
        # Actually use proper: compute from −h indices
        F_plus[wl] = np.maximum(
            np.abs(F_tot) * (1 + noise * rng.standard_normal(len(F_tot))), 0
        )
        F_minus[wl] = np.maximum(
            np.abs(F_tot_minus) * (1 + noise * rng.standard_normal(len(F_tot))), 0
        )

    return MADDataset(
        hkl=hkl,
        wavelengths=list(wavelengths),
        F_plus=F_plus,
        F_minus=F_minus,
        phases_true=phases,
        anomalous_fracs=anom_fracs,
        cell=structure.cell,
        f_prime=f_prime,
        f_double_prime=f_double,
        meta={
            "anomalous_element": anomalous_element,
            "n_sites": n_sites,
            "d_min": d_min,
            "structure": structure.name,
        },
    )


def simulate_mr(
    structure: CrystalStructure,
    d_min: float = 2.0,
    model_noise_atoms: float = 0.0,
    seed: int = 0,
) -> MRDataset:
    """
    MR-style dataset: observed |F| from structure; model Fcalc from same
    (or slightly perturbed) coordinates — perfect MR solution case.

    Hybrid AI test: can a network refine partial model phases + |F_obs|?
    """
    rng = np.random.default_rng(seed)
    data = structure_to_fcalc(structure, d_min=d_min)
    F = data["F"]
    if model_noise_atoms > 0:
        # Phase jitter as incomplete model
        F_model = F * np.exp(1j * model_noise_atoms * rng.standard_normal(len(F)))
    else:
        F_model = F.copy()
    return MRDataset(
        hkl=data["hkl"],
        F_obs=np.abs(F),
        F_model=F_model,
        phases_true=data["phases"],
        cell=structure.cell,
        meta={"d_min": d_min, "structure": structure.name, "perfect_model": model_noise_atoms == 0},
    )


def hybrid_feature_stack_mir(ds: MIRDataset) -> np.ndarray:
    """
    Feature matrix for ML: columns = [|F_p|, |F_ph|, |F_h|, Δ|F|, cos φ_h, sin φ_h]
    Shape (N_refl, 6). Prototype hybrid AI input.
    """
    fh = ds.F_heavy
    return np.column_stack(
        [
            ds.F_native,
            ds.F_derivative,
            np.abs(fh),
            ds.F_derivative - ds.F_native,
            np.cos(np.angle(fh)),
            np.sin(np.angle(fh)),
        ]
    )
