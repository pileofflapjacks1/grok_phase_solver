"""
Synthetic structure and diffraction data generation.

Phase 1: small random organics in common space groups for pipeline tests.
Phase 2 will scale to millions of physically valid structures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.io.cif import AtomSite, CrystalStructure
from grok_phase_solver.io.hkl import ReflectionTable
from grok_phase_solver.physics.reciprocal import generate_hkl
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc


# Approximate covalent radii (Å) for clash checks
_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "CL": 1.02, "Cl": 1.02}


def _random_cell(
    rng: np.random.Generator,
    volume_range: Tuple[float, float] = (200.0, 800.0),
    system: str = "monoclinic",
) -> np.ndarray:
    """Sample a plausible unit cell."""
    V_target = rng.uniform(*volume_range)
    if system == "cubic":
        a = V_target ** (1.0 / 3.0)
        return np.array([a, a, a, 90.0, 90.0, 90.0])
    if system == "orthorhombic":
        a = rng.uniform(5.0, 12.0)
        b = rng.uniform(5.0, 12.0)
        c = V_target / (a * b)
        return np.array([a, b, c, 90.0, 90.0, 90.0])
    # monoclinic default
    a = rng.uniform(5.0, 12.0)
    b = rng.uniform(5.0, 12.0)
    beta = rng.uniform(90.0, 120.0)
    c = V_target / (a * b * np.sin(np.deg2rad(beta)))
    return np.array([a, b, max(c, 4.0), 90.0, beta, 90.0])


def _min_image_dist(f1: np.ndarray, f2: np.ndarray, cell: np.ndarray) -> float:
    """Minimum-image distance between fractional coords."""
    from grok_phase_solver.io.cif import CrystalStructure as CS

    df = (f1 - f2 + 0.5) % 1.0 - 0.5
    M = CS("t", cell, "P1").orth_matrix
    return float(np.linalg.norm(M @ df))


def generate_random_organic(
    n_atoms: int = 12,
    seed: int = 0,
    space_group: str = "P1",
    elements: Optional[Sequence[str]] = None,
    min_dist: float = 1.1,
    volume_per_atom: float = 18.0,
    max_trials: int = 5000,
) -> CrystalStructure:
    """
    Place non-overlapping atoms randomly in a unit cell (asymmetric unit = cell for P1).

    This is a *minimal* synthetic generator for pipeline validation — not chemically
    accurate molecules. Phase 2 will add fragment databases and bonding constraints.
    """
    rng = np.random.default_rng(seed)
    if elements is None:
        # Organic-like composition
        pool = ["C"] * 6 + ["N"] * 1 + ["O"] * 2 + ["H"] * 3
        elements = [pool[i % len(pool)] for i in range(n_atoms)]

    V = volume_per_atom * n_atoms
    cell = _random_cell(rng, volume_range=(0.8 * V, 1.2 * V), system="monoclinic")
    atoms: List[AtomSite] = []
    fracs: List[np.ndarray] = []

    for i, el in enumerate(elements):
        placed = False
        for _ in range(max_trials // max(n_atoms, 1)):
            f = rng.random(3)
            ok = True
            for j, f2 in enumerate(fracs):
                dmin = 0.7 * (_RADII.get(el, 0.7) + _RADII.get(elements[j], 0.7))
                dmin = max(dmin, min_dist * 0.8)
                if _min_image_dist(f, f2, cell) < dmin:
                    ok = False
                    break
            if ok:
                fracs.append(f)
                atoms.append(
                    AtomSite(
                        label=f"{el}{i+1}",
                        element=el,
                        fract=f,
                        occupancy=1.0,
                        u_iso=float(rng.uniform(0.02, 0.06)),
                    )
                )
                placed = True
                break
        if not placed:
            # Force place with soft constraint
            f = rng.random(3)
            fracs.append(f)
            atoms.append(AtomSite(label=f"{el}{i+1}", element=el, fract=f, u_iso=0.04))

    return CrystalStructure(
        name=f"synth_n{n_atoms}_s{seed}",
        cell=cell,
        space_group_hm=space_group,
        atoms=atoms,
        z=1,
        wavelength=0.71073,
    )


def simulate_diffraction(
    structure: CrystalStructure,
    d_min: float = 1.0,
    noise_level: float = 0.0,
    completeness: float = 1.0,
    missing_wedge_axis: Optional[int] = None,
    missing_wedge_angle: float = 0.0,
    seed: int = 0,
) -> ReflectionTable:
    """
    Simulate |F| observations from a structure with optional degradations.

    Parameters
    ----------
    noise_level : relative Gaussian noise on |F|
    completeness : random subset fraction
    missing_wedge_axis : if set (0/1/2 for h/k/l), drop reflections in a wedge
    missing_wedge_angle : half-angle (degrees) of excluded wedge about axis
    """
    data = structure_to_fcalc(structure, d_min=d_min)
    hkl = data["hkl"]
    F = data["F"]
    amp = np.abs(F)
    phases = np.angle(F)
    rng = np.random.default_rng(seed)

    mask = np.ones(len(amp), dtype=bool)

    if missing_wedge_axis is not None and missing_wedge_angle > 0:
        # Simple reciprocal-space wedge: exclude based on angle in perpendicular plane
        axis = missing_wedge_axis
        coords = hkl.astype(float)
        # Project orthogonal to axis
        axes = [0, 1, 2]
        axes.remove(axis)
        ang = np.rad2deg(np.arctan2(coords[:, axes[1]], coords[:, axes[0]] + 1e-12))
        half = missing_wedge_angle
        # Exclude |ang| < half or near 180
        mask &= np.abs(ang) > half

    if completeness < 1.0:
        keep = rng.random(len(mask)) < completeness
        mask &= keep

    hkl = hkl[mask]
    amp = amp[mask]
    phases = phases[mask]
    F = F[mask]

    if noise_level > 0:
        amp = amp * (1.0 + noise_level * rng.standard_normal(len(amp)))
        amp = np.maximum(amp, 0.0)

    return ReflectionTable(
        hkl=hkl,
        F_meas=amp,
        F_calc=F,
        phase=phases,
        cell=structure.cell,
        space_group_hm=structure.space_group_hm,
        wavelength=structure.wavelength,
        meta={
            "synthetic": True,
            "d_min": d_min,
            "noise_level": noise_level,
            "completeness": completeness,
            "structure": structure.name,
        },
    )


def degradation_suite(
    structure: CrystalStructure,
    d_min: float = 1.2,
    seed: int = 0,
) -> Dict[str, ReflectionTable]:
    """Generate a suite of degraded datasets for robustness testing."""
    return {
        "perfect": simulate_diffraction(structure, d_min=d_min, seed=seed),
        "noise_5pct": simulate_diffraction(structure, d_min=d_min, noise_level=0.05, seed=seed),
        "noise_15pct": simulate_diffraction(structure, d_min=d_min, noise_level=0.15, seed=seed),
        "comp_80": simulate_diffraction(structure, d_min=d_min, completeness=0.8, seed=seed),
        "comp_50": simulate_diffraction(structure, d_min=d_min, completeness=0.5, seed=seed),
        "low_res_2A": simulate_diffraction(structure, d_min=2.0, seed=seed),
    }
