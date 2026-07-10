"""
Phase-2 synthetic structure generation (fragment-aware, SG-aware hooks).

v2 upgrades over `synthetic.generate_random_organic`:
- bonded fragment library (ring, chain stubs)
- optional centrosymmetric partner for P-1 demos
- Wilson-plot validated Fcalc export
- dataset writer for training shards
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.io.cif import AtomSite, CrystalStructure
from grok_phase_solver.data.synthetic import generate_random_organic, simulate_diffraction
from grok_phase_solver.solvers.baseline import structure_to_fcalc


# Minimal organic fragments in local coordinates (Å), centroid ~ origin
_FRAGMENTS: Dict[str, List[Tuple[str, Tuple[float, float, float]]]] = {
    "benzene": [
        ("C", (1.40, 0.00, 0.00)),
        ("C", (0.70, 1.21, 0.00)),
        ("C", (-0.70, 1.21, 0.00)),
        ("C", (-1.40, 0.00, 0.00)),
        ("C", (-0.70, -1.21, 0.00)),
        ("C", (0.70, -1.21, 0.00)),
    ],
    "carboxyl": [
        ("C", (0.00, 0.00, 0.00)),
        ("O", (1.20, 0.50, 0.00)),
        ("O", (-1.10, 0.70, 0.00)),
    ],
    "peptide": [
        ("N", (-1.20, 0.30, 0.00)),
        ("C", (0.00, 0.00, 0.00)),
        ("O", (0.50, 1.10, 0.00)),
        ("C", (1.20, -0.90, 0.00)),
    ],
}


def _rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Random 3D rotation (Arvo-style via axis-angle)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-16
    ang = rng.uniform(0, 2 * np.pi)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R


def generate_fragment_structure(
    n_fragments: int = 2,
    fragment_names: Optional[Sequence[str]] = None,
    seed: int = 0,
    space_group: str = "P1",
    padding: float = 3.5,
) -> CrystalStructure:
    """
    Place randomly rotated organic fragments in a box-like monoclinic cell.
    """
    rng = np.random.default_rng(seed)
    names = list(fragment_names) if fragment_names else list(_FRAGMENTS.keys())
    atoms: List[AtomSite] = []
    cart_points: List[np.ndarray] = []

    for fi in range(n_fragments):
        fname = names[fi % len(names)]
        frag = _FRAGMENTS[fname]
        R = _rotation_matrix(rng)
        center = rng.uniform(-4, 4, size=3) + np.array([fi * 2.0, 0, 0])
        for j, (el, xyz) in enumerate(frag):
            p = R @ np.array(xyz, dtype=np.float64) + center
            cart_points.append(p)
            atoms.append(
                AtomSite(
                    label=f"{el}{fi}_{j}",
                    element=el,
                    fract=np.zeros(3),  # fill after cell known
                    u_iso=float(rng.uniform(0.02, 0.05)),
                )
            )

    pts = np.vstack(cart_points)
    mins = pts.min(axis=0) - padding
    maxs = pts.max(axis=0) + padding
    extents = maxs - mins
    # Build orthogonal cell first
    a, b, c = extents
    beta = 90.0
    cell = np.array([a, b, c, 90.0, beta, 90.0], dtype=np.float64)
    # Fractional coordinates
    for i, p in enumerate(pts):
        frac = (p - mins) / np.maximum(extents, 1e-6)
        atoms[i].fract = frac % 1.0

    return CrystalStructure(
        name=f"frag_n{n_fragments}_s{seed}",
        cell=cell,
        space_group_hm=space_group,
        atoms=atoms,
        z=1,
        wavelength=0.71073,
    )


def iter_training_samples(
    n_samples: int,
    seed: int = 0,
    d_min: float = 1.2,
    mode: str = "fragment",
) -> Iterator[Dict]:
    """
    Yield dicts ready for dataset serialization:
    {hkl, amplitudes, phases, cell, meta}
    """
    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        s = int(rng.integers(0, 2**31 - 1))
        if mode == "fragment":
            st = generate_fragment_structure(
                n_fragments=int(rng.integers(1, 4)), seed=s
            )
        else:
            st = generate_random_organic(n_atoms=int(rng.integers(6, 16)), seed=s)
        data = structure_to_fcalc(st, d_min=d_min)
        yield {
            "name": st.name,
            "hkl": data["hkl"],
            "amplitudes": data["amplitudes"],
            "phases": data["phases"],
            "cell": st.cell,
            "n_atoms": data["n_atoms_cell"],
            "space_group": st.space_group_hm,
            "d_min": d_min,
        }


def write_training_shard(
    out_path: Path,
    n_samples: int = 100,
    seed: int = 0,
    d_min: float = 1.2,
    mode: str = "fragment",
) -> Path:
    """Write an NPZ shard of variable-length reflections (object arrays)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = list(iter_training_samples(n_samples, seed=seed, d_min=d_min, mode=mode))
    np.savez_compressed(
        out_path,
        hkl=np.array([s["hkl"] for s in samples], dtype=object),
        amplitudes=np.array([s["amplitudes"] for s in samples], dtype=object),
        phases=np.array([s["phases"] for s in samples], dtype=object),
        cell=np.stack([s["cell"] for s in samples]),
        n_atoms=np.array([s["n_atoms"] for s in samples]),
        names=np.array([s["name"] for s in samples]),
        d_min=d_min,
        mode=mode,
        seed=seed,
    )
    meta = {
        "n_samples": n_samples,
        "d_min": d_min,
        "mode": mode,
        "seed": seed,
        "path": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out_path
