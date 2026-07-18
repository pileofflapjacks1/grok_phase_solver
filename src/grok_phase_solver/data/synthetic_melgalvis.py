"""
Melgalvis & Rekis (2026) style artificial crystal structure generation.

Reference
---------
Melgalvis, D.M. & Rekis, T. (2026). "On artificial crystal structure generation
for solving the phase problem with deep learning." Acta Cryst. A 82, 32–40.

Implemented techniques (transparent, physics-grounded subset):
1. **Volume-first lattice sampling** — sample unit-cell volume V from a log-normal
   distribution fitted to experimental small-molecule stats (COD-like), then
   derive a,b,c with realistic axis ratios and monoclinic/triclinic skew.
2. **Artificial-molecule clusters** — grow bonded clusters from a seed atom using
   covalent radii / bond distances, empirical element frequencies, optional
   inversion-centre special positions, H addition, isotropic B sampling, and
   volume-per-non-H density constraints.
3. **Rejection baseline** — keep legacy random placement as ``mode="rejection"``.

On-the-fly generation: no pre-storage required. Falls back to
``generate_random_organic`` if cluster packing fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grok_phase_solver.io.cif import AtomSite, CrystalStructure

# Covalent radii (Å) — Cordero et al. style approximations
_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "Cl": 1.02,
    "BR": 1.20,
    "Br": 1.20,
    "I": 1.39,
}

# Element frequencies (general positions) — organic-biased, COD-inspired
_ELEM_FREQ_GENERAL = {
    "C": 0.55,
    "H": 0.22,
    "O": 0.12,
    "N": 0.07,
    "S": 0.02,
    "CL": 0.01,
    "P": 0.005,
    "F": 0.005,
}

# Special positions (near inversion / high symmetry): fewer H, more heavy
_ELEM_FREQ_SPECIAL = {
    "C": 0.45,
    "O": 0.20,
    "N": 0.12,
    "S": 0.08,
    "CL": 0.05,
    "P": 0.04,
    "F": 0.03,
    "BR": 0.02,
    "H": 0.01,
}

# Default log-normal volume params fitted loosely to small-molecule COD-like range
# mean log(V) ≈ log(450), sigma ≈ 0.55 → median ~450 Å³, typical 150–1500
_DEFAULT_LOG_V_MU = float(np.log(450.0))
_DEFAULT_LOG_V_SIGMA = 0.55


@dataclass
class MelgalvisGenConfig:
    """Controls Melgalvis-style synthetic realism."""

    # Volume / lattice
    log_v_mu: float = _DEFAULT_LOG_V_MU
    log_v_sigma: float = _DEFAULT_LOG_V_SIGMA
    v_min: float = 120.0
    v_max: float = 2500.0
    systems: Tuple[str, ...] = ("monoclinic", "orthorhombic", "triclinic", "triclinic")
    # Axis ratio bounds a/b, c/b after volume scaling
    ratio_lo: float = 0.55
    ratio_hi: float = 1.85
    # Density: volume per non-H atom (Å³)
    vol_per_nonh_lo: float = 7.0
    vol_per_nonh_hi: float = 22.0
    # Cluster builder
    n_nonh_lo: int = 6
    n_nonh_hi: int = 24
    p_special_seed: float = 0.12  # chance molecule seed near inversion center
    bond_scale: float = 1.0  # scale covalent sum for bond length
    min_nonbond: float = 0.85  # fraction of covalent sum for clash
    b_iso_lo: float = 0.01  # Å² (as U_iso ≈ B/8π²; we store u_iso)
    b_iso_hi: float = 0.10
    add_hydrogens: bool = True
    max_pack_trials: int = 40
    max_atom_trials: int = 80
    # Mode
    mode: str = "cluster"  # "cluster" | "rejection" | "hybrid"
    hybrid_cluster_frac: float = 0.7
    wavelength: float = 0.71073
    name_prefix: str = "melg"


def _sample_weighted(rng: np.random.Generator, freq: Dict[str, float]) -> str:
    keys = list(freq.keys())
    p = np.array([freq[k] for k in keys], dtype=np.float64)
    p = p / p.sum()
    return str(rng.choice(keys, p=p))


def sample_volume(rng: np.random.Generator, cfg: MelgalvisGenConfig) -> float:
    """Sample unit-cell volume from truncated log-normal."""
    for _ in range(100):
        V = float(np.exp(rng.normal(cfg.log_v_mu, cfg.log_v_sigma)))
        if cfg.v_min <= V <= cfg.v_max:
            return V
    return float(np.clip(V, cfg.v_min, cfg.v_max))


def sample_lattice_from_volume(
    rng: np.random.Generator,
    V: float,
    cfg: MelgalvisGenConfig,
    system: Optional[str] = None,
) -> np.ndarray:
    """
    Derive lattice parameters from volume with realistic ratios and skew.

    For monoclinic/triclinic, apply small skew via angle deviations so that
    V = abc √(1 − cos²α − cos²β − cos²γ + 2 cosα cosβ cosγ) matches target.
    """
    system = system or str(rng.choice(cfg.systems))
    # Sample axis ratios relative to geometric mean
    r_ab = rng.uniform(cfg.ratio_lo, cfg.ratio_hi)
    r_cb = rng.uniform(cfg.ratio_lo, cfg.ratio_hi)
    # b free; a = r_ab * b; c = r_cb * b; for orthogonal V = a b c = r_ab r_cb b³
    if system == "cubic":
        a = V ** (1.0 / 3.0)
        return np.array([a, a, a, 90.0, 90.0, 90.0], dtype=np.float64)

    if system == "orthorhombic":
        b = (V / (r_ab * r_cb)) ** (1.0 / 3.0)
        a, c = r_ab * b, r_cb * b
        # random axis permutation (Melgalvis-style)
        edges = np.array([a, b, c])
        rng.shuffle(edges)
        return np.array([*edges, 90.0, 90.0, 90.0], dtype=np.float64)

    if system == "monoclinic":
        beta = float(rng.uniform(92.0, 125.0))
        sbeta = np.sin(np.deg2rad(beta))
        # V = a b c sin(β)
        b = (V / (r_ab * r_cb * sbeta)) ** (1.0 / 3.0)
        a, c = r_ab * b, r_cb * b
        # permute a/c only (b unique monoclinic axis convention P21/c style optional)
        if rng.random() < 0.5:
            a, c = c, a
        return np.array([a, b, c, 90.0, beta, 90.0], dtype=np.float64)

    # triclinic: sample angles with mild correlations
    alpha = float(rng.uniform(70.0, 110.0))
    beta = float(rng.uniform(70.0, 120.0))
    gamma = float(rng.uniform(70.0, 110.0))
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    root = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    root = max(root, 0.05)
    factor = np.sqrt(root)
    b = (V / (r_ab * r_cb * factor)) ** (1.0 / 3.0)
    a, c = r_ab * b, r_cb * b
    edges = np.array([a, b, c])
    rng.shuffle(edges)
    return np.array([*edges, alpha, beta, gamma], dtype=np.float64)


def _cell_volume(cell: np.ndarray) -> float:
    a, b, c, al, be, ga = cell
    ca, cb, cg = np.cos(np.deg2rad([al, be, ga]))
    root = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    return float(a * b * c * np.sqrt(max(root, 0.0)))


def _orth_matrix(cell: np.ndarray) -> np.ndarray:
    """Fractional → Cartesian matrix (same convention as CrystalStructure)."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    ca, cb, cg = np.cos([al, be, ga])
    sg = np.sin(ga)
    v = np.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, 1e-16))
    return np.array(
        [
            [a, b * cg, c * cb],
            [0, b * sg, c * (ca - cb * cg) / (sg + 1e-16)],
            [0, 0, c * v / (sg + 1e-16)],
        ],
        dtype=np.float64,
    )


def _frac_from_cart(cart: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.linalg.solve(M, cart)


def _min_image_cart(f1: np.ndarray, f2: np.ndarray, M: np.ndarray) -> float:
    df = (f1 - f2 + 0.5) % 1.0 - 0.5
    return float(np.linalg.norm(M @ df))


def _bond_length(el1: str, el2: str, cfg: MelgalvisGenConfig) -> float:
    r1 = _COVALENT_RADII.get(el1, 0.75)
    r2 = _COVALENT_RADII.get(el2, 0.75)
    return cfg.bond_scale * (r1 + r2)


def _u_iso_from_b(rng: np.random.Generator, cfg: MelgalvisGenConfig) -> float:
    """Sample isotropic U from B-range (B = 8 π² U)."""
    B = rng.uniform(cfg.b_iso_lo, cfg.b_iso_hi) * 8.0 * np.pi**2  # if b_iso is U
    # Config documents b_iso as Å² B-factor style 0.01–0.1 — treat as U directly for small-mol
    # Melgalvis uses isotropic B 0.01–0.1 Å² which is unusually small for B;
    # interpret as U_iso in Å² for SHELXL-like storage.
    return float(rng.uniform(cfg.b_iso_lo, cfg.b_iso_hi))


def build_artificial_molecule(
    rng: np.random.Generator,
    n_nonh: int,
    cfg: MelgalvisGenConfig,
    special_seed: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Grow a bonded cluster in Cartesian Å (origin-centered).

    Returns (elements, coords) including optional hydrogens.
    """
    freq = _ELEM_FREQ_SPECIAL if special_seed else _ELEM_FREQ_GENERAL
    # First non-H atom
    elements: List[str] = []
    coords: List[np.ndarray] = []

    def add_nonh(el: str, xyz: np.ndarray) -> bool:
        for j, c in enumerate(coords):
            if elements[j] == "H":
                continue
            dmin = cfg.min_nonbond * (_COVALENT_RADII.get(el, 0.75) + _COVALENT_RADII.get(elements[j], 0.75))
            if np.linalg.norm(xyz - c) < dmin * 0.95:
                return False
        elements.append(el)
        coords.append(xyz.astype(np.float64))
        return True

    seed_el = _sample_weighted(rng, {k: v for k, v in freq.items() if k != "H"})
    add_nonh(seed_el, np.zeros(3))

    while sum(1 for e in elements if e != "H") < n_nonh:
        # Attach to a random existing non-H
        nonh_idx = [i for i, e in enumerate(elements) if e != "H"]
        parent = int(rng.choice(nonh_idx))
        el = _sample_weighted(rng, {k: v for k, v in freq.items() if k != "H"})
        bl = _bond_length(elements[parent], el, cfg)
        # Random direction
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-16
        candidate = coords[parent] + bl * direction
        placed = False
        for _ in range(cfg.max_atom_trials):
            if add_nonh(el, candidate):
                placed = True
                break
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-16
            candidate = coords[parent] + bl * direction
        if not placed:
            # force attach far from centroid
            centroid = np.mean(np.vstack(coords), axis=0)
            direction = candidate - centroid
            direction /= np.linalg.norm(direction) + 1e-16
            add_nonh(el, coords[parent] + bl * direction)

    if cfg.add_hydrogens:
        # Simple: each C/N/O gets 0–2 H at covalent distance if valency space
        nonh = [(i, e) for i, e in enumerate(elements) if e != "H"]
        for i, el in nonh:
            if el not in ("C", "N", "O"):
                continue
            n_h = int(rng.integers(0, 3 if el == "C" else 2))
            for _ in range(n_h):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction) + 1e-16
                bl = _bond_length(el, "H", cfg)
                hpos = coords[i] + bl * direction
                ok = True
                for j, c in enumerate(coords):
                    dmin = 0.7 * (_COVALENT_RADII["H"] + _COVALENT_RADII.get(elements[j], 0.7))
                    if np.linalg.norm(hpos - c) < dmin:
                        ok = False
                        break
                if ok:
                    elements.append("H")
                    coords.append(hpos)

    # Center molecule
    xyz = np.vstack(coords)
    xyz = xyz - xyz.mean(axis=0)
    return elements, xyz


def pack_molecule_in_cell(
    rng: np.random.Generator,
    elements: Sequence[str],
    cart: np.ndarray,
    cell: np.ndarray,
    cfg: MelgalvisGenConfig,
    special_seed: bool = False,
) -> Optional[List[AtomSite]]:
    """
    Place molecule in cell with random rotation/translation; optional inversion partner.
    """
    M = _orth_matrix(cell)
    # Random rotation
    from grok_phase_solver.data.synthetic_v2 import _rotation_matrix

    R = _rotation_matrix(rng)
    xyz = (R @ cart.T).T

    for _ in range(cfg.max_pack_trials):
        if special_seed:
            # Seed near inversion center (0,0,0) with small offset
            t = rng.normal(scale=0.08, size=3)
        else:
            t = rng.random(3)
        # Map molecule centroid to fractional t
        # Put first atom at fractional t, rest relative
        fracs = []
        atoms: List[AtomSite] = []
        ok = True
        for i, el in enumerate(elements):
            # Use relative cart from atom 0
            dcart = xyz[i] - xyz[0]
            # Convert offset to fractional via M
            dfrac = _frac_from_cart(dcart, M)
            f = (t + dfrac) % 1.0
            # Clash with already placed
            for j, f2 in enumerate(fracs):
                dmin = cfg.min_nonbond * (
                    _COVALENT_RADII.get(el, 0.75) + _COVALENT_RADII.get(elements[j], 0.75)
                )
                if _min_image_cart(f, f2, M) < dmin * 0.9:
                    ok = False
                    break
            if not ok:
                break
            fracs.append(f)
            atoms.append(
                AtomSite(
                    label=f"{el}{i+1}",
                    element=el if el != "CL" else "Cl",
                    fract=f,
                    occupancy=1.0,
                    u_iso=_u_iso_from_b(rng, cfg),
                )
            )
        if not ok:
            continue

        # Optional inversion image of a subset (simulate special-position symmetry content)
        if special_seed and rng.random() < 0.5:
            extra: List[AtomSite] = []
            for a in atoms:
                if a.element == "H":
                    continue
                f_inv = (-a.fract) % 1.0
                # skip if too close to existing
                clash = False
                for f2 in fracs:
                    if _min_image_cart(f_inv, f2, M) < 0.9:
                        clash = True
                        break
                if not clash:
                    fracs.append(f_inv)
                    extra.append(
                        AtomSite(
                            label=f"{a.element}i{len(extra)+1}",
                            element=a.element,
                            fract=f_inv,
                            u_iso=a.u_iso,
                        )
                    )
            atoms.extend(extra)
        return atoms
    return None


def generate_melgalvis_structure(
    seed: int = 0,
    cfg: Optional[MelgalvisGenConfig] = None,
    space_group: str = "P1",
    n_nonh: Optional[int] = None,
) -> CrystalStructure:
    """
    Generate one artificial crystal structure (Melgalvis-style or hybrid).

    Parameters
    ----------
    seed : RNG seed
    cfg : generator config
    space_group : HM symbol (P1 default; P-1 may add centrosym copy externally)
    n_nonh : override non-H atom count
    """
    cfg = cfg or MelgalvisGenConfig()
    rng = np.random.default_rng(seed)

    mode = cfg.mode
    if mode == "hybrid":
        mode = "cluster" if rng.random() < cfg.hybrid_cluster_frac else "rejection"

    if mode == "rejection":
        from grok_phase_solver.data.synthetic import generate_random_organic

        n_atoms = int(n_nonh or rng.integers(cfg.n_nonh_lo, cfg.n_nonh_hi + 1))
        # volume-informed cell: map to volume_per_atom
        V = sample_volume(rng, cfg)
        vpa = V / max(n_atoms, 1)
        vpa = float(np.clip(vpa, cfg.vol_per_nonh_lo, cfg.vol_per_nonh_hi * 1.5))
        st = generate_random_organic(
            n_atoms=n_atoms,
            seed=seed,
            space_group=space_group,
            volume_per_atom=vpa,
        )
        st.name = f"{cfg.name_prefix}_rej_n{n_atoms}_s{seed}"
        return st

    # Cluster mode
    n_nonh = int(n_nonh or rng.integers(cfg.n_nonh_lo, cfg.n_nonh_hi + 1))
    special = bool(rng.random() < cfg.p_special_seed)
    # Volume from density constraint
    vpa = float(rng.uniform(cfg.vol_per_nonh_lo, cfg.vol_per_nonh_hi))
    V = vpa * n_nonh
    # blend with log-normal prior
    V_ln = sample_volume(rng, cfg)
    V = 0.5 * V + 0.5 * V_ln
    V = float(np.clip(V, cfg.v_min, cfg.v_max))

    cell = sample_lattice_from_volume(rng, V, cfg)
    elements, cart = build_artificial_molecule(rng, n_nonh, cfg, special_seed=special)
    atoms = pack_molecule_in_cell(rng, elements, cart, cell, cfg, special_seed=special)
    if atoms is None:
        # fallback rejection
        from grok_phase_solver.data.synthetic import generate_random_organic

        st = generate_random_organic(
            n_atoms=n_nonh,
            seed=seed + 1,
            space_group=space_group,
            volume_per_atom=vpa,
        )
        st.name = f"{cfg.name_prefix}_fb_n{n_nonh}_s{seed}"
        return st

    return CrystalStructure(
        name=f"{cfg.name_prefix}_cl_n{n_nonh}_s{seed}",
        cell=cell,
        space_group_hm=space_group,
        atoms=atoms,
        z=1,
        wavelength=cfg.wavelength,
    )


def iter_melgalvis_samples(
    n_samples: int,
    seed: int = 0,
    d_min: float = 1.2,
    cfg: Optional[MelgalvisGenConfig] = None,
    n_nonh_range: Optional[Tuple[int, int]] = None,
    d_min_range: Optional[Tuple[float, float]] = None,
    include_p_minus1: float = 0.25,
) -> List[Dict]:
    """
    On-the-fly training samples: structures → Fcalc.

    Returns list of dicts compatible with strong_prior / training loops.
    """
    from grok_phase_solver.data.synthetic_v2 import make_centrosymmetric_copy
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    cfg = cfg or MelgalvisGenConfig()
    if n_nonh_range:
        cfg = MelgalvisGenConfig(**{**cfg.__dict__, "n_nonh_lo": n_nonh_range[0], "n_nonh_hi": n_nonh_range[1]})
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    for i in range(n_samples):
        s = int(rng.integers(0, 2**31 - 1))
        st = generate_melgalvis_structure(seed=s, cfg=cfg, space_group="P1")
        sg = "P1"
        if rng.random() < include_p_minus1:
            try:
                st = make_centrosymmetric_copy(st)
                sg = "P-1"
            except Exception:
                pass
        d = float(d_min)
        if d_min_range is not None:
            d = float(rng.uniform(*d_min_range))
        data = structure_to_fcalc(st, d_min=d)
        out.append(
            {
                "name": st.name,
                "hkl": data["hkl"],
                "amplitudes": data["amplitudes"],
                "phases": data["phases"],
                "cell": st.cell,
                "n_atoms": data["n_atoms_cell"],
                "d_min": d,
                "region": "hard" if d >= 1.45 or data["n_atoms_cell"] >= 12 else "bridge",
                "space_group": sg,
                "structure_seed": s,
                "fracs": data["fracs"],
                "elements": data["elements"],
                "difficulty": float(data["n_atoms_cell"]) * d,
                "generator": "melgalvis2026",
                "cell_volume": _cell_volume(st.cell),
            }
        )
    return out
