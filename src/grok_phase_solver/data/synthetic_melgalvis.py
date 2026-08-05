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
    """Controls Melgalvis-style synthetic realism (v0.7 curriculum extensions)."""

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
    # v0.7 curriculum: heavy atoms, partial occupancy, protein-like volumes
    p_heavy_atom: float = 0.18  # inject Br/Cl/S/P as HA-like
    heavy_elements: Tuple[str, ...] = ("BR", "CL", "S", "P", "I")
    p_partial_occupancy: float = 0.12  # random atoms get occ ∈ [0.4, 0.9]
    partial_occ_lo: float = 0.40
    partial_occ_hi: float = 0.90
    # Match experimental COD volume distribution more tightly (optional preset)
    cod_like_volumes: bool = False
    # Larger ASU for hard curriculum (GraPhAI / low-res panels)
    p_large_molecule: float = 0.15  # chance to sample n_nonh toward hi end ×1.5
    n_nonh_hard_cap: int = 48
    # v0.9: multi-fragment packing + Acta 2026-style denser COD volume band
    p_multi_fragment: float = 0.18  # chance to pack 2 independent clusters
    multi_frag_n_extra: Tuple[int, int] = (3, 10)
    # Tighter axis ratios for more realistic packing (artificial structure gen)
    prefer_realistic_angles: bool = True
    # v0.11: database-guided ring / functional-group scaffolds + packing quality
    p_ring_fragment: float = 0.22  # chance to seed from a small ring scaffold
    min_contact_frac: float = 0.88  # clash threshold vs covalent sum (packing)
    void_check: bool = True  # reject packs with large empty voids (centroid gap)
    max_void_frac: float = 0.55  # max empty volume fraction before re-pack
    # v0.12: experimental realism degradations (synth→exp gap)
    p_b_factor_inflate: float = 0.15  # chance to inflate U_iso (radiation-damage-ish)
    b_inflate_lo: float = 1.4
    b_inflate_hi: float = 2.8
    p_amp_noise: float = 0.0  # applied in iter_melgalvis_samples if >0
    amp_noise_frac: float = 0.04  # relative Gaussian noise on |F|
    # v0.13: intermolecular contact density (Acta 2026 packing realism)
    enforce_intermol_contacts: bool = True
    target_contacts_per_nonh: float = 1.2  # min nonbond neighbors in 2.5–4.0 Å
    p_solvent_void: float = 0.08  # chance to leave a larger solvent channel


def _sample_weighted(rng: np.random.Generator, freq: Dict[str, float]) -> str:
    keys = list(freq.keys())
    p = np.array([freq[k] for k in keys], dtype=np.float64)
    p = p / p.sum()
    return str(rng.choice(keys, p=p))


def cod_like_config(**overrides) -> MelgalvisGenConfig:
    """
    Preset fitted more tightly to COD-like organic small-molecule volumes.

    Melgalvis & Rekis (2026) emphasize matching experimental volume distributions
    to reduce domain gap for DL phasing. Median ~600 Å³, tail to ~4000 Å³.
    """
    base = dict(
        log_v_mu=float(np.log(600.0)),
        log_v_sigma=0.65,
        v_min=150.0,
        v_max=4500.0,
        vol_per_nonh_lo=8.0,
        vol_per_nonh_hi=20.0,
        n_nonh_lo=8,
        n_nonh_hi=32,
        p_special_seed=0.15,
        p_heavy_atom=0.22,
        p_partial_occupancy=0.10,
        cod_like_volumes=True,
        name_prefix="melg_cod",
        mode="hybrid",
        hybrid_cluster_frac=0.75,
    )
    base.update(overrides)
    return MelgalvisGenConfig(**base)


def hard_curriculum_config(**overrides) -> MelgalvisGenConfig:
    """Hard-region / larger-Z curriculum (low-res friendly volumes)."""
    base = dict(
        log_v_mu=float(np.log(1400.0)),
        log_v_sigma=0.48,
        v_min=500.0,
        v_max=5000.0,
        n_nonh_lo=14,
        n_nonh_hi=44,
        p_heavy_atom=0.30,
        p_partial_occupancy=0.15,
        p_large_molecule=0.38,
        p_special_seed=0.18,
        p_multi_fragment=0.28,
        p_ring_fragment=0.30,
        prefer_realistic_angles=True,
        name_prefix="melg_hard",
        mode="hybrid",
    )
    base.update(overrides)
    return MelgalvisGenConfig(**base)


def ha_heavy_config(**overrides) -> MelgalvisGenConfig:
    """
    Heavy-atom / metal-organic curriculum (GraPhAI Z≥19 success regime).

    Emphasizes Br/Cl/I/S injection and mid-large volumes for centrosymmetric
    HA-friendly training of graph priors. v0.11: guarantee Z≥19 path via
    forced Br/I and larger cells for HA-stratified scoreboards.
    """
    base = dict(
        log_v_mu=float(np.log(1100.0)),
        log_v_sigma=0.52,
        v_min=280.0,
        v_max=4800.0,
        n_nonh_lo=12,
        n_nonh_hi=40,
        p_heavy_atom=0.70,
        heavy_elements=("BR", "I", "CL", "S", "P"),
        p_partial_occupancy=0.10,
        p_special_seed=0.22,
        p_multi_fragment=0.28,
        p_ring_fragment=0.28,
        prefer_realistic_angles=True,
        cod_like_volumes=True,
        name_prefix="melg_ha",
        mode="hybrid",
        hybrid_cluster_frac=0.82,
    )
    base.update(overrides)
    return MelgalvisGenConfig(**base)


def actas2026_config(**overrides) -> MelgalvisGenConfig:
    """
    Curriculum tuned toward improved artificial structure generation (2026).

    Emphasizes COD-like volumes, multi-fragment packing, and HA injection for
    better domain match when training graph priors / PhAI-like models.
    v0.11: denser large-cell tail (Vol up to ~3500–4200 Å³) + ring fragments.
    """
    base = dict(
        log_v_mu=float(np.log(850.0)),
        log_v_sigma=0.60,
        v_min=180.0,
        v_max=4200.0,
        n_nonh_lo=8,
        n_nonh_hi=40,
        p_heavy_atom=0.26,
        p_partial_occupancy=0.12,
        p_multi_fragment=0.30,
        p_ring_fragment=0.35,
        prefer_realistic_angles=True,
        cod_like_volumes=True,
        name_prefix="melg_acta2026",
        mode="hybrid",
        hybrid_cluster_frac=0.80,
    )
    base.update(overrides)
    return MelgalvisGenConfig(**base)


def xdxd_lowres_config(**overrides) -> MelgalvisGenConfig:
    """
    Low-resolution / larger-cell curriculum (XDXD-inspired training domain).

    Emphasizes Vol ~1500–3500 Å³, lower d_min bands, multi-fragment packing —
    for generative coordinate proposal and hard AI-PhaSeed panels.
    """
    base = dict(
        log_v_mu=float(np.log(2200.0)),
        log_v_sigma=0.38,
        v_min=1200.0,
        v_max=3800.0,
        n_nonh_lo=18,
        n_nonh_hi=52,
        n_nonh_hard_cap=60,
        p_large_molecule=0.50,
        p_heavy_atom=0.35,
        p_multi_fragment=0.45,
        p_ring_fragment=0.40,
        p_b_factor_inflate=0.22,
        p_amp_noise=0.15,
        amp_noise_frac=0.05,
        prefer_realistic_angles=True,
        cod_like_volumes=True,
        enforce_intermol_contacts=True,
        name_prefix="melg_xdxd",
        mode="hybrid",
        hybrid_cluster_frac=0.88,
    )
    base.update(overrides)
    return MelgalvisGenConfig(**base)


def large_cell_config(**overrides) -> MelgalvisGenConfig:
    """
    Large-cell curriculum (Vol ~1000–3500 Å³, Z≥19 HA-friendly).

    Targets Carrozzini / AI-PhaSeed hybrid-friendly volume band and
    Melgalvis/Rekis larger-cell generalization. Prefer multi-fragment
    packing with ring scaffolds and elevated HA rate.
    """
    base = dict(
        log_v_mu=float(np.log(2000.0)),
        log_v_sigma=0.40,
        v_min=900.0,
        v_max=3500.0,
        n_nonh_lo=16,
        n_nonh_hi=48,
        n_nonh_hard_cap=56,
        p_large_molecule=0.45,
        p_heavy_atom=0.42,
        heavy_elements=("BR", "I", "CL", "S", "P"),
        p_partial_occupancy=0.12,
        p_multi_fragment=0.40,
        multi_frag_n_extra=(5, 14),
        p_ring_fragment=0.45,
        p_special_seed=0.18,
        prefer_realistic_angles=True,
        cod_like_volumes=True,
        vol_per_nonh_lo=9.0,
        vol_per_nonh_hi=18.0,
        name_prefix="melg_large",
        mode="hybrid",
        hybrid_cluster_frac=0.85,
        max_pack_trials=60,
    )
    base.update(overrides)
    return MelgalvisGenConfig(**base)


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


def build_ring_scaffold(
    rng: np.random.Generator,
    kind: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Small database-guided molecule-like scaffolds (COD-inspired organic motifs).

    Not a full CSD fragment library — transparent geometry priors for
    Melgalvis-style packing realism (v0.11).
    """
    kind = kind or str(
        rng.choice(["phenyl", "pyridine", "carboxyl", "imidazole", "chain"])
    )
    if kind == "phenyl":
        # planar hexagon ~1.39 Å
        els = ["C"] * 6
        ang = np.linspace(0, 2 * np.pi, 7)[:-1]
        r = 1.39
        xyz = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(6)])
    elif kind == "pyridine":
        els = ["N", "C", "C", "C", "C", "C"]
        ang = np.linspace(0, 2 * np.pi, 7)[:-1]
        r = 1.36
        xyz = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(6)])
    elif kind == "carboxyl":
        # C(=O)–OH-like flat fragment
        els = ["C", "O", "O"]
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0], [-0.65, 1.10, 0.0]],
            dtype=np.float64,
        )
    elif kind == "imidazole":
        # 5-membered N-heterocycle approx
        els = ["N", "C", "N", "C", "C"]
        ang = np.linspace(0, 2 * np.pi, 6)[:-1]
        r = 1.32
        xyz = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(5)])
    else:
        # short aliphatic chain
        n = int(rng.integers(3, 6))
        els = ["C"] * n
        xyz = np.zeros((n, 3), dtype=np.float64)
        for i in range(1, n):
            xyz[i] = xyz[i - 1] + np.array([1.54, 0.15 * (i % 2), 0.0])
    # small random rigid tilt
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-16
    ang = float(rng.uniform(0, 2 * np.pi))
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    xyz = (R @ xyz.T).T
    xyz = xyz - xyz.mean(axis=0)
    return els, xyz


def build_artificial_molecule(
    rng: np.random.Generator,
    n_nonh: int,
    cfg: MelgalvisGenConfig,
    special_seed: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Grow a bonded cluster in Cartesian Å (origin-centered).

    v0.11: optional ring / functional-group scaffold seed (database-guided),
    then grow remaining non-H atoms by covalent attachment.

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

    # Optional ring / motif scaffold (Melgalvis-style molecule-like packing)
    use_ring = (
        float(getattr(cfg, "p_ring_fragment", 0.0)) > 0
        and rng.random() < float(cfg.p_ring_fragment)
        and n_nonh >= 4
    )
    if use_ring:
        sc_els, sc_xyz = build_ring_scaffold(rng)
        for el, p in zip(sc_els, sc_xyz):
            if sum(1 for e in elements if e != "H") >= n_nonh:
                break
            add_nonh(el, p)
    if not elements:
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


def _packing_void_fraction(
    fracs: Sequence[np.ndarray],
    M: np.ndarray,
    n_probe: int = 48,
    contact_r: float = 2.2,
) -> float:
    """
    Crude empty-volume fraction via random probes (min-image).

    Used to reject packs with large voids (v0.11 Melgalvis packing quality).
    """
    if not fracs:
        return 1.0
    F = np.asarray(fracs, dtype=np.float64)
    empty = 0
    rng = np.random.default_rng(abs(hash(tuple(F[0].round(4)))) % (2**31))
    for _ in range(n_probe):
        p = rng.random(3)
        dmin = min(_min_image_cart(p, f, M) for f in F)
        if dmin > contact_r:
            empty += 1
    return float(empty) / float(n_probe)


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

    v0.11: tighter short-contact rejection + optional void-fraction check
    (avoid large empty regions / unphysical packing).
    """
    M = _orth_matrix(cell)
    # Random rotation
    from grok_phase_solver.data.synthetic_v2 import _rotation_matrix

    clash_frac = float(getattr(cfg, "min_contact_frac", 0.88))
    do_void = bool(getattr(cfg, "void_check", True))
    max_void = float(getattr(cfg, "max_void_frac", 0.55))

    for trial in range(cfg.max_pack_trials):
        R = _rotation_matrix(rng)
        xyz = (R @ cart.T).T
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
                dmin = clash_frac * (
                    _COVALENT_RADII.get(el, 0.75) + _COVALENT_RADII.get(elements[j], 0.75)
                )
                if _min_image_cart(f, f2, M) < dmin * 0.9:
                    ok = False
                    break
            if not ok:
                break
            fracs.append(f)
            el_store = el
            if el.upper() == "CL":
                el_store = "Cl"
            elif el.upper() == "BR":
                el_store = "Br"
            atoms.append(
                AtomSite(
                    label=f"{el_store}{i+1}",
                    element=el_store,
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

        # Void / empty-space rejection (large cells especially)
        if do_void and len(fracs) >= 4:
            vf = _packing_void_fraction(fracs, M)
            # optional solvent channel: allow higher voids occasionally
            thr = max_void + 0.12 * (1.0 - trial / max(cfg.max_pack_trials, 1))
            if rng.random() < float(getattr(cfg, "p_solvent_void", 0.0)):
                thr = min(0.75, thr + 0.12)
            if vf > thr:
                continue
        # v0.13: require some intermolecular contacts (packing density)
        if (
            bool(getattr(cfg, "enforce_intermol_contacts", False))
            and len(fracs) >= 6
            and trial > cfg.max_pack_trials // 3
        ):
            nonh_f = [
                fracs[i]
                for i, el in enumerate(elements)
                if str(el).upper() not in ("H", "D")
            ]
            if len(nonh_f) >= 4:
                contacts = 0
                for i in range(len(nonh_f)):
                    for j in range(i + 1, len(nonh_f)):
                        d = _min_image_cart(nonh_f[i], nonh_f[j], M)
                        if 2.5 <= d <= 4.0:
                            contacts += 1
                need = float(getattr(cfg, "target_contacts_per_nonh", 1.2)) * len(nonh_f) * 0.5
                if contacts < need * 0.35:
                    continue
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

    def _inject_ha_on_structure(st: CrystalStructure) -> CrystalStructure:
        """Apply HA injection post-generation (cluster, rejection, or fallback)."""
        if rng.random() >= float(getattr(cfg, "p_heavy_atom", 0.0)) or not st.atoms:
            return st
        heavies = list(getattr(cfg, "heavy_elements", ("BR", "CL", "S")))
        weights = []
        for h in heavies:
            hu = h.upper()
            weights.append(2.5 if hu in ("BR", "I") else 1.0)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        nonh = [a for a in st.atoms if a.element.upper() not in ("H", "D")]
        if not nonh:
            return st
        n_ha = 1 + int(rng.random() < 0.30 and len(nonh) >= 2)
        chosen = list(rng.choice(len(nonh), size=min(n_ha, len(nonh)), replace=False))
        for j in chosen:
            ha = str(rng.choice(heavies, p=w)).upper()
            if ha == "CL":
                nonh[j].element = "Cl"
            elif ha == "BR":
                nonh[j].element = "Br"
            elif ha == "I":
                nonh[j].element = "I"
            else:
                nonh[j].element = ha if len(ha) == 1 else ha.title()
            nonh[j].label = f"{nonh[j].element}ha{j+1}"
        return st

    if mode == "rejection":
        from grok_phase_solver.data.synthetic import generate_random_organic

        n_atoms = int(n_nonh or rng.integers(cfg.n_nonh_lo, cfg.n_nonh_hi + 1))
        # volume-informed cell: prefer log-normal V, then vpa (do not collapse large cells)
        V = sample_volume(rng, cfg)
        vpa = V / max(n_atoms, 1)
        # allow higher vpa for large-cell curricula (keep Vol near target)
        vpa = float(np.clip(vpa, cfg.vol_per_nonh_lo, max(cfg.vol_per_nonh_hi * 2.5, vpa)))
        st = generate_random_organic(
            n_atoms=n_atoms,
            seed=seed,
            space_group=space_group,
            volume_per_atom=vpa,
        )
        st.name = f"{cfg.name_prefix}_rej_n{n_atoms}_s{seed}"
        return _inject_ha_on_structure(st)

    # Cluster mode
    n_nonh = int(n_nonh or rng.integers(cfg.n_nonh_lo, cfg.n_nonh_hi + 1))
    if rng.random() < float(getattr(cfg, "p_large_molecule", 0.0)):
        n_nonh = int(min(cfg.n_nonh_hard_cap, int(n_nonh * 1.5) + 2))
    special = bool(rng.random() < cfg.p_special_seed)
    # Volume: prefer log-normal for COD-like / large curricula (Melgalvis volume-first)
    V_ln = sample_volume(rng, cfg)
    vpa = float(rng.uniform(cfg.vol_per_nonh_lo, cfg.vol_per_nonh_hi))
    V_den = vpa * n_nonh
    w_ln = 0.75 if getattr(cfg, "cod_like_volumes", False) else 0.5
    # large-cell band: force volume-first (Acta / Carrozzini regime)
    if cfg.v_min >= 800.0:
        w_ln = max(w_ln, 0.85)
    V = (1.0 - w_ln) * V_den + w_ln * V_ln
    V = float(np.clip(V, cfg.v_min, cfg.v_max))

    cell = sample_lattice_from_volume(rng, V, cfg)
    if getattr(cfg, "prefer_realistic_angles", False):
        # Soft nudge monoclinic β toward 90–120° if cell is monoclinic-like
        a, b, c, al, be, ga = cell
        if abs(al - 90) < 1e-6 and abs(ga - 90) < 1e-6 and abs(be - 90) > 1.0:
            be = float(np.clip(be, 95.0, 125.0))
            cell = np.array([a, b, c, al, be, ga], dtype=np.float64)
    elements, cart = build_artificial_molecule(rng, n_nonh, cfg, special_seed=special)
    # v0.9 multi-fragment: second independent cluster (Acta-style packing variety)
    if rng.random() < float(getattr(cfg, "p_multi_fragment", 0.0)):
        lo, hi = getattr(cfg, "multi_frag_n_extra", (3, 10))
        n2 = int(rng.integers(int(lo), int(hi) + 1))
        el2, cart2 = build_artificial_molecule(rng, n2, cfg, special_seed=False)
        # offset second fragment in Cartesian space
        offset = rng.normal(size=3)
        offset = 3.5 * offset / (np.linalg.norm(offset) + 1e-16)
        cart2 = cart2 + offset
        elements = list(elements) + list(el2)
        cart = np.vstack([cart, cart2])
        n_nonh = sum(1 for e in elements if e.upper() not in ("H", "D"))
    atoms = pack_molecule_in_cell(rng, elements, cart, cell, cfg, special_seed=special)
    if atoms is None:
        # fallback rejection — preserve large-cell volume when configured
        from grok_phase_solver.data.synthetic import generate_random_organic

        vpa_fb = max(vpa, V / max(n_nonh, 1))
        st = generate_random_organic(
            n_atoms=n_nonh,
            seed=seed + 1,
            space_group=space_group,
            volume_per_atom=vpa_fb,
        )
        st.name = f"{cfg.name_prefix}_fb_n{n_nonh}_s{seed}"
        return _inject_ha_on_structure(st)

    # Partial occupancy injection (domain-gap realism)
    p_occ = float(getattr(cfg, "p_partial_occupancy", 0.0))
    if p_occ > 0 and atoms:
        for a in atoms:
            if a.element.upper() in ("H", "D"):
                continue
            if rng.random() < p_occ:
                a.occupancy = float(
                    rng.uniform(
                        getattr(cfg, "partial_occ_lo", 0.4),
                        getattr(cfg, "partial_occ_hi", 0.9),
                    )
                )

    # v0.12 radiation-damage-ish B-factor inflation (subset of non-H atoms)
    if float(getattr(cfg, "p_b_factor_inflate", 0.0)) > 0 and atoms:
        for a in atoms:
            if a.element.upper() in ("H", "D"):
                continue
            if rng.random() < float(cfg.p_b_factor_inflate):
                fac = float(rng.uniform(cfg.b_inflate_lo, cfg.b_inflate_hi))
                a.u_iso = float(min(0.35, max(a.u_iso, 0.01) * fac))

    st = CrystalStructure(
        name=f"{cfg.name_prefix}_cl_n{n_nonh}_s{seed}",
        cell=cell,
        space_group_hm=space_group,
        atoms=atoms,
        z=1,
        wavelength=cfg.wavelength,
    )
    return _inject_ha_on_structure(st)


def iter_melgalvis_samples(
    n_samples: int,
    seed: int = 0,
    d_min: float = 1.2,
    cfg: Optional[MelgalvisGenConfig] = None,
    n_nonh_range: Optional[Tuple[int, int]] = None,
    d_min_range: Optional[Tuple[float, float]] = None,
    include_p_minus1: float = 0.25,
    include_low_res: float = 0.0,
    low_res_range: Tuple[float, float] = (1.8, 2.5),
    preset: Optional[str] = None,
) -> List[Dict]:
    """
    On-the-fly training samples: structures → Fcalc.

    Returns list of dicts compatible with strong_prior / training loops.

    Parameters
    ----------
    include_p_minus1 : fraction expanded to centrosymmetric P−1
    include_low_res : fraction forced into low-resolution shells (GraPhAI-like)
    preset : ``"cod"`` | ``"hard"`` | None — apply curriculum config
    """
    from grok_phase_solver.data.synthetic_v2 import make_centrosymmetric_copy
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    if cfg is None:
        if preset == "cod":
            cfg = cod_like_config()
        elif preset == "hard":
            cfg = hard_curriculum_config()
        elif preset in ("acta2026", "acta", "improved"):
            cfg = actas2026_config()
        elif preset in ("ha", "heavy", "graphai_ha"):
            cfg = ha_heavy_config()
        elif preset in ("large", "large_cell", "vol3500"):
            cfg = large_cell_config()
        elif preset in ("xdxd", "lowres", "generative"):
            cfg = xdxd_lowres_config()
        else:
            cfg = MelgalvisGenConfig()
    if n_nonh_range:
        cfg = MelgalvisGenConfig(**{**cfg.__dict__, "n_nonh_lo": n_nonh_range[0], "n_nonh_hi": n_nonh_range[1]})
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    n_ha = 0
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
        if include_low_res > 0 and rng.random() < include_low_res:
            d = float(rng.uniform(*low_res_range))
        data = structure_to_fcalc(st, d_min=d)
        amp = np.asarray(data["amplitudes"], dtype=np.float64)
        # Optional relative amplitude noise (experimental realism; phases untouched)
        p_noise = float(getattr(cfg, "p_amp_noise", 0.0))
        if p_noise > 0 and rng.random() < p_noise:
            nf = float(getattr(cfg, "amp_noise_frac", 0.04))
            amp = amp * (1.0 + nf * rng.normal(size=amp.shape))
            amp = np.maximum(amp, 1e-8)
        els = list(data["elements"])
        has_ha = any(e.upper() in ("BR", "CL", "I", "S", "P") for e in els)
        if has_ha:
            n_ha += 1
        out.append(
            {
                "name": st.name,
                "hkl": data["hkl"],
                "amplitudes": amp,
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
                "has_heavy": has_ha,
                "centrosymmetric": sg in ("P-1", "P−1"),
            }
        )
    # attach batch meta on last element for callers that inspect out[0]
    if out:
        out[0]["_batch_meta"] = {
            "n_samples": n_samples,
            "frac_heavy": n_ha / max(n_samples, 1),
            "preset": preset,
            "include_p_minus1": include_p_minus1,
        }
    return out
