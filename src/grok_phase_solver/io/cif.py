"""
CIF reader / writer helpers.

Uses gemmi for robust parsing of small-molecule COD CIFs. Falls back to a
minimal pure-Python path only where gemmi is unavailable (not recommended).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def parse_cif_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    """
    Parse CIF numeric values that may include uncertainty, e.g. ``0.84040(10)``.
    """
    if value is None or value in ("?", ".", ""):
        return default
    s = str(value).strip()
    # Strip parenthetical uncertainty: 1.234(5) → 1.234
    if "(" in s:
        s = s.split("(", 1)[0]
    try:
        return float(s)
    except ValueError:
        return default


@dataclass
class AtomSite:
    """Asymmetric-unit atom site in fractional coordinates."""

    label: str
    element: str
    fract: np.ndarray  # shape (3,), fractional x,y,z
    occupancy: float = 1.0
    u_iso: float = 0.05  # Å² isotropic displacement (U_iso)
    b_iso: Optional[float] = None  # B = 8 π² U if set

    def __post_init__(self) -> None:
        self.fract = np.asarray(self.fract, dtype=np.float64).reshape(3)
        if self.b_iso is None:
            self.b_iso = 8.0 * np.pi**2 * self.u_iso
        else:
            self.u_iso = self.b_iso / (8.0 * np.pi**2)


@dataclass
class CrystalStructure:
    """Minimal crystal model used throughout the framework."""

    name: str
    cell: np.ndarray  # (6,) a,b,c,alpha,beta,gamma in Å / degrees
    space_group_hm: str
    atoms: List[AtomSite] = field(default_factory=list)
    z: int = 1
    wavelength: float = 0.71073  # Mo Kα default (Å)
    source_path: Optional[str] = None
    space_group_number: Optional[int] = None

    @property
    def volume(self) -> float:
        """Unit-cell volume in Å³."""
        a, b, c, al, be, ga = self.cell
        al, be, ga = np.deg2rad([al, be, ga])
        cos_al, cos_be, cos_ga = np.cos(al), np.cos(be), np.cos(ga)
        sin_al, sin_be, sin_ga = np.sin(al), np.sin(be), np.sin(ga)
        v2 = (
            1.0
            - cos_al**2
            - cos_be**2
            - cos_ga**2
            + 2.0 * cos_al * cos_be * cos_ga
        )
        return float(a * b * c * np.sqrt(max(v2, 0.0)))

    @property
    def orth_matrix(self) -> np.ndarray:
        """Fractional → Cartesian (Å) transformation matrix (rows = basis vectors)."""
        a, b, c, al, be, ga = self.cell
        al, be, ga = np.deg2rad([al, be, ga])
        cos_al, cos_be, cos_ga = np.cos(al), np.cos(be), np.cos(ga)
        sin_ga = np.sin(ga)
        v = self.volume
        # Standard orthogonalization (International Tables)
        m = np.zeros((3, 3), dtype=np.float64)
        m[0, 0] = a
        m[0, 1] = b * cos_ga
        m[0, 2] = c * cos_be
        m[1, 1] = b * sin_ga
        m[1, 2] = c * (cos_al - cos_be * cos_ga) / sin_ga
        m[2, 2] = v / (a * b * sin_ga)
        return m

    def fractional_coords(self) -> np.ndarray:
        """(N, 3) fractional coordinates of asymmetric unit."""
        if not self.atoms:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack([a.fract for a in self.atoms])

    def elements(self) -> List[str]:
        return [a.element for a in self.atoms]

    def b_factors(self) -> np.ndarray:
        return np.array([a.b_iso if a.b_iso is not None else 0.0 for a in self.atoms])

    def occupancies(self) -> np.ndarray:
        return np.array([a.occupancy for a in self.atoms])

    def summary(self) -> str:
        return (
            f"CrystalStructure({self.name!r}, SG={self.space_group_hm!r}, "
            f"cell={self.cell.tolist()}, Z={self.z}, natoms={len(self.atoms)}, "
            f"V={self.volume:.2f} Å³)"
        )


def load_cif(path: PathLike) -> CrystalStructure:
    """
    Load a small-molecule CIF (COD-style) into a :class:`CrystalStructure`.

    Prefer gemmi's small-structure path, which correctly handles
    ``_atom_site_*`` without requiring a PDB-like model hierarchy.
    """
    path = Path(path)
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            "gemmi is required for CIF I/O. Install with: pip install gemmi"
        ) from exc

    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()
    small = gemmi.make_small_structure_from_block(block)

    cell = small.cell
    cell_arr = np.array(
        [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        dtype=np.float64,
    )

    sg_hm = small.spacegroup_hm or block.find_value("_symmetry_space_group_name_H-M") or "P 1"
    sg_num_raw = block.find_value("_space_group_IT_number")
    sg_num_f = parse_cif_float(sg_num_raw)
    sg_num = int(sg_num_f) if sg_num_f is not None else None

    z_raw = block.find_value("_cell_formula_units_Z")
    z_f = parse_cif_float(z_raw, 1.0)
    z = int(z_f) if z_f is not None else 1

    wl_raw = block.find_value("_diffrn_radiation_wavelength")
    wavelength = parse_cif_float(wl_raw, 0.71073) or 0.71073

    atoms: List[AtomSite] = []
    for site in small.sites:
        element = site.type_symbol or site.element.name
        # Prefer anisotropic/isotropic U from gemmi if present
        u_iso = 0.05
        b_iso = None
        if hasattr(site, "u_iso") and site.u_iso is not None and site.u_iso > 0:
            u_iso = float(site.u_iso)
        occ = float(site.occ) if site.occ is not None else 1.0
        atoms.append(
            AtomSite(
                label=site.label or element,
                element=element.strip(),
                fract=np.array([site.fract.x, site.fract.y, site.fract.z], dtype=np.float64),
                occupancy=occ,
                u_iso=u_iso,
                b_iso=b_iso,
            )
        )

    name = path.stem
    data_name = block.name
    if data_name:
        name = data_name

    return CrystalStructure(
        name=name,
        cell=cell_arr,
        space_group_hm=sg_hm.replace(" ", "") if " " in sg_hm else sg_hm,
        atoms=atoms,
        z=z,
        wavelength=wavelength,
        source_path=str(path.resolve()),
        space_group_number=sg_num,
    )


def save_atoms_xyz(
    path: PathLike,
    atoms: Sequence[AtomSite],
    cell: Optional[np.ndarray] = None,
    comment: str = "grok_phase_solver",
) -> None:
    """Write asymmetric-unit atoms as XYZ (Cartesian if cell given, else fractional*10)."""
    path = Path(path)
    lines = [str(len(atoms)), comment]
    if cell is not None:
        tmp = CrystalStructure("tmp", np.asarray(cell, dtype=np.float64), "P1", list(atoms))
        M = tmp.orth_matrix
        for a in atoms:
            xyz = M @ a.fract
            lines.append(f"{a.element:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}")
    else:
        for a in atoms:
            lines.append(
                f"{a.element:2s} {a.fract[0]*10:12.6f} {a.fract[1]*10:12.6f} {a.fract[2]*10:12.6f}"
            )
    path.write_text("\n".join(lines) + "\n")


def expand_asymmetric_unit(
    structure: CrystalStructure,
    include_identity_only: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Expand asymmetric unit to full unit cell using space-group operators.

    Returns
    -------
    fracs : (N, 3) fractional coordinates in [0, 1)
    elements : list of element symbols
    b_isos : (N,) B-factors
    occs : (N,) occupancies
    """
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError("gemmi required for symmetry expansion") from exc

    sg = gemmi.SpaceGroup(structure.space_group_hm)
    ops = sg.operations()
    if include_identity_only:
        ops = [gemmi.Op()]

    fracs: List[np.ndarray] = []
    elements: List[str] = []
    b_isos: List[float] = []
    occs: List[float] = []

    seen = set()
    for atom in structure.atoms:
        for op in ops:
            # gemmi Op.apply_to_xyz expects list/tuple
            x, y, z = op.apply_to_xyz(atom.fract.tolist())
            f = np.array([x % 1.0, y % 1.0, z % 1.0], dtype=np.float64)
            # Deduplicate on rounded fractional coords
            key = (atom.element, round(f[0], 5), round(f[1], 5), round(f[2], 5))
            if key in seen:
                continue
            seen.add(key)
            fracs.append(f)
            elements.append(atom.element)
            b_isos.append(float(atom.b_iso if atom.b_iso is not None else 0.0))
            occs.append(float(atom.occupancy))

    if not fracs:
        return (
            np.zeros((0, 3)),
            [],
            np.zeros(0),
            np.zeros(0),
        )
    return np.vstack(fracs), elements, np.array(b_isos), np.array(occs)
