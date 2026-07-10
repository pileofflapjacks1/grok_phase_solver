"""
Reflection table I/O (.hkl CIF from COD, SHELX-style free-format, simple text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class ReflectionTable:
    """
    Miller indices and associated amplitude/intensity data.

    Phases are optional (radians). When only intensities are known, set
    ``F_meas`` from sqrt(I) where I > 0.
    """

    hkl: np.ndarray  # (N, 3) int
    F_meas: Optional[np.ndarray] = None  # measured |F|
    sigF: Optional[np.ndarray] = None
    I_meas: Optional[np.ndarray] = None
    sigI: Optional[np.ndarray] = None
    F_calc: Optional[np.ndarray] = None  # complex calculated F
    phase: Optional[np.ndarray] = None  # radians
    free_flag: Optional[np.ndarray] = None
    cell: Optional[np.ndarray] = None  # (6,) if known
    space_group_hm: Optional[str] = None
    wavelength: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hkl = np.asarray(self.hkl, dtype=np.int32).reshape(-1, 3)
        n = len(self.hkl)
        for name in ("F_meas", "sigF", "I_meas", "sigI", "phase", "free_flag"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr).reshape(n))
        if self.F_calc is not None:
            self.F_calc = np.asarray(self.F_calc, dtype=np.complex128).reshape(n)

    def __len__(self) -> int:
        return int(self.hkl.shape[0])

    @property
    def amplitudes(self) -> np.ndarray:
        """Best available |F| estimate."""
        if self.F_meas is not None:
            return np.asarray(self.F_meas, dtype=np.float64)
        if self.F_calc is not None:
            return np.abs(self.F_calc)
        if self.I_meas is not None:
            I = np.asarray(self.I_meas, dtype=np.float64)
            return np.sqrt(np.maximum(I, 0.0))
        raise ValueError("No amplitude or intensity data available")

    def resolution_d(self) -> np.ndarray:
        """d-spacings (Å) for each reflection; requires cell."""
        if self.cell is None:
            raise ValueError("cell parameters required for resolution")
        from grok_phase_solver.physics.reciprocal import d_spacing

        return d_spacing(self.hkl, self.cell)

    def filter_resolution(
        self,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
    ) -> "ReflectionTable":
        """Keep reflections with d_min ≤ d ≤ d_max (Å)."""
        d = self.resolution_d()
        mask = np.ones(len(self), dtype=bool)
        if d_min is not None:
            mask &= d >= d_min
        if d_max is not None:
            mask &= d <= d_max
        return self.subset(mask)

    def subset(self, mask: np.ndarray) -> "ReflectionTable":
        mask = np.asarray(mask, dtype=bool)
        def take(x):
            return None if x is None else x[mask]

        return ReflectionTable(
            hkl=self.hkl[mask],
            F_meas=take(self.F_meas),
            sigF=take(self.sigF),
            I_meas=take(self.I_meas),
            sigI=take(self.sigI),
            F_calc=take(self.F_calc),
            phase=take(self.phase),
            free_flag=take(self.free_flag),
            cell=self.cell,
            space_group_hm=self.space_group_hm,
            wavelength=self.wavelength,
            meta=dict(self.meta),
        )

    def remove_systematic_absences(self, space_group_hm: Optional[str] = None) -> "ReflectionTable":
        """Filter reflections that are systematically absent."""
        try:
            import gemmi
        except ImportError:
            return self
        sg_name = space_group_hm or self.space_group_hm
        if not sg_name:
            return self
        sg = gemmi.SpaceGroup(sg_name)
        ops = sg.operations()
        keep = []
        for h, k, l in self.hkl:
            keep.append(ops.is_systematically_absent([int(h), int(k), int(l)]) is False)
        return self.subset(np.array(keep, dtype=bool))


def load_hkl_cif(path: PathLike) -> ReflectionTable:
    """
    Load COD-style structure-factor CIF (``_refln_*`` loops).

    Supports F_meas, F_squared_meas, etc.
    """
    path = Path(path)
    import gemmi

    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()

    def col(tag: str):
        col = block.find_values(tag)
        if col is None or len(col) == 0:
            # try without leading underscore variants already handled by gemmi
            return None
        return col

    # Miller indices are required
    h = block.find_values("_refln_index_h")
    k = block.find_values("_refln_index_k")
    l = block.find_values("_refln_index_l")
    if h is None or len(h) == 0:
        raise ValueError(f"No _refln_index_h/k/l in {path}")

    n = len(h)
    hkl = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        hkl[i] = (int(float(h[i])), int(float(k[i])), int(float(l[i])))

    def _parse_num(v: str) -> float:
        if v in ("?", ".", ""):
            return np.nan
        s = str(v).strip()
        if "(" in s:
            s = s.split("(", 1)[0]
        try:
            return float(s)
        except ValueError:
            return np.nan

    def float_col(*tags: str) -> Optional[np.ndarray]:
        for tag in tags:
            c = block.find_values(tag)
            if c is not None and len(c) == n:
                out = np.zeros(n, dtype=np.float64)
                for i in range(n):
                    out[i] = _parse_num(c[i])
                return out
        return None

    F_meas = float_col("_refln_F_meas", "_refln_F_meas_au")
    sigF = float_col("_refln_F_sigma", "_refln_F_meas_sigma")
    I_meas = float_col(
        "_refln_F_squared_meas",
        "_refln_intensity_meas",
        "_refln_F_squared_calc",  # last resort if only calc present
    )
    sigI = float_col("_refln_F_squared_sigma", "_refln_intensity_sigma")
    phase_deg = float_col("_refln_phase_calc", "_refln_phase_meas")

    # If only I present, convert positive I → |F|
    if F_meas is None and I_meas is not None:
        F_meas = np.sqrt(np.maximum(I_meas, 0.0))
        # Prefer measured F^2 if available under different tag
        I_true = float_col("_refln_F_squared_meas", "_refln_intensity_meas")
        if I_true is not None:
            I_meas = I_true
            F_meas = np.sqrt(np.maximum(I_true, 0.0))

    phase = None
    if phase_deg is not None:
        phase = np.deg2rad(phase_deg)

    # Cell from block if present
    cell = None
    try:
        a = block.find_value("_cell_length_a")
        b = block.find_value("_cell_length_b")
        c = block.find_value("_cell_length_c")
        al = block.find_value("_cell_angle_alpha") or "90"
        be = block.find_value("_cell_angle_beta") or "90"
        ga = block.find_value("_cell_angle_gamma") or "90"
        if a and b and c:
            cell = np.array(
                [float(a), float(b), float(c), float(al), float(be), float(ga)],
                dtype=np.float64,
            )
    except (TypeError, ValueError):
        cell = None

    sg = block.find_value("_symmetry_space_group_name_H-M")
    wl = block.find_value("_diffrn_radiation_wavelength")
    wavelength = float(wl) if wl not in (None, "?", ".") else None

    # Drop (000) and NaN amplitudes
    mask = np.ones(n, dtype=bool)
    mask &= ~((hkl[:, 0] == 0) & (hkl[:, 1] == 0) & (hkl[:, 2] == 0))
    if F_meas is not None:
        mask &= np.isfinite(F_meas)
        mask &= F_meas >= 0

    def apply(x):
        return None if x is None else x[mask]

    return ReflectionTable(
        hkl=hkl[mask],
        F_meas=apply(F_meas),
        sigF=apply(sigF),
        I_meas=apply(I_meas),
        sigI=apply(sigI),
        phase=apply(phase),
        cell=cell,
        space_group_hm=sg,
        wavelength=wavelength,
        meta={"source": str(path.resolve()), "block": block.name},
    )


def load_hkl_shelx(path: PathLike, cell: Optional[np.ndarray] = None) -> ReflectionTable:
    """
    Load free-format SHELX-style HKL: h k l I sigI [batch] per line.
    Stops at 0 0 0.
    """
    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            try:
                h, k, l = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
            except ValueError:
                continue
            if h == 0 and k == 0 and l == 0:
                break
            I = float(parts[3])
            sig = float(parts[4]) if len(parts) > 4 else np.nan
            rows.append((h, k, l, I, sig))
    if not rows:
        raise ValueError(f"No reflections parsed from {path}")
    arr = np.array(rows, dtype=np.float64)
    hkl = arr[:, :3].astype(np.int32)
    I_meas = arr[:, 3]
    sigI = arr[:, 4]
    F_meas = np.sqrt(np.maximum(I_meas, 0.0))
    return ReflectionTable(
        hkl=hkl,
        F_meas=F_meas,
        I_meas=I_meas,
        sigI=sigI,
        cell=cell,
        meta={"source": str(path.resolve()), "format": "shelx"},
    )


def write_hkl_simple(path: PathLike, table: ReflectionTable, use_intensity: bool = False) -> None:
    """Write simple text: h k l |F| [sigF] or h k l I [sigI]."""
    path = Path(path)
    amp = table.amplitudes
    lines = []
    for i in range(len(table)):
        h, k, l = table.hkl[i]
        if use_intensity and table.I_meas is not None:
            lines.append(f"{h:4d} {k:4d} {l:4d} {table.I_meas[i]:12.4f}")
        else:
            lines.append(f"{h:4d} {k:4d} {l:4d} {amp[i]:12.4f}")
    lines.append("   0    0    0")
    path.write_text("\n".join(lines) + "\n")
