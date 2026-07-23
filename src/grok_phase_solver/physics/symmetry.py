"""
Space-group symmetry helpers via gemmi (general SG beyond P1/P−1 special cases).

Capabilities
------------
- Parse HM / Hall names → gemmi.SpaceGroup
- Centrosymmetry / chiral / enantiomorphic flags
- Expand asymmetric-unit fractional coords to unit-cell atoms
- Apply centrosymmetric phase constraints (0/π) when appropriate
- Systematic absence filtering (thin wrapper)
- Origin-choice notes for mapCC (origin-invariant metrics remain preferred)

Honest limits
-------------
- Full reciprocal-space symmetry averaging of noisy Fobs is not a drop-in
  SHELXL MERGE replacement; this module supports **physics constraints** and
  fragment expansion for seeding.
- Origin ambiguity in non-P1 groups: use origin-invariant mapCC / free FOM.

References: International Tables; gemmi SpaceGroup / GroupOps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SpaceGroupInfo:
    """Normalized space-group metadata for pipeline diagnostics."""

    hm: str
    number: Optional[int] = None
    hall: Optional[str] = None
    is_centrosymmetric: bool = False
    is_chiral: bool = False
    n_sym_ops: int = 1
    crystal_system: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    available: bool = True  # False if gemmi missing / parse failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hm": self.hm,
            "number": self.number,
            "hall": self.hall,
            "is_centrosymmetric": self.is_centrosymmetric,
            "is_chiral": self.is_chiral,
            "n_sym_ops": self.n_sym_ops,
            "crystal_system": self.crystal_system,
            "notes": list(self.notes),
            "available": self.available,
        }


def gemmi_available() -> bool:
    try:
        import gemmi  # noqa: F401

        return True
    except Exception:
        return False


def parse_space_group(name: Optional[str]) -> SpaceGroupInfo:
    """
    Parse a space-group string (HM preferred). Fallback: P1 info without gemmi.
    """
    raw = (name or "P 1").strip() or "P 1"
    if not gemmi_available():
        return SpaceGroupInfo(
            hm=raw,
            is_centrosymmetric=_heuristic_centro(raw),
            n_sym_ops=1,
            available=False,
            notes=["gemmi unavailable; using heuristic centro flag only"],
        )
    try:
        import gemmi

        sg = gemmi.SpaceGroup(raw)
        ops = sg.operations()
        info = SpaceGroupInfo(
            hm=sg.xhm() if hasattr(sg, "xhm") else raw,
            number=int(sg.number) if hasattr(sg, "number") else None,
            hall=sg.hall if hasattr(sg, "hall") else None,
            is_centrosymmetric=bool(sg.is_centrosymmetric()),
            is_chiral=bool(getattr(sg, "is_chiral", lambda: False)()),
            n_sym_ops=int(len(ops)),
            crystal_system=str(sg.crystal_system_str())
            if hasattr(sg, "crystal_system_str")
            else None,
            available=True,
        )
        if info.is_centrosymmetric:
            info.notes.append("centrosymmetric: phases ideally 0/π after origin choice")
        return info
    except Exception as e:
        return SpaceGroupInfo(
            hm=raw,
            is_centrosymmetric=_heuristic_centro(raw),
            available=False,
            notes=[f"parse failed ({e}); heuristic fallback"],
        )


def _heuristic_centro(name: str) -> bool:
    s = name.replace(" ", "").upper()
    # Common centro markers: bar, /c, /n, /a, /m, P-1, etc.
    if "P-1" in s or "P1-" in s or s in ("P-1", "P  -1"):
        return True
    if "/" in s:  # P21/c, C2/c, ...
        return True
    if "BAR" in s or "-1" in s or "-3" in s or "-4" in s or "-6" in s:
        return True
    return False


def is_centrosymmetric(space_group: Optional[str]) -> bool:
    return parse_space_group(space_group).is_centrosymmetric


def expand_fractional_coords(
    fracs: np.ndarray,
    space_group: Optional[str],
    *,
    elements: Optional[Sequence[str]] = None,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Expand asymmetric-unit atoms to the full unit cell via space-group ops.

    Returns (fracs_expanded, elements_expanded, meta).
    If gemmi fails, returns input unchanged.
    """
    fracs = np.asarray(fracs, dtype=np.float64).reshape(-1, 3)
    n0 = len(fracs)
    els = list(elements) if elements is not None else ["C"] * n0
    if len(els) != n0:
        els = (els + ["C"] * n0)[:n0]

    info = parse_space_group(space_group)
    if not info.available or info.n_sym_ops <= 1:
        return fracs.copy(), list(els), {
            "expanded": False,
            "n_in": n0,
            "n_out": n0,
            "sg": info.to_dict(),
        }

    try:
        import gemmi

        sg = gemmi.SpaceGroup(space_group or "P 1")
        ops = sg.operations()
        out_f: List[np.ndarray] = []
        out_e: List[str] = []
        for i, xyz in enumerate(fracs):
            el = els[i]
            for op in ops:
                # gemmi apply on fractional
                p = op.apply_to_xyz([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                f = np.array(p, dtype=np.float64) % 1.0
                # de-duplicate
                if out_f:
                    d = np.linalg.norm(np.vstack(out_f) - f, axis=1)
                    if np.any(d < tol):
                        continue
                out_f.append(f)
                out_e.append(el)
        if not out_f:
            return fracs.copy(), list(els), {
                "expanded": False,
                "n_in": n0,
                "n_out": n0,
                "sg": info.to_dict(),
            }
        return np.vstack(out_f), out_e, {
            "expanded": True,
            "n_in": n0,
            "n_out": len(out_f),
            "n_ops": info.n_sym_ops,
            "sg": info.to_dict(),
        }
    except Exception as e:
        return fracs.copy(), list(els), {
            "expanded": False,
            "error": str(e),
            "n_in": n0,
            "n_out": n0,
            "sg": info.to_dict(),
        }


def apply_centro_phase_constraint(
    phases: np.ndarray,
    space_group: Optional[str],
    *,
    force: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Snap phases to {0, π} when the space group is centrosymmetric.

    ``force=True`` applies even if SG parse is uncertain but heuristic says centro.
    """
    info = parse_space_group(space_group)
    ph = np.asarray(phases, dtype=np.float64).copy()
    if not (info.is_centrosymmetric or force):
        return ph, {"applied": False, "sg": info.to_dict()}
    c = np.cos(ph)
    out = np.where(c >= 0.0, 0.0, np.pi)
    return out, {
        "applied": True,
        "sg": info.to_dict(),
        "frac_zero": float(np.mean(out == 0.0)),
    }


def filter_systematic_absences(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    space_group: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Remove reflections that are systematically absent for the space group.

    Returns filtered (hkl, amplitudes, meta). On failure, returns inputs.
    """
    hkl = np.asarray(hkl, dtype=int).reshape(-1, 3)
    amp = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if len(hkl) != len(amp):
        raise ValueError("hkl/amplitudes length mismatch")
    if not space_group or not gemmi_available():
        return hkl, amp, {"filtered": False, "n_removed": 0}
    try:
        import gemmi

        sg = gemmi.SpaceGroup(space_group)
        ops = sg.operations()
        keep = []
        for i, (h, k, l) in enumerate(hkl):
            # gemmi: is_systematically_absent on GroupOps
            if hasattr(ops, "is_systematically_absent"):
                absent = ops.is_systematically_absent(gemmi.Miller(int(h), int(k), int(l)))
            else:
                absent = False
            if not absent:
                keep.append(i)
        if len(keep) == len(hkl):
            return hkl, amp, {"filtered": True, "n_removed": 0}
        idx = np.asarray(keep, dtype=int)
        return hkl[idx], amp[idx], {
            "filtered": True,
            "n_removed": int(len(hkl) - len(idx)),
            "n_kept": int(len(idx)),
        }
    except Exception as e:
        return hkl, amp, {"filtered": False, "error": str(e), "n_removed": 0}


def space_group_diagnostics(space_group: Optional[str]) -> Dict[str, Any]:
    """Compact dict for report.md / free-FOM context."""
    info = parse_space_group(space_group)
    d = info.to_dict()
    d["gemmi"] = gemmi_available()
    return d
