"""Q-peaks .res for Olex2 hand-build / peak picking (Mark/Bragg).

Fold unique-ASU, keep strongest non-H peaks (default n=34), write as Q1…Qn
with SFAC C only. Always writes the Q list — no packing fail-closed gate.
Not a SHELXL starting model; no CrystalX typing / Br/Cl/S.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from grok_phase_solver.pipeline.solve import SolveResult

DEFAULT_N_NON_H_BUDGET = 34

# Soft isotropic U for hand-build placeholders (Å²).
_SOFT_U = 0.05


def write_shelxl_res_budgeted(
    result: "SolveResult",
    element: str = "Q",
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    n_non_h_budget: int = DEFAULT_N_NON_H_BUDGET,
) -> str:
    """
    Build a non-claiming .res of Q peaks for Olex2 hand-build / peak picking.

    After unique-ASU fold, keep the strongest ``n_non_h_budget`` non-H peaks
    (default 34 = COD 2200001 C12H18N2O3 × Z′=2). Written as Q labels (or
    ``element``) with ``SFAC C`` only — no CrystalX typing / Br/Cl/S.
    LATT / SYMM come from ``result.space_group_hm`` (or an explicit .ins
    lattice/symm passthrough). Identity is omitted (SHELX convention).

    This is **not** a SHELXL starting model or refinable molecule claim.
    """
    from grok_phase_solver.physics.shelx_cards import format_shelx_latt_symm_lines
    from grok_phase_solver.physics.unique_asu import (
        DEFAULT_N_NON_H_BUDGET as _DEFAULT,
        budget_peaks,
        unique_peaks,
    )

    if n_non_h_budget is None:
        n_non_h_budget = _DEFAULT

    a, b, c, al, be, ga = result.cell
    sg = result.space_group_hm or "P1"
    ins_latt = lattice if lattice is not None else getattr(result, "shelx_lattice", None)
    ins_symm = list(symm) if symm is not None else getattr(result, "shelx_symm", None)
    peaks, umeta = unique_peaks(
        result.peaks, result.cell, sg, lattice=ins_latt, symm=ins_symm
    )
    peaks, bmeta = budget_peaks(peaks, n_non_h_budget=int(n_non_h_budget))
    lines = [
        "TITL gps-solve hand-build peaks (not a SHELXL start)",
        f"CELL 0.71073 {a:.4f} {b:.4f} {c:.4f} {al:.2f} {be:.2f} {ga:.2f}",
        "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
    ]
    lines.extend(format_shelx_latt_symm_lines(sg, lattice=ins_latt, symm=ins_symm))
    lines.extend(
        [
            "SFAC C",
            "UNIT 1",
            "FVAR 1.0",
            f"REM hand_build_peaks n={int(n_non_h_budget)} method={result.method}",
            f"REM unique_asu n_in={umeta.get('n_in')} n_out={umeta.get('n_out')} "
            f"n_ops={umeta.get('n_ops')}",
            f"REM peak_budget n={int(n_non_h_budget)} n_in={bmeta.get('n_in')} "
            f"n_out={bmeta.get('n_out')}",
            f"REM space_group_hint={sg}",
            f"REM free_fom_composite={result.diagnostics.get('free_fom_composite', 'n/a')}",
        ]
    )
    for i, p in enumerate(peaks):
        label = f"Q{i+1}" if element.upper() == "Q" else f"{element}{i+1}"
        lines.append(
            f"{label:6s} 1 {p.fract[0]:10.6f} {p.fract[1]:10.6f} {p.fract[2]:10.6f} "
            f"11.00000 {_SOFT_U:.5f}"
        )
    lines.append("HKLF 4")
    lines.append("END")
    return "\n".join(lines) + "\n"
