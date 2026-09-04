"""Peak-budget trial.res writer (Mark/Bragg COD 2200001 n=34).

Fold unique-ASU, keep strongest non-H peaks, pack one discrete molecule via
covalent connectivity, write as Q with SFAC C H N O. No CrystalX typing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from grok_phase_solver.pipeline.solve import SolveResult

DEFAULT_N_NON_H_BUDGET = 34


def write_shelxl_res_budgeted(
    result: "SolveResult",
    element: str = "Q",
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    n_non_h_budget: int = DEFAULT_N_NON_H_BUDGET,
) -> str:
    """
    Build a minimal SHELXL-style .res trial model from density peaks.

    After unique-ASU fold, keep the strongest ``n_non_h_budget`` non-H peaks
    (default 34 = COD 2200001 C12H18N2O3 \u00d7 Z\u2032=2), then pack largest finite
    covalent component(s) (C\u2013C cutoff; SYMM/LATT \u00b11 cell). Written as Q
    labels (or ``element``) with simple ``SFAC C H N O`` \u2014 no CrystalX
    typing / Br/Cl/S. Raises ConnectivityAsuError on infinite polymer
    (fail closed; does not invent a fake discrete molecule).
    """
    from grok_phase_solver.physics.connectivity_asu import (
        ConnectivityAsuError,
        format_trial_res_gate,
        pack_discrete_asu,
    )
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
    try:
        peaks, cmeta = pack_discrete_asu(
            peaks,
            result.cell,
            sg,
            lattice=ins_latt,
            symm=ins_symm,
            n_non_h_budget=int(n_non_h_budget),
        )
    except ConnectivityAsuError:
        gate = format_trial_res_gate(
            sg=sg, non_h=0, finite=False, pass_=False
        )
        print(gate, flush=True)
        raise
    non_h = int(cmeta.get("n_out", len(peaks)))
    gate = format_trial_res_gate(sg=sg, non_h=non_h, finite=True, pass_=True)
    print(gate, flush=True)
    lines = [
        f"TITL gps-solve trial ({result.method})",
        f"CELL 0.71073 {a:.4f} {b:.4f} {c:.4f} {al:.2f} {be:.2f} {ga:.2f}",
        f"ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
    ]
    lines.extend(format_shelx_latt_symm_lines(sg, lattice=ins_latt, symm=ins_symm))
    lines.extend(
        [
            f"SFAC C H N O",
            f"UNIT 1 1 1 1",
            f"FVAR 1.0",
            f"REM free_fom_composite={result.diagnostics.get('free_fom_composite', 'n/a')}",
            f"REM method={result.method} n_peaks={len(peaks)}",
            f"REM unique_asu n_in={umeta.get('n_in')} n_out={umeta.get('n_out')} n_ops={umeta.get('n_ops')}",
            f"REM peak_budget n={int(n_non_h_budget)} n_in={bmeta.get('n_in')} n_out={bmeta.get('n_out')}",
            f"REM connectivity_asu n_in={cmeta.get('n_in')} n_out={cmeta.get('n_out')} "
            f"n_components={cmeta.get('n_components')}",
            f"REM gate {gate.split(' ', 1)[1]}",
            f"REM space_group_hint={sg}",
        ]
    )
    for i, p in enumerate(peaks):
        label = f"Q{i+1}" if element.upper() == "Q" else f"{element}{i+1}"
        u = max(0.02, 0.08 / max(p.height_sigma / 3.0, 0.5))
        lines.append(
            f"{label:6s} 1 {p.fract[0]:10.6f} {p.fract[1]:10.6f} {p.fract[2]:10.6f} "
            f"11.00000 {u:.5f}"
        )
    lines.append("HKLF 4")
    lines.append("END")
    return "\n".join(lines) + "\n"
