"""
Export phased results for crystallographers.

Writes:
  - phases.csv          h k l |F| phase_deg A B
  - structure_factors.F simple complex F list
  - density.npz         rho grid + cell
  - density.map         CCP4/MRC map (PyMOL / Coot)
  - density_slice.png   central slice (if matplotlib available)
  - peaks.xyz / peaks.csv / peaks.pdb  density peak list
  - open_in_pymol.pml / open_in_coot.sh  viewer handoff
  - report.md           human-readable summary + next steps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .solve import SolveResult

from .export_report import _render_report


def export_solution(result: "SolveResult", out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    hkl = result.hkl
    amp = result.amplitudes
    phases = result.phases
    A = amp * np.cos(phases)
    B = amp * np.sin(phases)
    phase_deg = np.rad2deg(phases)

    # CSV
    csv_path = out_dir / "phases.csv"
    with csv_path.open("w") as f:
        f.write("h,k,l,F_meas,phase_deg,A,B\n")
        for i in range(len(hkl)):
            h, k, l = hkl[i]
            f.write(
                f"{int(h)},{int(k)},{int(l)},{amp[i]:.6f},{phase_deg[i]:.4f},{A[i]:.6f},{B[i]:.6f}\n"
            )
    written.append(csv_path)

    # Simple .F file
    f_path = out_dir / "structure_factors.F"
    with f_path.open("w") as f:
        f.write("# h k l Freal Fimag  (gps-solve)\n")
        for i in range(len(hkl)):
            h, k, l = map(int, hkl[i])
            f.write(f"{h:4d} {k:4d} {l:4d} {A[i]:12.4f} {B[i]:12.4f}\n")
    written.append(f_path)

    # Density
    dens_path = out_dir / "density.npz"
    np.savez_compressed(
        dens_path,
        rho=result.density,
        cell=result.cell,
        method=result.method,
    )
    written.append(dens_path)

    # Slice plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rho = result.density
        z = rho.shape[2] // 2
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(rho[:, :, z].T, origin="lower", cmap="magma")
        ax.set_title(f"Density slice z={z} ({result.method})")
        fig.colorbar(im, ax=ax, fraction=0.046, label="\u03c1")
        fig.tight_layout()
        png = out_dir / "density_slice.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        written.append(png)
    except Exception:
        pass

    # Peaks
    if result.peaks:
        peaks_csv = out_dir / "peaks.csv"
        with peaks_csv.open("w") as f:
            f.write("rank,x_frac,y_frac,z_frac,height,height_sigma\n")
            for p in result.peaks:
                f.write(
                    f"{p.rank},{p.fract[0]:.6f},{p.fract[1]:.6f},{p.fract[2]:.6f},"
                    f"{p.height:.6g},{p.height_sigma:.3f}\n"
                )
        written.append(peaks_csv)

        from .peaks import peaks_to_xyz_lines

        xyz_path = out_dir / "peaks.xyz"
        xyz_path.write_text("\n".join(peaks_to_xyz_lines(result.peaks, result.cell)) + "\n")
        written.append(xyz_path)

        # SHELXL-style trial .res \u2014 fold + peak budget + connectivity/ASU.
        # Mark/Bragg: Q (or C) placeholders only; SFAC C H N O; keep LATT/SYMM.
        # Fail closed on infinite polymer: GATE printed once by writer; no fake trial.res.
        from grok_phase_solver.physics.connectivity_asu import (
            ConnectivityAsuError,
            format_trial_res_gate,
        )

        res_path = out_dir / "trial.res"
        try:
            res_text = write_shelxl_res(result, element="Q")
            res_path.write_text(res_text)
            written.append(res_path)
            for line in res_text.splitlines():
                if line.startswith("REM gate "):
                    result.diagnostics["trial_res_gate"] = (
                        "GATE " + line[len("REM gate ") :]
                    )
                    break
        except ConnectivityAsuError as exc:
            # Writer already printed the single GATE fail-closed line.
            sg = result.space_group_hm or "P1"
            result.diagnostics["trial_res_gate"] = format_trial_res_gate(
                sg=sg, non_h=0, finite=False, pass_=False
            )
            result.warnings.append(str(exc))
            # do NOT write fake trial.res

    try:
        from grok_phase_solver.pipeline.map_export import write_map_handoff

        written.extend(write_map_handoff(result, out_dir))
    except Exception as exc:
        result.warnings.append(f"map export skipped: {exc}")

    from grok_phase_solver.pipeline.next_action import recommend_next_action

    next_act = recommend_next_action(
        cell=result.cell,
        d_min=result.d_min,
        method=result.method,
        n_reflections=len(hkl),
        n_peaks=len(result.peaks),
        diagnostics=result.diagnostics,
        space_group=result.space_group_hm,
    )
    result.diagnostics["next_action"] = next_act

    # JSON summary
    summary = {
        "method": result.method,
        "n_reflections": len(hkl),
        "cell": result.cell.tolist(),
        "space_group": result.space_group_hm,
        "d_min": result.d_min,
        "diagnostics": result.diagnostics,
        "warnings": result.warnings,
        "n_peaks": len(result.peaks),
        "next_action": next_act,
    }
    js = out_dir / "solve_summary.json"
    js.write_text(json.dumps(summary, indent=2))
    written.append(js)

    # Report
    report = out_dir / "report.md"
    report.write_text(_render_report(result))
    written.append(report)

    return written



def write_shelxl_res(
    result: "SolveResult",
    element: str = "Q",
    *,
    lattice: Optional[int] = None,
    symm: Optional[Sequence[str]] = None,
    n_non_h_budget: int = 34,
) -> str:
    """Fold unique-ASU, budget top non-H peaks, write Q with SFAC C H N O."""
    from grok_phase_solver.pipeline.peak_budget_res import write_shelxl_res_budgeted

    return write_shelxl_res_budgeted(
        result,
        element=element,
        lattice=lattice,
        symm=symm,
        n_non_h_budget=n_non_h_budget,
    )
