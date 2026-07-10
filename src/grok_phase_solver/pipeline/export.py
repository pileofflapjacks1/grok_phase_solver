"""
Export phased results for crystallographers.

Writes:
  - phases.csv          h k l |F| phase_deg A B
  - structure_factors.F simple complex F list
  - density.npz         rho grid + cell
  - density_slice.png   central slice (if matplotlib available)
  - peaks.xyz / peaks.csv  density peak list
  - report.md           human-readable summary + next steps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .solve import SolveResult


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
        fig.colorbar(im, ax=ax, fraction=0.046, label="ρ")
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
    }
    js = out_dir / "solve_summary.json"
    js.write_text(json.dumps(summary, indent=2))
    written.append(js)

    # Report
    report = out_dir / "report.md"
    report.write_text(_render_report(result))
    written.append(report)

    return written


def _render_report(result: "SolveResult") -> str:
    d = result.diagnostics
    lines = [
        f"# gps-solve report",
        "",
        f"**Method:** `{result.method}`  ",
        f"**Reflections:** {len(result.hkl)}  ",
        f"**Space group:** {result.space_group_hm or 'unknown'}  ",
        f"**d_min (Å):** {result.d_min if result.d_min else 'auto'}  ",
        "",
        "## Cell",
        "",
        "```",
        " ".join(f"{x:.4f}" for x in result.cell),
        "```",
        "",
        "## Diagnostics",
        "",
    ]
    for k, v in d.items():
        lines.append(f"- **{k}:** {v}")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        for w in result.warnings:
            lines.append(f"- ⚠️ {w}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "| File | Purpose |",
            "|------|---------|",
            "| `phases.csv` | hkl, \\|F\\|, phase (°) for downstream tools |",
            "| `structure_factors.F` | Complex F (A, B) |",
            "| `density.npz` | Electron density grid |",
            "| `density_slice.png` | Quick visual check |",
            "| `peaks.csv` / `peaks.xyz` | Strongest density maxima (trial atoms) |",
            "| `solve_summary.json` | Machine-readable summary |",
            "",
            "## Suggested next steps (crystallography practice)",
            "",
            "1. **Inspect** `density_slice.png` and peak heights in `peaks.csv` "
            "(look for chemically sensible peak counts).",
            "2. **Load** trial peaks into Olex2 / ShelXle / PyMOL and assign element types.",
            "3. **Refine** with SHELXL against your experimental intensities "
            "(this tool does **not** replace SHELXL refinement).",
            "4. If the map looks noisy or empty: try `--method charge_flipping` with more "
            "`--n-iter`, check space group/cell, or improve data completeness/resolution.",
            "5. Optional: install PhAI weights and re-run with `--method auto` or `--method phai` "
            "for P2₁/c small-molecule cases (see `third_party/phai/README.md`).",
            "",
            "## Honest scope",
            "",
            "gps-solve is an **open ab initio / hybrid phasing assistant**. It is strongest for "
            "small-molecule data at good resolution. It is **not** a guaranteed replacement for "
            "SHELXT/SHELXD or experimental phasing (MIR/MAD/MR) for proteins.",
            "",
        ]
    )
    return "\n".join(lines)
