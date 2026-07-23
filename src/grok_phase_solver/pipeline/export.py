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

        # SHELXL-style trial .res (Q peaks / carbon placeholders)
        res_path = out_dir / "trial.res"
        res_path.write_text(write_shelxl_res(result))
        written.append(res_path)

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
    # Flatten nested dicts for readability
    for k, v in d.items():
        if k in ("seed_quality",) and isinstance(v, dict):
            continue
        if isinstance(v, (dict, list)) and k not in ("holdout",):
            lines.append(f"- **{k}:** `{v}`")
        else:
            lines.append(f"- **{k}:** {v}")

    # Seed quality: Lane B partial-φ metrics and/or Carrozzini Class 0/1 predictor
    sq = d.get("seed_quality")
    if isinstance(sq, dict):
        # Carrozzini-style predictor (keys: predicted_class, success_probability, …)
        if "predicted_class" in sq:
            feats = sq.get("features") or {}
            lines.extend(
                [
                    "",
                    "## AI-PhaSeed seed quality (Carrozzini-style Class 0/1)",
                    "",
                    f"- **Predicted class:** {sq.get('predicted_class')} "
                    f"(1 ≈ high-success band; heuristic / optional RF)",
                    f"- **P(success) estimate:** {sq.get('success_probability')}",
                    f"- **Est. seed MPE (°):** {sq.get('predicted_mpe_deg')}",
                    f"- **Est. seed CORR:** {sq.get('predicted_corr')}",
                    f"- **max |E| (max W):** {feats.get('max_W')}",
                    f"- **Vol (Å³):** {feats.get('Vol')}",
                    f"- **Seed fraction:** {feats.get('seed_fraction')}",
                    f"- **Predictor:** {sq.get('method')}",
                    f"- **Final free-FOM composite:** {d.get('free_fom_composite')}",
                    "",
                ]
            )
            if sq.get("warning"):
                lines.append(f"- ⚠️ {sq['warning']}")
            for n in (sq.get("notes") or [])[:6]:
                lines.append(f"- note: {n}")
            if sq.get("recommend_fallback"):
                lines.extend(
                    [
                        "",
                        "**Action:** Class 0 seed — prefer partial-φ / fragment / HA, "
                        "or try `--ai-dm-hybrid --low-res-path` / ensemble. "
                        "Does not prove the structure is unsolvable.",
                        "",
                    ]
                )
        # Lane B partial-seed size metrics
        if "frac_strong_seeded" in sq or "size_meets_bar" in sq:
            lines.extend(
                [
                    "",
                    "## Partial seed quality (truth-free size bar)",
                    "",
                    f"- **Source / kind:** {d.get('seed_kind', d.get('seed_source', '—'))}",
                    f"- **Seeded reflections:** {sq.get('n_seed')} "
                    f"({100 * float(sq.get('fraction_all') or 0):.1f}% of all)",
                    f"- **Strong-|E| coverage:** {sq.get('n_strong_seeded')}/"
                    f"{sq.get('n_strong')} "
                    f"({100 * float(sq.get('frac_strong_seeded') or 0):.0f}%)",
                    f"- **Size vs 30% oracle bar:** "
                    f"{'OK' if sq.get('size_meets_bar') else 'BELOW BAR'}",
                    f"- **Seed free-FOM composite:** {sq.get('seed_free_fom_composite')}",
                    f"- **Final free-FOM composite:** {d.get('free_fom_composite')}",
                    "",
                ]
            )
            for h in sq.get("hints") or []:
                lines.append(f"- 💡 {h}")
        if sq.get("size_meets_bar") is False:
            lines.extend(
                [
                    "",
                    "**Action:** enlarge the seed — more known φ, heavier fragment, "
                    "or HA sites. Oracle: ≥~30% of strong |E| phases within ~20°.",
                    "",
                    "```bash",
                    "# From SHELXS fragment / trial.res",
                    "gps-make-seed --hkl your.hkl --ins your.ins --from-res model.res -o seed.csv",
                    "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed \\",
                    "  --phase-seed-csv seed.csv --out ./out_partial",
                    "```",
                ]
            )

    # Space group + device + uncertainty (v0.5)
    sg = d.get("space_group")
    if isinstance(sg, dict):
        lines.extend(
            [
                "",
                "## Space group",
                "",
                f"- **HM:** {sg.get('hm')}",
                f"- **Number:** {sg.get('number')}",
                f"- **Centrosymmetric:** {sg.get('is_centrosymmetric')}",
                f"- **N sym ops:** {sg.get('n_sym_ops')}",
                f"- **Crystal system:** {sg.get('crystal_system')}",
                "",
            ]
        )
    if d.get("device"):
        lines.append(f"- **Device:** {d.get('device')}")
    uq = d.get("phase_uncertainty")
    if isinstance(uq, dict):
        lines.extend(
            [
                "",
                "## Phase uncertainty (multistart circular)",
                "",
                f"- **Mean resultant length R̄:** {uq.get('mean_resultant_length')}",
                f"- **Mean phase probability:** {uq.get('mean_phase_probability')}",
                f"- **Mean circular std (°):** {uq.get('mean_circular_std_deg')}",
                f"- **Frac high confidence (R̄≥0.7):** {uq.get('frac_high_confidence')}",
                f"- **Strong-set confident frac:** {uq.get('strong_frac_confident')}",
                f"- **Note:** {uq.get('note')}",
                "",
            ]
        )
    boot = d.get("free_fom_bootstrap")
    if isinstance(boot, dict) and boot.get("n_boot"):
        lines.extend(
            [
                "",
                "## Free-FOM bootstrap stability",
                "",
                f"- **Mean ± std:** {boot.get('mean')} ± {boot.get('std')} "
                f"(n={boot.get('n_boot')})",
                f"- **Range:** [{boot.get('min')}, {boot.get('max')}]",
                "",
            ]
        )
    if d.get("method_used") in ("diffusion_hybrid", "diffusion_phaseed") or (
        isinstance(d.get("method_used"), str) and "diffusion" in str(d.get("method_used"))
    ):
        lines.extend(
            [
                "",
                "## Diffusion hybrid (experimental)",
                "",
                "Physics Langevin reverse process with positivity + modulus projection. "
                "**Not** a claim of PXRDnet/XRDSol parity. Prefer partial-φ when seeds "
                "meet the strong-|E| bar.",
                "",
            ]
        )

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
            "| `trial.res` | SHELXL-style trial model (Q peaks) for Olex2/SHELXL |",
            "| `solve_summary.json` | Machine-readable summary |",
            "",
            "## Suggested next steps (crystallography practice)",
            "",
            "1. **Inspect** `density_slice.png` and peak heights in `peaks.csv`.",
            "2. **Open** `trial.res` in Olex2 / ShelXle — assign element types to Q peaks.",
            "3. **Refine** with **SHELXL** (local `ShelX/shelxl` if installed):",
            "   ```bash",
            "   cp trial.res work.ins && cp your.hkl work.hkl && ShelX/shelxl work",
            "   ```",
            "4. If the map is poor:",
            "   - Easy/high-res: `--method ensemble` (default `auto` already prefers this).",
            "   - Hard ab initio failed — **seed paths**:",
            "     - `--phase-seed-csv known.csv` (known φ)",
            "     - `--phase-seed-res model.res` (SHELXS/fragment atoms → Fcalc)",
            "     - `--seed-peaks-csv peaks.csv` (recycle density peaks)",
            "     - `--native-hkl` + `--derivative-hkl` (isomorphous HA)",
            "     - `gps-make-seed …` then `partial_phaseed`",
            "   - Classical external: `--method shelxs` or `shelxs+shelxe`.",
            "5. Free-FOM composite is a **truth-free** ranking score, not proof of solution.",
            "6. Demo hard + partial-φ: `examples/partial_seed_demo/`.",
            "",
            "## Decision tree",
            "",
        ]
    )
    try:
        from grok_phase_solver.solvers.workflow import (
            shelxl_refinement_instructions,
            workflow_decision_tree_md,
        )

        lines.append(workflow_decision_tree_md())
        lines.append("")
        lines.append(shelxl_refinement_instructions(Path(".")))
    except Exception:
        pass
    lines.extend(
        [
            "",
            "## Honest scope",
            "",
            "gps-solve is an **open ab initio / hybrid phasing assistant**. Strongest for "
            "small-molecule data at good resolution (ensemble). Hard cells need better "
            "seeds (partial-φ) or external SHELXS — pure priors still lag the 30%/20° bar. "
            "Not a general protein ab initio solver.",
            "",
        ]
    )
    return "\n".join(lines)


def write_shelxl_res(result: "SolveResult", element: str = "C") -> str:
    """
    Build a minimal SHELXL-style .res trial model from density peaks.

    Peaks are written as Q labels (or ``element`` if specified) so Olex2/SHELXL
    can load them as a starting model. Not a refined structure.
    """
    a, b, c, al, be, ga = result.cell
    sg = result.space_group_hm or "P1"
    # LATT / SYMM omitted for simplicity — user should paste cell into real .ins
    lines = [
        f"TITL gps-solve trial ({result.method})",
        f"CELL 0.71073 {a:.4f} {b:.4f} {c:.4f} {al:.2f} {be:.2f} {ga:.2f}",
        f"ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        f"LATT -1",
        f"SFAC C H N O",
        f"UNIT 1 1 1 1",
        f"FVAR 1.0",
        f"REM free_fom_composite={result.diagnostics.get('free_fom_composite', 'n/a')}",
        f"REM method={result.method} n_peaks={len(result.peaks)}",
        f"REM space_group_hint={sg}",
    ]
    # Element index for SFAC C = 1
    for i, p in enumerate(result.peaks):
        # Q peaks as carbon placeholders (sfac 1); Uiso rough
        label = f"Q{i+1}" if element.upper() == "Q" else f"{element}{i+1}"
        u = max(0.02, 0.08 / max(p.height_sigma / 3.0, 0.5))
        lines.append(
            f"{label:6s} 1 {p.fract[0]:10.6f} {p.fract[1]:10.6f} {p.fract[2]:10.6f} "
            f"11.00000 {u:.5f}"
        )
    lines.append("HKLF 4")
    lines.append("END")
    return "\n".join(lines) + "\n"
