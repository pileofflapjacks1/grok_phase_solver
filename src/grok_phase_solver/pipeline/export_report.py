"""Report rendering for gps-solve export."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grok_phase_solver.pipeline.solve import SolveResult

def _render_report(result: "SolveResult") -> str:
    from grok_phase_solver.pipeline.next_action import (
        format_next_action_md,
        next_action_banner,
        recommend_next_action,
    )

    d = result.diagnostics
    next_act = d.get("next_action")
    if not isinstance(next_act, dict):
        next_act = recommend_next_action(
            cell=result.cell,
            d_min=result.d_min,
            method=result.method,
            n_reflections=len(result.hkl),
            n_peaks=len(result.peaks),
            diagnostics=d,
            space_group=result.space_group_hm,
        )
    lines = [
        f"# gps-solve report",
        "",
        f"**Method:** `{result.method}`  ",
        f"**Reflections:** {len(result.hkl)}  ",
        f"**Space group:** {result.space_group_hm or 'unknown'}  ",
        f"**d_min (Å):** {result.d_min if result.d_min else 'auto'}  ",
        f"**{next_action_banner(next_act)}**",
        "",
        format_next_action_md(next_act),
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

    sq = d.get("seed_quality")
    if isinstance(sq, dict):
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
            "| `density.map` | CCP4 map for **PyMOL / Coot** |",
            "| `open_in_pymol.pml` | `pymol open_in_pymol.pml` |",
            "| `open_in_coot.sh` | `sh open_in_coot.sh` |",
            "| `density_slice.png` | Quick visual check |",
            "| `peaks.csv` / `peaks.xyz` / `peaks.pdb` | Strongest density maxima |",
            "| `trial.res` | SHELXL-style trial model (Q peaks) for Olex2/SHELXL |",
            "| `solve_summary.json` | Machine-readable summary |",
            "",
            "## Suggested next steps (crystallography practice)",
            "",
            "1. **Inspect** `density_slice.png` and peak heights in `peaks.csv`.",
            "2. **3D map:** `pymol open_in_pymol.pml` or `sh open_in_coot.sh` (`density.map`).",
            "3. **Open** `trial.res` in Olex2 / ShelXle — assign element types to Q peaks.",
            "4. **Refine** with **SHELXL** (local `ShelX/shelxl` if installed):",
            "   ```bash",
            "   cp trial.res work.ins && cp your.hkl work.hkl && ShelX/shelxl work",
            "   ```",
            "5. If the map is poor: follow **Next action** above (Vol-band chooser).",
            "   Catalog: `--phase-seed-res` / `--predicted-model` / `--phase-seed-csv` /",
            "   `--seed-peaks-csv` / HA pair / `gps-make-seed` / `--method shelxs`.",
            "6. Free-FOM composite is a **truth-free** ranking score, not proof of solution.",
            "7. Demo hard + partial-φ: `examples/partial_seed_demo/`.",
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
