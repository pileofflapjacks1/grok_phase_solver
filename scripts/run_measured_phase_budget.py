#!/usr/bin/env python3
"""
Measured-phase information-budget sweep (P2).

Simulates φ_meas = φ_true + wrapped noise on a fraction of the strong-|E|
set, then extends with the existing partial_phaseed path.

This is a **simulator**. A phase-sensitive detector does not exist in this
repository. Do not quote σ_φ as experimental.

Reuses:
  - hard P1 cells from the partial-seed benchmark generator
  - COD ids already in the stratified Vol-band panel (data/raw/cod)

Writes:
  data/processed/measured_phase_budget.{json,md}
  docs/figures/measured_phase_budget.png

Regenerate (default = --quick)::

    python scripts/run_measured_phase_budget.py --quick
    python scripts/run_measured_phase_budget.py --full
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.measured_phase_noise import C4_FRAC_LE_20, WINDOW_DEG
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.partial_seed import measured_partial_seed, partial_phaseed_solve
from grok_phase_solver.solvers.projectors import unit_cell_volume

# Existing stratified-bench catalog (do not invent new cells)
COD_STRATIFIED_IDS = [
    "2012000",
    "2013000",
    "2016452",
    "2017775",
    "2100301",
    "2200000",
]
# True mid-band Vol 1000–3500 Å³ on that catalog
COD_MIDBAND_IDS = ["2012000", "2013000"]
QUICK_COD_ID = "2012000"  # P2₁, Vol ~1027 Å³, HKL present in-repo


def make_hard_case(seed: int, n_atoms: int = 14, d_min: float = 1.7):
    """Same generator as scripts/run_partial_seed_benchmark.py."""
    st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
    data = structure_to_fcalc(st, d_min=d_min)
    data["d_min"] = d_min
    return st, data


def _score(hkl, amp, ph, ph_t, cell, fracs, elements, rho, d_min) -> Dict[str, Any]:
    rho_t = density_from_structure_factors(
        hkl, amp * np.exp(1j * ph_t), cell, shape=rho.shape
    )
    if rho.shape != rho_t.shape:
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, shape=rho_t.shape
        )
    cc, _ = map_correlation_origin_invariant(rho, rho_t)
    rep = evaluate_success(
        hkl, amp, ph, ph_t, cell, fracs, density=rho, elements=elements,
        thresholds=SuccessThresholds(),
    )
    fom = None
    r_pos = None
    try:
        from grok_phase_solver.solvers.free_fom import free_fom

        fm = free_fom(hkl, amp, ph, cell, density=rho)
        fom = float(fm.get("composite"))
        r_pos = float(fm.get("R_pos"))
    except Exception:
        pass
    return {
        "mapcc_oi": float(cc),
        "peak_recovery": float(rep.peak_recovery),
        "r1": None if rep.r1 is None else float(rep.r1),
        "solved": bool(rep.solved),
        "free_fom_composite": fom,
        "free_fom_R_pos": r_pos,
    }


def load_cod_fcalc(cod_id: str, d_min: float = 1.0):
    cif = ROOT / "data" / "raw" / "cod" / f"{cod_id}.cif"
    if not cif.exists():
        return None, f"missing {cif}"
    try:
        st = load_cif(str(cif))
        data = structure_to_fcalc(st, d_min=d_min)
        data["d_min"] = d_min
        return (st, data), None
    except Exception as e:
        return None, f"{cod_id}: {e}"


def _cell_meta(label: str, st, data) -> Dict[str, Any]:
    cell = np.asarray(st.cell, dtype=np.float64)
    vol = float(unit_cell_volume(cell))
    n_nonh = int(data.get("n_atoms_cell") or len(data.get("elements") or []))
    return {
        "cell_id": label,
        "space_group": getattr(st, "space_group_hm", None) or "P1",
        "vol": vol,
        "d_min": float(data.get("d_min") or 1.7),
        "n_atoms": n_nonh,
        "n_refl": int(len(data["hkl"])),
    }


def run_grid(
    cases: List[Tuple[str, Any, Dict]],
    sigmas: List[float],
    fracs: List[float],
    *,
    n_extend: int,
    n_polish: int,
    n_starts: int,
    polish: str,
) -> List[Dict]:
    rows: List[Dict] = []
    for label, st, data in cases:
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = np.asarray(st.cell, dtype=np.float64)
        meta_c = _cell_meta(label, st, data)
        print(
            f"=== {label}  n={meta_c['n_atoms']}  d_min={meta_c['d_min']}  "
            f"Vol={meta_c['vol']:.0f}  n_refl={meta_c['n_refl']} ===",
            flush=True,
        )
        for sigma in sigmas:
            for frac in fracs:
                t0 = time.time()
                seed_ph, mask, smeta = measured_partial_seed(
                    hkl, amp, cell, ph_t,
                    frac_strong=frac,
                    sigma_phi_deg=sigma,
                    model="von_mises",
                    seed=0,
                )
                qual = smeta.get("seed_quality") or {}
                ph, rho, info = partial_phaseed_solve(
                    hkl, amp, cell, seed_ph,
                    mask=mask if mask.sum() >= 1 else None,
                    n_extend=n_extend,
                    n_polish=n_polish,
                    n_starts=n_starts,
                    polish=polish,
                    seed=0,
                    d_min=data.get("d_min"),
                    meta=smeta,
                    verbose=False,
                )
                sc = _score(
                    hkl, amp, ph, ph_t, cell,
                    data["fracs"], data["elements"], rho, data.get("d_min"),
                )
                row = {
                    **meta_c,
                    "sigma_phi_deg": float(sigma),
                    "frac_strong_measured": float(frac),
                    "n_seed": int(mask.sum()),
                    "frac_le_20": qual.get("frac_le_20"),
                    "frac_le_20_all_strong": qual.get("frac_le_20_all_strong"),
                    "circular_mean_error_deg": qual.get("circular_mean_error_deg"),
                    "n_strong": qual.get("n_strong"),
                    "n_measured_strong": qual.get("n_measured_strong"),
                    "meets_c4_bar": qual.get("meets_c4_bar"),
                    "seconds": time.time() - t0,
                    "free_fom_rank_only": sc.get("free_fom_composite"),
                    **sc,
                }
                rows.append(row)
                print(
                    f"  σ={sigma:.0f}° frac={frac:.2f}  "
                    f"≤20°(all strong)={row['frac_le_20_all_strong']}  "
                    f"CC={sc['mapcc_oi']:.3f}  sol={sc['solved']}  "
                    f"t={row['seconds']:.1f}s",
                    flush=True,
                )
    return rows


def summarize(rows: List[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_sigma": {}, "by_frac": {}, "by_cell": {}, "c4_crossings": []}
    from collections import defaultdict

    gs = defaultdict(list)
    gf = defaultdict(list)
    gc = defaultdict(list)
    for r in rows:
        gs[float(r["sigma_phi_deg"])].append(r)
        gf[float(r["frac_strong_measured"])].append(r)
        gc[r["cell_id"]].append(r)
        if r.get("meets_c4_bar"):
            out["c4_crossings"].append(
                {
                    "cell_id": r["cell_id"],
                    "sigma_phi_deg": r["sigma_phi_deg"],
                    "frac_strong_measured": r["frac_strong_measured"],
                    "frac_le_20_all_strong": r.get("frac_le_20_all_strong"),
                    "solved": r.get("solved"),
                }
            )

    def _agg(rs: List[Dict]) -> Dict[str, Any]:
        sols = [bool(r.get("solved")) for r in rs]
        return {
            "n": len(rs),
            "n_solved": int(sum(sols)),
            "solve_rate": float(np.mean(sols)) if sols else None,
            "mean_mapcc": float(np.mean([r["mapcc_oi"] for r in rs])),
            "mean_peak": float(np.mean([r["peak_recovery"] for r in rs])),
            "mean_r1": float(np.nanmean([r["r1"] if r["r1"] is not None else np.nan for r in rs])),
            "mean_frac_le_20_all_strong": float(
                np.nanmean([r["frac_le_20_all_strong"] for r in rs])
            ),
            "mean_free_fom": float(
                np.nanmean(
                    [r["free_fom_composite"] if r.get("free_fom_composite") is not None else np.nan for r in rs]
                )
            ),
        }

    out["by_sigma"] = {str(k): _agg(v) for k, v in sorted(gs.items())}
    out["by_frac"] = {str(k): _agg(v) for k, v in sorted(gf.items())}
    out["by_cell"] = {str(k): _agg(v) for k, v in gc.items()}
    n_sol = sum(1 for r in rows if r.get("solved"))
    out["n_rows"] = len(rows)
    out["n_solved"] = n_sol
    out["strict_success"] = (
        "YES" if n_sol == len(rows) and rows else (
            "NO" if n_sol == 0 else "mixed across grid"
        )
    )
    return out


def write_md(
    path: Path,
    *,
    summary: Dict[str, Any],
    rows: List[Dict],
    skipped: List[str],
    quick: bool,
    cells_run: List[Dict[str, Any]],
) -> None:
    # Status block (ticket template)
    frac20_vals = [r.get("frac_le_20_all_strong") for r in rows if r.get("frac_le_20_all_strong") is not None]
    mean_f20 = float(np.mean(frac20_vals)) if frac20_vals else float("nan")
    n_c4 = sum(1 for r in rows if r.get("meets_c4_bar"))
    mapccs = [r["mapcc_oi"] for r in rows]
    peaks = [r["peak_recovery"] for r in rows]
    r1s = [r["r1"] for r in rows if r.get("r1") is not None]
    foms = [r.get("free_fom_composite") for r in rows if r.get("free_fom_composite") is not None]
    rpos = [r.get("free_fom_R_pos") for r in rows if r.get("free_fom_R_pos") is not None]
    cell_line = "; ".join(
        f"{c['cell_id']} Vol={c['vol']:.0f} Å³ d_min={c['d_min']} N={c['n_atoms']}"
        for c in cells_run
    )
    crossings = summary.get("c4_crossings") or []
    if crossings:
        c4_s = "; ".join(
            f"{c['cell_id']} σ={c['sigma_phi_deg']:.0f}° frac={c['frac_strong_measured']:.2f} "
            f"≤20°={100 * float(c['frac_le_20_all_strong'] or 0):.0f}%"
            for c in crossings[:8]
        )
    else:
        c4_s = "none on this grid (need higher frac and/or lower σ_φ)"

    lines = [
        "# Measured-phase information budget",
        "",
        "```",
        "Ticket:              P2 (+ optional P1 audit / P3 trail / hygiene)",
        "Information source:  measured-φ (simulated) / fragment / n/a (docs)",
        "Method:              partial_phaseed + measured_phase_noise",
        f"Vol-band / d_min / N: {cell_line or '—'}",
        f"Seed quality:        frac≤20° (all strong, mean) = {100 * mean_f20:.0f}%   "
        f"(bar = 30%)   C4 crossings = {n_c4}/{len(rows)}",
        f"Strict success:      {summary.get('strict_success')}  "
        f"({summary.get('n_solved')}/{summary.get('n_rows')} grid points)",
        f"  mapCC_OI= {float(np.mean(mapccs)):.3f} (mean)   "
        f"peak recovery= {float(np.mean(peaks)):.3f} (mean)   "
        f"R1= {float(np.mean(r1s)) if r1s else float('nan'):.3f} (mean)",
        f"Free FOM:            rank only; composite={float(np.mean(foms)) if foms else float('nan'):.3f}  "
        f"R₊={float(np.mean(rpos)) if rpos else float('nan'):.3f}",
        "Claim status:        does not touch freeze C1–C25; new simulator only",
        "PR / path:           feat/measured-phase-budget",
        "Next action:         If a lab measured-φ seed has ≥~30% of strong |E| within 20°, "
        "run partial_phaseed then inspect trial.res in Olex2 / SHELXL.",
        "What we will not do: GraphPhaseNet v12; claim detector exists",
        f"Blocker for Joe:     {('; '.join(skipped) if skipped else 'none')}",
        "```",
        "",
        "Hardware **does not exist**. This sweep is a simulator plugged into the",
        "existing partial-φ path. See [docs/math/measured_phase_budget.md](../../docs/math/measured_phase_budget.md).",
        "",
        f"Mode: **{'quick' if quick else 'full'}**. C4 bar: ≥ {100 * C4_FRAC_LE_20:.0f}% of",
        f"strong-|E| phases correct within {WINDOW_DEG:.0f}° (non-centro TREF window).",
        "",
        f"C4 crossings on this run: {c4_s}",
        "",
        "## Cells",
        "",
        "| id | SG | Vol (Å³) | d_min | N | n_refl |",
        "|----|----|----------|-------|---|--------|",
    ]
    for c in cells_run:
        lines.append(
            f"| {c['cell_id']} | {c.get('space_group', '')} | {c['vol']:.0f} | "
            f"{c['d_min']} | {c['n_atoms']} | {c['n_refl']} |"
        )
    if skipped:
        lines.extend(["", "**Skipped:** " + "; ".join(skipped), ""])

    lines.extend(
        [
            "",
            "## Grid results",
            "",
            "| cell | σ_φ (°) | frac measured | ≤20° meas | ≤20° all strong | C4 | mapCC | peak | R1 | solved | FOM | s |",
            "|------|---------|---------------|-----------|-----------------|----|-------|------|----|--------|-----|---|",
        ]
    )
    for r in rows:
        f20 = r.get("frac_le_20")
        f20a = r.get("frac_le_20_all_strong")
        f20_s = "" if f20 is None else f"{100 * f20:.0f}%"
        f20a_s = "" if f20a is None else f"{100 * f20a:.0f}%"
        r1_s = "" if r.get("r1") is None else f"{r['r1']:.2f}"
        fom_s = "" if r.get("free_fom_composite") is None else f"{r['free_fom_composite']:.3f}"
        c4_s = "yes" if r.get("meets_c4_bar") else "no"
        lines.append(
            f"| {r['cell_id']} | {r['sigma_phi_deg']:.0f} | {r['frac_strong_measured']:.2f} | "
            f"{f20_s} | {f20a_s} | {c4_s} | "
            f"{r['mapcc_oi']:.3f} | {r['peak_recovery']:.2f} | "
            f"{r1_s} | {r['solved']} | {fom_s} | {r['seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Summary by σ_φ",
            "",
            "| σ_φ (°) | n | solved | rate | mean ≤20° all strong | mean mapCC |",
            "|---------|---|--------|------|----------------------|------------|",
        ]
    )
    for k, s in summary.get("by_sigma", {}).items():
        lines.append(
            f"| {k} | {s['n']} | {s['n_solved']} | {s['solve_rate']:.0%} | "
            f"{100 * s['mean_frac_le_20_all_strong']:.0f}% | {s['mean_mapcc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Summary by measured fraction",
            "",
            "| frac | n | solved | rate | mean ≤20° all strong | mean mapCC |",
            "|------|---|--------|------|----------------------|------------|",
        ]
    )
    for k, s in summary.get("by_frac", {}).items():
        lines.append(
            f"| {k} | {s['n']} | {s['n_solved']} | {s['solve_rate']:.0%} | "
            f"{100 * s['mean_frac_le_20_all_strong']:.0f}% | {s['mean_mapcc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## How to regenerate",
            "",
            "```bash",
            "python scripts/run_measured_phase_budget.py --quick",
            "python scripts/run_measured_phase_budget.py --full",
            "```",
            "",
            "Strict success is never redefined: mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5",
            "**and** R1 ≤ 0.45. Free FOM ranks trials only.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_figure(rows: List[Dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    sigmas = sorted({float(r["sigma_phi_deg"]) for r in rows})
    fracs = sorted({float(r["frac_strong_measured"]) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    # Left: strict-success vs σ_φ (one line per frac); C4 on twin of seed quality
    ax = axes[0]
    ax2 = ax.twinx()
    for frac in fracs:
        sub = [r for r in rows if abs(float(r["frac_strong_measured"]) - frac) < 1e-9]
        xs, ys, f20 = [], [], []
        for s in sigmas:
            chunk = [r for r in sub if abs(float(r["sigma_phi_deg"]) - s) < 1e-9]
            if not chunk:
                continue
            xs.append(s)
            ys.append(float(np.mean([bool(r["solved"]) for r in chunk])))
            f20.append(float(np.nanmean([r["frac_le_20_all_strong"] for r in chunk])))
        ax.plot(xs, ys, marker="o", label=f"frac={frac:.2f}")
        ax2.plot(xs, f20, marker="s", linestyle="--", alpha=0.7)
    ax2.axhline(C4_FRAC_LE_20, color="0.2", linestyle=":", linewidth=1.4, label="C4 30% ≤20°")
    ax.set_xlabel(r"$\sigma_\varphi$ (deg)")
    ax.set_ylabel("strict-success rate")
    ax2.set_ylabel("frac ≤20° (all strong |E|)")
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("strict success vs phase noise")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="lower left", fontsize=8)

    # Right: strict-success vs measured fraction
    ax = axes[1]
    ax2 = ax.twinx()
    for s in sigmas:
        sub = [r for r in rows if abs(float(r["sigma_phi_deg"]) - s) < 1e-9]
        xs, ys, f20 = [], [], []
        for frac in fracs:
            chunk = [r for r in sub if abs(float(r["frac_strong_measured"]) - frac) < 1e-9]
            if not chunk:
                continue
            xs.append(frac)
            ys.append(float(np.mean([bool(r["solved"]) for r in chunk])))
            f20.append(float(np.nanmean([r["frac_le_20_all_strong"] for r in chunk])))
        ax.plot(xs, ys, marker="o", label=rf"$\sigma$={s:.0f}°")
        ax2.plot(xs, f20, marker="s", linestyle="--", alpha=0.7)
    ax2.axhline(C4_FRAC_LE_20, color="0.2", linestyle=":", linewidth=1.4, label="C4 30% ≤20°")
    ax.set_xlabel("fraction of strong-|E| measured")
    ax.set_ylabel("strict-success rate")
    ax2.set_ylabel("frac ≤20° (all strong |E|)")
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("strict success vs measured fraction")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Measured-φ information budget (simulator). Hardware does not exist.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small grid + 1 hard P1 + 1 COD mid-band (default if --full is off)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Full σ/frac grid and existing bench cell lists",
    )
    args = p.parse_args()
    quick = not bool(args.full)

    if quick:
        sigmas = [10.0, 20.0, 45.0]
        fracs = [0.10, 0.30, 1.00]
        n_extend, n_polish, n_starts = 8, 20, 1
        polish = "charge_flipping"
        hard_specs = [(2026, 14, 1.7)]
        cod_ids = [QUICK_COD_ID]
    else:
        sigmas = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 90.0]
        fracs = [0.10, 0.20, 0.30, 0.50, 1.00]
        n_extend, n_polish, n_starts = 12, 40, 1
        polish = "charge_flipping"
        rng = np.random.default_rng(2026)
        hard_specs = []
        for _ in range(4):
            n_atoms = int(rng.integers(12, 17))
            d_min = float(rng.choice([1.5, 1.7, 2.0]))
            s = int(rng.integers(0, 2**31 - 1))
            hard_specs.append((s, n_atoms, d_min))
        cod_ids = list(COD_STRATIFIED_IDS)

    skipped: List[str] = []
    cases: List[Tuple[str, Any, Dict]] = []
    cells_run: List[Dict[str, Any]] = []

    for i, (s, n_atoms, d_min) in enumerate(hard_specs):
        st, data = make_hard_case(s, n_atoms=n_atoms, d_min=d_min)
        label = f"hard_p1_{i}_n{n_atoms}_d{d_min}"
        cases.append((label, st, data))
        cells_run.append(_cell_meta(label, st, data))
        print(f"loaded {label}", flush=True)

    for cid in cod_ids:
        loaded, err = load_cod_fcalc(cid, d_min=1.0)
        if err:
            print(f"skip {err}", flush=True)
            skipped.append(err)
            continue
        st, data = loaded
        label = f"cod_{cid}_fcalc"
        cases.append((label, st, data))
        cells_run.append(_cell_meta(label, st, data))
        print(f"loaded {label}", flush=True)

    rows = run_grid(
        cases, sigmas, fracs,
        n_extend=n_extend, n_polish=n_polish, n_starts=n_starts, polish=polish,
    )
    summary = summarize(rows)
    payload = {
        "quick": quick,
        "sigmas_deg": sigmas,
        "fracs": fracs,
        "window_deg": WINDOW_DEG,
        "c4_bar": C4_FRAC_LE_20,
        "hardware_exists": False,
        "note": (
            "Simulated measured phases on Fcalc truth. "
            "A phase-sensitive detector does not exist in this repository."
        ),
        "skipped": skipped,
        "cells": cells_run,
        "summary": summary,
        "rows": rows,
    }
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "measured_phase_budget.json"
    mp = out_dir / "measured_phase_budget.md"
    jp.write_text(json.dumps(payload, indent=2, default=float))
    write_md(mp, summary=summary, rows=rows, skipped=skipped, quick=quick, cells_run=cells_run)
    fig_path = ROOT / "docs" / "figures" / "measured_phase_budget.png"
    try:
        plot_figure(rows, fig_path)
        print(f"Wrote {fig_path}", flush=True)
    except Exception as e:
        print(f"figure failed: {e}", flush=True)
        skipped.append(f"figure: {e}")
    print(f"Wrote {jp}\nWrote {mp}", flush=True)
    print("strict_success:", summary.get("strict_success"), flush=True)
    print("C4 crossings:", len(summary.get("c4_crossings") or []), flush=True)


if __name__ == "__main__":
    main()
