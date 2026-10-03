#!/usr/bin/env python3
"""Per-gate breakdown of mid-band Fobs fragment_half, plus one side R1.

The strict bar is unchanged: mapCC_OI ≥ 0.7, peak recovery ≥ 0.5, and
carbon-peak R1 ≤ 0.45. Official gate values are copied from
``data/processed/cod_stratified_bench.json``. This script does not rewrite
that scoreboard and does not flip ``solved``.

The side residual places the same strongest peaks used by ``r1_from_peaks``
and assigns deposited non-H elements where a peak matches a true site.
Unmatched peaks stay carbon. B = 5, same scaled R1.

COD 1544230 is kept. The bench and ``solve_structure`` share
``d_min_keep_mask`` (d ≥ d_min − 1e-9). A re-run should score R1 and peak
recovery on that cell. The committed diagnostic note from before that
alignment described the one-reflection mismatch.

Usage (from repo root)::

    python scripts/run_r1_gate_diagnostic.py
    python scripts/run_r1_gate_diagnostic.py --ids 2012000
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.io.cif import CrystalStructure, load_cif
from grok_phase_solver.io.experiment import load_experiment
from grok_phase_solver.io.hkl import ReflectionTable, d_min_keep_mask, write_hkl_simple
from grok_phase_solver.metrics.r1_diagnostic import r1_from_typed_peaks
from grok_phase_solver.metrics.success import (
    SuccessThresholds,
    evaluate_success,
    peak_recovery_score,
    r1_from_peaks,
)
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.pipeline.peaks import pick_density_peaks
from grok_phase_solver.pipeline.solve import SolveConfig, _filter_dmin, solve_structure
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.projectors import unit_cell_volume
from grok_phase_solver.solvers.seed_import import select_fragment_atoms

D_MIN = 1.0
SCOREBOARD = ROOT / "data" / "processed" / "cod_stratified_bench.json"
# Must match scripts/run_cod_stratified_bench.py run_dataset (non-hard cells).
N_ITER_FRAGMENT = 90
N_EXTEND_FRAGMENT = 26


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x


def _fmt(x: Any, digits: int = 3) -> str:
    if not _finite(x):
        return "—"
    return f"{float(x):.{digits}f}"


def _hkl_key(h: Sequence[float]) -> Tuple[int, int, int]:
    return (int(h[0]), int(h[1]), int(h[2]))


def _match_phases(hkl_obs: np.ndarray, data: Dict) -> np.ndarray:
    key = {_hkl_key(h): i for i, h in enumerate(data["hkl"])}
    ph = np.zeros(len(hkl_obs), dtype=np.float64)
    for i, h in enumerate(hkl_obs):
        t = _hkl_key(h)
        if t in key:
            ph[i] = data["phases"][key[t]]
            continue
        tf = (-t[0], -t[1], -t[2])
        if tf in key:
            ph[i] = -data["phases"][key[tf]]
    return ph


def _write_fragment_cif(path: Path, st: CrystalStructure) -> int:
    els = [a.element for a in st.atoms if a.element.upper() not in ("H", "D")]
    fracs = np.array(
        [a.fract for a in st.atoms if a.element.upper() not in ("H", "D")],
        dtype=np.float64,
    )
    n_frag = min(40, max(3, len(els) // 2))
    fr_sel, el_sel, _ = select_fragment_atoms(
        fracs, els, max_atoms=n_frag, mode="heaviest_cluster", seed=0
    )
    a, b, c, al, be, ga = st.cell
    lines = [
        "data_fragment",
        f"_cell_length_a {a}",
        f"_cell_length_b {b}",
        f"_cell_length_c {c}",
        f"_cell_angle_alpha {al}",
        f"_cell_angle_beta {be}",
        f"_cell_angle_gamma {ga}",
        f"_symmetry_space_group_name_H-M '{st.space_group_hm}'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_U_iso_or_equiv",
    ]
    for i, el in enumerate(el_sel):
        x, y, z = fr_sel[i]
        lines.append(f"{el}{i+1} {el} {x:.6f} {y:.6f} {z:.6f} 1.000 0.05000")
    path.write_text("\n".join(lines) + "\n")
    return n_frag


def _bench_reflections(st: CrystalStructure, hkl_path: Path):
    table, _ = load_experiment(
        str(hkl_path),
        cell=",".join(str(x) for x in st.cell),
        space_group=st.space_group_hm,
    )
    d = d_spacing(table.hkl, table.cell)
    keep = d_min_keep_mask(d, D_MIN)
    return table.hkl[keep], table.amplitudes[keep], d[keep]


def _dropped_by_solver(hkl: np.ndarray, amp: np.ndarray, d: np.ndarray, st: CrystalStructure, tmp: Path):
    """Reflection kept by the bench epsilon and removed by solve's d >= d_min."""
    write_hkl_simple(
        tmp,
        ReflectionTable(hkl=hkl, F_meas=amp, cell=st.cell, space_group_hm=st.space_group_hm),
    )
    reloaded, _ = load_experiment(
        str(tmp),
        cell=",".join(str(x) for x in st.cell),
        space_group=st.space_group_hm,
    )
    reloaded = _filter_dmin(reloaded, D_MIN)
    have = {_hkl_key(h) for h in reloaded.hkl}
    dropped = []
    for i, h in enumerate(hkl):
        key = _hkl_key(h)
        if key not in have:
            dropped.append(
                {
                    "hkl": list(key),
                    "d": float(d[i]),
                    "amplitude": float(amp[i]),
                }
            )
    return dropped, len(reloaded)


def _non_h(fracs: np.ndarray, elements: Sequence[str]):
    mask = np.array([str(e).upper() not in ("H", "D") for e in elements], dtype=bool)
    return fracs[mask], [e for e, m in zip(elements, mask) if m]


def _score_row(
    label: str,
    official: Dict,
    st: CrystalStructure,
    work: Path,
) -> Dict:
    hkl_path = ROOT / "data" / "raw" / "cod" / f"{label}.hkl"
    hkl_b, amp_b, d_b = _bench_reflections(st, hkl_path)
    tmp_hkl = work / f"{label}.hkl"
    dropped, n_solver_list = _dropped_by_solver(hkl_b, amp_b, d_b, st, tmp_hkl)
    frag = work / f"{label}_half.cif"
    _write_fragment_cif(frag, st)
    vol = float(unit_cell_volume(np.asarray(st.cell, dtype=np.float64)))
    hard = vol > 3500 or sum(
        1 for a in st.atoms if a.element.upper() not in ("H", "D")
    ) > 40
    if hard:
        raise RuntimeError(f"{label} would use the large-cell budget; this diagnostic is mid-band only")

    t0 = time.time()
    res = solve_structure(
        str(tmp_hkl),
        cell=",".join(str(x) for x in st.cell),
        space_group=st.space_group_hm,
        config=SolveConfig(
            method="partial_phaseed",
            predicted_model_cif=str(frag),
            expand_model_symmetry=True,
            d_min=D_MIN,
            n_iter=N_ITER_FRAGMENT,
            n_extend=N_EXTEND_FRAGMENT,
            n_starts=1,
            prior_weight=0.52,
            dm_ai_weight=0.42,
            verbose=False,
            seed=0,
            compute_uncertainty=False,
        ),
    )
    seconds = time.time() - t0
    fdata = structure_to_fcalc(st, d_min=D_MIN)
    true_fracs, true_els = _non_h(fdata["fracs"], fdata["elements"])
    ph = _match_phases(res.hkl, fdata)
    thr = SuccessThresholds()
    n_atoms = len(true_fracs)
    n_pick = max(n_atoms + 5, int(thr.n_peaks_factor * n_atoms))
    peaks = pick_density_peaks(res.density, n_peaks=n_pick, min_sigma=thr.min_peak_sigma)
    peak_fracs = np.array([p.fract for p in peaks], dtype=np.float64) if peaks else np.zeros((0, 3))
    _, shift = peak_recovery_score(
        peak_fracs, true_fracs, tol=thr.peak_tol, n_origin_shifts=6
    )
    # Scoreboard R1 uses the bench |F| list, not the amplitudes solve_structure
    # reloads (write_hkl_simple stores |F|; the SHELX reader treats that column
    # as intensity). Pair original |F| with the reflections the solver kept.
    if not dropped:
        hkl_r1, amp_r1 = hkl_b, amp_b
    else:
        amp_by = {_hkl_key(h): float(amp_b[i]) for i, h in enumerate(hkl_b)}
        hkl_r1 = np.asarray(res.hkl)
        amp_r1 = np.array([amp_by[_hkl_key(h)] for h in hkl_r1], dtype=np.float64)
    carbon = float(
        r1_from_peaks(hkl_r1, amp_r1, res.cell, peak_fracs, n_atoms=max(n_atoms, 1))
    )
    typed = r1_from_typed_peaks(
        hkl_r1,
        amp_r1,
        res.cell,
        peak_fracs,
        true_fracs,
        true_els,
        n_atoms=max(n_atoms, 1),
        origin_shift=shift,
        tol=thr.peak_tol,
        b_iso=5.0,
    )
    eval_error = None
    eval_r1 = None
    eval_peak = None
    try:
        # Bench lists are the scoreboard's amplitude vector. A length mismatch
        # is the 1544230 failure mode and must be reported, not patched here.
        ph_bench = _match_phases(hkl_b, fdata)
        rep = evaluate_success(
            hkl_b,
            amp_b,
            res.phases,
            ph_bench,
            st.cell,
            fdata["fracs"],
            density=res.density,
            elements=fdata["elements"],
            thresholds=thr,
        )
        eval_r1 = float(rep.r1) if rep.r1 is not None else None
        eval_peak = float(rep.peak_recovery)
    except Exception as exc:
        eval_error = f"{type(exc).__name__}: {exc}"

    sb_r1 = official.get("r1")
    carbon_delta = None
    if _finite(sb_r1):
        carbon_delta = float(carbon) - float(sb_r1)
    note = ""
    if dropped:
        bits = []
        for item in dropped:
            h, k, l = item["hkl"]
            bits.append(
                f"({h} {k} {l}) d={item['d']:.12f} Å |F|={item['amplitude']:.3g}"
            )
        note = (
            f"COD {label} stays in the mapCC mean. Scoreboard R1 and peak recovery "
            "are missing because of a reflection-list mismatch, not an empty density. "
            "The bench keeps d ≥ d_min − 1e-9. solve_structure reloads the same file "
            f"and keeps d ≥ d_min, dropping {', '.join(bits)}. "
            f"The solver returns {len(res.phases)} phases and the bench list has "
            f"{len(amp_b)} amplitudes. "
        )
        if eval_error:
            note += f"evaluate_success raises `{eval_error}`. "
        note += (
            "The bench stores that failure as missing R1 and peak recovery. "
            "The typed R1 in this table uses original |F| on the reflections the "
            "solver kept. It is not a scoreboard R1 and it does not set solved."
        )
    elif eval_error:
        note = f"evaluate_success raises {eval_error}."

    return {
        "dataset": label,
        "scoreboard_mapcc": official.get("mapcc_oi"),
        "scoreboard_peak_recovery": official.get("peak_recovery"),
        "scoreboard_r1": sb_r1,
        "scoreboard_solved": bool(official.get("solved")),
        "pass_mapcc": bool(_finite(official.get("mapcc_oi")) and official["mapcc_oi"] >= 0.7),
        "pass_peak": bool(_finite(official.get("peak_recovery")) and official["peak_recovery"] >= 0.5),
        "pass_r1": bool(_finite(sb_r1) and sb_r1 <= 0.45),
        "n_bench_reflections": int(len(hkl_b)),
        "n_solver_reflections": int(len(res.hkl)),
        "n_reload_reflections": int(n_solver_list),
        "dropped_reflections": dropped,
        "diag_r1_carbon": carbon,
        "carbon_minus_scoreboard": carbon_delta,
        "diag_r1_typed": typed.r1,
        "n_typed_matched": typed.n_matched,
        "n_peaks_used": typed.n_peaks_used,
        "eval_error": eval_error,
        "eval_r1_on_bench_list": eval_r1,
        "eval_peak_on_bench_list": eval_peak,
        "seconds": seconds,
        "note": note,
    }


def _markdown(payload: Dict) -> str:
    rows: List[Dict] = payload["rows"]
    lines = [
        "# R1 gate diagnostic — mid-band Fobs fragment_half",
        "",
        "Strict success is unchanged: mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 "
        "**and** carbon-peak R1 ≤ 0.45. Official columns are copied from "
        "`cod_stratified_bench.json` (d_min = 1.0 Å). `solved` is not recomputed.",
        "",
        "Carbon-peak R1 places the strongest peaks as carbon with B = 5. "
        "The side column uses those same peaks and assigns a deposited non-H "
        "element when the peak matches a true site within the peak-recovery "
        "tolerance (0.15, fractional min-image) after that search's origin shift. "
        "Unmatched peaks stay carbon. This side R1 is not a strict-bar input.",
        "",
        "| COD | mapCC | peak | R1 carbon | mapCC≥0.7 | peak≥0.5 | R1≤0.45 | solved | R1 typed | matched |",
        "|-----|-------|------|-----------|-----------|----------|---------|--------|----------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | **{_fmt(r['scoreboard_mapcc'])}** | "
            f"{_fmt(r['scoreboard_peak_recovery'])} | {_fmt(r['scoreboard_r1'])} | "
            f"{'yes' if r['pass_mapcc'] else 'no'} | "
            f"{'yes' if r['pass_peak'] else 'no'} | "
            f"{'yes' if r['pass_r1'] else 'no'} | "
            f"{r['scoreboard_solved']} | **{_fmt(r['diag_r1_typed'])}**"
            f"{' †' if r.get('eval_error') else ''} | "
            f"{r['n_typed_matched']}/{r['n_peaks_used']} |"
        )
    if any(r.get("eval_error") for r in rows):
        lines.extend(
            [
                "",
                "† Typed R1 on a row whose scoreboard R1 is missing. "
                "It is not a strict-bar value.",
            ]
        )
    counts = payload["gate_counts"]
    lines.extend(
        [
            "",
            f"## Gate counts (scoreboard, n = {counts['n']})",
            "",
            f"- mapCC ≥ 0.7: **{counts['mapcc']}**",
            f"- peak recovery ≥ 0.5: **{counts['peak']}**",
            f"- R1 ≤ 0.45: **{counts['r1']}**",
            f"- strict solved: **{counts['solved']}**",
            "",
            "## Side R1 (typed peaks)",
            "",
            payload["typed_summary"],
            "",
            "## COD 1544230",
            "",
            payload["cell_1544230"],
            "",
            "Regenerate:",
            "",
            "```bash",
            "python scripts/run_r1_gate_diagnostic.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _typed_summary(rows: List[Dict]) -> str:
    typed = [r["diag_r1_typed"] for r in rows if _finite(r.get("diag_r1_typed"))]
    carbon_sb = [r["scoreboard_r1"] for r in rows if _finite(r.get("scoreboard_r1"))]
    n_under = sum(1 for r in rows if _finite(r.get("diag_r1_typed")) and r["diag_r1_typed"] <= 0.45)
    mean_s = _fmt(float(np.mean(typed)) if typed else float("nan"))
    min_s = _fmt(float(np.min(typed)) if typed else float("nan"))
    parts = [
        f"Typed-peak R1 was computed for {len(typed)} of {len(rows)} rows "
        f"(mean {mean_s}, minimum {min_s}). "
        f"{n_under} of those side values are ≤ 0.45. "
        "Unmatched peaks stay carbon, so the side residual stays close to the "
        "frozen carbon-peak R1. That count does not change `solved`.",
    ]
    if carbon_sb:
        parts.append(
            f" Scoreboard carbon-peak R1, where it exists (n={len(carbon_sb)}), "
            f"has mean {_fmt(float(np.mean(carbon_sb)))} and minimum {_fmt(float(np.min(carbon_sb)))}."
        )
    return "".join(parts)


def _cell_note(rows: List[Dict]) -> str:
    hit = next((r for r in rows if r["dataset"] == "1544230"), None)
    if hit is None:
        return "COD 1544230 was not in this run."
    if hit.get("note"):
        return hit["note"]
    return "COD 1544230 scored on the same reflection list as the bench."


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Mid-band Fobs fragment_half R1 gate diagnostic")
    p.add_argument("--ids", type=str, default="", help="Comma-separated COD ids (default: all 12 mid-band Fobs fragment_half rows)")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "processed"))
    args = p.parse_args()

    board = json.loads(SCOREBOARD.read_text())
    official = [
        r for r in board["rows"]
        if r.get("vol_band") == "vol_1000_3500"
        and r.get("run") == "fragment_half"
        and r.get("amp_mode") == "fobs"
        and "mapcc_oi" in r
    ]
    if args.ids.strip():
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        official = [r for r in official if r["dataset"] in want]
    if not official:
        raise SystemExit("No matching scoreboard rows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_tmp_r1_gate"
    work.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for off in official:
        label = str(off["dataset"])
        print(f"=== {label} fragment_half fobs ===", flush=True)
        st = load_cif(str(ROOT / "data" / "raw" / "cod" / f"{label}.cif"))
        row = _score_row(label, off, st, work)
        rows.append(row)
        print(
            f"  scoreboard R1={_fmt(row['scoreboard_r1'])} "
            f"typed={row['diag_r1_typed']:.3f} "
            f"matched={row['n_typed_matched']}/{row['n_peaks_used']} "
            f"t={row['seconds']:.1f}s",
            flush=True,
        )
        if row.get("eval_error"):
            print(f"  eval: {row['eval_error']}", flush=True)
        elif _finite(row.get("eval_r1_on_bench_list")) and _finite(row.get("diag_r1_carbon")):
            delta = abs(row["diag_r1_carbon"] - row["eval_r1_on_bench_list"])
            if delta > 1e-6:
                print(f"  carbon R1 drifted from evaluate_success by {delta:.3e}", flush=True)

    payload = {
        "d_min": D_MIN,
        "strict_bar": {
            "mapcc_min": 0.7,
            "peak_recovery_min": 0.5,
            "r1_max": 0.45,
            "r1_model": "strongest peaks as carbon, B=5, scaled |F| residual",
        },
        "side_r1": (
            "Same peaks. Deposited non-H element if the peak matches a true site "
            "within tol 0.15 after the peak-recovery origin shift; otherwise carbon. "
            "Not an input to solved."
        ),
        "source_scoreboard": "data/processed/cod_stratified_bench.json",
        "rows": rows,
        "gate_counts": {
            "n": len(rows),
            "mapcc": sum(1 for r in rows if r["pass_mapcc"]),
            "peak": sum(1 for r in rows if r["pass_peak"]),
            "r1": sum(1 for r in rows if r["pass_r1"]),
            "solved": sum(1 for r in rows if r["scoreboard_solved"]),
        },
        "n_typed_r1_le_0.45": sum(
            1 for r in rows if _finite(r.get("diag_r1_typed")) and r["diag_r1_typed"] <= 0.45
        ),
    }
    payload["typed_summary"] = _typed_summary(rows)
    payload["cell_1544230"] = _cell_note(rows)

    # Partial --ids runs stay out of the published scoreboard names.
    suffix = "" if len(rows) == 12 and not args.ids.strip() else "_partial"
    jp = out_dir / f"r1_gate_diagnostic{suffix}.json"
    mp = out_dir / f"r1_gate_diagnostic{suffix}.md"
    jp.write_text(json.dumps(payload, indent=2) + "\n")
    mp.write_text(_markdown(payload))
    print(f"Wrote {jp}", flush=True)
    print(f"Wrote {mp}", flush=True)


if __name__ == "__main__":
    main()
