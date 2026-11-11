#!/usr/bin/env python3
"""Honest multi-seed ab_initio COD panel: min/median/max per method and chosen."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from grok_phase_solver.io.experiment import load_experiment
from grok_phase_solver.metrics.qpeak_benchmark import (
    DEFAULT_N_RANDOM,
    DEFAULT_SEED,
    read_cif_nonh,
    score_q_peaks,
)
from grok_phase_solver.solvers.ab_initio_cctbx import (
    BENCH_OUTER_SEEDS,
    TIME_BUDGET_S,
    ab_initio_solve,
)

COD_DIR = ROOT / "data" / "raw" / "cod_qpeak"
IDS = ["2200239", "2200001", "2230859"]
OUT = ROOT / "data" / "processed" / "ab_initio_bench"
OUTER_SEEDS = list(BENCH_OUTER_SEEDS)


def write_peaks_artifacts(out_dir: Path, peaks, cell, sg: str, stem: str = "peaks"):
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(peaks[0], "fract"):
        rows = [
            {
                "rank": p.rank,
                "fract": list(map(float, p.fract)),
                "height": float(p.height),
                "height_sigma": float(getattr(p, "height_sigma", 0.0)),
            }
            for p in peaks
        ]
    else:
        rows = []
        for i, p in enumerate(peaks):
            rows.append(
                {
                    "rank": i,
                    "fract": list(map(float, p["fract"])),
                    "height": float(p["height"]),
                    "height_sigma": float(p.get("height_sigma") or 0.0),
                }
            )
    csv = out_dir / f"{stem}.csv"
    with csv.open("w") as fh:
        fh.write("rank,x_frac,y_frac,z_frac,height,height_sigma\n")
        for p in rows:
            f = p["fract"]
            fh.write(
                f"{p['rank']},{f[0]:.6f},{f[1]:.6f},{f[2]:.6f},"
                f"{p['height']:.6f},{p['height_sigma']:.6f}\n"
            )
    a, b, c, al, be, ga = cell
    lines = [
        f"TITL ab_initio {stem}",
        f"CELL 0.71073 {a:.4f} {b:.4f} {c:.4f} {al:.3f} {be:.3f} {ga:.3f}",
        "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        "LATT 1",
        "SFAC C",
        "UNIT 1",
        f"REM space_group_hint={sg}",
        "FVAR 1.0",
    ]
    for i, p in enumerate(rows):
        f = p["fract"]
        lines.append(
            f"Q{i+1:<5d} 1 {f[0]:9.6f} {f[1]:9.6f} {f[2]:9.6f} 11.00000 0.05000"
        )
    lines += ["HKLF 4", "END"]
    (out_dir / f"{stem}.res").write_text("\n".join(lines) + "\n")
    return csv, out_dir / f"{stem}.res"


def score_peak_rows(cid, cif, out_dir, peaks, cell, sg, stem) -> Dict[str, Any]:
    if not peaks:
        return {"id": cid, "verdict": "FAIL", "error": "no peaks", "stem": stem}
    csv, res = write_peaks_artifacts(out_dir, peaks, cell, sg, stem=stem)
    sc = score_q_peaks(
        cid, cif, res, csv, n_random=DEFAULT_N_RANDOM, seed=DEFAULT_SEED
    )
    sc["stem"] = stem
    return sc



def _hit05(score: Dict[str, Any]) -> Optional[int]:
    real = score.get("real")
    if isinstance(real, dict):
        return real.get("hit05")
    return None


def _summary(vals: List[Optional[int]]) -> Dict[str, Any]:
    nums = [int(v) for v in vals if v is not None]
    if not nums:
        return {"min": None, "median": None, "max": None, "n": 0, "values": vals}
    return {
        "min": min(nums),
        "median": float(statistics.median(nums)),
        "max": max(nums),
        "n": len(nums),
        "values": vals,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel: Dict[str, Any] = {
        "outer_seeds": OUTER_SEEDS,
        "time_budget_s": TIME_BUDGET_S,
        "selection": "emap_check",
        "cods": {},
        "bad_selection_events": [],
    }
    n_bad = 0
    n_comparable = 0

    for cid in IDS:
        cif = COD_DIR / f"{cid}.cif"
        hkl_path = COD_DIR / f"{cid}.hkl"
        cell, sg, atoms = read_cif_nonh(cif)
        n_nonH = len(atoms)
        cell = np.asarray(cell, dtype=np.float64)
        table, _ins = load_experiment(str(hkl_path))
        print(
            f"=== {cid} SG={sg} n_nonH={n_nonH} n_refl={len(table)} "
            f"seeds={OUTER_SEEDS} ===",
            flush=True,
        )
        per_seed = []
        chosen_hits, tan_hits, cf_hits = [], [], []

        for seed in OUTER_SEEDS:
            out_dir = OUT / cid / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            phases, density, info, peaks = ab_initio_solve(
                table.hkl,
                table.amplitudes,
                cell,
                space_group=sg,
                n_nonH=n_nonH,
                seed=int(seed),
                time_budget_s=TIME_BUDGET_S,
                verbose=True,
                work_dir=out_dir / "work",
            )
            dt = time.time() - t0
            chosen_sc = score_peak_rows(
                cid, cif, out_dir, peaks, cell, sg or "?", "chosen"
            )
            # Per-method: best tangent candidate + best CF candidate by emap_score
            cands = list(info.get("candidates") or [])
            tan_cands = [c for c in cands if (c.get("method") or c.get("name")) == "tangent"]
            cf_cands = [
                c
                for c in cands
                if (c.get("method") == "charge_flip")
                or str(c.get("name", "")).startswith("charge_flip")
            ]

            def best_by_emap(group):
                if not group:
                    return None
                return max(group, key=lambda c: float(c.get("emap_score") or -1e99))

            tan_best = best_by_emap(tan_cands)
            cf_best = best_by_emap(cf_cands)
            tan_sc = (
                score_peak_rows(
                    cid, cif, out_dir, tan_best.get("peaks") or [], cell, sg or "?", "tangent"
                )
                if tan_best
                else {"verdict": "FAIL", "error": "no tangent candidate"}
            )
            cf_sc = (
                score_peak_rows(
                    cid, cif, out_dir, cf_best.get("peaks") or [], cell, sg or "?", "charge_flip"
                )
                if cf_best
                else {"verdict": "FAIL", "error": "no CF candidate"}
            )

            ch = _hit05(chosen_sc)
            th = _hit05(tan_sc)
            fh = _hit05(cf_sc)
            chosen_hits.append(ch)
            tan_hits.append(th)
            cf_hits.append(fh)

            chosen_pass = chosen_sc.get("verdict") == "PASS"
            tan_pass = tan_sc.get("verdict") == "PASS"
            cf_pass = cf_sc.get("verdict") == "PASS"
            any_pass = tan_pass or cf_pass
            bad = (not chosen_pass) and any_pass
            if any_pass:
                n_comparable += 1
            if bad:
                n_bad += 1
                panel["bad_selection_events"].append(
                    {
                        "cod": cid,
                        "seed": seed,
                        "chosen": info.get("winner"),
                        "chosen_verdict": chosen_sc.get("verdict"),
                        "chosen_hit05": ch,
                        "tangent_verdict": tan_sc.get("verdict"),
                        "tangent_hit05": th,
                        "cf_verdict": cf_sc.get("verdict"),
                        "cf_hit05": fh,
                    }
                )

            row = {
                "seed": seed,
                "wall_s": round(dt, 3),
                "winner": info.get("winner"),
                "winner_method": info.get("winner_method"),
                "emap_score": info.get("emap_score"),
                "chosen": {
                    "verdict": chosen_sc.get("verdict"),
                    "hit05": ch,
                    "ghost": (chosen_sc.get("real") or {}).get("ghost")
                    if isinstance(chosen_sc.get("real"), dict)
                    else None,
                },
                "tangent": {
                    "verdict": tan_sc.get("verdict"),
                    "hit05": th,
                    "emap_score": (tan_best or {}).get("emap_score"),
                    "fom": (tan_best or {}).get("fom"),
                },
                "charge_flip": {
                    "verdict": cf_sc.get("verdict"),
                    "hit05": fh,
                    "emap_score": (cf_best or {}).get("emap_score"),
                    "fom": (cf_best or {}).get("fom"),
                    "cf_seed": (cf_best or {}).get("cf_seed"),
                    "n_cf_candidates": len(cf_cands),
                },
                "bad_selection": bad,
            }
            (out_dir / "row.json").write_text(json.dumps(row, indent=2) + "\n")
            per_seed.append(row)
            print(
                f"[{cid} seed={seed}] chosen={row['winner']} "
                f"hit05={ch} ({chosen_sc.get('verdict')}) "
                f"tan={th}/{n_nonH} cf={fh}/{n_nonH} "
                f"bad_sel={bad} wall={dt:.1f}s",
                flush=True,
            )

        panel["cods"][cid] = {
            "n_nonH": n_nonH,
            "sg": sg,
            "chosen_hit05": _summary(chosen_hits),
            "tangent_hit05": _summary(tan_hits),
            "charge_flip_hit05": _summary(cf_hits),
            "per_seed": per_seed,
        }

    panel["bad_selection_count"] = n_bad
    panel["comparable_runs"] = n_comparable
    panel["bad_selection_rate"] = (
        (n_bad / n_comparable) if n_comparable else None
    )
    (OUT / "results_multiseed.json").write_text(json.dumps(panel, indent=2) + "\n")
    # Human table
    lines = [
        "# ab_initio multi-seed bench",
        f"seeds={OUTER_SEEDS} budget={TIME_BUDGET_S}s selection=emap_check",
        f"bad_selection={n_bad}/{n_comparable}",
        "",
        "| COD | N | chosen min/med/max | tangent min/med/max | CF min/med/max |",
        "|---|---:|---|---|---|",
    ]
    for cid in IDS:
        block = panel["cods"][cid]
        n = block["n_nonH"]

        def fmt(s):
            if s["min"] is None:
                return "n/a"
            return f"{s['min']}/{s['median']}/{s['max']} (of {n})"

        lines.append(
            f"| {cid} | {n} | {fmt(block['chosen_hit05'])} | "
            f"{fmt(block['tangent_hit05'])} | {fmt(block['charge_flip_hit05'])} |"
        )
    summary_txt = "\n".join(lines) + "\n"
    (OUT / "SUMMARY.md").write_text(summary_txt)
    print(summary_txt)
    print("Wrote", OUT / "results_multiseed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
