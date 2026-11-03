#!/usr/bin/env python3
"""Benchmark ab_initio on the COD qpeak panel (honest scores)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from grok_phase_solver.solvers.ab_initio_cctbx import ab_initio_solve

COD_DIR = ROOT / "data" / "raw" / "cod_qpeak"
IDS = ["2200239", "2200001", "2230859"]
OUT = ROOT / "data" / "processed" / "ab_initio_bench"


def write_peaks_artifacts(out_dir: Path, peaks, cell, sg: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "peaks.csv"
    with csv.open("w") as fh:
        fh.write("rank,x_frac,y_frac,z_frac,height,height_sigma\n")
        for p in peaks:
            fh.write(
                f"{p.rank},{p.fract[0]:.6f},{p.fract[1]:.6f},{p.fract[2]:.6f},"
                f"{p.height:.6f},{p.height_sigma:.6f}\n"
            )
    a, b, c, al, be, ga = cell
    lines = [
        "TITL ab_initio peaks",
        f"CELL 0.71073 {a:.4f} {b:.4f} {c:.4f} {al:.3f} {be:.3f} {ga:.3f}",
        "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01",
        "LATT 1",
        "SFAC C",
        "UNIT 1",
        f"REM space_group_hint={sg}",
        "FVAR 1.0",
    ]
    for i, p in enumerate(peaks):
        lines.append(
            f"Q{i+1:<5d} 1 {p.fract[0]:9.6f} {p.fract[1]:9.6f} {p.fract[2]:9.6f} 11.00000 0.05000"
        )
    lines += ["HKLF 4", "END"]
    (out_dir / "trial.res").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in IDS:
        cif = COD_DIR / f"{cid}.cif"
        hkl_path = COD_DIR / f"{cid}.hkl"
        out_dir = OUT / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        cell, sg, atoms = read_cif_nonh(cif)
        n_nonH = len(atoms)
        cell = np.asarray(cell, dtype=np.float64)
        table, _ins = load_experiment(str(hkl_path))
        # Prefer CIF cell/SG; amplitudes from HKL table
        print(
            f"=== {cid} SG={sg} n_nonH={n_nonH} n_refl={len(table)} ===",
            flush=True,
        )
        t0 = time.time()
        phases, density, info, peaks = ab_initio_solve(
            table.hkl,
            table.amplitudes,
            cell,
            space_group=sg,
            n_nonH=n_nonH,
            seed=int(cid) % 100000,
            time_budget_s=60.0,
            verbose=True,
            work_dir=out_dir / "work",
        )
        dt = time.time() - t0
        write_peaks_artifacts(out_dir, peaks, cell, sg or "?")
        if len(peaks) == 0:
            score = {
                "id": cid,
                "verdict": "FAIL",
                "error": "no peaks returned",
                "wall_s": round(dt, 3),
                "info": {
                    k: info.get(k)
                    for k in (
                        "winner", "fom", "candidates", "defaults",
                        "worker", "status", "cctbx_runner",
                    )
                },
            }
        else:
            score = score_q_peaks(
                cid,
                cif,
                out_dir / "trial.res",
                out_dir / "peaks.csv",
                n_random=DEFAULT_N_RANDOM,
                seed=DEFAULT_SEED,
            )
            score["wall_s"] = round(dt, 3)
            score["info"] = {
                k: info.get(k)
                for k in (
                    "winner", "fom", "candidates", "defaults",
                    "worker", "status", "cctbx_runner",
                )
            }
        (out_dir / "score.json").write_text(json.dumps(score, indent=2) + "\n")
        rows.append(score)
        print(
            f"[{cid}] verdict={score.get('verdict')} "
            f"hits05={score.get('real', {}).get('hit05') if isinstance(score.get('real'), dict) else score.get('real')} "
            f"winner={info.get('winner')} fom={info.get('fom')} "
            f"peaks={len(peaks)} wall={dt:.1f}s "
            f"cf_err={(info.get('worker') or {}).get('charge_flip', {}).get('error') if isinstance(info.get('worker'), dict) else None}",
            flush=True,
        )

    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Wrote", OUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
