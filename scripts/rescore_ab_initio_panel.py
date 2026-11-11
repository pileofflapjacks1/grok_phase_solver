#!/usr/bin/env python3
"""Re-score finished ab_initio panel runs with the OLD (discrete v0) and NEW
(polar-shift-v1) grader without re-running any solve.

Reads the per-run peak files written by ``run_ab_initio_panel.py``
(``runs/<id>/seed_<n>/trial_winner.res`` + ``peaks_winner.csv`` and
``trial_cand_<name>.res`` + ``peaks_cand_<name>.csv``) from one or more ``--runs``
output dirs (later dirs override the same COD x seed, e.g. a re-run) and prints
centro PASS, non-centro PASS, wrong picks and the random baseline for both
graders.

    python scripts/rescore_ab_initio_panel.py --panel <PANEL_DIR> \\
        --runs <OUT_DIR> [<RERUN_DIR> ...] --report <report.md>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_ab_initio_panel as rap  # noqa: E402
import numpy as np  # noqa: E402

from grok_phase_solver.metrics.qpeak_benchmark import (  # noqa: E402
    DEFAULT_N_RANDOM, DEFAULT_SEED, evaluate, read_res_q, strength_order, verdict)


def score_files(cid, cif, res, csv, polar_shift):
    if not Path(res).exists():
        return {"verdict": "FAIL", "hit05": 0, "error": "no peak file", "rand_hit05_mean": None}
    g = rap._geom(cif)
    q_all = read_res_q(res)
    order, _ = strength_order(csv, g["ops"], q_all)
    n_use = min(g["n"], len(q_all))
    q = q_all[order][:n_use]
    real = evaluate(g["af"], g["a_orb"], q, g["ops"], g["orth"], polar_shift=polar_shift)
    ck = (str(cid), int(n_use), int(DEFAULT_N_RANDOM), int(DEFAULT_SEED), bool(polar_shift))
    if ck not in rap._RAND:
        rng = np.random.default_rng(DEFAULT_SEED)
        rap._RAND[ck] = np.array([
            evaluate(g["af"], g["a_orb"], rng.random((n_use, 3)), g["ops"], g["orth"],
                     polar_shift=polar_shift)["hit05"] for _ in range(DEFAULT_N_RANDOM)])
    r05 = rap._RAND[ck]
    n = g["n"]
    rate05 = real["hit05"] / n
    base = r05.mean() / n
    ratio = rate05 / base if base > 0 else float("inf")
    gf = real["ghost"] / n_use
    return {"hit05": real["hit05"], "ghost": real["ghost"], "rate05": rate05, "ratio05": ratio,
            "ghost_frac": gf, "rand_hit05_mean": float(r05.mean()),
            "verdict": verdict(rate05, ratio, gf)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, nargs="+")
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    sel = json.loads((a.panel / "SELECTION.json").read_text())["selected"]
    meta = {str(m["id"]): m for m in sel}
    cod = a.panel / "data" / "cod_qpeak"
    rows = {}
    for d in a.runs:
        for r in json.loads((d / "results_raw.json").read_text()):
            rows[(r["id"], r["seed"])] = (d, r)
    out = {"old": [], "new": []}
    for (cid, seed), (d, r) in sorted(rows.items()):
        rd = d / "runs" / cid / f"seed_{seed}"
        cif = cod / f"{cid}.cif"
        for key, polar in (("old", False), ("new", True)):
            win = score_files(cid, cif, rd / "trial_winner.res", rd / "peaks_winner.csv", polar)
            cands = []
            for c in r.get("candidates", []):
                tag = "cand_" + str(c["name"]).replace("/", "_")
                cands.append(score_files(cid, cif, rd / f"trial_{tag}.res", rd / f"peaks_{tag}.csv", polar))
            wrong = win["verdict"] == "FAIL" and any(c["verdict"] == "PASS" for c in cands)
            out[key].append({"id": cid, "seed": seed, "centro": bool(meta[cid].get("centro")),
                             "verdict": win["verdict"], "hit05": win["hit05"],
                             "pass": win["verdict"] == "PASS", "wrong_pick": wrong,
                             "rand": win["rand_hit05_mean"]})
        print(cid, seed, "old", out["old"][-1]["verdict"], out["old"][-1]["hit05"],
              "new", out["new"][-1]["verdict"], out["new"][-1]["hit05"], flush=True)
    lines = ["| grader | centro PASS | non-centro PASS | wrong picks | runs |", "|---|---:|---:|---:|---:|"]
    for key, label in (("old", "old (discrete v0)"), ("new", "new (polar-shift-v1)")):
        rs = out[key]
        c = [r for r in rs if r["centro"]]
        n = [r for r in rs if not r["centro"]]
        lines.append(f"| {label} | {sum(r['pass'] for r in c)}/{len(c)} | {sum(r['pass'] for r in n)}/{len(n)} | "
                     f"{sum(r['wrong_pick'] for r in rs)}/{len(rs)} | {len(rs)} |")
    lines += ["", "| COD | SG | centro | PASS old | PASS new | wrong old | wrong new | random hit05 mean old | new |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for cid in [str(m["id"]) for m in sel]:
        o = [r for r in out["old"] if r["id"] == cid]
        nw = [r for r in out["new"] if r["id"] == cid]
        if not o:
            continue
        g = rap._geom(cod / f"{cid}.cif")
        ro = [r["rand"] for r in o if r["rand"] is not None]
        rn = [r["rand"] for r in nw if r["rand"] is not None]
        lines.append(f"| {cid} | {g['hm']} | {meta[cid].get('centro')} | {sum(r['pass'] for r in o)}/{len(o)} | "
                     f"{sum(r['pass'] for r in nw)}/{len(nw)} | {sum(r['wrong_pick'] for r in o)} | "
                     f"{sum(r['wrong_pick'] for r in nw)} | {np.mean(ro):.2f} | {np.mean(rn):.2f} |")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if a.report:
        a.report.write_text("# Re-score of existing ab_initio panel runs: old vs new grader\n\n" + text)
        (a.report.with_suffix(".json")).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
