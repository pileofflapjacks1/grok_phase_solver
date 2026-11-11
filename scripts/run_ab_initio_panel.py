#!/usr/bin/env python3
"""Run ``--method ab_initio`` over a locked COD panel and print the four bar numbers.

Panel layout (same as the held-out panels used for PR #17)::

    <panel>/SELECTION.json            {"selected": [{"id": "2019828", "centro": true, ...}, ...]}
    <panel>/data/cod_qpeak/<id>.cif   COD CIF (cell, space group, non-H atoms)
    <panel>/data/cod_qpeak/<id>.hkl   COD reflection file

Usage (Olex2 bundle provides cctbx/smtbx; ~1 min per run, 5 seeds x 8 files)::

    export OLEX2_ROOT=/workspace/olex2/olex2
    python scripts/run_ab_initio_panel.py --panel /path/to/panel --out /path/to/out

Outputs in ``--out``: ``results_raw.json`` (one row per COD x seed),
``SUMMARY.md`` and the same block printed at the end::

    centro PASS      18/20   bar >=18   OK
    non-centro PASS  12/20   bar >=12   OK
    wrong picks       1/40   bar <=2    OK
    runs > budget+30s 0/40   bar 0      OK

Scorer = the unchanged ``metrics.qpeak_benchmark`` (top-N, 0.5 A, PASS needs
>=50% hits, >=3x random, ghosts <=40%). **Wrong pick** = chosen candidate FAILs
while at least one candidate of that run PASSes. Re-runs are resumable
(finished COD x seed rows in ``results_raw.json`` are skipped).

This script never selects, swaps or tunes COD entries: it runs exactly the
list in SELECTION.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# solvers <-> pipeline import cycle: import pipeline first (as the tests do).
import grok_phase_solver.pipeline.peaks  # noqa: F401,E402
import numpy as np  # noqa: E402

from grok_phase_solver.io.experiment import load_reflections  # noqa: E402
from grok_phase_solver.metrics.qpeak_benchmark import (  # noqa: E402
    DEFAULT_N_RANDOM,
    DEFAULT_SEED,
    evaluate,
    orbit,
    orth_matrix,
    read_cif_nonh,
    read_res_q,
    space_group_ops,
    strength_order,
    verdict,
)
from grok_phase_solver.solvers import ab_initio_cctbx as ab  # noqa: E402

DEFAULT_SEEDS = (0, 1, 7, 42, 2025)
os.environ.setdefault("OLEX2_ROOT", "/workspace/olex2/olex2")

_GEOM = {}
_RAND = {}


def _geom(cif: Path):
    key = str(cif)
    if key not in _GEOM:
        cell, hm, atoms = read_cif_nonh(cif)
        ops = space_group_ops(hm)
        af = np.array([a[2] for a in atoms])
        _GEOM[key] = dict(cell=cell, hm=hm, atoms=atoms, orth=orth_matrix(cell),
                          ops=ops, af=af, a_orb=orbit(af, ops), n=len(af))
    return _GEOM[key]


def _write_peaks(out_dir: Path, peaks, cell, sg: str, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"peaks_{tag}.csv"
    with csv.open("w") as fh:
        fh.write("rank,x_frac,y_frac,z_frac,height,height_sigma\n")
        for i, p in enumerate(peaks):
            f = p["fract"]
            h = float(p["height"])
            fh.write(f"{i},{f[0]:.6f},{f[1]:.6f},{f[2]:.6f},{h:.6f},{h:.6f}\n")
    a, b, c, al, be, ga = cell
    lines = [f"TITL ab_initio {tag}",
             f"CELL 0.71073 {a:.4f} {b:.4f} {c:.4f} {al:.3f} {be:.3f} {ga:.3f}",
             "ZERR 1 0.001 0.001 0.001 0.01 0.01 0.01", "LATT 1", "SFAC C", "UNIT 1",
             f"REM space_group_hint={sg}", "FVAR 1.0"]
    for i, p in enumerate(peaks):
        f = p["fract"]
        lines.append(f"Q{i+1:<5d} 1 {f[0]:9.6f} {f[1]:9.6f} {f[2]:9.6f} 11.00000 0.05000")
    lines += ["HKLF 4", "END"]
    res = out_dir / f"trial_{tag}.res"
    res.write_text("\n".join(lines) + "\n")
    return res, csv


def score_peaks(cid, cif: Path, cell, sg, peaks, out_dir: Path, tag: str, n_nonH: int):
    """Unchanged qpeak_benchmark scoring of the top-N peaks (plain 1.2 A pick)."""
    if not peaks:
        return {"verdict": "FAIL", "hit05": 0, "error": "no peaks"}
    sites = [p["fract"] for p in peaks]
    heights = [float(p["height"]) for p in peaks]
    picked = ab.plain_peak_pick(sites, heights, cell, n_keep=n_nonH)
    plist = [{"fract": list(p.fract), "height": p.height} for p in picked]
    res, csv = _write_peaks(out_dir, plist, cell, sg or "?", tag)
    g = _geom(cif)
    q_all = read_res_q(res)
    order, _how = strength_order(csv, g["ops"], q_all)
    n_use = min(g["n"], len(q_all))
    q = q_all[order][:n_use]
    real = evaluate(g["af"], g["a_orb"], q, g["ops"], g["orth"])
    ck = (str(cid), int(n_use), int(DEFAULT_N_RANDOM), int(DEFAULT_SEED))
    if ck not in _RAND:
        rng = np.random.default_rng(DEFAULT_SEED)
        _RAND[ck] = np.array([
            evaluate(g["af"], g["a_orb"], rng.random((n_use, 3)), g["ops"], g["orth"])["hit05"]
            for _ in range(DEFAULT_N_RANDOM)])
    r05 = _RAND[ck]
    n = g["n"]
    rate05 = real["hit05"] / n
    base = r05.mean() / n
    ratio05 = rate05 / base if base > 0 else float("inf")
    ghost_frac = real["ghost"] / n_use
    return {"hit05": real["hit05"], "ghost": real["ghost"], "N_nonH": n,
            "rate05": rate05, "ratio05": ratio05, "ghost_frac": ghost_frac,
            "verdict": verdict(rate05, ratio05, ghost_frac)}


def run_one(panel: Path, out: Path, meta: dict, seed: int, budget: float, grace: float):
    cid = str(meta["id"])
    cod = panel / "data" / "cod_qpeak"
    cif, hkl = cod / f"{cid}.cif", cod / f"{cid}.hkl"
    run_dir = out / "runs" / cid / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cell, sg, atoms = read_cif_nonh(cif)
    cell = np.asarray(cell, dtype=np.float64)
    n_nonH = len(atoms)
    table = load_reflections(str(hkl), cell=cell, space_group=sg)
    limit = budget + grace
    print(f"=== {cid} seed={seed} SG={sg} N={n_nonH} ===", flush=True)
    row = {"id": cid, "seed": seed, "sg": sg, "N_nonH": n_nonH,
           "centro": bool(meta.get("centro")), "limit_s": limit}
    t0 = time.time()
    info = None
    try:
        _ph, _den, info, peaks = ab.ab_initio_solve(
            table.hkl, table.amplitudes, cell, space_group=sg, n_nonH=n_nonH,
            seed=seed, time_budget_s=budget, grace_s=grace, work_dir=run_dir / "work")
        row["status"] = info.get("status")
    except ab.AbInitioTimeoutError as e:
        info = dict(e.info)
        peaks = []
        row["status"] = "timeout_no_result"
        row["error"] = str(e)
    except Exception as e:  # noqa: BLE001 - recorded, counts as non-pass
        traceback.print_exc()
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        info = {}
        peaks = []
    wall = time.time() - t0
    row["wall_s"] = round(wall, 3)
    row["over_limit"] = bool(wall > limit)
    for k in ("child_wall_s", "child_monotonic_s", "clock_jump_s", "suspend_suspected",
              "host_killed", "timed_out", "winner", "selection", "refined_wR2",
              "refined_R1", "peak_r1", "emap_score"):
        row[k] = info.get(k)

    win = score_peaks(cid, cif, cell, sg,
                      [{"fract": list(p.fract), "height": p.height} for p in peaks],
                      run_dir, "winner", n_nonH)
    cands = []
    for c in (info.get("candidates") or []):
        sc = score_peaks(cid, cif, cell, sg, c.get("peaks") or [], run_dir,
                         "cand_" + str(c.get("name")).replace("/", "_"), n_nonH)
        cands.append({"name": c.get("name"), "verdict": sc.get("verdict"),
                      "hit05": sc.get("hit05"), "refined_wR2": c.get("refined_wR2"),
                      "peak_r1": c.get("peak_r1"), "emap_score": c.get("emap_score")})
    any_pass = any(c["verdict"] == "PASS" for c in cands)
    row["winner_score"] = win
    row["candidates"] = cands
    row["pass"] = win.get("verdict") == "PASS"
    row["wrong_pick"] = bool(win.get("verdict") == "FAIL" and any_pass)
    (run_dir / "row.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(f"[{cid} seed={seed}] {win.get('verdict')} hit05={win.get('hit05')}/{n_nonH} "
          f"winner={row['winner']} sel={row['selection']} wrong_pick={row['wrong_pick']} "
          f"timed_out={row['timed_out']} wall={wall:.1f}s over_limit={row['over_limit']} "
          f"clock_jump={row['clock_jump_s']}", flush=True)
    return row


def summarize(rows, ids_order, bar, budget, grace):
    limit = budget + grace
    ok = [r for r in rows]
    c = [r for r in ok if r.get("centro")]
    n = [r for r in ok if not r.get("centro")]
    c_pass, n_pass = sum(r["pass"] for r in c), sum(r["pass"] for r in n)
    wrong = sum(bool(r.get("wrong_pick")) for r in ok)
    over = sum(bool(r.get("over_limit")) for r in ok)
    walls = [r["wall_s"] for r in ok if r.get("wall_s") is not None]
    monos = [r["child_monotonic_s"] for r in ok if r.get("child_monotonic_s") is not None]
    susp = sum(bool(r.get("suspend_suspected")) for r in ok)
    killed = sum(bool(r.get("host_killed")) for r in ok)
    errs = sum(1 for r in ok if r.get("status") in ("error", "timeout_no_result"))
    res = {
        "centro": (c_pass, len(c)), "noncentro": (n_pass, len(n)),
        "wrong_picks": (wrong, len(ok)), "over_limit": (over, len(ok)),
        "checks": {
            "centro": c_pass >= bar["centro"],
            "noncentro": n_pass >= bar["noncentro"],
            "wrong_picks": wrong <= bar["wrong"],
            "over_limit": over <= bar["over"],
        },
    }
    res["overall_pass"] = all(res["checks"].values())
    f = lambda b: "OK" if b else "FAIL"  # noqa: E731
    lines = [
        f"PANEL RESULT  (budget {budget:g}s + grace {grace:g}s = {limit:g}s limit)",
        f"  centro PASS        {c_pass:>2}/{len(c):<3} bar >= {bar['centro']:<2}  {f(res['checks']['centro'])}",
        f"  non-centro PASS    {n_pass:>2}/{len(n):<3} bar >= {bar['noncentro']:<2}  {f(res['checks']['noncentro'])}",
        f"  wrong picks        {wrong:>2}/{len(ok):<3} bar <= {bar['wrong']:<2}  {f(res['checks']['wrong_picks'])}",
        f"  runs > {limit:g}s      {over:>2}/{len(ok):<3} bar <= {bar['over']:<2}  {f(res['checks']['over_limit'])}",
        f"  OVERALL: {'PASS' if res['overall_pass'] else 'FAIL'}",
        "",
        f"  wall_s median/max: {statistics.median(walls):.1f} / {max(walls):.1f}" if walls else "  wall_s: n/a",
        f"  child monotonic_s median/max: {statistics.median(monos):.1f} / {max(monos):.1f}" if monos else "",
        f"  host-killed runs: {killed}; error/no-result runs: {errs}; "
        f"runs with wall-vs-monotonic clock jump > {ab.CLOCK_JUMP_SUSPECT_S:g}s (machine suspended?): {susp}",
        "  (bar is defined for 20 centro / 20 non-centro / 40 runs; counts above are actual)",
    ]
    block = "\n".join(l for l in lines if l != "")
    md = ["# ab_initio panel summary", "", "```", block, "```", "",
          "| COD | SG | centro | N | PASS | wrong | max wall_s | timed_out |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for cid in ids_order:
        rs = [r for r in ok if r["id"] == cid]
        if not rs:
            continue
        md.append(
            f"| {cid} | {rs[0]['sg']} | {rs[0]['centro']} | {rs[0]['N_nonH']} | "
            f"{sum(r['pass'] for r in rs)}/{len(rs)} | {sum(bool(r.get('wrong_pick')) for r in rs)} | "
            f"{max(r['wall_s'] for r in rs):.1f} | {sum(bool(r.get('timed_out')) for r in rs)} |")
    md += ["", "Per-run rows: `results_raw.json`.", ""]
    return res, block, "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--ids", nargs="*", default=None, help="subset of SELECTION.json ids (default: all)")
    ap.add_argument("--budget", type=float, default=ab.TIME_BUDGET_S)
    ap.add_argument("--grace", type=float, default=ab.CF_SUBPROCESS_GRACE_S)
    ap.add_argument("--bar-centro", type=int, default=18)
    ap.add_argument("--bar-noncentro", type=int, default=12)
    ap.add_argument("--bar-wrong", type=int, default=2)
    ap.add_argument("--bar-over", type=int, default=0)
    a = ap.parse_args(argv)

    sel = json.loads((a.panel / "SELECTION.json").read_text())["selected"]
    ids_order = [str(m["id"]) for m in sel]
    metas = [m for m in sel if a.ids is None or str(m["id"]) in set(a.ids)]
    a.out.mkdir(parents=True, exist_ok=True)
    raw_path = a.out / "results_raw.json"
    rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
    done = {(r["id"], r["seed"]) for r in rows if r.get("status") not in (None,)}
    for m in metas:
        for seed in a.seeds:
            if (str(m["id"]), seed) in done:
                print(f"skip done {m['id']} seed={seed}", flush=True)
                continue
            row = run_one(a.panel, a.out, m, seed, a.budget, a.grace)
            rows = [r for r in rows if not (r["id"] == row["id"] and r["seed"] == seed)] + [row]
            raw_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    bar = {"centro": a.bar_centro, "noncentro": a.bar_noncentro,
           "wrong": a.bar_wrong, "over": a.bar_over}
    res, block, md = summarize(rows, ids_order, bar, a.budget, a.grace)
    (a.out / "SUMMARY.md").write_text(md)
    print("\n" + block)
    return 0 if res["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
