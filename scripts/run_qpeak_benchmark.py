#!/usr/bin/env python3
"""Q-peak benchmark: do ``gps-solve`` ``trial.res`` peaks sit on deposited atoms?

Runs ``gps-solve`` (default ``--method auto``) on three COD Fobs sets, then
scores the N strongest Q peaks against the deposited non-H atoms with the
pre-registered bar in ``grok_phase_solver.metrics.qpeak_benchmark``:
PASS needs >= 50% of atoms within 0.5 A, >= 3x a random baseline (mean of
``--n-random`` point sets, fixed seed), and <= 40% ghosts.

The CIF/HKL inputs are COD entries (CC0) committed under ``data/raw/cod_qpeak/``.
Writes ``data/processed/qpeak_benchmark.{md,json}``.

Usage (from repo root)::

    python scripts/run_qpeak_benchmark.py                  # full: 1000 random sets
    python scripts/run_qpeak_benchmark.py --ids 2200239
    python scripts/run_qpeak_benchmark.py --n-random 50 --no-write   # quick look
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.cli import solve_main  # noqa: E402
from grok_phase_solver.metrics.qpeak_benchmark import (  # noqa: E402
    DEFAULT_N_RANDOM,
    DEFAULT_SEED,
    planted_selftest,
    score_q_peaks,
)

COD_DIR = ROOT / "data" / "raw" / "cod_qpeak"
OUT_MD = ROOT / "data" / "processed" / "qpeak_benchmark.md"
OUT_JSON = ROOT / "data" / "processed" / "qpeak_benchmark.json"

# Cell and space group exactly as passed to gps-solve in the original run.
PANEL = {
    "2200239": ("11.3070,9.4750,14.9190,90,99.340,90", "P 21/c"),
    "2200001": ("8.920,16.282,18.504,90,90,90", "P 21 21 21"),
    "2230859": ("7.7275,9.4705,16.729,90,101.416,90", "P 21/n"),
}


def run_solve(cid: str, out: Path) -> float:
    cell, sg = PANEL[cid]
    argv = ["--hkl", str(COD_DIR / f"{cid}.hkl"), "--cell", cell, "--sg", sg, "--out", str(out)]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        solve_main(argv)
    return time.time() - t0


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)"


def render_md(rows, selftests, n_random: int, seed: int) -> str:
    lines = [
        "# Q-peak benchmark (`trial.res` vs deposited COD atoms)",
        "",
        "Written by `scripts/run_qpeak_benchmark.py`. Do not edit numbers by hand.",
        "",
        "Inputs: COD CIF + Fobs HKL (CC0) in `data/raw/cod_qpeak/`. Each set runs",
        "`gps-solve --hkl <id>.hkl --cell … --sg … --out …` (default `--method auto`), then the",
        "scorer in `grok_phase_solver.metrics.qpeak_benchmark` reads `trial.res` + `peaks.csv`.",
        "Full run: `python scripts/run_qpeak_benchmark.py` (~2.5 min CPU). Fast CI check:",
        "`pytest tests/test_qpeak_benchmark.py` (planted atoms + 20-draw baseline).",
        "",
        "**Bar (pre-registered, not tuned):** N = deposited non-H atoms; take the N strongest Q",
        "peaks; best alignment over origin shifts {0,½}³ × {identity, inversion} with full",
        "space-group orbits. **PASS:** ≥ 50% of atoms have a Q within 0.5 Å, that rate is",
        f"≥ 3× the random baseline (mean of {n_random} uniform point sets, seed {seed}, same",
        "alignment search), and ghosts (Q > 1.0 Å from every atom image) ≤ 40%.",
        "**PARTIAL:** 25–50% within 0.5 Å and > 3× baseline. **FAIL:** anything else.",
        "",
        (
            "| COD | SG | N non-H | ≤0.5 Å | ≤1.0 Å | ghosts | random ≤0.5 mean (sd, max) "
            "| random ghost | rate / baseline | P(random ≥ real) | verdict |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in rows:
        n, r, rd = o["N_nonH"], o["real"], o["rand"]
        lines.append(
            f"| {o['id']} | {o['sg']} | {n} | {_pct(r['hit05'], n)} | {_pct(r['hit10'], n)} "
            f"| {_pct(r['ghost'], o['n_Q_used'])} "
            f"| {rd['hit05_mean']:.2f} ({rd['hit05_sd']:.2f}, {rd['hit05_max']}) "
            f"| {100 * rd['ghost_frac_mean']:.0f}% | {o['ratio05']:.2f}× "
            f"| {o['p_random_ge_real_hit05']:.3f} | **{o['verdict']}** |"
        )
    lines += [
        "",
        "## Planted-atom self-test",
        "",
        "Deposited atoms planted as Q peaks after a random symmetry image per atom, inversion",
        "and an origin shift. The scorer must recover N/N within 0.5 Å; jitter adds Gaussian",
        "noise per axis.",
        "",
        "| COD | N | exact ≤0.5 Å | jitter σ | jitter ≤0.5 Å |",
        "|---|---|---|---|---|",
    ]
    for cid, s in selftests.items():
        lines.append(
            f"| {cid} | {s['N_nonH']} | {s['exact']['hit05']}/{s['N_nonH']} "
            f"| {s['jitter_sigma']} Å | {s['jitter']['hit05']}/{s['N_nonH']} |"
        )
    lines += [
        "",
        "Caveats: Q peaks sit on the map grid (~0.3–0.4 Å steps), which limits 0.5 Å hits even",
        "for a correct map. `trial.res` is `SFAC C` only, so element types are not scored.",
        "A random point set already has ~70–75% ghosts at 1.0 Å, so the 40% ghost bound is strict.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="*", default=list(PANEL))
    ap.add_argument("--n-random", type=int, default=DEFAULT_N_RANDOM)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--work", default=str(ROOT / ".tmp" / "qpeak_benchmark"),
                    help="gps-solve output root (one subfolder per COD id)")
    ap.add_argument("--skip-solve", action="store_true",
                    help="score existing <work>/<id>/trial.res instead of re-running gps-solve")
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not update data/processed")
    args = ap.parse_args(argv)

    rows, selftests = [], {}
    for cid in args.ids:
        out = Path(args.work) / cid
        if not args.skip_solve:
            dt = run_solve(cid, out)
            print(f"[{cid}] gps-solve {dt:.1f} s -> {out}")
        cif = COD_DIR / f"{cid}.cif"
        t0 = time.time()
        o = score_q_peaks(cid, cif, out / "trial.res", out / "peaks.csv",
                          n_random=args.n_random, seed=args.seed)
        o["solve_args"] = {"cell": PANEL[cid][0], "sg": PANEL[cid][1], "method": "auto"}
        rows.append(o)
        selftests[cid] = planted_selftest(cif)
        r = o["real"]
        print(f"[{cid}] {o['verdict']}: {r['hit05']}/{o['N_nonH']} within 0.5 A, "
              f"random mean {o['rand']['hit05_mean']:.2f}, ghosts {r['ghost']}/{o['n_Q_used']}, "
              f"self-test {selftests[cid]['exact']['hit05']}/{o['N_nonH']} "
              f"({time.time() - t0:.1f} s scoring)")

    if not args.no_write:
        OUT_JSON.write_text(json.dumps(
            {"n_random": args.n_random, "seed": args.seed, "rows": rows, "selftest": selftests},
            indent=1) + "\n")
        OUT_MD.write_text(render_md(rows, selftests, args.n_random, args.seed))
        print(f"wrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
