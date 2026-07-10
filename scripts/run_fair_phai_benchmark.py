#!/usr/bin/env python3
"""
Fair PhAI benchmark vs classical methods.

Protocol alignment with public PhAI notebook:
  - merge: reindex_monoclinic + average + /max(|F|)
  - grid packing max_index=10
  - n_cycles = 5
  - cell for COD 2016452 matches notebook defaults

Compares on:
  1) COD 2016452 Fcalc truth (P21/c — PhAI's native SG)
  2) COD 2100301 Fcalc at several d_min
  3) Optional: PhAI sample HKL file (no truth phases — diagnostics only)

Writes data/processed/fair_phai_benchmark.{json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.io.cif import load_cif, expand_asymmetric_unit
from grok_phase_solver.io.hkl import load_hkl_shelx
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval
from grok_phase_solver.solvers.phase_recycle import phase_recycle
from grok_phase_solver.models.phai_fair import run_phai_fair, pack_phai_amplitudes_fair
from grok_phase_solver.models.phai_runner import phai_available, find_model_path
from grok_phase_solver.physics.density import density_from_structure_factors


def _eval_method(name, hkl, amp, ph_true, cell, true_fracs, phases, density=None, extra=None):
    if density is None:
        density = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases), cell
        )
    rep = evaluate_success(
        hkl, amp, phases, ph_true, cell, true_fracs, density=density,
        thresholds=SuccessThresholds(),
    )
    row = {
        "dataset": name,
        "mapcc_oi": rep.mapcc_oi,
        "mpe_oi_deg": rep.mpe_oi_deg,
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": rep.solved,
    }
    if extra:
        row.update(extra)
    return row


def benchmark_structure(label, st, d_min, seed=0):
    data = structure_to_fcalc(st, d_min=d_min)
    hkl, amp, ph_true = data["hkl"], data["amplitudes"], data["phases"]
    true_fracs = data["fracs"]
    elements = data["elements"]
    cell = st.cell
    rows = []
    print(f"\n=== {label} @ {d_min} Å  Nrefl={len(amp)} Natoms_cell={data['n_atoms_cell']} ===")

    def _eval(name_method, phases, density=None, **extra):
        if density is None:
            density = density_from_structure_factors(
                hkl, amp * np.exp(1j * phases), cell, d_min=d_min
            )
        rep = evaluate_success(
            hkl, amp, phases, ph_true, cell, true_fracs, density=density,
            thresholds=SuccessThresholds(), elements=elements,
        )
        row = {
            "dataset": label,
            "method": name_method,
            "d_min": d_min,
            "mapcc_oi": rep.mapcc_oi,
            "mpe_oi_deg": rep.mpe_oi_deg,
            "peak_recovery": rep.peak_recovery,
            "r1": rep.r1,
            "solved": rep.solved,
        }
        row.update(extra)
        return row

    # CF
    t0 = time.time()
    ph, rho, hist = charge_flipping_solve(
        hkl, amp, cell, n_iter=100, seed=seed, d_min=d_min, centrosymmetric=True, verbose=False
    )
    row = _eval(
        "charge_flipping", ph, rho, seconds=time.time() - t0,
        final_R=hist.get("R", [None])[-1],
    )
    print(f"  CF          CC={row['mapcc_oi']:.3f} peak={row['peak_recovery']:.2f} "
          f"R1={row['r1']:.2f} solved={row['solved']}")
    rows.append(row)

    # recycle
    t0 = time.time()
    ph, rho, hist = phase_recycle(hkl, amp, cell, n_cycles=8, seed=seed, d_min=d_min)
    row = _eval("phase_recycle", ph, rho, seconds=time.time() - t0)
    print(f"  recycle     CC={row['mapcc_oi']:.3f} peak={row['peak_recovery']:.2f} "
          f"R1={row['r1']:.2f} solved={row['solved']}")
    rows.append(row)

    if not phai_available():
        rows.append({
            "dataset": label, "d_min": d_min, "method": "phai_fair",
            "status": "unavailable", "weights": str(find_model_path()),
        })
        print("  PhAI unavailable")
        return rows

    # Fair PhAI
    t0 = time.time()
    ph, meta = run_phai_fair(hkl, amp, n_cycles=5, random_init=True, seed=seed)
    rho = density_from_structure_factors(hkl, amp * np.exp(1j * ph), cell, d_min=d_min)
    row = _eval(
        "phai_fair", ph, rho, seconds=time.time() - t0,
        frac_mapped=meta.get("frac_input_mapped"),
        n_mapped=meta.get("n_mapped"),
        protocol=meta.get("protocol"),
    )
    print(f"  phai_fair   CC={row['mapcc_oi']:.3f} peak={row['peak_recovery']:.2f} "
          f"R1={row['r1']:.2f} solved={row['solved']} mapped={meta.get('frac_input_mapped'):.2f}")
    rows.append(row)

    # PhAI fair + CF polish
    t0 = time.time()
    ph2, rho2, _ = hybrid_phase_retrieval(
        hkl, amp, cell, ph, polish="charge_flipping", n_iter=80, seed=seed, verbose=False
    )
    row = _eval("phai_fair+CF", ph2, rho2, seconds=time.time() - t0)
    print(f"  phai_fair+CF CC={row['mapcc_oi']:.3f} peak={row['peak_recovery']:.2f} "
          f"R1={row['r1']:.2f} solved={row['solved']}")
    rows.append(row)

    # Also report packing meta once
    _, _, _, _, pack_meta = pack_phai_amplitudes_fair(hkl, amp)
    rows.append({
        "dataset": label, "d_min": d_min, "method": "_pack_meta", **pack_meta,
    })
    return rows


def main():
    rows = []
    print("Fair PhAI benchmark")
    print(f"PhAI available: {phai_available()}  weights: {find_model_path()}")

    # Primary: COD 2016452 (PhAI demo structure)
    cif_path = ROOT / "data/raw/cod/2016452.cif"
    if not cif_path.exists():
        import urllib.request

        urllib.request.urlretrieve(
            "https://www.crystallography.net/cod/2016452.cif", cif_path
        )
    st = load_cif(cif_path)
    print(st.summary())
    for d_min in (0.9, 1.2, 1.5, 2.0):
        rows.extend(benchmark_structure("COD_2016452", st, d_min=d_min))

    # Secondary: COD 2100301
    cif2 = ROOT / "data/raw/cod/2100301.cif"
    if cif2.exists():
        st2 = load_cif(cif2)
        for d_min in (0.9, 1.2, 1.5):
            rows.extend(benchmark_structure("COD_2100301", st2, d_min=d_min))

    # Sample HKL from PhAI (no truth)
    sample = ROOT / "data/raw/phai_samples/COD_2016452.hkl"
    if sample.exists() and phai_available():
        print("\n=== PhAI sample HKL (no ground-truth phases) ===")
        # notebook cell params
        cell = np.array([9.748, 8.89, 7.566, 90, 112.74, 90.0])
        table = load_hkl_shelx(sample, cell=cell)
        # file stores F not I? values look like F (0-1 range after possible prior scale)
        # load_hkl_shelx treats col3 as I and takes sqrt — FIX for this file
        # Re-read as F directly
        raw = []
        for line in sample.read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            h, k, l = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
            if h == k == l == 0:
                break
            raw.append((h, k, l, float(parts[3])))
        arr = np.array(raw)
        hkl = arr[:, :3].astype(int)
        fabs = arr[:, 3]
        ph, meta = run_phai_fair(hkl, fabs, n_cycles=5, seed=0)
        print(f"  Ran PhAI on sample HKL: mapped={meta['frac_input_mapped']:.2f} n={meta['n_mapped']}")
        rows.append({
            "dataset": "COD_2016452_phai_sample_hkl",
            "method": "phai_fair",
            "note": "no ground truth phases — smoke test only",
            **{k: meta[k] for k in ("frac_input_mapped", "n_mapped", "n_merged", "protocol")},
        })

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "fair_phai_benchmark.json"
    jp.write_text(json.dumps(rows, indent=2, default=float))

    # Markdown
    md = [
        "# Fair PhAI benchmark",
        "",
        "Protocol: PhAI official **reindex_monoclinic → merge average → /max(|F|) → "
        "notebook grid (max_index=10) → 5 recycle cycles**.",
        "",
        "Success criterion (same as solvability diagram): "
        "mapCC_OI≥0.70, peak_recovery≥0.50, R1≤0.45.",
        "",
        f"PhAI weights: `{find_model_path()}`",
        "",
        "| Dataset | d_min | Method | mapCC_OI | peak_rec | R1 | solved |",
        "|---------|-------|--------|----------|----------|----|--------|",
    ]
    for r in rows:
        if r.get("method") == "_pack_meta" or r.get("note"):
            continue
        if "error" in r or r.get("status") == "unavailable":
            md.append(
                f"| {r.get('dataset')} | {r.get('d_min')} | {r.get('method')} | — | — | — | no |"
            )
            continue
        md.append(
            f"| {r['dataset']} | {r.get('d_min', '')} | `{r['method']}` | "
            f"{r['mapcc_oi']:.3f} | {r['peak_recovery']:.2f} | {r['r1']:.2f} | "
            f"{'**yes**' if r['solved'] else 'no'} |"
        )

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Fair packing** is necessary but not sufficient for reproducing Science paper numbers:",
            "  training distribution, multi-start seeds, and experimental vs Fcalc still matter.",
            "- If `phai_fair` ≪ CF at atomic resolution on Fcalc, CF remains the right default.",
            "- If `phai_fair` > CF at low resolution / incomplete data, that is the target regime.",
            "- `phai_fair+CF` tests whether PhAI provides a useful **seed** for classical polish.",
            "",
            f"JSON: `{jp.relative_to(ROOT)}`",
            "",
        ]
    )
    mp = out_dir / "fair_phai_benchmark.md"
    mp.write_text("\n".join(md))
    print(f"\nWrote {jp}")
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
