#!/usr/bin/env python3
"""
High-impact scoreboard: multi-method, multi-resolution comparison.

Methods:
  random, direct_methods, charge_flipping, phase_recycle (physics),
  phai (if weights available), phai+CF / recycle hybrids

Writes:
  data/processed/scoreboard.json
  data/processed/scoreboard.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc, run_physics_baseline
from grok_phase_solver.solvers.phase_recycle import phase_recycle
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval
from grok_phase_solver.models.phai_interface import PhAIInterface
from grok_phase_solver.models.phai_runner import phai_available, PhAIRunner, find_model_path


def _eval(hkl, amp, phases_pred, phases_true, cell):
    Fp = amp * np.exp(1j * phases_pred)
    Ft = amp * np.exp(1j * phases_true)
    rho_p = density_from_structure_factors(hkl, Fp, cell)
    rho_t = density_from_structure_factors(hkl, Ft, cell, shape=rho_p.shape)
    cc, _ = map_correlation_origin_invariant(rho_p, rho_t)
    mpe, _ = mean_phase_error_origin_invariant(
        phases_pred, phases_true, hkl, weights=amp
    )
    return {"mapCC_OI": float(cc), "MPE_OI_deg": float(mpe)}


def run_methods_on_data(name, hkl, amp, phases_true, cell, d_min, seed=0):
    rows = []
    # Classical via baseline helpers where possible
    # We have Fcalc truth already — build a lightweight structure-free path

    # random
    rng = np.random.default_rng(seed)
    ph = rng.uniform(-np.pi, np.pi, len(amp))
    m = _eval(hkl, amp, ph, phases_true, cell)
    rows.append({"dataset": name, "d_min": d_min, "method": "random", **m})

    # charge flipping
    t0 = time.time()
    from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

    ph, _, hist = charge_flipping_solve(
        hkl, amp, cell, n_iter=100, seed=seed, d_min=d_min, verbose=False
    )
    m = _eval(hkl, amp, ph, phases_true, cell)
    rows.append(
        {
            "dataset": name,
            "d_min": d_min,
            "method": "charge_flipping",
            **m,
            "seconds": time.time() - t0,
            "final_R": hist.get("R", [None])[-1],
        }
    )

    # physics phase recycle (positivity)
    t0 = time.time()
    ph, _, hist = phase_recycle(
        hkl, amp, cell, n_cycles=8, seed=seed, d_min=d_min, use_positivity=True
    )
    m = _eval(hkl, amp, ph, phases_true, cell)
    rows.append(
        {
            "dataset": name,
            "d_min": d_min,
            "method": "phase_recycle",
            **m,
            "seconds": time.time() - t0,
            "final_R": hist.get("final_R"),
        }
    )

    # direct methods
    t0 = time.time()
    from grok_phase_solver.solvers.direct_methods import direct_methods_solve

    dm = direct_methods_solve(
        hkl, amp, cell, n_atoms_approx=50, n_trials=25, seed=seed, verbose=False
    )
    m = _eval(hkl, amp, dm.phases_full, phases_true, cell)
    rows.append(
        {
            "dataset": name,
            "d_min": d_min,
            "method": "direct_methods",
            **m,
            "seconds": time.time() - t0,
            "dm_fom": dm.history.get("best_fom"),
        }
    )

    # PhAI if available
    if phai_available():
        try:
            t0 = time.time()
            runner = PhAIRunner(device="cpu")
            ph, info = runner.predict(hkl, amp, n_cycles=5, random_init=True, seed=seed)
            m = _eval(hkl, amp, ph, phases_true, cell)
            rows.append(
                {
                    "dataset": name,
                    "d_min": d_min,
                    "method": "phai",
                    **m,
                    "seconds": time.time() - t0,
                    "n_mapped": info["n_mapped"],
                }
            )
            # PhAI + CF polish
            t0 = time.time()
            ph2, _, _ = hybrid_phase_retrieval(
                hkl, amp, cell, ph, polish="charge_flipping", n_iter=60, seed=seed
            )
            m = _eval(hkl, amp, ph2, phases_true, cell)
            rows.append(
                {
                    "dataset": name,
                    "d_min": d_min,
                    "method": "phai+CF",
                    **m,
                    "seconds": time.time() - t0,
                }
            )
            # PhAI + physics recycle
            t0 = time.time()
            ph3, _, _ = phase_recycle(
                hkl, amp, cell, n_cycles=5, phase_init=ph, seed=seed, d_min=d_min
            )
            m = _eval(hkl, amp, ph3, phases_true, cell)
            rows.append(
                {
                    "dataset": name,
                    "d_min": d_min,
                    "method": "phai+recycle",
                    **m,
                    "seconds": time.time() - t0,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "dataset": name,
                    "d_min": d_min,
                    "method": "phai",
                    "error": str(e),
                }
            )
    else:
        rows.append(
            {
                "dataset": name,
                "d_min": d_min,
                "method": "phai",
                "status": "unavailable",
                "weights_found": str(find_model_path()),
                "note": "Install torch+einops; download PhAI_model.pth (see third_party/phai/README.md)",
            }
        )

    return rows


def main():
    out_rows = []
    print("=" * 70)
    print("SCOREBOARD — high-impact multi-method comparison")
    print(f"PhAI available: {phai_available()}  weights: {find_model_path()}")
    print("=" * 70)

    # --- Synthetic tiny (should be solvable by CF) ---
    print("\n[1] Synthetic 5-atom P1 @ 1.0 Å")
    st = generate_random_organic(n_atoms=5, seed=0)
    data = structure_to_fcalc(st, d_min=1.0)
    rows = run_methods_on_data(
        "synth_n5", data["hkl"], data["amplitudes"], data["phases"], st.cell, 1.0
    )
    for r in rows:
        print(" ", r)
    out_rows.extend(rows)

    # --- COD 2100301 resolution series (flagship) ---
    cif = ROOT / "data/raw/cod/2100301.cif"
    if cif.exists():
        st = load_cif(cif)
        for d_min in (0.9, 1.2, 1.5, 2.0):
            print(f"\n[2] COD 2100301 @ {d_min} Å")
            data = structure_to_fcalc(st, d_min=d_min)
            print(f"  n_atoms_cell={data['n_atoms_cell']}  n_refl={len(data['amplitudes'])}")
            rows = run_methods_on_data(
                "COD_2100301",
                data["hkl"],
                data["amplitudes"],
                data["phases"],
                st.cell,
                d_min,
            )
            for r in rows:
                print(" ", r)
            out_rows.extend(rows)
    else:
        print("COD 2100301 missing")

    # Write JSON
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scoreboard.json"

    def conv(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    json_path.write_text(json.dumps(out_rows, indent=2, default=conv))

    # Write markdown table
    md_lines = [
        "# Phase problem scoreboard",
        "",
        f"Generated by `scripts/run_scoreboard.py`. PhAI available: **{phai_available()}**.",
        "",
        "Metrics: **mapCC_OI** = origin/enantiomorph-invariant map correlation (primary); "
        "**MPE_OI** = origin-invariant mean phase error (degrees).",
        "",
        "| Dataset | d_min (Å) | Method | mapCC_OI | MPE_OI (°) | notes |",
        "|---------|-----------|--------|----------|------------|-------|",
    ]
    for r in out_rows:
        if "error" in r or r.get("status") == "unavailable":
            md_lines.append(
                f"| {r.get('dataset','')} | {r.get('d_min','')} | {r.get('method','')} | — | — | {r.get('error') or r.get('note','unavailable')} |"
            )
            continue
        note = ""
        if "final_R" in r and r["final_R"] == r["final_R"]:
            note = f"R={r['final_R']:.3f}" if isinstance(r["final_R"], float) else ""
        if "n_mapped" in r:
            note = f"mapped {r['n_mapped']}"
        md_lines.append(
            f"| {r['dataset']} | {r['d_min']} | `{r['method']}` | "
            f"{r.get('mapCC_OI', float('nan')):.3f} | {r.get('MPE_OI_deg', float('nan')):.1f} | {note} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation (truth-seeking)",
            "",
            "- **mapCC_OI > 0.8**: density essentially solved (up to origin/hand).",
            "- **0.4–0.8**: partial phase information; may need hybrid polish or better prior.",
            "- **~0.3**: near random for amplitude-only maps (Patterson-like residual correlation).",
            "- PhAI public model is **P21/c-oriented** and index-limited (max_index=10); "
            "COD 2100301 is P21/c — a fair-ish test when reflections fit the grid.",
            "- Physics `phase_recycle` is ER-style positivity recycling, **not** the PhAI network.",
            "",
            f"JSON: `{json_path.relative_to(ROOT)}`",
            "",
        ]
    )
    md_path = out_dir / "scoreboard.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
