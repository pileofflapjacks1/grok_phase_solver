#!/usr/bin/env python3
"""
Extended AI-PhaSeed benchmark (Carrozzini 2025 alignment).

Targets:
  - Synthetic easy / hard cells
  - Optional COD CIFs under data/raw/cod/ (larger Vol / P2₁/c when present)
  - Stratify by resolution, volume, and predicted seed Class (0/1)

Compares:
  - charge_flipping
  - ai_phaseed (oracle partial / true seed when available)
  - ai_phaseed + dm_ai hybrid
  - phai_phaseed (if PhAI available)

Writes:
  data/processed/ai_phaseed_extended_benchmark.{json,md}

Does **not** claim to reproduce the full 1505-structure COD panel of
Carrozzini et al. (2025); this is a reproducible subset harness.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.seed_quality import (
    oracle_seed_metrics,
    predict_seed_quality,
)
from grok_phase_solver.metrics.success import SuccessThresholds, evaluate_success
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.ai_phaseed import (
    ai_phaseed_solve,
    phai_phaseed_solve,
    select_seed_indices,
)
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.projectors import unit_cell_volume


def _eval(hkl, amp, ph, ph_true, cell, fracs, elements, rho, d_min):
    if rho is None:
        rho = density_from_structure_factors(
            hkl, amp * np.exp(1j * ph), cell, d_min=d_min
        )
    rep = evaluate_success(
        hkl, amp, ph, ph_true, cell, fracs, density=rho,
        elements=elements, thresholds=SuccessThresholds(),
    )
    return {
        "mapcc_oi": rep.mapcc_oi,
        "peak_recovery": rep.peak_recovery,
        "r1": rep.r1,
        "solved": rep.solved,
    }


def _vol_band(vol: float) -> str:
    if vol < 800:
        return "small_<800"
    if vol <= 3500:
        return "mid_800_3500"  # Carrozzini hybrid-friendly
    if vol <= 8000:
        return "large_3500_8000"
    return "xl_>8000"


def run_one_structure(
    name: str,
    hkl,
    amp,
    ph_true,
    cell,
    fracs,
    elements,
    d_min: float,
    *,
    n_extend: int = 12,
    n_starts: int = 1,
    seed: int = 0,
    n_atoms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    vol = float(unit_cell_volume(np.asarray(cell, dtype=np.float64)))
    seed_idx = select_seed_indices(hkl, amp, cell, seed_fraction=0.25)
    # Seed quality on true phases (oracle upper bound) and noisy partial
    sq_true = predict_seed_quality(
        hkl, amp, cell, ph_true, seed_idx=seed_idx, d_min=d_min, n_atoms_user=n_atoms
    )
    oracle = oracle_seed_metrics(ph_true, ph_true, seed_idx, amplitudes=amp)

    def add(method: str, ph, rho, t0, **extra):
        r = _eval(hkl, amp, ph, ph_true, cell, fracs, elements, rho, d_min)
        r.update(
            {
                "dataset": name,
                "d_min": d_min,
                "volume": vol,
                "vol_band": _vol_band(vol),
                "n_atoms": n_atoms,
                "method": method,
                "seconds": time.time() - t0,
                "seed_class_pred": sq_true.get("predicted_class"),
                "seed_p_success": sq_true.get("success_probability"),
                "oracle_MPE_seed": oracle.get("MPE_seed_deg"),
                "oracle_CORR_seed": oracle.get("CORR_seed"),
            }
        )
        r.update(extra)
        rows.append(r)
        flag = "SOLVED" if r["solved"] else "fail"
        print(
            f"  {flag:6s} {method:22s} CC={r['mapcc_oi']:.3f} "
            f"R1={r['r1']:.2f} class={r['seed_class_pred']} "
            f"Vol={vol:.0f} {r['vol_band']}"
        )
        return r

    # CF
    t0 = time.time()
    ph, rho, _ = charge_flipping_solve(
        hkl, amp, cell, n_iter=60, seed=seed, d_min=d_min
    )
    add("charge_flipping", ph, rho, t0)

    # Oracle AI-PhaSeed (true phases as AI prior)
    t0 = time.time()
    ph, rho, info = ai_phaseed_solve(
        hkl, amp, cell, ph_true,
        seed_fraction=0.25, n_extend=n_extend, polish="none",
        n_starts=n_starts, seed=seed, d_min=d_min, dm_ai_weight=0.0,
    )
    add("ai_phaseed_oracle", ph, rho, t0, dm_ai_weight=0.0)

    # Oracle + DM-AI hybrid
    t0 = time.time()
    ph, rho, info = ai_phaseed_solve(
        hkl, amp, cell, ph_true,
        seed_fraction=0.25, n_extend=n_extend, polish="none",
        n_starts=n_starts, seed=seed, d_min=d_min, dm_ai_weight=0.5,
        low_res_path=(d_min >= 1.3),
    )
    add(
        "ai_phaseed_oracle_dm",
        ph, rho, t0,
        dm_ai_weight=0.5,
        low_res_path=info.get("low_res_path"),
    )

    # Partial noisy seed (~55% quality)
    rng = np.random.default_rng(seed + 7)
    ph_r = rng.uniform(-np.pi, np.pi, len(amp))
    ph_partial = np.angle(0.55 * np.exp(1j * ph_true) + 0.45 * np.exp(1j * ph_r))
    sq_p = predict_seed_quality(
        hkl, amp, cell, ph_partial, seed_idx=seed_idx, d_min=d_min, n_atoms_user=n_atoms
    )
    t0 = time.time()
    ph, rho, info = ai_phaseed_solve(
        hkl, amp, cell, ph_partial,
        seed_fraction=0.25, n_extend=n_extend, polish="none",
        n_starts=n_starts, seed=seed, d_min=d_min, dm_ai_weight=0.5,
    )
    add(
        "ai_phaseed_partial_dm",
        ph, rho, t0,
        dm_ai_weight=0.5,
        seed_class_pred=sq_p.get("predicted_class"),
        seed_p_success=sq_p.get("success_probability"),
    )

    # PhAI path if available
    try:
        from grok_phase_solver.models.phai_runner import phai_available

        if phai_available():
            t0 = time.time()
            ph, rho, info = phai_phaseed_solve(
                hkl, amp, cell,
                n_extend=n_extend, polish="charge_flipping", n_polish=40,
                n_starts=1, seed=seed, d_min=d_min, dm_ai_weight=0.5,
                fallback="cf",
            )
            add(
                "phai_phaseed_dm",
                ph, rho, t0,
                seed_source=info.get("seed_source"),
                seed_class_pred=(info.get("seed_quality") or {}).get("predicted_class"),
            )
    except Exception as e:
        print(f"  (phai skipped: {e})")

    return rows


def load_cod_optional() -> List[Dict[str, Any]]:
    """Load a few COD CIFs if present for Fcalc controls."""
    out = []
    cod_dir = ROOT / "data" / "raw" / "cod"
    if not cod_dir.is_dir():
        return out
    try:
        from grok_phase_solver.io.cif import load_cif
    except Exception:
        return out
    for cif in sorted(cod_dir.glob("*.cif"))[:6]:
        try:
            st = load_cif(str(cif))
            data = structure_to_fcalc(st, d_min=1.0)
            out.append(
                {
                    "name": f"cod_{cif.stem}_fcalc",
                    "hkl": data["hkl"],
                    "amp": data["amplitudes"],
                    "ph_true": data["phases"],
                    "cell": st.cell,
                    "fracs": data["fracs"],
                    "elements": data["elements"],
                    "d_min": 1.0,
                    "n_atoms": len(data["elements"]),
                }
            )
            print(f"  loaded COD {cif.name}")
        except Exception as e:
            print(f"  skip {cif.name}: {e}")
    return out


def main():
    print("=== Extended AI-PhaSeed benchmark (v0.4) ===")
    all_rows: List[Dict[str, Any]] = []

    # Synthetic panel
    for n_atoms, d_min, seed, tag in [
        (6, 1.0, 0, "synth_easy"),
        (8, 1.0, 1, "synth_med"),
        (12, 1.2, 2, "synth_harder"),
        (10, 1.5, 3, "synth_lowres"),
    ]:
        print(f"\n-- {tag} n={n_atoms} d_min={d_min} --")
        st = generate_random_organic(n_atoms=n_atoms, seed=seed, space_group="P1")
        data = structure_to_fcalc(st, d_min=d_min)
        rows = run_one_structure(
            tag,
            data["hkl"],
            data["amplitudes"],
            data["phases"],
            st.cell,
            data["fracs"],
            data["elements"],
            d_min,
            n_atoms=n_atoms,
            seed=seed,
        )
        all_rows.extend(rows)

    print("\n-- optional COD Fcalc --")
    for item in load_cod_optional():
        print(f"\n-- {item['name']} --")
        all_rows.extend(
            run_one_structure(
                item["name"],
                item["hkl"],
                item["amp"],
                item["ph_true"],
                item["cell"],
                item["fracs"],
                item["elements"],
                item["d_min"],
                n_atoms=item.get("n_atoms"),
            )
        )

    # Stratified summary
    by_method: Dict[str, List] = {}
    by_class: Dict[str, List] = {}
    by_vol: Dict[str, List] = {}
    for r in all_rows:
        by_method.setdefault(r["method"], []).append(r)
        by_class.setdefault(str(r.get("seed_class_pred")), []).append(r)
        by_vol.setdefault(r.get("vol_band", "?"), []).append(r)

    def _summ(rows: List[Dict]) -> Dict:
        if not rows:
            return {}
        return {
            "n": len(rows),
            "solve_rate": float(np.mean([1.0 if x["solved"] else 0.0 for x in rows])),
            "mean_mapcc": float(np.mean([x["mapcc_oi"] for x in rows])),
            "mean_r1": float(np.mean([x["r1"] for x in rows])),
        }

    summary = {
        "by_method": {k: _summ(v) for k, v in by_method.items()},
        "by_seed_class": {k: _summ(v) for k, v in by_class.items()},
        "by_vol_band": {k: _summ(v) for k, v in by_vol.items()},
        "n_rows": len(all_rows),
        "notes": (
            "Subset harness aligned with Carrozzini et al. J. Appl. Cryst. 2025 "
            "(Class 0/1 features, DM+AI hybrid). Not a 1505-structure replication."
        ),
    }

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rows": all_rows, "summary": summary, "version": "0.4.0"}
    json_path = out_dir / "ai_phaseed_extended_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_lines = [
        "# Extended AI-PhaSeed benchmark",
        "",
        "Carrozzini *et al.* (2025) alignment harness — hybrid DM+AI, seed Class,",
        "volume/resolution stratification. Honest subset (not full COD 1505 panel).",
        "",
        "## Summary by method",
        "",
        "| method | n | solve_rate | mean_mapCC | mean_R1 |",
        "|--------|---|------------|------------|---------|",
    ]
    for m, s in sorted(summary["by_method"].items()):
        md_lines.append(
            f"| `{m}` | {s['n']} | {s['solve_rate']:.2f} | "
            f"{s['mean_mapcc']:.3f} | {s['mean_r1']:.3f} |"
        )
    md_lines.extend(
        [
            "",
            "## By predicted seed class (oracle true-phase features on that row)",
            "",
            "```json",
            json.dumps(summary["by_seed_class"], indent=2),
            "```",
            "",
            "## By volume band",
            "",
            "```json",
            json.dumps(summary["by_vol_band"], indent=2),
            "```",
            "",
            "## References",
            "",
            "1. Carrozzini et al. (2025). J. Appl. Cryst. **58**, 1859–1869. "
            "DOI: 10.1107/S1600576725008271",
            "2. Larsen et al. (2024). Science **385**, 522–528 (PhAI).",
            "",
            f"Generated rows: {len(all_rows)}. JSON: `{json_path.name}`.",
        ]
    )
    md_path = out_dir / "ai_phaseed_extended_benchmark.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(summary["by_method"], indent=2))


if __name__ == "__main__":
    main()
