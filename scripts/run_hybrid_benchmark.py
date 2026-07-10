#!/usr/bin/env python3
"""
Hybrid benchmark suites A (ab initio) and B (MIR) — hybrid_ai_tests.md.

Truth-seeking report: prints method × metric tables; does not claim SOTA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.experimental_phasing import (
    simulate_mir,
    mir_phase_indication,
)
from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.synthetic_v2 import generate_fragment_structure
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.metrics.map_cc import map_correlation_origin_invariant
from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import run_physics_baseline, structure_to_fcalc
from grok_phase_solver.solvers.difference_patterson import locate_heavy_atom_vectors
from grok_phase_solver.solvers.hybrid import hybrid_phase_retrieval
from grok_phase_solver.solvers.mir_blow_crick import (
    DerivativeData,
    combine_mir_phases,
    single_isomorphous_replacement,
)
from grok_phase_solver.models.phase_mlp import (
    PhaseMLP,
    reflection_features,
    train_phase_mlp_on_structure,
)


def _map_metrics(hkl, amp, phases_pred, phases_true, cell):
    F_p = amp * np.exp(1j * phases_pred)
    F_t = amp * np.exp(1j * phases_true)
    rho_p = density_from_structure_factors(hkl, F_p, cell)
    rho_t = density_from_structure_factors(hkl, F_t, cell, shape=rho_p.shape)
    cc, _ = map_correlation_origin_invariant(rho_p, rho_t)
    mpe, _ = mean_phase_error_origin_invariant(phases_pred, phases_true, hkl, weights=amp)
    return {"mapCC_OI": cc, "MPE_OI_deg": mpe}


def suite_A(seed: int = 0) -> list:
    """Ab initio classical methods on tiny synthetic."""
    rows = []
    st = generate_random_organic(n_atoms=5, seed=seed)
    for method in ("random", "direct_methods", "charge_flipping"):
        res = run_physics_baseline(
            st, method=method, d_min=1.0, n_iter=80 if method != "random" else 1,
            seed=seed, verbose=False,
        )
        rows.append(
            {
                "suite": "A_abinitio",
                "structure": st.name,
                "method": method,
                "mapCC_OI": res.map_cc,
                "MPE_OI_deg": res.mean_phase_error_origin_invariant_deg,
                "n_refl": res.n_reflections,
            }
        )
    # MLP trained on same structure (overfit sanity — not generalization claim)
    data = structure_to_fcalc(st, d_min=1.0)
    mlp = PhaseMLP(hidden=48, seed=seed)
    train_phase_mlp_on_structure(
        mlp, data["hkl"], data["amplitudes"], data["phases"], st.cell,
        n_epochs=150, lr=5e-3, verbose=False,
    )
    X = reflection_features(data["hkl"], data["amplitudes"], st.cell)
    ph = mlp.predict_phases(X)
    # hybrid polish
    ph2, _, _ = hybrid_phase_retrieval(
        data["hkl"], data["amplitudes"], st.cell, ph,
        polish="charge_flipping", n_iter=40, seed=seed,
    )
    m0 = _map_metrics(data["hkl"], data["amplitudes"], ph, data["phases"], st.cell)
    m1 = _map_metrics(data["hkl"], data["amplitudes"], ph2, data["phases"], st.cell)
    rows.append({"suite": "A_abinitio", "structure": st.name, "method": "mlp_overfit", **m0})
    rows.append({"suite": "A_abinitio", "structure": st.name, "method": "mlp+CF", **m1})
    return rows


def suite_B(seed: int = 1) -> list:
    """MIR: Harker, Blow–Crick SIR, difference Patterson diagnostics."""
    rows = []
    st = generate_fragment_structure(n_fragments=2, seed=seed)
    mir = simulate_mir(st, heavy_element="AU", n_heavy=1, d_min=1.5, noise=0.02, seed=seed)

    # simple Harker indication
    ph_h, fom_h = mir_phase_indication(mir.F_native, mir.F_derivative, mir.F_heavy)
    m = _map_metrics(mir.hkl, mir.F_native, ph_h, mir.phases_true, mir.cell)
    rows.append({"suite": "B_MIR", "method": "harker_sign", **m, "mean_fom": float(fom_h.mean())})

    # Blow–Crick SIR
    ph_bc, fom_bc = single_isomorphous_replacement(
        mir.F_native, mir.F_derivative, mir.F_heavy, sigma=float(np.std(mir.F_native) * 0.1 + 1.0)
    )
    m = _map_metrics(mir.hkl, mir.F_native, ph_bc, mir.phases_true, mir.cell)
    rows.append({"suite": "B_MIR", "method": "blow_crick_SIR", **m, "mean_fom": float(fom_bc.mean())})

    # hybrid: BC seed + DM polish
    ph_hyb, _, _ = hybrid_phase_retrieval(
        mir.hkl, mir.F_native, mir.cell, ph_bc,
        polish="density_modification", n_iter=20, solvent_fraction=0.3, seed=seed,
    )
    m = _map_metrics(mir.hkl, mir.F_native, ph_hyb, mir.phases_true, mir.cell)
    rows.append({"suite": "B_MIR", "method": "SIR+solvent_flatten", **m})

    peaks, _, info = locate_heavy_atom_vectors(
        mir.hkl, mir.F_native, mir.F_derivative, mir.cell, n_peaks=8
    )
    rows.append(
        {
            "suite": "B_MIR",
            "method": "diff_patterson",
            "n_peaks": len(peaks),
            "mean_abs_delta_F": info["mean_abs_delta_F"],
            "map_max": info["map_max"],
        }
    )
    return rows


def suite_COD(d_min: float = 1.2) -> list:
    """COD 2100301 CF baseline if present."""
    path = ROOT / "data/raw/cod/2100301.cif"
    if not path.exists():
        return []
    st = load_cif(path)
    rows = []
    for method in ("random", "charge_flipping"):
        res = run_physics_baseline(
            st, method=method, d_min=d_min, n_iter=60 if method != "random" else 1,
            seed=0, verbose=False,
        )
        rows.append(
            {
                "suite": "COD_2100301",
                "method": method,
                "d_min": d_min,
                "mapCC_OI": res.map_cc,
                "MPE_OI_deg": res.mean_phase_error_origin_invariant_deg,
                "n_refl": res.n_reflections,
            }
        )
    return rows


def main():
    out = []
    print("=" * 60)
    print("Hybrid benchmark (truth-seeking, not SOTA claims)")
    print("=" * 60)
    for name, fn in (("A", suite_A), ("B", suite_B)):
        print(f"\n--- Suite {name} ---")
        rows = fn()
        for r in rows:
            print(r)
            out.append(r)
    print("\n--- COD ---")
    for r in suite_COD():
        print(r)
        out.append(r)

    out_path = ROOT / "data/processed/hybrid_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON-serializable
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    out_path.write_text(json.dumps(out, indent=2, default=conv))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
