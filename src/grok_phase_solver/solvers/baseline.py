"""
Physics baseline pipeline: structure → Fcalc → strip phases → recover → metrics.

This is the Phase-1 reproducibility core used before / alongside neural methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from grok_phase_solver.io.cif import CrystalStructure, expand_asymmetric_unit, load_cif
from grok_phase_solver.metrics.map_cc import map_correlation, map_correlation_origin_invariant
from grok_phase_solver.metrics.phase_error import mean_phase_error, mean_phase_error_origin_invariant
from grok_phase_solver.metrics.rfactor import r_factor
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import generate_hkl
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve
from grok_phase_solver.solvers.direct_methods import direct_methods_solve
from grok_phase_solver.solvers.hio import hio_solve
from grok_phase_solver.solvers.patterson import patterson_solve


@dataclass
class BaselineResult:
    """Container for a single baseline phasing run."""

    name: str
    method: str
    d_min: float
    n_reflections: int
    n_atoms_cell: int
    mean_phase_error_deg: float
    mean_phase_error_origin_invariant_deg: float
    map_cc: float
    r_factor: float
    history: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.method}] {self.name} @ {self.d_min:.2f} Å  "
            f"Nrefl={self.n_reflections}  "
            f"MPE={self.mean_phase_error_deg:.1f}°  "
            f"MPE_oi={self.mean_phase_error_origin_invariant_deg:.1f}°  "
            f"mapCC={self.map_cc:.3f}  R={self.r_factor:.3f}"
        )


def structure_to_fcalc(
    structure: CrystalStructure,
    d_min: float = 1.0,
    expand_symmetry: bool = True,
) -> Dict:
    """Compute full-cell Fcalc and true phases from a CrystalStructure."""
    fracs, elements, b_isos, occs = expand_asymmetric_unit(
        structure, include_identity_only=not expand_symmetry
    )
    hkl = generate_hkl(structure.cell, d_min=d_min, expand_friedel=True)
    # Remove systematic absences if possible
    try:
        import gemmi

        sg = gemmi.SpaceGroup(structure.space_group_hm)
        ops = sg.operations()
        keep = [
            not ops.is_systematically_absent([int(h), int(k), int(l)])
            for h, k, l in hkl
        ]
        hkl = hkl[np.array(keep, dtype=bool)]
    except Exception:
        pass

    F = compute_structure_factors(
        hkl, fracs, elements, structure.cell, b_isos=b_isos, occs=occs
    )
    return {
        "hkl": hkl,
        "F": F,
        "amplitudes": np.abs(F),
        "phases": np.angle(F),
        "fracs": fracs,
        "elements": elements,
        "n_atoms_cell": len(elements),
    }


def run_physics_baseline(
    structure: CrystalStructure,
    method: str = "charge_flipping",
    d_min: float = 1.2,
    n_iter: int = 150,
    seed: int = 0,
    noise_level: float = 0.0,
    completeness: float = 1.0,
    verbose: bool = True,
) -> BaselineResult:
    """
    Full baseline: Fcalc → corrupt → phase retrieval → metrics vs truth.

    Parameters
    ----------
    noise_level : Gaussian relative noise σ on amplitudes (0 = perfect)
    completeness : fraction of reflections to keep (random subset)
    """
    data = structure_to_fcalc(structure, d_min=d_min)
    hkl = data["hkl"]
    amp = data["amplitudes"].copy()
    phases_true = data["phases"]

    notes: List[str] = []
    rng = np.random.default_rng(seed)

    if noise_level > 0:
        amp = amp * (1.0 + noise_level * rng.standard_normal(len(amp)))
        amp = np.maximum(amp, 0.0)
        notes.append(f"amplitude noise σ={noise_level}")

    if completeness < 1.0:
        n_keep = max(10, int(completeness * len(amp)))
        idx = rng.choice(len(amp), size=n_keep, replace=False)
        # Keep Friedel pairs roughly: simple random subset for Phase 1
        hkl = hkl[idx]
        amp = amp[idx]
        phases_true = phases_true[idx]
        notes.append(f"completeness={completeness:.0%} → {n_keep} refl")

    if verbose:
        print(f"Running {method} on {structure.name}: {len(amp)} reflections, d_min={d_min} Å")

    if method == "charge_flipping":
        phases_pred, rho_pred, history = charge_flipping_solve(
            hkl, amp, structure.cell, n_iter=n_iter, seed=seed, verbose=verbose
        )
    elif method == "hio":
        phases_pred, rho_pred, history = hio_solve(
            hkl, amp, structure.cell, n_iter=n_iter, seed=seed, verbose=verbose
        )
    elif method == "direct_methods":
        dm = direct_methods_solve(
            hkl,
            amp,
            structure.cell,
            n_atoms_approx=max(data["n_atoms_cell"], 4),
            n_trials=max(20, n_iter // 5),
            seed=seed,
            verbose=verbose,
        )
        phases_pred = dm.phases_full
        rho_pred = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases_pred), structure.cell, d_min=d_min
        )
        history = dm.history
        notes.append(
            f"direct methods: {dm.history.get('n_strong')} strong, "
            f"{dm.history.get('n_triplets')} triplets, best FOM={dm.history.get('best_fom'):.3f}"
        )
    elif method == "patterson":
        phases_pred, Pmap, info = patterson_solve(
            hkl, amp, structure.cell, seed=seed, verbose=verbose
        )
        # Evaluate density from random phases but store Patterson map stats
        rho_pred = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases_pred), structure.cell, d_min=d_min
        )
        history = {
            "n_peaks": len(info["peaks"]),
            "patterson_max": float(np.max(Pmap)),
        }
        # Peak recovery vs true vectors
        score = None
        try:
            from grok_phase_solver.physics.patterson import patterson_peak_recovery_score

            score = patterson_peak_recovery_score(info["peaks"], data["fracs"])
            history["vector_recovery"] = score
        except Exception:
            pass
        notes.append(
            "Patterson: interatomic vectors only (not general phases); "
            f"peaks={len(info['peaks'])}"
            + (f", vector_recovery={score:.2f}" if score is not None else "")
        )
    elif method == "random":
        phases_pred = rng.uniform(-np.pi, np.pi, size=len(amp))
        rho_pred = density_from_structure_factors(
            hkl, amp * np.exp(1j * phases_pred), structure.cell, d_min=d_min
        )
        history = {}
        notes.append("random phases (null baseline)")
    else:
        raise ValueError(f"Unknown method: {method}")

    # True density for map CC
    F_true = amp * np.exp(1j * phases_true)
    # Use same amplitudes for fair density comparison
    rho_true = density_from_structure_factors(
        hkl, F_true, structure.cell, shape=rho_pred.shape
    )

    mpe = mean_phase_error(phases_pred, phases_true, weights=amp)
    mpe_oi, _ = mean_phase_error_origin_invariant(
        phases_pred, phases_true, hkl, weights=amp
    )
    # Fixed-origin CC (diagnostic) and origin/enantiomorph-invariant CC (primary)
    cc_fixed = map_correlation(rho_pred, rho_true)
    cc, shift = map_correlation_origin_invariant(rho_pred, rho_true, also_inverted=True)
    notes.append(f"mapCC_fixed_origin={cc_fixed:.3f}, shift={shift}")
    # R-factor of modulus projection is always ~0 if we force |F|; use final R from history
    R = history.get("R", [np.nan])[-1] if history.get("R") else float("nan")

    result = BaselineResult(
        name=structure.name,
        method=method,
        d_min=d_min,
        n_reflections=len(amp),
        n_atoms_cell=data["n_atoms_cell"],
        mean_phase_error_deg=mpe,
        mean_phase_error_origin_invariant_deg=mpe_oi,
        map_cc=cc,
        r_factor=R if R == R else r_factor(amp, amp * np.exp(1j * phases_pred)),
        history=history,
        notes=notes,
    )
    if verbose:
        print(result.summary())
        for n in notes:
            print(f"  note: {n}")
    return result


def run_baseline_from_cif(
    cif_path: str,
    method: str = "charge_flipping",
    **kwargs,
) -> BaselineResult:
    """Convenience: load CIF and run baseline."""
    structure = load_cif(cif_path)
    return run_physics_baseline(structure, method=method, **kwargs)
