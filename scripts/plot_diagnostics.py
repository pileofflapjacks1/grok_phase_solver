#!/usr/bin/env python3
"""Phase-error histograms, Wilson plots, density slices — matplotlib only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.wilson import wilson_plot
from grok_phase_solver.metrics.phase_error import wrap_phase
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.baseline import structure_to_fcalc, run_physics_baseline


def main():
    out_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    st = generate_random_organic(n_atoms=6, seed=0)
    data = structure_to_fcalc(st, d_min=1.0)
    res = run_physics_baseline(st, method="charge_flipping", d_min=1.0, n_iter=80, verbose=False)

    # Phase error hist (raw — origin sensitive; still educational)
    # Use origin-invariant alignment would need baseline internals; plot raw for honesty
    from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant

    # recompute pred phases via CF again for plot
    from grok_phase_solver.solvers.charge_flipping import charge_flipping_solve

    ph, rho, _ = charge_flipping_solve(
        data["hkl"], data["amplitudes"], st.cell, n_iter=80, seed=0
    )
    _, ph_al = mean_phase_error_origin_invariant(
        ph, data["phases"], data["hkl"], weights=data["amplitudes"]
    )
    dphi = np.rad2deg(wrap_phase(ph_al - data["phases"]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(dphi, bins=36, range=(-180, 180), color="steelblue", edgecolor="k", lw=0.3)
    axes[0].set_xlabel("Phase error (deg, origin-aligned)")
    axes[0].set_title(f"CF phase errors\n{res.summary()}")

    w = wilson_plot(data["hkl"], data["amplitudes"], st.cell)
    axes[1].plot(w["s2"], w["ln_mean"], "o-", ms=4)
    axes[1].set_xlabel(r"$s^2 = (\sin\theta/\lambda)^2$")
    axes[1].set_ylabel(r"$\ln\langle|F|^2\rangle$")
    axes[1].set_title(f"Wilson plot (slope={w['slope']:.2f})")

    z = rho.shape[2] // 2
    im = axes[2].imshow(rho[:, :, z].T, origin="lower", cmap="magma")
    axes[2].set_title("CF density slice")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    path = out_dir / "diagnostics_cf_synthetic.png"
    fig.savefig(path, dpi=140)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
