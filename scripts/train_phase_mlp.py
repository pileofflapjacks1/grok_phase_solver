#!/usr/bin/env python3
"""Train PhaseMLP on synthetic fragment structures; save weights + loss curve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic_v2 import generate_fragment_structure, iter_training_samples
from grok_phase_solver.models.phase_mlp import (
    PhaseMLP,
    reflection_features,
    train_phase_mlp_on_structure,
)
from grok_phase_solver.metrics.phase_error import mean_phase_error
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-structures", type=int, default=20)
    p.add_argument("--epochs-per", type=int, default=80)
    p.add_argument("--dmin", type=float, default=1.2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/processed/phase_mlp.npz")
    args = p.parse_args()

    model = PhaseMLP(hidden=args.hidden, seed=args.seed)
    all_losses = []
    print(f"Training on {args.n_structures} synthetic structures…")
    for i, sample in enumerate(
        iter_training_samples(args.n_structures, seed=args.seed, d_min=args.dmin, mode="fragment")
    ):
        losses = train_phase_mlp_on_structure(
            model,
            sample["hkl"],
            sample["amplitudes"],
            sample["phases"],
            sample["cell"],
            n_epochs=args.epochs_per,
            lr=3e-3,
            seed=args.seed + i,
            verbose=(i == 0),
        )
        all_losses.append(float(np.mean(losses[-10:])))
        # hold-out style: evaluate on same structure final loss
        X = reflection_features(sample["hkl"], sample["amplitudes"], sample["cell"])
        pred = model.predict_phases(X)
        mpe = mean_phase_error(pred, sample["phases"], weights=sample["amplitudes"])
        print(f"  [{i+1}/{args.n_structures}] {sample['name']}  loss≈{all_losses[-1]:.4f}  MPE={mpe:.1f}°")

    out = Path(args.out)
    model.save(out)
    meta = {
        "n_structures": args.n_structures,
        "epochs_per": args.epochs_per,
        "d_min": args.dmin,
        "final_losses": all_losses,
        "note": (
            "Supervised on synthetic φ_true. Transfer to experimental |F| is unproven. "
            "Use as seed for hybrid CF/DM polish, not as a claimed general solver."
        ),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
