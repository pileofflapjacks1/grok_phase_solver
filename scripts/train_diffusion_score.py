#!/usr/bin/env python3
"""
Train lightweight PhaseScoreNet for diffusion hybrid v2.

Saves data/processed/diffusion_score.npz (small; safe to commit optionally).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.models.diffusion_score import train_score_on_structures


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-structures", type=int, default=100)
    p.add_argument("--epochs-per", type=int, default=12)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/processed/diffusion_score.npz")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_structures = 30
        args.epochs_per = 6

    t0 = time.time()
    net, meta = train_score_on_structures(
        n_structures=args.n_structures,
        epochs_per=args.epochs_per,
        hidden=args.hidden,
        seed=args.seed,
        verbose=True,
    )
    meta["train_seconds"] = time.time() - t0
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out}  loss≈{meta.get('final_loss')}")


if __name__ == "__main__":
    main()
