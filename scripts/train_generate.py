#!/usr/bin/env python3
"""Train the small coordinate denoiser on one or more CIFs.

The checkpoint is a torch state dict. It is not a foundation model and it is
not used by gps-solve. Default output is under data/processed/, which gitignores
``.pt`` files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_generate.py` from a source checkout.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grok_phase_solver.generate.cohort import record_from_cif  # noqa: E402
from grok_phase_solver.generate.model import denoiser_state, train_denoiser  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the gps-generate coordinate denoiser")
    parser.add_argument("--cif", action="append", required=True, help="Repeat for each CIF")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument(
        "--out",
        default="data/processed/generate_denoiser.pt",
        help="Torch checkpoint path (gitignored under data/processed/)",
    )
    args = parser.parse_args(argv)

    records = []
    for raw in args.cif:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"CIF not found: {path}")
        records.append(record_from_cif(path))
    model = train_denoiser(
        records, steps=int(args.steps), hidden=int(args.hidden), seed=int(args.seed)
    )
    state = denoiser_state(model)
    state["names"] = [r["name"] for r in records]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(state, out)
    loss = float(state["final_loss"])
    print(f"Wrote {out}  steps={args.steps}  final_loss={loss:.6f}  structures={state['names']}")


if __name__ == "__main__":
    main()
