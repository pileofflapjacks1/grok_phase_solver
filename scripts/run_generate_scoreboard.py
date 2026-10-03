#!/usr/bin/env python3
"""Write data/processed/generate_scoreboard.md from the frozen local COD CIFs.

Numbers in that file come from this run. Do not hand-edit them.
The experimental diffusion_hybrid row uses deposited Fcalc (oracle amplitudes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grok_phase_solver.generate.cohort import cohort_from_dir  # noqa: E402
from grok_phase_solver.generate.scoreboard import run_scoreboard  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="COD coordinate-generation scoreboard")
    parser.add_argument("--cod-dir", default="data/raw/cod")
    parser.add_argument("--n", type=int, default=8, help="Samples per cell for random and denoiser")
    parser.add_argument("--steps", type=int, default=40, help="Denoiser training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/processed/generate_scoreboard.md")
    parser.add_argument(
        "--skip-diffusion",
        action="store_true",
        help="Leave the experimental diffusion row as not run",
    )
    args = parser.parse_args(argv)

    records, excluded = cohort_from_dir(args.cod_dir)
    if not records:
        raise SystemExit(f"No CIFs with 4–20 non-H atoms under {args.cod_dir}")
    result = run_scoreboard(
        records,
        n=int(args.n),
        steps=int(args.steps),
        seed=int(args.seed),
        run_diffusion=not args.skip_diffusion,
        excluded=excluded,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["markdown"])
    print(f"Wrote {out}  wall={result['wall_s']:.1f}s")
    print(result["comparison"])


if __name__ == "__main__":
    main()
