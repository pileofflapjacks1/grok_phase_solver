# Release notes — v0.13.3

**Theme:** CCP4 density map + PyMOL/Coot handoff on the published package (patch over 0.13.2).

## What’s in 0.13.3

All of **v0.13.2** (Vol-band next-action chooser), plus:

After `gps-solve --out ./solve_out`, that folder includes:

| File | Use |
|------|-----|
| `density.map` | CCP4 unit-cell map (gemmi MODE-2) |
| `peaks.pdb` | Peak list for viewers |
| `open_in_pymol.pml` | `pymol open_in_pymol.pml` |
| `open_in_coot.sh` | `sh open_in_coot.sh` |

Visualization only. `trial.res` → SHELXL / Olex2 is still the refine path.
Science freeze unchanged (GraphPhaseNet below the 30% seed bar).

## Install

```bash
python -m pip install -U "grok-phase-solver>=0.13.3"
gps-solve --help
# live: https://pypi.org/project/grok-phase-solver/0.13.3/
```

## Prior
[v0.13.2 notes](RELEASE_NOTES_v0.13.2.md) · [v0.13.1 notes](RELEASE_NOTES_v0.13.1.md)
