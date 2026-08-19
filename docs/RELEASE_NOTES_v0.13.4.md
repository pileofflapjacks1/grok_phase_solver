# Release notes — v0.13.4

**Theme:** one-command hard retry on the published package (patch over 0.13.3).

## What’s in 0.13.4

All of **v0.13.3** (CCP4 map + PyMOL/Coot handoff), plus:

```bash
gps-solve --hkl data.hkl --ins data.ins --retry-with-peaks --out ./solve_out
```

If the first pass looks weak (same gate as the GUI retry button), a second
`partial_phaseed` run uses this run’s `peaks.csv` and writes `solve_out/retry_peaks/`.
Skipped when free-FOM already looks healthy, peaks are too few, or you were
already on a seeded path.

Peaks-as-carbon is a cheap fallback. A real `.res` / predicted-model fragment
is still stronger (COD Vol-band C25). Science freeze unchanged.

## Install

```bash
python -m pip install -U "grok-phase-solver>=0.13.4"
gps-solve --help
# live: https://pypi.org/project/grok-phase-solver/0.13.4/
```

## Prior
[v0.13.3 notes](RELEASE_NOTES_v0.13.3.md) · [v0.13.2 notes](RELEASE_NOTES_v0.13.2.md)
