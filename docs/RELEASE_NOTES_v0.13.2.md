# Release notes — v0.13.2

**Theme:** Vol-band next-action chooser on the published package (patch over 0.13.1).

## What’s in 0.13.2

All of **v0.13.1** (COD Vol-band experimental panel + GraphPhaseNet v11 / CrystalX / XDXD), plus:

### Next action in `report.md`
After every `gps-solve`, the report opens with a **Next action** block:
volume band + one concrete command (not a generic flag dump).

| Situation | Recommendation |
|-----------|----------------|
| Vol 1000–3500, `auto` weak | Fragment `.res` or predicted-model CIF |
| Vol > 3500 | Larger fragment or HA pair |
| Small cell, good res, not ensemble | Retry `--method ensemble` |
| Seed below 30% size bar | Enlarge the seed |
| Seed applied, map still weak | Different source — don’t polish the same seed |
| Free-FOM looks healthy | Inspect `trial.res` → SHELXL |

Same object is written to `solve_summary.json` and the GUI banner.
Evidence: local COD Vol-band panel (C25). Free FOM remains a ranking score.

**Science freeze:** GraphPhaseNet v9–v11 still below the 30% seed bar. No v12.

## Install

```bash
python -m pip install -U "grok-phase-solver>=0.13.2"
gps-solve --help
# live: https://pypi.org/project/grok-phase-solver/0.13.2/
```

## Prior
[v0.13.1 notes](RELEASE_NOTES_v0.13.1.md) · [v0.13.0 notes](RELEASE_NOTES_v0.13.0.md)
