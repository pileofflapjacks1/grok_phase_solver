# Changelog

## 0.2.0 — 2026-07

### Product / pipeline
- `auto` prefers **ensemble** on easy/high-res; hard uses graph/hard-P1 priors or CF
- **Partial-φ hard path**: `partial_phaseed` + `--phase-seed-csv`; demo in `examples/partial_seed_demo/`
- **SHELXS / SHELXE** external runners (`ShelX/` gitignored); `shelxs+shelxe` method
- USER_GUIDE decision tree; SHELXL refinement instructions in `report.md`
- `trial.res` export for Olex2/SHELXL

### Science
- Free FOM v2.1, failure taxonomy A/B/C, AI-PhaSeed
- GraphPhaseNet strong prior (v3): Wilson match + strong-seed metrics/loss
- Oracle partial-φ curves: ≥30% strong φ within 20° → hard strict solves
- Wilson domain-gap closing (`wilson_match.py`)
- SHELXS head-to-head scoreboard vs CF/ensemble/priors

### Honest limits
- Hard ab initio still ~0% strict success under full priors (seed bar ~21% ≤20°)
- Not a general protein ab initio solver

## 0.1.0 — earlier

- Phase-1 baseline: CF/HIO, COD I/O, metrics, gps-solve core
