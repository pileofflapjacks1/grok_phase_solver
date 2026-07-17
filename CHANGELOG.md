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
- **Lane A v4 XL:** residual MP + Adam + d_in=10 + `--scale-xl` (1200 structs);
  fine-tune `--continue-from` / `--hard-only` / `--seed-focus`
- Oracle partial-φ curves: ≥30% strong φ within 20° → hard strict solves
- Wilson domain-gap closing (`wilson_match.py`)
- SHELXS head-to-head scoreboard vs CF/ensemble/priors

### Honest limits
- Hard ab initio still ~0% strict success under full priors
- **Seed bar ~21% ≤20° plateaus under 5× scale** (v3 → v4 XL); not cleared to 30%
- Not a general protein ab initio solver

## 0.1.0 — earlier

- Phase-1 baseline: CF/HIO, COD I/O, metrics, gps-solve core
