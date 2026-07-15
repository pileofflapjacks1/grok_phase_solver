# Strong phase prior (GraphPhaseNet) — v3 seed retarget

Triplet-graph prior with **Wilson-matched |F|** (optional), **strong-|E| loss reweight**, and **within-20°** focus. Success bar for the hard cliff: ≥30% of strong |E| phases within 20° of truth (oracle AI-PhaSeed threshold).

- Structures: **250** (packs: 250, Wilson-matched train: 250)
- Hidden=128, layers=3, max_refl=120
- wilson_match=**True**, scale=**v3_seed_retarget**
- Train strong MPE_OI: **64.6°**, frac≤20°: **21%**
- Hold-out full MPE_OI: **71.3°**, strong MPE: **59.3°**, frac≤20°: **21%**, would_seed_solve: **5%**
- Weights: `data/processed/strong_prior.npz`

## Strong-seed bar (hold-out hard)

| Metric | Graph prior | Graph+PhaSeed |
|--------|-------------|---------------|
| strong MPE_OI | 58.2° | — |
| frac ≤20° (top 30% \|E\|) | 19% | 21% |
| would_seed_solve (≥30% within 20°) | 0/8 | 1/8 |

## Hold-out hard-region comparison

| Method | Solved | Rate | mean mapCC |
|--------|--------|------|------------|
| Graph prior only | — | — | 0.460 (full MPE 72°) |
| **Graph + AI-PhaSeed** | 0 | 0% | **0.502** |
| CF | 0 | 0% | 0.479 |
| hard_p1 PhaseMLP + PhaSeed | 0 | 0% | 0.512 |

## Per-case

| n | d_min | strongMPE | frac20 | seedOK | prior CC | +PS CC | solved | CF |
|---|-------|-----------|--------|--------|----------|--------|--------|-----|
| 13 | 2.0 | 59° | 16% | False | 0.52 | 0.54 | False | 0.44 |
| 16 | 1.7 | 61° | 21% | False | 0.41 | 0.45 | False | 0.45 |
| 17 | 2.0 | 56° | 20% | False | 0.47 | 0.50 | False | 0.45 |
| 17 | 1.5 | 63° | 16% | False | 0.45 | 0.48 | False | 0.44 |
| 12 | 2.0 | 51° | 17% | False | 0.58 | 0.59 | False | 0.57 |
| 15 | 1.5 | 66° | 19% | False | 0.35 | 0.42 | False | 0.51 |
| 16 | 1.5 | 51° | 27% | False | 0.46 | 0.55 | False | 0.52 |
| 13 | 1.7 | 58° | 20% | False | 0.43 | 0.49 | False | 0.46 |

## Notes (A+B)

- **A:** train with `--wilson-match` so |F| match experimental Wilson template.
- **B:** loss targets strong-|E| accuracy (E² weights, top-half boost, extra weight when error >20°).
- Oracle bar: ≥30% strong phases within 20° → AI-PhaSeed can strict-solve hard cells.

Train 115.9s + eval 16.3s
