# Strong phase prior (GraphPhaseNet)

Triplet-graph message-passing network trained on **hard multi-SG** synthetic cells (P1 + P−1) with origin-invariant loss.

- Structures: **50**
- Hidden=80, layers=2, max strong reflections=90
- Mean train MPE_OI: **60.2°**
- Mean hold-out MPE_OI: **72.9°**
- Weights: `data/processed/strong_prior.npz`

## Hold-out hard-region comparison

| Method | Solved | Rate | mean mapCC |
|--------|--------|------|------------|
| Graph prior only | — | — | 0.447 (MPE_OI 71°) |
| **Graph + AI-PhaSeed** | 0 | 0% | **0.480** |
| CF | 0 | 0% | 0.475 |
| hard_p1 PhaseMLP + PhaSeed | — | — | 0.511 |

## Per-case

| n | d_min | prior MPE | prior CC | strong+PS CC | solved | CF CC |
|---|-------|-----------|----------|--------------|--------|-------|
| 13 | 2.0 | 66° | 0.48 | 0.49 | False | 0.44 |
| 16 | 1.7 | 73° | 0.41 | 0.46 | False | 0.45 |
| 17 | 2.0 | 63° | 0.48 | 0.50 | False | 0.45 |
| 17 | 1.5 | 77° | 0.42 | 0.48 | False | 0.44 |
| 12 | 2.0 | 71° | 0.56 | 0.60 | False | 0.57 |
| 15 | 1.5 | 77° | 0.32 | 0.35 | False | 0.51 |

## Scope

Stronger than per-reflection PhaseMLP by using **Cochran triplet graph** message passing. Still a synthetic hard-region prior — not a claimed general experimental solver. Use via `strong_prior_phaseed_solve`.

Train 86.9s + eval 12.1s
