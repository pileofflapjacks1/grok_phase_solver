# Strong phase prior (GraphPhaseNet) — scaled

Triplet-graph message-passing network trained on **hard multi-SG** synthetic cells (P1 + P−1) with origin-invariant + triplet auxiliary loss, curriculum multi-pass training, and vectorized adjacency aggregation.

- Structures: **250** (packs used: 250)
- Hidden=128, layers=3, max strong reflections=120
- Global passes: **3**, epochs/struct≈18, triplet_w=0.18
- Curriculum: **True**
- Mean train MPE_OI: **61.4°**
- Mean hold-out MPE_OI: **71.9°**
- Weights: `data/processed/strong_prior.npz`

## Hold-out hard-region comparison

| Method | Solved | Rate | mean mapCC |
|--------|--------|------|------------|
| Graph prior only | — | — | 0.465 (MPE_OI 70°) |
| **Graph + AI-PhaSeed** | 0 | 0% | **0.513** |
| CF | 0 | 0% | 0.479 |
| hard_p1 PhaseMLP + PhaSeed | 0 | 0% | 0.512 |

## Per-case

| n | d_min | prior MPE | prior CC | strong+PS CC | solved | CF CC |
|---|-------|-----------|----------|--------------|--------|-------|
| 13 | 2.0 | 68° | 0.49 | 0.53 | False | 0.44 |
| 16 | 1.7 | 74° | 0.40 | 0.45 | False | 0.45 |
| 17 | 2.0 | 66° | 0.46 | 0.50 | False | 0.45 |
| 17 | 1.5 | 73° | 0.44 | 0.52 | False | 0.44 |
| 12 | 2.0 | 69° | 0.58 | 0.62 | False | 0.57 |
| 15 | 1.5 | 68° | 0.42 | 0.45 | False | 0.51 |
| 16 | 1.5 | 73° | 0.47 | 0.54 | False | 0.52 |
| 13 | 1.7 | 70° | 0.45 | 0.49 | False | 0.46 |

## Scale notes

Compared to the first GraphPhaseNet pass (50 structs / H=80 / L=2), this run increases data, capacity, curriculum multi-pass SGD, and adds a Cochran triplet-consistency auxiliary. Strict hard-region success remains difficult; report mapCC honestly vs CF and hard-P1.

Still a synthetic hard-region prior — not a claimed general experimental solver. Use via `strong_prior_phaseed_solve` or `gps-solve --method strong_prior_phaseed`.

Train 159.8s + eval 20.5s
