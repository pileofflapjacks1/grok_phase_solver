# Hard-P1 domain-matched phase prior

PhaseMLP trained on synthetic **P1** cells in the hard region (\(n \in [12,20]\), \(d_{\min} \in [1.5, 2.0]\)), plus bridge samples.

- Structures: **60** (bridge every 4th)
- Hidden: 128, epochs/structure: 40+12
- Mean train MPE: **63.3°**
- Mean hold-out MPE: **64.3°**
- Weights: `data/processed/hard_p1_prior.npz`

## Hold-out hard-region solve rates

| Method | Solved | Rate | mean mapCC |
|--------|--------|------|------------|
| hard_p1 prior only | — | — | 0.506 (MPE 68°) |
| **hard_p1 + AI-PhaSeed** | 0 | 0% | **0.516** |
| CF | 0 | 0% | 0.497 |
| random + AI-PhaSeed | 0 | 0% | 0.374 |

## Per-case

| n | d_min | prior MPE | prior CC | hP1+PS CC | solved | CF CC |
|---|-------|-----------|----------|-----------|--------|-------|
| 17 | 1.7 | 64° | 0.53 | 0.54 | False | 0.46 |
| 14 | 1.5 | 60° | 0.46 | 0.47 | False | 0.42 |
| 14 | 2.0 | 68° | 0.52 | 0.53 | False | 0.44 |
| 13 | 1.5 | 70° | 0.50 | 0.52 | False | 0.57 |
| 14 | 1.7 | 65° | 0.51 | 0.53 | False | 0.45 |
| 15 | 1.7 | 76° | 0.46 | 0.47 | False | 0.50 |
| 12 | 1.7 | 78° | 0.53 | 0.53 | False | 0.63 |
| 12 | 1.7 | 62° | 0.53 | 0.55 | False | 0.51 |

## Scope

In-domain prior for **synthetic hard P1** only. Transfer to experimental or non-P1 space groups is unproven. Use via `hard_p1_phaseed_solve` / `predict_phases_hard_p1`.

Train time: 216.3s + eval 119.0s
