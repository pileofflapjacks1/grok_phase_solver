# Physics-recycle net (hard region)

- Trained on **16** synthetic structures with **n ∈ [12,20]**, **d_min ∈ [1.5, 2.0]**
- Hidden=96, epochs/structure=40
- Mean train MPE: **79.5°**
- Weights: `data/processed/recycle_net.npz`

## Hold-out comparison (strict success)

| n | d_min | recycle_net CC | pure recycle CC | CF CC | net solved |
|---|-------|----------------|-----------------|-------|------------|
| 13 | 1.8 | 0.354 | 0.542 | 0.599 | False |
| 15 | 2.0 | 0.331 | 0.488 | 0.363 | False |
| 13 | 1.5 | 0.258 | 0.359 | 0.610 | False |
| 14 | 1.5 | 0.234 | 0.383 | 0.504 | False |

## Scope

Supervised on synthetic φ_true in the hard region; physics recycle enforces |F| consistency. Not a claimed general experimental solver.
