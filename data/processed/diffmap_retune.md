# Difference Map retune (β, charge-flip P_S, δσ)

Grid search ranked by **truth-free free-FOM composite**. Truth metrics reported for diagnostics only.

## Per-structure best (free-FOM pick)

| n | d_min | best β | P_S | δσ | retuned CC | default DM CC | CF CC | retuned solved |
|---|-------|--------|-----|----|------------|---------------|-------|----------------|
| 6 | 1.0 | 1.2 | charge_flip | 1.0 | 0.317 | 0.208 | 0.737 | False |
| 8 | 0.9 | 0.7 | charge_flip | 1.0 | 0.266 | 0.169 | 0.957 | False |
| 12 | 1.5 | 0.5 | charge_flip | 1.0 | 0.429 | 0.231 | 0.477 | False |
| 12 | 1.5 | 0.5 | charge_flip | 1.0 | 0.410 | 0.249 | 0.544 | False |
| 16 | 1.5 | 0.5 | charge_flip | 1.0 | 0.361 | 0.239 | 0.511 | False |

## Global parameter ranking (mean free-FOM composite)

| Rank | β | P_S | δσ | mean composite | n |
|------|---|-----|----|----------------|---|
| 1 | 0.5 | charge_flip | 1.0 | 0.682 | 10 |
| 2 | 0.7 | charge_flip | 1.0 | 0.681 | 10 |
| 3 | 1.0 | charge_flip | 1.0 | 0.678 | 10 |
| 4 | 1.2 | charge_flip | 1.0 | 0.678 | 10 |
| 5 | 0.5 | charge_flip | 0.5 | 0.668 | 10 |
| 6 | 0.7 | charge_flip | 0.5 | 0.666 | 10 |
| 7 | 1.0 | charge_flip | 0.5 | 0.663 | 10 |
| 8 | 0.5 | charge_flip | 0.0 | 0.661 | 10 |
| 9 | 1.2 | charge_flip | 0.5 | 0.659 | 10 |
| 10 | 0.7 | charge_flip | 0.0 | 0.657 | 10 |
| 11 | 0.5 | positivity | 1.0 | 0.656 | 10 |
| 12 | 1.0 | charge_flip | 0.0 | 0.654 | 10 |

## Recommended defaults (from this search)

- **β** = 0.5, **real_proj** = `charge_flip`, **delta_sigma** = 1.0 (mean composite 0.682)

- Still multistart + free-FOM for production; retune is case-dependent.
- JSON: `data/processed/diffmap_retune.json`
