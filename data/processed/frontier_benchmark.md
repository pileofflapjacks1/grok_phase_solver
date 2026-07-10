# Frontier algorithm benchmark

Compares **RAAR**, **Difference Map**, κ-weighted DM, recycle, CF (+ PhAI conditional if available) under **strict SuccessThresholds**.

## Easy region (n≤8, d_min≤1.0) success rates

| Method | Solved | Total | Rate | mean mapCC |
|--------|--------|-------|------|------------|
| `cf` | 6 | 9 | 67% | 0.850 |
| `raar` | 5 | 9 | 56% | 0.610 |
| `diffmap` | 0 | 9 | 0% | 0.183 |
| `recycle` | 0 | 9 | 0% | 0.423 |
| `dm_kappa` | 0 | 9 | 0% | 0.344 |
| `phai_cond` | 0 | 9 | 0% | 0.300 |

## Hard region (n≥12, d_min≥1.5) success rates

| Method | Solved | Total | Rate | mean mapCC |
|--------|--------|-------|------|------------|
| `cf` | 0 | 12 | 0% | 0.476 |
| `raar` | 0 | 12 | 0% | 0.337 |
| `diffmap` | 0 | 12 | 0% | 0.253 |
| `recycle` | 0 | 12 | 0% | 0.476 |
| `dm_kappa` | 0 | 12 | 0% | 0.434 |
| `phai_cond` | 0 | 12 | 0% | 0.329 |

## Notes

- Easy region should remain CF-competitive (no regressions).
- Hard region is where new methods must show gains vs CF.
- PhAI conditional uses fair packing + free-FOM gate for polish.

JSON: `data/processed/frontier_benchmark.json`
