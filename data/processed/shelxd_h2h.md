# SHELXD head-to-head

Fair comparison of **gps-solve** methods vs **external SHELXD** (when installed) and an in-repo **dual-space** educational baseline.

- SHELXD binary: **not found**
- Install: [SHELX academic distribution](https://shelx.uni-goettingen.de/) then `export SHELXD=/path/to/shelxd`
- dual_space: multi-start peak↔phase recycling (Sheldrick-style idea); **not** SHELXD

## Summary

| Panel | Method | n | solved | rate | mean mapCC | mean peak | t (s) |
|-------|--------|---|--------|------|------------|-----------|-------|
| COD_2016452_Fcalc | `charge_flipping` | 1/1 | 0 | 0% | 0.505 | 0.97 | 3.2 |
| COD_2016452_Fcalc | `direct_methods` | 1/1 | 0 | 0% | 0.269 | 0.94 | 4.0 |
| COD_2016452_Fcalc | `dual_space` | 1/1 | 0 | 0% | 0.419 | 0.97 | 5.3 |
| COD_2016452_Fcalc | `ensemble` | 1/1 | 1 | 100% | 0.839 | 1.00 | 6.1 |
| COD_2016452_Fcalc | `hard_p1_phaseed` | 1/1 | 0 | 0% | 0.503 | 1.00 | 6.6 |
| COD_2016452_Fcalc | `strong_prior_phaseed` | 1/1 | 0 | 0% | 0.363 | 0.97 | 6.8 |
| synthetic_easy | `charge_flipping` | 4/4 | 0 | 0% | 0.636 | 0.89 | 0.6 |
| synthetic_easy | `direct_methods` | 4/4 | 0 | 0% | 0.338 | 0.86 | 4.0 |
| synthetic_easy | `dual_space` | 4/4 | 1 | 25% | 0.498 | 0.69 | 1.0 |
| synthetic_easy | `ensemble` | 4/4 | 2 | 50% | 0.777 | 0.89 | 1.8 |
| synthetic_easy | `hard_p1_phaseed` | 4/4 | 0 | 0% | 0.574 | 0.81 | 1.9 |
| synthetic_easy | `strong_prior_phaseed` | 4/4 | 0 | 0% | 0.431 | 0.78 | 1.9 |
| synthetic_hard | `charge_flipping` | 4/4 | 0 | 0% | 0.510 | 0.56 | 0.3 |
| synthetic_hard | `direct_methods` | 4/4 | 0 | 0% | 0.425 | 0.79 | 1.9 |
| synthetic_hard | `dual_space` | 4/4 | 0 | 0% | 0.462 | 0.65 | 0.6 |
| synthetic_hard | `ensemble` | 4/4 | 0 | 0% | 0.481 | 0.63 | 0.8 |
| synthetic_hard | `hard_p1_phaseed` | 4/4 | 0 | 0% | 0.504 | 0.52 | 0.8 |
| synthetic_hard | `strong_prior_phaseed` | 4/4 | 0 | 0% | 0.505 | 0.67 | 0.9 |

## Rankings by panel (mean mapCC)

### COD_2016452_Fcalc

1. **`ensemble`** — mapCC 0.839, solved 1/1, 6.1s
2. **`charge_flipping`** — mapCC 0.505, solved 0/1, 3.2s
3. **`hard_p1_phaseed`** — mapCC 0.503, solved 0/1, 6.6s
4. **`dual_space`** — mapCC 0.419, solved 0/1, 5.3s
5. **`strong_prior_phaseed`** — mapCC 0.363, solved 0/1, 6.8s
6. **`direct_methods`** — mapCC 0.269, solved 0/1, 4.0s

### synthetic_easy

1. **`ensemble`** — mapCC 0.777, solved 2/4, 1.8s
2. **`charge_flipping`** — mapCC 0.636, solved 0/4, 0.6s
3. **`hard_p1_phaseed`** — mapCC 0.574, solved 0/4, 1.9s
4. **`dual_space`** — mapCC 0.498, solved 1/4, 1.0s
5. **`strong_prior_phaseed`** — mapCC 0.431, solved 0/4, 1.9s
6. **`direct_methods`** — mapCC 0.338, solved 0/4, 4.0s

### synthetic_hard

1. **`charge_flipping`** — mapCC 0.510, solved 0/4, 0.3s
2. **`strong_prior_phaseed`** — mapCC 0.505, solved 0/4, 0.9s
3. **`hard_p1_phaseed`** — mapCC 0.504, solved 0/4, 0.8s
4. **`ensemble`** — mapCC 0.481, solved 0/4, 0.8s
5. **`dual_space`** — mapCC 0.462, solved 0/4, 0.6s
6. **`direct_methods`** — mapCC 0.425, solved 0/4, 1.9s

## Fairness notes

- Same synthetic structures and COD Fcalc control for all methods.
- Success uses strict `SuccessThresholds` (mapCC_OI + peak recovery + R1).
- SHELXD wall time depends on NTRY; dual_space uses multi-start × cycles.
- If SHELXD is missing, the `shelxd` method rows show errors; dual_space still runs.
- This is **not** a claim that gps-solve replaces SHELXD/SHELXT in production.

JSON: `data/processed/shelxd_h2h.json`
