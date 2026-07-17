# SHELXS head-to-head

Fair comparison of **gps-solve** methods vs external **SHELXS 2024/1** (Sheldrick direct methods) on the same synthetic cases.

- SHELXS binary: **`/Users/joe/Projects/grok_phase_solver/ShelX/shelxs`**
- TREF trials: **100**

## Summary

| Panel | Method | n | solved | rate | mean mapCC | mean peak | t (s) |
|-------|--------|---|--------|------|------------|-----------|-------|
| synthetic_easy | `charge_flipping` | 4/4 | 0 | 0% | 0.636 | 0.89 | 0.5 |
| synthetic_easy | `dual_space` | 4/4 | 1 | 25% | 0.498 | 0.69 | 1.0 |
| synthetic_easy | `ensemble` | 4/4 | 2 | 50% | 0.777 | 0.89 | 1.6 |
| synthetic_easy | `shelxs` | 4/4 | 0 | 0% | 0.642 | 0.90 | 0.3 |
| synthetic_easy | `strong_prior_phaseed` | 4/4 | 0 | 0% | 0.500 | 0.72 | 1.8 |
| synthetic_hard | `charge_flipping` | 4/4 | 0 | 0% | 0.510 | 0.56 | 0.3 |
| synthetic_hard | `dual_space` | 4/4 | 0 | 0% | 0.462 | 0.65 | 0.5 |
| synthetic_hard | `ensemble` | 4/4 | 0 | 0% | 0.481 | 0.63 | 0.7 |
| synthetic_hard | `shelxs` | 4/4 | 0 | 0% | 0.396 | 0.78 | 0.3 |
| synthetic_hard | `strong_prior_phaseed` | 4/4 | 0 | 0% | 0.516 | 0.59 | 0.8 |

## Rankings by panel (mean mapCC)

### synthetic_easy

1. **`ensemble`** — mapCC 0.777, solved 2/4, 1.6s
2. **`shelxs`** — mapCC 0.642, solved 0/4, 0.3s
3. **`charge_flipping`** — mapCC 0.636, solved 0/4, 0.5s
4. **`strong_prior_phaseed`** — mapCC 0.500, solved 0/4, 1.8s
5. **`dual_space`** — mapCC 0.498, solved 1/4, 1.0s

### synthetic_hard

1. **`strong_prior_phaseed`** — mapCC 0.516, solved 0/4, 0.8s
2. **`charge_flipping`** — mapCC 0.510, solved 0/4, 0.3s
3. **`ensemble`** — mapCC 0.481, solved 0/4, 0.7s
4. **`dual_space`** — mapCC 0.462, solved 0/4, 0.5s
5. **`shelxs`** — mapCC 0.396, solved 0/4, 0.3s

## Notes

- **SHELXS** = classical direct methods (not SHELXD dual-space).
- HKL written as fixed-format (3i4,2f8.2) with scaled intensities.
- Peaks from `.res` → equal-atom Fcalc phases for mapCC vs truth.
- Do not commit `ShelX/` binaries (academic license, gitignored).
- If SHELXS is missing, only in-repo methods are scored.

JSON: `data/processed/shelxs_h2h.json`
