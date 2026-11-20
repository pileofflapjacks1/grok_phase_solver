# SHELXS head-to-head

Fair comparison of **gps-solve** methods vs external **SHELXS 2024/1** (Sheldrick direct methods) on the same synthetic cases.

- SHELXS binary: **`/Users/joe/Projects/grok_phase_solver/ShelX/shelxs`**
- TREF trials: **100**

## Summary

| Panel | Method | n | solved | rate | mean mapCC | mean peak | t (s) |
|-------|--------|---|--------|------|------------|-----------|-------|
| synthetic_easy | `charge_flipping` | 4/4 | 4 | 100% | 0.861 | 1.00 | 0.9 |
| synthetic_easy | `dual_space` | 4/4 | 2 | 50% | 0.721 | 0.88 | 0.4 |
| synthetic_easy | `ensemble` | 4/4 | 4 | 100% | 0.859 | 1.00 | 1.5 |
| synthetic_easy | `shelxs` | 4/4 | 0 | 0% | 0.642 | 1.00 | 0.2 |
| synthetic_easy | `strong_prior_phaseed` | 4/4 | 1 | 25% | 0.557 | 0.72 | 0.9 |
| synthetic_hard | `charge_flipping` | 4/4 | 3 | 75% | 0.772 | 0.83 | 0.6 |
| synthetic_hard | `dual_space` | 4/4 | 0 | 0% | 0.557 | 0.52 | 0.3 |
| synthetic_hard | `ensemble` | 4/4 | 3 | 75% | 0.787 | 0.80 | 1.9 |
| synthetic_hard | `shelxs` | 4/4 | 0 | 0% | 0.396 | 0.59 | 0.1 |
| synthetic_hard | `strong_prior_phaseed` | 4/4 | 0 | 0% | 0.517 | 0.41 | 0.4 |

## Rankings by panel (mean mapCC)

### synthetic_easy

1. **`charge_flipping`** — mapCC 0.861, solved 4/4, 0.9s
2. **`ensemble`** — mapCC 0.859, solved 4/4, 1.5s
3. **`dual_space`** — mapCC 0.721, solved 2/4, 0.4s
4. **`shelxs`** — mapCC 0.642, solved 0/4, 0.2s
5. **`strong_prior_phaseed`** — mapCC 0.557, solved 1/4, 0.9s

### synthetic_hard

1. **`ensemble`** — mapCC 0.787, solved 3/4, 1.9s
2. **`charge_flipping`** — mapCC 0.772, solved 3/4, 0.6s
3. **`dual_space`** — mapCC 0.557, solved 0/4, 0.3s
4. **`strong_prior_phaseed`** — mapCC 0.517, solved 0/4, 0.4s
5. **`shelxs`** — mapCC 0.396, solved 0/4, 0.1s

## Notes

- **SHELXS** = classical direct methods (not SHELXD dual-space).
- HKL written as fixed-format (3i4,2f8.2) with scaled intensities.
- Peaks from `.res` → equal-atom Fcalc phases for mapCC vs truth.
- Do not commit `ShelX/` binaries (academic license, gitignored).
- If SHELXS is missing, only in-repo methods are scored.

JSON: `data/processed/shelxs_h2h.json`
