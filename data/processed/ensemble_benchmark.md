# Multistart ensemble benchmark (CF + RAAR, free-FOM pick)

Selection uses **truth-free composite FOM**; success rates use strict SuccessThresholds against synthetic truth.

## Easy region

| Method | Solved | Total | Rate | mean mapCC |
|--------|--------|-------|------|------------|
| `cf` | 7 | 9 | 78% | 0.848 |
| `raar` | 5 | 9 | 56% | 0.612 |
| `ensemble_cf_raar` | 6 | 9 | 67% | 0.771 |

## Hard region

| Method | Solved | Total | Rate | mean mapCC |
|--------|--------|-------|------|------------|
| `cf` | 0 | 6 | 0% | 0.517 |
| `raar` | 0 | 6 | 0% | 0.303 |
| `ensemble_cf_raar` | 0 | 6 | 0% | 0.520 |

## Notes

- Ensemble runs multiple CF and RAAR starts; picks highest free-FOM composite.
- Expect higher wall-clock (~ n_starts × 2 methods) and modest success gains.
- JSON: `data/processed/ensemble_benchmark.json`
