# AI-PhaSeed benchmark

Protocol: AI phases → strong-|E| seed subset → positivity phase extension with seed re-imposition + soft full prior → free-FOM–gated CF polish.

Literature: PhAI (Larsen *et al.* 2024) + phase-seeding / AI-PhaSeed (Carrozzini *et al.* 2025).

## Synthetic

| n | d_min | seed | method | mapCC | peak | solved |
|---|-------|------|--------|-------|------|--------|
| 6 | 1.0 | 0 | `cf` | 0.768 | 0.67 | True |
| 6 | 1.0 | 0 | `oracle_phaseed` | 0.999 | 0.67 | True |
| 6 | 1.0 | 0 | `partial_phaseed` | 0.869 | 0.67 | True |
| 6 | 1.0 | 0 | `phai_phaseed` | 0.387 | 0.83 | False |
| 6 | 1.0 | 0 | `phai_only` | 0.265 | 0.83 | False |
| 6 | 1.0 | 0 | `phai_cf_cond` | 0.707 | 0.67 | False |
| 6 | 1.0 | 1 | `cf` | 0.872 | 1.00 | True |
| 6 | 1.0 | 1 | `oracle_phaseed` | 1.000 | 0.83 | True |
| 6 | 1.0 | 1 | `partial_phaseed` | 0.899 | 0.67 | True |
| 6 | 1.0 | 1 | `phai_phaseed` | 0.318 | 0.83 | False |
| 6 | 1.0 | 1 | `phai_only` | 0.282 | 0.83 | False |
| 6 | 1.0 | 1 | `phai_cf_cond` | 0.684 | 1.00 | False |
| 8 | 0.9 | 0 | `cf` | 0.609 | 0.62 | False |
| 8 | 0.9 | 0 | `oracle_phaseed` | 0.999 | 0.50 | True |
| 8 | 0.9 | 0 | `partial_phaseed` | 0.882 | 0.62 | True |
| 8 | 0.9 | 0 | `phai_phaseed` | 0.342 | 0.62 | False |
| 8 | 0.9 | 0 | `phai_only` | 0.282 | 0.62 | False |
| 8 | 0.9 | 0 | `phai_cf_cond` | 0.602 | 0.62 | False |
| 12 | 1.5 | 0 | `cf` | 0.485 | 0.44 | False |
| 12 | 1.5 | 0 | `oracle_phaseed` | 0.996 | 0.67 | True |
| 12 | 1.5 | 0 | `partial_phaseed` | 0.899 | 0.67 | True |
| 12 | 1.5 | 0 | `phai_phaseed` | 0.333 | 0.67 | False |
| 12 | 1.5 | 0 | `phai_only` | 0.286 | 0.67 | False |
| 12 | 1.5 | 0 | `phai_cf_cond` | 0.472 | 0.44 | False |
| 12 | 1.5 | 1 | `cf` | 0.548 | 0.56 | False |
| 12 | 1.5 | 1 | `oracle_phaseed` | 0.996 | 0.56 | True |
| 12 | 1.5 | 1 | `partial_phaseed` | 0.861 | 0.56 | True |
| 12 | 1.5 | 1 | `phai_phaseed` | 0.465 | 0.56 | False |
| 12 | 1.5 | 1 | `phai_only` | 0.353 | 0.67 | False |
| 12 | 1.5 | 1 | `phai_cf_cond` | 0.353 | 0.67 | False |
| 16 | 1.5 | 0 | `cf` | 0.508 | 0.54 | False |
| 16 | 1.5 | 0 | `oracle_phaseed` | 0.994 | 0.69 | True |
| 16 | 1.5 | 0 | `partial_phaseed` | 0.817 | 0.69 | True |
| 16 | 1.5 | 0 | `phai_phaseed` | 0.376 | 0.77 | False |
| 16 | 1.5 | 0 | `phai_only` | 0.331 | 0.77 | False |
| 16 | 1.5 | 0 | `phai_cf_cond` | 0.331 | 0.77 | False |

## COD 2016452

| d_min | method | mapCC | peak | R1 | solved | notes |
|-------|--------|-------|------|----|--------|-------|
| 0.9 | `cf` | 0.350 | 0.94 | 0.63 | False |  |
| 0.9 | `phai_phaseed` | 0.542 | 1.00 | 0.56 | False | phai_fair; polish=False |
| 0.9 | `phai_only` | 0.558 | 1.00 | 0.61 | False |  |
| 0.9 | `phai_cf_cond` | 0.816 | 1.00 | 0.40 | True | polish=True |
| 1.2 | `cf` | 0.443 | 0.97 | 0.57 | False |  |
| 1.2 | `phai_phaseed` | 0.617 | 0.97 | 0.48 | False | phai_fair; polish=False |
| 1.2 | `phai_only` | 0.607 | 1.00 | 0.63 | False |  |
| 1.2 | `phai_cf_cond` | 0.607 | 1.00 | 0.63 | False | polish=False |
| 1.5 | `cf` | 0.535 | 0.88 | 0.54 | False |  |
| 1.5 | `phai_phaseed` | 0.626 | 0.97 | 0.47 | False | phai_fair; polish=False |
| 1.5 | `phai_only` | 0.621 | 0.94 | 0.48 | False |  |
| 1.5 | `phai_cf_cond` | 0.621 | 0.94 | 0.48 | False | polish=False |
| 2.0 | `cf` | 0.479 | 0.66 | 0.52 | False |  |
| 2.0 | `phai_phaseed` | 0.606 | 0.81 | 0.45 | False | phai_fair; polish=False |
| 2.0 | `phai_only` | 0.628 | 0.75 | 0.54 | False |  |
| 2.0 | `phai_cf_cond` | 0.628 | 0.75 | 0.54 | False | polish=False |

## Notes

- `oracle_phaseed`: true phases as AI seed (upper bound).
- `partial_phaseed`: 55% true + noise (simulated mediocre AI).
- `phai_phaseed`: PhAI fair + AI-PhaSeed + gated polish.
- Compare to `phai_only` / `phai_cf_cond` to isolate extension gain.

JSON: `data/processed/ai_phaseed_benchmark.json`
