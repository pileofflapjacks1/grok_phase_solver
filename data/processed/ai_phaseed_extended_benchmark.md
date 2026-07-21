# Extended AI-PhaSeed benchmark

Carrozzini *et al.* (2025) alignment harness — hybrid DM+AI, seed Class,
volume/resolution stratification. Honest subset (not full COD 1505 panel).

## Summary by method

| method | n | solve_rate | mean_mapCC | mean_R1 |
|--------|---|------------|------------|---------|
| `ai_phaseed_oracle` | 7 | 0.71 | 0.998 | 0.360 |
| `ai_phaseed_oracle_dm` | 7 | 0.71 | 0.997 | 0.360 |
| `ai_phaseed_partial_dm` | 7 | 0.86 | 0.896 | 0.351 |
| `charge_flipping` | 7 | 0.29 | 0.547 | 0.487 |
| `phai_phaseed_dm` | 7 | 0.00 | 0.369 | 0.532 |

## By predicted seed class (oracle true-phase features on that row)

```json
{
  "1": {
    "n": 34,
    "solve_rate": 0.5294117647058824,
    "mean_mapcc": 0.780952190739508,
    "mean_r1": 0.41259094470279567
  },
  "0": {
    "n": 1,
    "solve_rate": 0.0,
    "mean_mapcc": 0.09129944107861578,
    "mean_r1": 0.6038180879568121
  }
}
```

## By volume band

```json
{
  "small_<800": {
    "n": 30,
    "solve_rate": 0.6,
    "mean_mapcc": 0.7857922772285585,
    "mean_r1": 0.3938575779954262
  },
  "large_3500_8000": {
    "n": 5,
    "solve_rate": 0.0,
    "mean_mapcc": 0.6139811218730262,
    "mean_r1": 0.563236573597816
  }
}
```

## References

1. Carrozzini et al. (2025). J. Appl. Cryst. **58**, 1859–1869. DOI: 10.1107/S1600576725008271
2. Larsen et al. (2024). Science **385**, 522–528 (PhAI).

Generated rows: 35. JSON: `ai_phaseed_extended_benchmark.json`.
