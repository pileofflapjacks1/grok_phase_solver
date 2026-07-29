# Seed quality RF (v0.8)

Carrozzini-aligned Class 0/1 features; **synthetic oracle labels**.

| Metric | Value |
|--------|-------|
| n_total | 200 |
| test accuracy | **78.0%** |
| ROC-AUC | nan |
| Class 1 rate (train set) | 72.0% |
| bar frac≤20° | 0.28 |

## Top feature importance

- `free_fom_composite`: 0.421
- `excess_kurtosis`: 0.220
- `n_seed`: 0.125
- `d_min`: 0.051
- `max_W`: 0.050
- `R_pos`: 0.033
- `median_E_seed`: 0.027
- `N_asym`: 0.021

## Honest limits

- Not the published RF trained on 1505 COD structures.
- Heuristic predictor remains the default fallback without sklearn/joblib.
- Bundle: `seed_quality_rf.npz`
