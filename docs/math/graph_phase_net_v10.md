# GraphPhaseNet v10 (v0.12)

v10 extends the residual GNN + κ-gated multipath stack with **d_in=30** features
(feature_version=9) and stronger multipath edges for large-cell / HA curricula.

## Features

| Block | d | Cues |
|-------|---|------|
| v4–v8 | 26 | … through log Vol, shell E std, κ×E, low-res rank |
| **v9** | **+4 → 30** | **hop3 local E, multipath span, Wilson B proxy, E-outlier ratio** |

Edges: κ power ≥1.65, self-loop ≥0.14, multipath E-boost exponent 1/5.
Training: Carrozzini-style bin CE weight default **0.24** on `run_strong_prior_v10.py`.

## Train

```bash
# laptop smoke
python scripts/run_strong_prior_v10.py --quick --melgalvis-preset large

# pilot
python scripts/run_strong_prior_v10.py --pilot --melgalvis-preset ha

# cluster
python scripts/run_strong_prior_v10.py --scale-xl --melgalvis-preset large \
  --continue-from data/processed/strong_prior_v10.npz
```

## Reporting

Hold-out: frac≤20°, seedOK, strong MPE, strict solve rate; stratified by HA/Z,
centrosymmetric panel (P−1 mix), and Vol bands.

## Honest limits

- Laptop pilots are expected near the **~18–26%** frac≤20° band.
- **30% oracle bar is not claimed** unless the committed scoreboard says YES.
- Partial-φ / fragment / HA remains the practical hard path.
- Official GraPhAI weights are external only (`graphai_external.md`).
