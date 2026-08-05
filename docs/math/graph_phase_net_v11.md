# GraphPhaseNet v11 (v0.13)

v11 uses **d_in=34** (feature_version=10): v10 multipath depth plus intensity
moments and centrosymmetric/HA cues aligned with GraPhAI reporting.

## Features

| Block | d | New cues |
|-------|---|---------|
| …v9 | 30 | hop3, multipath span, Wilson B, E-outlier |
| **v10** | **+4 → 34** | **⟨E⁴⟩ moment, shell skew, deg×E, centro intensity cue** |

Edges: κ power ≥1.70, self-loop ≥0.15, multipath E-boost exponent 1/4.8.
Training bin CE weight default **0.26**.

## Train

```bash
python scripts/run_strong_prior_v11.py --quick --melgalvis-preset large
python scripts/run_strong_prior_v11.py --pilot --melgalvis-preset ha
python scripts/run_strong_prior_v11.py --scale-xl --melgalvis-preset xdxd \
  --continue-from data/processed/strong_prior_v11.npz
```

## Honest limits

- Target: raise frac≤20° toward the **30%** practical bar; **do not claim**
  the bar is cleared unless the committed scoreboard says YES.
- GraPhAI official weights remain external (`third_party/graphai/README.md`).
