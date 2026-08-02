# GraphPhaseNet v9 strong prior

**Scale tag:** `v9_quick` · **N train:** 80
**Features:** v9 d_in=26 · large-cell / HA multipath · Melgalvis gen
**Preset:** `large` · low-res frac=0.2

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **17.7%** |
| seedOK rate (≥30% of strong ≤20°) | **0.0%** |
| mean strong MPE OI | 66.3° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | no / comparable |

Train wall time: 31s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- Practical hard path: `partial_phaseed` / fragment / HA (≥~30% strong ≤20°).
- Cluster scale: `python scripts/run_strong_prior_v9.py --scale-xl --melgalvis-preset large`
- Resume: `--continue-from data/processed/strong_prior_v9.npz`

Weights: `strong_prior_v9.npz`

## Stratified seed quality (v0.11)

| Cohort | n | frac≤20° | seedOK | strong MPE |
|--------|---|----------|--------|------------|
| `all` | 8 | **17.7%** | 0.0% | 66.3° |
| `ha_bearing_Zge17` | 2 | **17.5%** | 0.0% | 65.6° |
| `organic_light` | 6 | **17.8%** | 0.0% | 66.6° |
| `max_Z_ge19` | 2 | **17.5%** | 0.0% | 65.6° |
| `max_Z_lt19` | 6 | **17.8%** | 0.0% | 66.6° |
| `centrosymmetric` | 0 | — | — | — |
| `non_centrosymmetric_panel` | 8 | **17.7%** | 0.0% | 66.3° |

Stratified synthetic hold-out (GraPhAI-style reporting). Not a claim of published GraPhAI COD/Z≥19 success rates.


## Volume-band stratification

| Band | n | frac≤20° | seedOK | strong MPE |
|------|---|----------|--------|------------|
| `Vol < 1000` | 6 | **18.7%** | 0.0% | 64.2° |
| `Vol 1000–3500` | 2 | **14.9%** | 0.0% | 72.9° |
| `Vol > 3500` | 0 | — | — | — |