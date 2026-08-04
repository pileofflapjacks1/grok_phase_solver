# GraphPhaseNet v10 strong prior

**Scale tag:** `v10_quick` · **N train:** 80
**Features:** v10 d_in=30 · large-cell / HA multipath · Melgalvis gen
**Preset:** `large` · low-res frac=0.2

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **19.1%** |
| seedOK rate (≥30% of strong ≤20°) | **0.0%** |
| mean strong MPE OI | 68.9° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | no / comparable |

Train wall time: 33s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- Practical hard path: `partial_phaseed` / fragment / HA (≥~30% strong ≤20°).
- Cluster scale: `python scripts/run_strong_prior_v10.py --scale-xl --melgalvis-preset large`
- Resume: `--continue-from data/processed/strong_prior_v10.npz`

Weights: `strong_prior_v10.npz`

## Stratified seed quality (v0.11)

| Cohort | n | frac≤20° | seedOK | strong MPE |
|--------|---|----------|--------|------------|
| `all` | 8 | **19.1%** | 0.0% | 68.9° |
| `ha_bearing_Zge17` | 2 | **16.5%** | 0.0% | 76.4° |
| `organic_light` | 6 | **20.0%** | 0.0% | 66.4° |
| `max_Z_ge19` | 2 | **16.5%** | 0.0% | 76.4° |
| `max_Z_lt19` | 6 | **20.0%** | 0.0% | 66.4° |
| `centrosymmetric` | 1 | **20.2%** | 0.0% | 75.5° |
| `non_centrosymmetric_panel` | 7 | **18.9%** | 0.0% | 68.0° |

Stratified synthetic hold-out (GraPhAI-style reporting). Not a claim of published GraPhAI COD/Z≥19 success rates.


## Volume-band stratification

| Band | n | frac≤20° | seedOK | strong MPE |
|------|---|----------|--------|------------|
| `Vol < 1000` | 5 | **22.1%** | 0.0% | 64.5° |
| `Vol 1000–3500` | 3 | **14.0%** | 0.0% | 76.3° |
| `Vol > 3500` | 0 | — | — | — |