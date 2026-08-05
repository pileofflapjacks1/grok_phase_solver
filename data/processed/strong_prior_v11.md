# GraphPhaseNet v11 strong prior

**Scale tag:** `v11_quick` · **N train:** 80
**Features:** v11 d_in=34 · GraPhAI moments + large/HA · Melgalvis gen
**Preset:** `large` · low-res frac=0.2

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **18.0%** |
| seedOK rate (≥30% of strong ≤20°) | **0.0%** |
| mean strong MPE OI | 73.7° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | no / comparable |

Train wall time: 30s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- Practical hard path: `partial_phaseed` / fragment / HA (≥~30% strong ≤20°).
- Cluster scale: `python scripts/run_strong_prior_v11.py --scale-xl --melgalvis-preset large`
- Resume: `--continue-from data/processed/strong_prior_v11.npz`

Weights: `strong_prior_v11.npz`

## Stratified seed quality (v0.11)

| Cohort | n | frac≤20° | seedOK | strong MPE |
|--------|---|----------|--------|------------|
| `all` | 8 | **18.0%** | 0.0% | 73.7° |
| `ha_bearing_Zge17` | 0 | — | — | — |
| `organic_light` | 8 | **18.0%** | 0.0% | 73.7° |
| `max_Z_ge19` | 0 | — | — | — |
| `max_Z_lt19` | 8 | **18.0%** | 0.0% | 73.7° |
| `centrosymmetric` | 1 | **20.0%** | 0.0% | 76.8° |
| `non_centrosymmetric_panel` | 7 | **17.7%** | 0.0% | 73.3° |

Stratified synthetic hold-out (GraPhAI-style reporting). Not a claim of published GraPhAI COD/Z≥19 success rates.


## Volume-band stratification

| Band | n | frac≤20° | seedOK | strong MPE |
|------|---|----------|--------|------------|
| `Vol < 1000` | 3 | **17.8%** | 0.0% | 75.1° |
| `Vol 1000–3500` | 5 | **18.1%** | 0.0% | 72.9° |
| `Vol > 3500` | 0 | — | — | — |