# GraphPhaseNet v8 strong prior

**Scale tag:** `v8_quick` · **N train:** 80
**Features:** v8 d_in=22 · GraPhAI multipath · Melgalvis gen · κ-gated edges
**Preset:** `ha` · low-res frac=0.18

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **19.1%** |
| seedOK rate (≥30% of strong ≤20°) | **12.5%** |
| mean strong MPE OI | 63.1° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | no / comparable |

Train wall time: 16s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- Cluster scale: `python scripts/run_strong_prior_v8.py --scale-xl --melgalvis-preset cod`
- Resume: `--continue-from data/processed/strong_prior_v8.npz`

Weights: `strong_prior_v8.npz`

## Stratified seed quality (v0.10)

| Cohort | n | frac≤20° | seedOK | strong MPE |
|--------|---|----------|--------|------------|
| `all` | 8 | **19.1%** | 12.5% | 63.1° |
| `ha_bearing_Zge17` | 0 | — | — | — |
| `organic_light` | 8 | **19.1%** | 12.5% | 63.1° |
| `max_Z_ge19` | 0 | — | — | — |
| `max_Z_lt19` | 8 | **19.1%** | 12.5% | 63.1° |
| `centrosymmetric` | 0 | — | — | — |
| `non_centrosymmetric_panel` | 8 | **19.1%** | 12.5% | 63.1° |

Stratified synthetic hold-out (GraPhAI-style reporting). Not a claim of published GraPhAI COD/Z≥19 success rates.
