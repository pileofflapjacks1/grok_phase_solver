# GraphPhaseNet v5 strong prior

**Scale tag:** `v5_pilot` · **N train:** 200
**Features:** v5 d_in=14 · Melgalvis gen · Wilson match · κ-gated edges

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **24.3%** |
| seedOK rate (≥30% of strong ≤20°) | **30.0%** |
| mean strong MPE OI | 60.0° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | yes |

Train wall time: 127s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- For full 5k–10k runs: `python scripts/run_strong_prior_v5.py --scale-xl`

Weights: `strong_prior_v5.npz`
