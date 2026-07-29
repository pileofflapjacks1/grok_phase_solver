# GraphPhaseNet v6 strong prior

**Scale tag:** `v6_pilot` · **N train:** 200
**Features:** v6 d_in=18 · GraPhAI HA cues · Melgalvis gen · κ-gated edges
**Preset:** `cod` · low-res frac=0.15

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **24.1%** |
| seedOK rate (≥30% of strong ≤20°) | **20.0%** |
| mean strong MPE OI | 58.3° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | yes |

Train wall time: 375s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- Cluster scale: `python scripts/run_strong_prior_v6.py --scale-xl --melgalvis-preset cod`
- Resume: `--continue-from data/processed/strong_prior_v6.npz`

Weights: `strong_prior_v6.npz`
