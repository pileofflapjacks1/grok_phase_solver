# GraphPhaseNet v7 strong prior

**Scale tag:** `v7_pilot` · **N train:** 180
**Features:** v7 d_in=22 · GraPhAI multipath · bin CE · Melgalvis · κ×E edges
**Preset:** `acta2026` · low-res frac=0.15

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **22.5%** |
| seedOK rate (≥30% of strong ≤20°) | **0.0%** |
| mean strong MPE OI | 60.6° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | yes |

Train wall time: 234s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- Cluster scale: `python scripts/run_strong_prior_v7.py --scale-xl --melgalvis-preset cod`
- Resume: `--continue-from data/processed/strong_prior_v7.npz`

Weights: `strong_prior_v7.npz`
