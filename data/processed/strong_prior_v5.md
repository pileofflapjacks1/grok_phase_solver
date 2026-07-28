# GraphPhaseNet v5 strong prior

**Scale tag:** `v5_pilot` · **N train:** 160
**Features:** v5 d_in=14 · Melgalvis gen · Wilson match · κ-gated edges

## Hold-out seed quality

| Metric | Value |
|--------|-------|
| mean frac ≤20° | **22.1%** |
| seedOK rate (≥30% of strong ≤20°) | **0.0%** |
| mean strong MPE OI | 62.5° |
| strict solve rate (PhaSeed polish none) | 0.0% |
| clears 30% oracle bar? | **NO** |
| above legacy ~22% plateau? | yes |

Train wall time: 160s

## Honest limits

- Hard ab initio strict solves remain rare without partial-φ.
- Numbers are synthetic hold-out; experimental COD may differ.
- **Hard curriculum** (`--melgalvis-preset hard --low-res-frac 0.2`) can *lower*
  hold-out frac≤20° on the fixed legacy hard-P1 panel while improving
  in-domain hard/low-res training loss — domain shift, not a bug.
- COD-like preset (N≈160, v0.7): mean frac≤20° ≈**22%** (comparable to legacy
  plateau; 30% bar not cleared).
- For full 5k–10k runs: `python scripts/run_strong_prior_v5.py --scale-xl --melgalvis-preset cod`

Weights: `strong_prior_v5.npz`
