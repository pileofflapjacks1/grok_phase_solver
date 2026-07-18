# Strong phase prior (GraphPhaseNet) — summary

## Default weights

`data/processed/strong_prior.npz` — **legacy v4 XL** (1200 random-organic synth).

| Run | N | Generator | frac≤20° | seedOK rate | Strict hard |
|-----|---|-----------|----------|-------------|-------------|
| v4 XL (default) | 1200 | legacy random | ~21% | 5–12% | 0% |
| Melg pilot | 120 | Melgalvis hybrid | ~22% | 0–12% | 0% |
| **Melg XL** | **1200** | **Melgalvis hybrid** | **22%** | **12%** | **0%** |
| Target bar | — | — | ≥30% | — | >0% |

Melg XL details: [`strong_prior_melg_xl.md`](strong_prior_melg_xl.md) · weights `strong_prior_melg_xl.npz`.

## Strong-seed bar

Oracle partial-φ: ≥**30%** of top-|E| phases within **20°** → AI-PhaSeed can strict-solve hard cells.
Melgalvis XL improves infrastructure and slightly raises seedOK rate; **does not clear the bar**.

## Reproduce

```bash
# Legacy XL
python scripts/train_strong_prior.py --scale-xl --wilson-match

# Melgalvis XL (~30 min)
python scripts/train_strong_prior.py --scale-xl --use-melgalvis-gen --wilson-match \
  --out data/processed/strong_prior_melg_xl.npz
```
