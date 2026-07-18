# Strong phase prior (GraphPhaseNet) — Lane A v4 + Melgalvis pilot

## Default weights

`data/processed/strong_prior.npz` — **v4 XL** (1200 legacy synth, residual H=192 L=4).

Hold-out mean **frac ≤20° ≈ 21%** (below 30% oracle bar). Strict hard solves 0%.

## Melgalvis & Rekis (2026) pilot

See **`strong_prior_melg.md`**: training with `--use-melgalvis-gen` (N=120 pilot)
gives hold-out frac≤20° ≈ **22%**, mapCC ≈ 0.49, still **0%** hard strict solves.

Generator: `data/synthetic_melgalvis.py` · docs: `docs/math/synthetic_melgalvis.md`.

## Strong-seed bar (reminder)

Oracle partial-φ: ≥**30%** of top-|E| phases within **20°** → AI-PhaSeed can strict-solve hard cells.

| Version | N train | Generator | frac≤20° | Strict hard |
|---------|---------|-----------|----------|-------------|
| v3 A+B | 250 | legacy | ~21% | 0% |
| v4 XL | 1200 | legacy | **21%** | 0% |
| v0.3 melg pilot | 120 | Melgalvis hybrid | **~22%** | 0% |
| Target | — | — | ≥30% | >0% |

## Reproduce XL (legacy)

```bash
python scripts/train_strong_prior.py --scale-xl --wilson-match
```

## Reproduce Melgalvis pilot

```bash
python scripts/train_strong_prior.py --use-melgalvis-gen --wilson-match \
  --n-structures 120 --out data/processed/strong_prior_melg.npz
```
