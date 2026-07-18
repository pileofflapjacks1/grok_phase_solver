# Strong phase prior — Melgalvis generator (v0.3 pilot)

Trained GraphPhaseNet with **Melgalvis & Rekis (2026)** style on-the-fly synthetic
structures (`--use-melgalvis-gen --melgalvis-mode hybrid`) + Wilson match.

## Config (this run)

- Structures: **120** · packs: 120
- Hidden=128, layers=3, residual=True
- wilson_match=**True**, scale=**v4_scale_seed_melg**
- use_melgalvis_gen=**True**, mode=**hybrid**

## Hold-out strong-seed metrics

| Metric | Melg pilot (N=120) | v4 XL legacy synth (N=1200) | Target bar |
|--------|--------------------|-----------------------------|------------|
| strong MPE (meta hold) | 63.5° | ~60° | ↓ |
| frac ≤20° | **22%** | **~21%** | ≥30% |
| would_seed_solve rate | 0% | 5–12% | ↑ |
| mean mapCC prior | 0.489 | ~0.46 | ↑ |

## Solve panel (8 hard cells, graph+PhaSeed)

From training script eval: **0/8 strict** solved; mean +PS mapCC ≈ 0.52 (vs CF ≈ 0.47).

## Honest read

- Melgalvis generation is **wired and training-stable**; pilot seed frac≤20° ≈ **22%**
  is **comparable** to legacy v4 XL (~21%), not yet past the 30% bar.
- Larger N (10³) + cluster-only curriculum + longer retrain is the natural next scale step
  (same as Melgalvis: on-the-fly data + recycling helps more at scale).
- Weights: `data/processed/strong_prior_melg.npz` (pilot; default production prior remains
  `strong_prior.npz` until a fuller retrain clears or matches XL mapCC).

## Reproduce

```bash
python scripts/train_strong_prior.py --use-melgalvis-gen --wilson-match --scale-xl \
  --out data/processed/strong_prior_melg.npz
```

Cite: Melgalvis & Rekis, Acta Cryst. A 82, 32–40 (2026). Math: `docs/math/synthetic_melgalvis.md`.
