# Strong phase prior (GraphPhaseNet) — Lane A v4 scale-xl

**Lane A result (honest):** scaling + residual GNN + Adam **does not** clear the
hard-cliff seed bar on mean metrics. Mean hold-out **frac ≤20° remains ~21%**
(same as v3). Occasional individual cells hit ≥30% (would_seed_solve 5–12%).
Strict ab initio hard solves remain **0%**.

Oracle bar (partial-φ): ≥**30%** of top-|E| phases within **20°** → AI-PhaSeed
can strict-solve. See [`docs/math/partial_seed.md`](../../docs/math/partial_seed.md).

## Scoreboard (hard hold-out)

| Version | N train | Arch | strong MPE | frac≤20° | seedOK rate | +PhaSeed mapCC | Strict |
|---------|---------|------|------------|----------|-------------|----------------|--------|
| v3 A+B | 250 | H128 L3 | ~59° | **~21%** | ~5–12% | ~0.50 | 0% |
| **v4 XL** (default weights) | **1200** | H192 L4 residual Adam d_in=10 | **60°** | **21%** | **5%** (1/12) | **0.51** | **0%** |
| v4 XL→hard seed-focus FT | +600 | same | 63° | 21% | **12%** (1/12 eval; 12% hold meta) | 0.50 | 0% |

Default weights: `data/processed/strong_prior.npz` (= v4 XL).  
Checkpoints: `strong_prior_v4_xl.npz`, `strong_prior_v4_ft.npz`.

## v4 XL config

- Structures: **1200** Wilson-matched (bridge + hard multi-SG)
- Hidden=**192**, layers=**4**, residual=**True**, d_in=**10** (deg + E² feats)
- Adam, hard_oversample=1.4, max_refl=140, 4 global passes
- Train ~31 min on laptop CPU

## Strong-seed bar (v4 XL hold-out hard, n=12)

| Metric | Graph prior | Graph+PhaSeed |
|--------|-------------|---------------|
| strong MPE_OI | ~59° | — |
| frac ≤20° (top 30% \|E\|) | **22%** | ~21% |
| would_seed_solve (≥30% within 20°) | **1/12** | 0/12 |

## Hold-out hard-region comparison (v4 XL)

| Method | Solved | Rate | mean mapCC |
|--------|--------|------|------------|
| Graph prior only | — | — | ~0.46 (full MPE ~72°) |
| **Graph + AI-PhaSeed** | 0 | 0% | **~0.51** |
| CF | 0 | 0% | ~0.50 |
| hard_p1 PhaseMLP + PhaSeed | 0 | 0% | ~0.52 |

## Interpretation (Lane A)

1. **5× data + residual + Adam + richer features** did **not** move mean
   frac≤20° off the ~21% plateau (v3 → v4).
2. **Tail** improves slightly: more frequent individual seedOK cells during train
   (32–42% on some packs); hold-out seedOK rate up to ~12% after hard FT.
3. **Strict hard solves** still require **partial φ** (oracle ≥30%/20°) or
   external classical seed — not pure GraphPhaseNet scale alone.
4. Product path unchanged: **easy → ensemble**; **hard → partial_phaseed**.

## Reproduce

```bash
# Full Lane A XL train (~30 min CPU)
python scripts/train_strong_prior.py --scale-xl --wilson-match

# Optional hard seed-focus fine-tune from XL checkpoint
python scripts/train_strong_prior.py \
  --continue-from data/processed/strong_prior_v4_xl.npz \
  --wilson-match --hard-only --seed-focus \
  --n-structures 600 --epochs-per 12 --n-passes 3
```

## Notes

- **Wilson:** `--wilson-match` aligns synthetic |F| to experimental template.
- **Seed loss:** E^p weights, top-|E| boost, within-20° reweight.
- Not a general experimental multi-SG production solver.
