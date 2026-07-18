# Strong phase prior — Melgalvis XL (v0.3)

**Long retrain** of GraphPhaseNet on Melgalvis & Rekis (2026) style synthetics.

## Config

- Structures: **1200** (packs 1200), Wilson-matched **1200**
- Hidden=192, layers=4, residual=True, d_in=10
- max_refl=140, passes=4, epochs/struct=10
- optimizer=adam, hard_oversample=1.4
- **use_melgalvis_gen=True**, mode=**hybrid**
- scale tag: `v4_scale_xl_melg`
- Weights: `data/processed/strong_prior_melg_xl.npz`

## Hold-out hard metrics (meta)

| Metric | Melg XL (N=1200) | Melg pilot (N=120) | Legacy v4 XL (N=1200) | Bar |
|--------|------------------|--------------------|------------------------|-----|
| strong MPE | **60.8°** | ~63.5° | ~60° | ↓ |
| frac ≤20° | **22%** | ~22% | ~21% | ≥30% |
| would_seed_solve | **12%** | 0–12% | 5–12% | ↑ |
| mapCC prior | **0.491** | ~0.49 | ~0.46 | ↑ |

## Solve panel (12 hard cells)

From train script: **0/12 strict** solved · mean Graph+PhaSeed mapCC **≈0.53** · CF ≈0.50 · hard_p1 ≈0.52.

Per-case seedOK (prior): **2/12** cells with frac20 ≥30% (e.g. 34%, 33%).

## Honest read

1. Full Melgalvis-scale training is **stable** and matches or slightly beats legacy XL on mean seed fraction (~**22%** vs ~21%) and seedOK rate (~**12%**).
2. Still **below the 30% oracle bar** — pure ab initio hard strict solves remain **0%**.
3. Individual hard cells can hit seedOK; extension mapCC is competitive with CF/hP1 but not transformative.
4. Default production weights stay **`strong_prior.npz`** (legacy v4 XL) unless you promote melg_xl after further eval.

## Reproduce

```bash
python scripts/train_strong_prior.py --scale-xl --use-melgalvis-gen \
  --melgalvis-mode hybrid --wilson-match \
  --out data/processed/strong_prior_melg_xl.npz
# wall-clock ~30 min CPU on laptop
```

Cite: Melgalvis & Rekis, Acta Cryst. A **82**, 32–40 (2026).  
Math: `docs/math/synthetic_melgalvis.md`.
