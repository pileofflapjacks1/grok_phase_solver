# Strong phase prior (GraphPhaseNet)

## Motivation

Per-reflection PhaseMLP (hard-P1 prior) cannot use **direct-methods geometry**:
strong reflections are coupled by Cochran triplets
$\varphi_h + \varphi_k \approx \varphi_{h+k}$. A graph net with those edges is a
principled upgrade for an ab initio **seed prior**.

## Architecture

- **Nodes:** strong reflections (top $|E|$, up to $M \approx 100$–120)
- **Edges:** undirected pairs from enumerated triplets
- **Message passing:** $L$ layers with row-normalized weighted adjacency
  ($\mathrm{agg} = \hat{A} h$), vectorized NumPy
- **Output:** $(\cos\varphi, \sin\varphi)$ per node
- **Weak reflections:** nearest strong reflection in Miller index space
- **Training losses:**
  1. Origin/enantiomorph-invariant MSE on $(\cos,\sin)$
  2. Unit-norm penalty on output vectors
  3. **Triplet auxiliary:** MSE on $\cos(\varphi_i+\varphi_j-\varphi_k)$ vs true invariant
- **Curriculum:** bridge cells first, then hard; multi-pass SGD with LR anneal
- **Inference:** free-FOM origin search → AI-PhaSeed

Code: `models/graph_phase_net.py`, `models/strong_prior.py`

## Scale recipe (v3 = A+B)

```bash
# A: Wilson-matched |F|  +  B: strong-seed loss retarget
python scripts/train_strong_prior.py --scale --wilson-match
# 250 structs · H=128 · L=3 · E² weights · within-20° boost · Wilson template
# → data/processed/strong_prior.{npz,json,md}
```

| Lever | v1 | v2 scale | **v3 A+B** |
|-------|----|----------|------------|
| Structures | 50 | 250 | 250 |
| Hidden / layers | 80/2 | 128/3 | 128/3 |
| Wilson match | no | no | **yes** |
| Loss focus | uniform amp | amp + triplet | **E² + top boost + >20° boost** |
| Primary metric | full MPE | full MPE | **strong MPE + frac≤20°** |

## Strong-seed bar

Oracle partial-φ: hard cells **strict-solve** when ≥30% of top-|E| phases are
within **20°** of truth. Metrics in `metrics/strong_seed.py`:

- `strong_mpe_oi` — OI MAPE on top 30% \|E\|
- `frac_within_deg` — weighted fraction ≤20°
- `would_seed_solve` — `frac ≥ 0.30`

## Empirical (v3 A+B, this repo)

| Method | mean mapCC | strong MPE | frac≤20° | Strict solved |
|--------|------------|------------|----------|---------------|
| Graph prior | ~0.46 | **~59°** | **~21%** | — |
| Graph + AI-PhaSeed | **~0.50** | — | ~21% | 0/8 |
| hard_p1 + PhaSeed | ~0.51 | — | — | 0/8 |
| CF | ~0.48 | — | — | 0/8 |

**Honest read:**

- Strong-subset MPE (~59°) is better than full-map (~71°); frac≤20° (~21%) is
  **below the 30% oracle bar** — still not seed-ready for hard solves.
- Wilson match + retarget improves the *objective* and reporting; mapCC remains
  comparable to CF/hP1. Need more capacity/data or better architecture to hit 30%/20°.
- Internal train occasionally hits `seedOK=True` on individual cells (frac≤20°≥30%).

## Usage

```bash
python scripts/train_strong_prior.py --scale
# → data/processed/strong_prior.{npz,json,md}

gps-solve --hkl data.hkl --ins data.ins --method strong_prior_phaseed --out out/
```

```python
from grok_phase_solver.models.strong_prior import strong_prior_phaseed_solve
phases, rho, info = strong_prior_phaseed_solve(hkl, amp, cell)
```

## Next upgrades (if pursuing this line)

1. 10³–10⁴ synthetic multi-SG cells (HDF5 shards; overnight job)
2. Optional torch GNN when NumPy/torch versions align
3. Patterson voxel channels as node features
4. Partial-phase / fragment-conditioned seeding (hybrid with classical DM)
5. External SHELXD head-to-head on experimental HKL
