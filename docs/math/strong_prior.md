# Strong phase prior (GraphPhaseNet)

## Motivation

Per-reflection PhaseMLP (hard-P1 prior) cannot use **direct-methods geometry**:
strong reflections are coupled by Cochran triplets
$\varphi_h + \varphi_k \approx \varphi_{h+k}$. A graph net with those edges is a
principled upgrade for an ab initio **seed prior**.

## Architecture (v4)

- **Nodes:** strong reflections (top $|E|$, up to $M \approx 120$–140)
- **Edges:** undirected pairs from enumerated triplets
- **Message passing:** $L$ residual layers with row-normalized weighted adjacency
  ($\mathrm{agg} = \hat{A} h$), vectorized NumPy + **Adam**
- **Features ($d_{\mathrm{in}}=10$):** $E$, $s$, $s^2$, $|h|$, amp, Miller
  fractions, **triplet degree**, **$E^2$**
- **Output:** $(\cos\varphi, \sin\varphi)$ per node
- **Weak reflections:** nearest strong reflection in Miller index space
- **Training losses:**
  1. Origin/enantiomorph-invariant MSE on $(\cos,\sin)$
  2. Unit-norm penalty on output vectors
  3. **Triplet auxiliary:** MSE on $\cos(\varphi_i+\varphi_j-\varphi_k)$ vs true invariant
  4. Strong-|E| reweight + within-20° boost
- **Curriculum:** bridge cells first, then hard; multi-pass with LR anneal;
  optional hard oversample / seed-focus fine-tune
- **Inference:** free-FOM origin search → AI-PhaSeed

Code: `models/graph_phase_net.py`, `models/strong_prior.py`

## Scale recipes

```bash
# v3-scale (250 structs)
python scripts/train_strong_prior.py --scale --wilson-match

# Lane A XL (~1200 structs, H=192, L=4 residual+Adam)
python scripts/train_strong_prior.py --scale-xl --wilson-match

# Fine-tune hard-only + seed-set loss focus
python scripts/train_strong_prior.py \
  --continue-from data/processed/strong_prior.npz \
  --wilson-match --hard-only --seed-focus \
  --n-structures 600 --epochs-per 12 --n-passes 3
```

| Lever | v1 | v2 | v3 A+B | **v4 XL (Lane A)** |
|-------|----|----|--------|---------------------|
| Structures | 50 | 250 | 250 | **1200** |
| Hidden / layers | 80/2 | 128/3 | 128/3 | **192/4 residual** |
| Optimizer | SGD | SGD | SGD | **Adam** |
| d_in | 8 | 8 | 8 | **10** |
| Wilson match | no | no | **yes** | **yes** |
| Loss focus | uniform | amp + triplet | E² + top + >20° | same + hard OS |
| Primary metric | full MPE | full MPE | **strong MPE + frac≤20°** | same |

## Strong-seed bar

Oracle partial-φ: hard cells **strict-solve** when ≥30% of top-|E| phases are
within **20°** of truth. Metrics in `metrics/strong_seed.py`:

- `strong_mpe_oi` — OI MAPE on top 30% \|E\|
- `frac_within_deg` — weighted fraction ≤20°
- `would_seed_solve` — `frac ≥ 0.30`

## Empirical (Lane A)

| Method | mean mapCC | strong MPE | frac≤20° | Strict solved |
|--------|------------|------------|----------|---------------|
| Graph prior v4 XL | ~0.46 | **~60°** | **~21%** | — |
| Graph + AI-PhaSeed | **~0.51** | — | ~21% | **0/12** |
| hard_p1 + PhaSeed | ~0.52 | — | — | 0/12 |
| CF | ~0.50 | — | — | 0/12 |

**Honest read (Lane A conclusion):**

- 5× data + residual GNN + Adam **does not** lift mean frac≤20° above the
  **~21% plateau** established at v3 (still well below the 30% oracle bar).
- Individual cells can hit seedOK (train packs at 32–42%; hold-out seedOK rate
  5–12% after hard fine-tune) but **mean** seed quality is insufficient for
  reliable pure ab initio hard solves.
- **Open path for hard data remains partial φ / fragment seed**, not more
  pure-prior scale alone at this capacity class.

Scoreboard: [`data/processed/strong_prior.md`](../../data/processed/strong_prior.md)

## Usage

```bash
python scripts/train_strong_prior.py --scale-xl --wilson-match
# → data/processed/strong_prior.{npz,json,md}

gps-solve --hkl data.hkl --ins data.ins --method strong_prior_phaseed --out out/
```

```python
from grok_phase_solver.models.strong_prior import strong_prior_phaseed_solve
phases, rho, info = strong_prior_phaseed_solve(hkl, amp, cell)
```

## Next upgrades (if still pursuing pure ab initio)

1. 10⁴-scale synthetic multi-SG (overnight / shard pipeline)
2. Torch GNN + equivariant layers when stack is stable
3. Patterson voxel / DM tangent hybrid as node features or post-process
4. Fragment / HA-conditioned priors (merge with partial-φ product path)
5. Accept seed bar ceiling for |F|-only GNN and invest in partial-φ UX
