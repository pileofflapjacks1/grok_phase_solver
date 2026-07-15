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

## Scale recipe (v2)

```bash
python scripts/train_strong_prior.py --scale
# 250 structures · hidden=128 · layers=3 · 3 global passes · triplet_w≈0.18
# → data/processed/strong_prior.{npz,json,md}
```

| Lever | First pass | Scale v2 |
|-------|------------|----------|
| Structures | 50 | **250** |
| Hidden / layers | 80 / 2 | **128 / 3** |
| Strong reflections | 90 | **120** |
| Training schedule | 1 pass online | **curriculum multi-pass** |
| Triplet aux | no | **yes** |
| Aggregation | Python loops | **dense $\hat{A}$ matmul** |

## Empirical (scale v2, this repo)

| Method | mean mapCC (hold-out hard) | Strict solved |
|--------|----------------------------|---------------|
| Graph prior only | ~0.46 | — |
| **Graph + AI-PhaSeed** | **~0.51** | 0/8 |
| hard_p1 MLP + PhaSeed | ~0.51 | 0/8 |
| CF | ~0.48 | 0/8 |

**Honest read:**

- Scale **matched hard-P1** and **beat CF by ~0.03 mapCC** on the same hard hold-out.
- Still **0% strict success** under `SuccessThresholds` — the B/C failure cliff is not
  closed by data/capacity alone at this regime.
- Train MPE_OI on strong nodes is often ~55–65°, but full-map OI MPE stays ~70°
  (weak reflections + origin residual).
- Oracle/partial AI-PhaSeed seeds still prove the *extension* path works when the
  seed is good enough; the bottleneck remains prior quality on hard cells.

First-pass baseline (50 structs / H=80 / L=2) for comparison: Graph+PhaSeed mapCC
~0.48, CF ~0.47, hard-P1 ~0.51, 0/6 strict.

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
