# Strong phase prior (GraphPhaseNet)

## Motivation

Per-reflection PhaseMLP (hard-P1 prior) cannot use **direct-methods geometry**:
strong reflections are coupled by Cochran triplets
\(\varphi_h + \varphi_k \approx \varphi_{h+k}\). A graph net with those edges is a
principled upgrade for an ab initio **seed prior**.

## Architecture

- **Nodes:** strong reflections (top \(|E|\), up to \(M \approx 90\)–120)
- **Edges:** undirected pairs from enumerated triplets
- **Message passing:** 2 layers, κ/|EEE|-weighted neighbor aggregation
- **Output:** \((\cos\varphi, \sin\varphi)\) per node
- **Weak reflections:** nearest strong reflection in Miller index space
- **Training:** origin/enantiomorph-invariant targets; multi-SG (P1 + P−1)
- **Inference:** free-FOM origin search → AI-PhaSeed

Code: `models/graph_phase_net.py`, `models/strong_prior.py`

## Empirical (50 structures, hidden 80, this repo)

| Method | mean mapCC (hold-out hard) | Strict solved |
|--------|----------------------------|---------------|
| Graph prior only | ~0.43 | — |
| Graph + AI-PhaSeed | ~0.48 | 0/6 |
| hard_p1 MLP + PhaSeed | ~0.51 | 0/6 |
| CF | ~0.47 | 0/6 |

**Honest read:** the graph architecture is correct and trainable; at this data/compute
budget it is **comparable** to the domain MLP, not yet a clear win on strict
success. Gains should come from **scale** (more structures/epochs), richer node
features (Patterson channels), or a larger/torch backbone — not from more classical
polish alone. Oracle/partial AI-PhaSeed seeds still prove the *extension* path works
when the prior is good enough.

## Usage

```bash
python scripts/train_strong_prior.py
# → data/processed/strong_prior.{npz,json,md}

gps-solve --hkl data.hkl --ins data.ins --method strong_prior_phaseed --out out/
```

```python
from grok_phase_solver.models.strong_prior import strong_prior_phaseed_solve
phases, rho, info = strong_prior_phaseed_solve(hkl, amp, cell)
```

## Next upgrades (if pursuing this line)

1. 10×–100× more synthetic multi-SG cells  
2. Triplet-consistency auxiliary loss on edges  
3. Torch GNN (PyG / DGL) for deeper models  
4. Condition on low-res Patterson voxel features  
