# v0.7.0 — grok-phase-solver

**Melgalvis curriculum, GraPhAI-style edges, AI-PhaSeed seed heuristics, protein-aware DM.** MIT.

## Scientific / engineering gains

| Area | Change |
|------|--------|
| Synthetic data | COD-like + hard Melgalvis presets; HA injection; partial occ; low-res fraction |
| Graph prior | κ power-law edges + self-loops (v5.1); curriculum train flags |
| Partial-φ | Seed-fraction heuristic; multi-seed agreement boost; explicit 30% bar text |
| DM | Auto / protein-mode solvent fraction |
| Diffusion | SE(3) stub marked research-only (not default) |

## Honest limits

- GraphPhaseNet hold-out frac≤20° remains near mid-20% on laptop pilots — **not**
  a claim of ≥30% seed bar.
- Hard ab initio strict solves without partial information stay rare.
- No PhAI / SHELX redistribution.

## How to test the hard-region / graph-prior path

```bash
# Melgalvis curriculum pilot retrain
python scripts/run_strong_prior_v5.py --n-structures 200 \
  --melgalvis-preset hard --low-res-frac 0.2 \
  --out data/processed/strong_prior_v5.npz

# Hard path with predicted model
gps-solve --hkl data.hkl --ins data.ins \
  --predicted-model model.cif --method partial_phaseed -o ./out

# Partial-φ demo
python scripts/run_partial_seed_demo.py

pytest -q
```

## Cluster scale

```bash
python scripts/run_strong_prior_v5.py --scale-xl --melgalvis-preset cod
```
