# v0.7.0 — grok-phase-solver

**Melgalvis curriculum, GraPhAI-style edges, AI-PhaSeed seed heuristics, protein-aware DM.** MIT.

## Install

```bash
pip install -U grok-phase-solver==0.7.0
# Anaconda / no bare pip:
#   /path/to/python -m pip install -U grok-phase-solver==0.7.0
```

- **PyPI:** https://pypi.org/project/grok-phase-solver/0.7.0/
- **Tag:** `v0.7.0` (package release commit)
- **Source:** https://github.com/pileofflapjacks1/grok_phase_solver

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

## After this tag (on `main`, not in the 0.7.0 sdist)

Post-release work landed on `main` after the tag (will ship in a future 0.7.1+):

- Experimental COD hard-path validation (`scripts/run_cod_hard_path_validation.py`)
- Fragment / predicted-model seeding: full Fcalc soft prior + SG expand so
  `fragment_half` approaches oracle `partial_30` mapCC on COD Fobs

Use `main` or wait for a patch release for those improvements.
