# v0.8.0 — grok-phase-solver

**GraphPhaseNet v6 (GraPhAI HA features), Carrozzini-style seed classifier,
diffusion research path, hard-path polish.** MIT.

## Install

```bash
pip install -U grok-phase-solver==0.8.0
# or from source
pip install -e .
```

- **PyPI:** https://pypi.org/project/grok-phase-solver/ (when published)
- **Source:** https://github.com/pileofflapjacks1/grok_phase_solver

## Scientific / engineering gains

| Area | Change |
|------|--------|
| Graph prior | **v6** d_in=18 HA/low-res features + stronger κ-gated edges; `run_strong_prior_v6.py` scale path |
| Seed quality | Trainable Class 0/1 classifier (NumPy logistic / sklearn RF) + feature importance; Carrozzini feature list |
| Partial-φ | Full Fcalc soft prior (from 0.7.1), richer `recommend_seed_fraction` |
| Diffusion | Extended score features; SE(3) research helpers; trial completion research mode |
| SG / I/O | Common space-group aliases (P21/c, Pbca, P212121, …) |
| Hard path | COD fragment path remains the practical route when ab initio fails |

## Honest limits

- GraphPhaseNet hold-out frac≤20° on laptop pilots may stay near mid-20% or
  below on hard P1 panels — **not** a claim of ≥30% seed bar unless the
  scoreboard reports it.
- Hard ab initio strict solves without partial information stay rare.
- Seed RF is **synthetic/oracle-labeled**, not the published 1505-COD forest.
- Diffusion / SE(3) / trial completion are **research-only**, off default auto.
- No PhAI / SHELX redistribution.

## How to test

```bash
pytest -q

# Graph prior pilot
python scripts/run_strong_prior_v6.py --quick --melgalvis-preset cod

# Seed classifier
python scripts/train_seed_quality_rf.py --n 200

# Hard path COD
python scripts/run_cod_hard_path_validation.py

gps-solve --help
gps-gui   # optional .[gui]
```

## Cluster scale

```bash
python scripts/run_strong_prior_v6.py --scale-xl --melgalvis-preset cod
```
