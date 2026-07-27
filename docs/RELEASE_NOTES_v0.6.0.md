# v0.6.0 — grok-phase-solver

**GraphPhaseNet v5, trainable diffusion score, density viewer, MERGE helpers.** MIT.

## Highlights

### Seed quality / GraphPhaseNet v5
- Richer diffraction-graph node features (d_in=14) and κ-gated edges
- Melgalvis + Wilson-match training CLI: `scripts/run_strong_prior_v5.py`
- Hold-out metrics documented in `data/processed/strong_prior_v5.md`
- **Does not claim** clearing the 30% strong-phase ≤20° oracle bar on current pilots

### Diffusion generative path
- Trainable `PhaseScoreNet` + physics Langevin hybrid (`diffusion_hybrid_v2`)
- Train: `python scripts/train_diffusion_score.py`
- Pure physics fallback if no weights

### Scientist path
- Streamlit multi-plane slices + optional plotly 3D volume HTML
- `merge_symmetry_equivalents` MERGE helper
- Ensemble threaded multistart (`n_jobs`)
- Dockerfile

## Remaining honest limits

- Hard synthetic cells: pure ab initio strict solves still rare
- Partial-φ / fragment / HA remains the production hard path
- No PhAI / SHELX binaries redistributed
- Full 5k–10k prior retrain: use `--scale-xl` on a cluster

## Install

```bash
python -m pip install -U grok-phase-solver
# or from source
pip install -e ".[gui]"
```

## Links

| | |
|--|--|
| Repo | https://github.com/pileofflapjacks1/grok_phase_solver |
| Math | docs/math/graph_phase_net_v5.md · docs/math/diffusion_phase.md |
