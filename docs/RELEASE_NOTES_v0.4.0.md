# v0.4.0 — grok-phase-solver

**Carrozzini 2025 AI-PhaSeed alignment** — hybrid DM+AI tangent, seed-quality
Class 0/1 diagnostics, low-res EDM path, expanded docs. MIT.

## Highlights

- `dm_ai_weight` / `--ai-dm-hybrid`: modified tangent with AI a priori phases
- `metrics/seed_quality.py`: Class 0/1 heuristic (+ optional sklearn RF extra)
- `--low-res-path`, `--prior-weight`, `--seed-quality-filter`
- GUI seed-quality panel; report.md Class diagnostics
- `scripts/run_ai_phaseed_extended_benchmark.py` stratified subset harness
- Docs: `docs/math/ai_phaseed.md`, FOR_REVIEWERS C11, references.bib

## Honest limits (unchanged)

- Hard ab initio seed bar still ~21–22% ≤20° on strong |E|
- Partial-φ remains the hard-data path
- Seed Class predictor is operational UX, not the published 1505-COD RF
- Not a general protein ab initio solver

## Install

```bash
python -m pip install -U grok-phase-solver
# optional RF extras:
python -m pip install "grok-phase-solver[seed-quality]"
```

## Links

| | |
|--|--|
| **Repo** | https://github.com/pileofflapjacks1/grok_phase_solver |
| **PyPI** | https://pypi.org/project/grok-phase-solver/ |
| **Paper PDF** | docs/paper/arxiv_draft.pdf |
| **Math** | docs/math/ai_phaseed.md |
